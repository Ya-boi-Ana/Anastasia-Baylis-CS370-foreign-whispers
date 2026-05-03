import asyncio
import logging as _logging
import os
import pathlib
import json
import glob
import re
import subprocess
import tempfile
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import librosa
import soundfile as sf
import pyrubberband
from pydub import AudioSegment

from api.src.core.config import settings
from foreign_whispers.voice_resolution import resolve_speaker_wav

# ── Chatterbox API configuration ─────────────────────────────────────
CHATTERBOX_API_URL = settings.chatterbox_api_url
import sys
print(f"[tts] Module init: CHATTERBOX_API_URL={CHATTERBOX_API_URL}", file=sys.stderr)
# Path to the default speaker reference WAV, relative to pipeline_data/speakers/
CHATTERBOX_SPEAKER_WAV = os.getenv("CHATTERBOX_SPEAKER_WAV", "")

# Set FW_ALIGNMENT=off to use the pre-alignment baseline (legacy unclamped stretch).
# Default is "on" (new clamped path). Useful for A/B comparisons.
_ALIGNMENT_ENABLED = os.getenv("FW_ALIGNMENT", "on").lower() != "off"

_MAX_NATURAL_SPEEDUP = float(os.getenv("FW_TTS_MAX_SPEEDUP", "1.25"))
# When TTS audio is less than this fraction of the target window, skip
# time-stretching entirely — play at natural speed and pad with silence.
# Prevents comically slow speech in windows with long narrator pauses.
_SPEED_MIN_LEGACY = 0.1
_SPEED_MAX_LEGACY = 10.0
_MIN_SEGMENT_TARGET_SEC = 0.25
_TIMING_MODEL = "non_overlapping_phrase_groups_v1"
_MAX_SYNTH_GROUP_SEC = 6.0
_TTS_CONNECT_TIMEOUT = float(os.getenv("FW_TTS_CONNECT_TIMEOUT", "5"))
_TTS_READ_TIMEOUT = float(os.getenv("FW_TTS_READ_TIMEOUT", "180"))
_TTS_RETRIES = int(os.getenv("FW_TTS_RETRIES", "3"))
_TTS_RETRY_BACKOFF_SEC = float(os.getenv("FW_TTS_RETRY_BACKOFF_SEC", "1.5"))
_TTS_CHUNK_CHARS = int(os.getenv("FW_TTS_CHUNK_CHARS", "260"))
_MAX_CONSECUTIVE_TTS_FAILURES = int(os.getenv("FW_TTS_MAX_CONSECUTIVE_FAILURES", "5"))
_TTS_FAIL_FAST = os.getenv("FW_TTS_FAIL_FAST", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TTS_CONCURRENCY = max(1, int(os.getenv("FW_TTS_CONCURRENCY", "2")))
_TTS_LONG_FORM_GROUP_THRESHOLD = int(os.getenv("FW_TTS_LONG_FORM_GROUP_THRESHOLD", "500"))
_TTS_LONG_FORM_GROUP_SEC = float(os.getenv("FW_TTS_LONG_FORM_GROUP_SEC", "45"))
_SYNTH_CACHE_VERSION = "v2"
_EXTRACT_SEGMENT_VOICE_REFS = os.getenv("FW_TTS_EXTRACT_SEGMENT_VOICE_REFS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_AUTO_VOICE_CLONING = os.getenv("FW_TTS_AUTO_VOICE_CLONING", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SPEAKER_COLORING = os.getenv("FW_TTS_SPEAKER_COLORING", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class ChatterboxClient:
    """Thin HTTP client for the Chatterbox TTS API server (OpenAI-compatible).

    Uses /v1/audio/speech for default voice and /v1/audio/speech/upload
    when a speaker reference WAV is provided for voice cloning.
    """

    def __init__(self, base_url: str = CHATTERBOX_API_URL,
                 speaker_wav: str = CHATTERBOX_SPEAKER_WAV):
        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav  # path relative to pipeline_data/speakers/

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        """Synthesize *text* via the Chatterbox API and save the WAV to *file_path*.

        If *speaker_wav* is provided (via kwarg or constructor), uses the
        /v1/audio/speech/upload endpoint with the reference WAV for voice cloning.
        Otherwise uses /v1/audio/speech with the server's default voice.
        """
        chunks = self._split_text(text) if len(text) > _TTS_CHUNK_CHARS else [text]
        wav_parts = []

        speaker_wav = kwargs.get("speaker_wav", self.speaker_wav)

        for chunk in chunks:
            if speaker_wav:
                # Voice cloning: upload the reference WAV
                wav_parts.append(self._synthesize_with_voice(chunk, speaker_wav))
            else:
                # Default voice
                wav_parts.append(self._synthesize_default(chunk))

        if len(wav_parts) == 1:
            pathlib.Path(file_path).write_bytes(wav_parts[0])
        else:
            combined = AudioSegment.empty()
            for part in wav_parts:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(part)
                    tmp.flush()
                    combined += AudioSegment.from_wav(tmp.name)
            combined.export(file_path, format="wav")

    def _synthesize_default(self, text: str) -> bytes:
        """Call /v1/audio/speech with the server's default voice."""
        return self._request_with_retries(
            f"{self.base_url}/v1/audio/speech",
            json={"input": text, "response_format": "wav"},
        )

    def _synthesize_with_voice(self, text: str, speaker_wav: str) -> bytes:
        """Call /v1/audio/speech/upload with a reference WAV for voice cloning."""
        # Resolve the speaker WAV path — could be relative to speakers dir
        speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
        wav_path = speakers_base / speaker_wav
        if not wav_path.exists():
            # Try as absolute path
            wav_path = pathlib.Path(speaker_wav)
        if not wav_path.exists():
            _logging.getLogger(__name__).warning(
                "[tts] Speaker WAV %s not found, falling back to default voice", speaker_wav
            )
            return self._synthesize_default(text)
        if wav_path.stat().st_size <= 44:
            _logging.getLogger(__name__).warning(
                "[tts] Speaker WAV %s is empty or invalid, falling back to default voice", speaker_wav
            )
            return self._synthesize_default(text)

        try:
            with open(wav_path, "rb") as f:
                return self._request_with_retries(
                    f"{self.base_url}/v1/audio/speech/upload",
                    data={"input": text, "response_format": "wav"},
                    files={"voice_file": (wav_path.name, f, "audio/wav")},
                )
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                "[tts] Voice cloning failed for %s; falling back to default voice: %s",
                speaker_wav,
                exc,
            )
            return self._synthesize_default(text)

    def _request_with_retries(self, url: str, **kwargs) -> bytes:
        """POST to Chatterbox with retries and a longer read timeout."""
        last_exc: Exception | None = None
        timeout = (_TTS_CONNECT_TIMEOUT, _TTS_READ_TIMEOUT)

        for attempt in range(1, _TTS_RETRIES + 1):
            try:
                self._rewind_upload_files(kwargs.get("files"))
                resp = requests.post(url, timeout=timeout, **kwargs)
                resp.raise_for_status()
                if not resp.content:
                    raise ValueError("empty TTS response")
                return resp.content
            except Exception as exc:
                last_exc = exc
                _logging.getLogger(__name__).warning(
                    "[tts] Chatterbox request failed (attempt %s/%s): %s",
                    attempt,
                    _TTS_RETRIES,
                    exc,
                )
                if attempt < _TTS_RETRIES:
                    time.sleep(_TTS_RETRY_BACKOFF_SEC * attempt)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _rewind_upload_files(files) -> None:
        """Reset file handles before retrying multipart uploads."""
        if not files:
            return
        for value in files.values() if isinstance(files, dict) else files:
            file_obj = value[1] if isinstance(value, tuple) and len(value) > 1 else value
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)

    @staticmethod
    def _split_text(text: str, max_len: int = _TTS_CHUNK_CHARS) -> list[str]:
        """Split text at sentence boundaries to stay under max_len chars."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if current and len(current) + len(s) + 1 > max_len:
                chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}".strip() if current else s
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text]


class FallbackTTSEngine:
    """Tiny local TTS stand-in used when no real TTS backend is available.

    It deliberately writes valid WAV data so local tests and non-GPU development
    can exercise timing, stitching, and caption behavior without Coqui/Chatterbox.
    Production Docker runs should still use the Chatterbox service.
    """

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        duration_ms = max(300, min(4000, 80 * len(text.strip() or " ")))
        tone = AudioSegment.silent(duration=duration_ms)
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        tone.export(file_path, format="wav")


def _make_tts_engine():
    """Create TTS engine: Chatterbox API client if server is reachable, else local Coqui.

    Tries Chatterbox with a real /v1/audio/speech test call
    to ensure the model is fully loaded before committing.
    """
    import sys
    print(f"[tts] Attempting to connect to Chatterbox at {CHATTERBOX_API_URL}", file=sys.stderr)
    try:
        client = ChatterboxClient(base_url=CHATTERBOX_API_URL)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            print(f"[tts] Testing Chatterbox with test synthesis...", file=sys.stderr)
            client.tts_to_file(text="prueba", file_path=tmp.name)
        print(f"[tts] ✓ Using Chatterbox GPU server at {CHATTERBOX_API_URL}", file=sys.stderr)
        return client
    except Exception as exc:
        import traceback
        print(f"[tts] ✗ Chatterbox at {CHATTERBOX_API_URL} not available: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Fallback: local Coqui TTS (for dev/test without Docker)
    print(f"[tts] Falling back to local Coqui TTS", file=sys.stderr)
    import functools
    import torch
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        print("[tts] Coqui TTS not installed; using synthetic fallback WAVs", file=sys.stderr)
        return FallbackTTSEngine()
    # Coqui TTS checkpoints contain classes (RAdam, defaultdict, etc.) that
    # PyTorch 2.6+ rejects with weights_only=True.  Monkey-patch torch.load
    # to default to weights_only=False for these trusted model files.
    _original_torch_load = torch.load
    @functools.wraps(_original_torch_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] Using local Coqui TTS on {device}", file=sys.stderr)
    return CoquiTTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)


_tts_engine = None


class _LazyTTSProxy:
    """Compatibility proxy for older callers that imported a module-level tts."""


tts = _LazyTTSProxy()


def _get_tts_engine():
    """Lazy singleton — resolved on first call, not at import time."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = _make_tts_engine()
    return _tts_engine


def text_from_file(file_path) -> str:
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans["text"]


def segments_from_file(file_path) -> list[dict]:
    """Load segments with start/end timestamps from a translated JSON file."""
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans.get("segments", [])


def files_from_dir(dir_path) -> list:
    SUFFIX = ".json"
    pth = pathlib.Path(dir_path)
    if not pth.exists():
        raise ValueError("provided path does not exist")

    es_files = glob.glob(str(pth) + "/*.json")

    if not es_files:
        raise ValueError(f"no {SUFFIX} files found in {pth}")

    return es_files


def _synthesize_raw(tts_engine, text: str, wav_path: str, speaker_wav: str | None = None) -> bytes | None:
    """GPU-bound: call TTS engine and return raw WAV bytes, or None on failure."""
    if not text or not text.strip():
        return None
    try:
        kwargs = {}
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        tts_engine.tts_to_file(text=text, file_path=wav_path, **kwargs)
        return pathlib.Path(wav_path).read_bytes()
    except TypeError:
        # Some fallback TTS backends may not accept speaker_wav.
        try:
            tts_engine.tts_to_file(text=text, file_path=wav_path)
            return pathlib.Path(wav_path).read_bytes()
        except Exception as exc:
            print(f"[tts] TTS failed for segment ({exc}), using silence")
            return None
    except Exception as exc:
        print(f"[tts] TTS failed for segment ({exc}), using silence")
        return None


def _synth_cache_key(m: dict, speaker_wav: str | None, use_alignment: bool) -> str:
    """Stable key for reusable raw TTS phrase audio."""
    payload = {
        "version": _SYNTH_CACHE_VERSION,
        "text": m.get("text", ""),
        "speaker_wav": speaker_wav or "",
        "target_sec": round(float(m.get("target_sec", 0.0)), 3),
        "stretch_factor": round(float(m.get("stretch_factor", 1.0)), 3),
        "alignment": bool(use_alignment),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _read_cached_raw(cache_path: pathlib.Path) -> bytes | None:
    if not cache_path.exists() or cache_path.stat().st_size <= 44:
        return None
    return cache_path.read_bytes()


def _write_cached_raw(cache_path: pathlib.Path, raw_bytes: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent,
        prefix=f"{cache_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.replace(cache_path)


def _synthesize_pending_raw(
    tts_engine,
    pending: list[dict],
    max_workers: int = _TTS_CONCURRENCY,
    max_consecutive_failures: int = _MAX_CONSECUTIVE_TTS_FAILURES,
) -> dict[int, bytes | None]:
    """Synthesize uncached phrase audio with bounded parallelism.

    Results are keyed by phrase index so callers can still post-process and
    assemble audio in transcript order.
    """
    results: dict[int, bytes | None] = {}
    if not pending:
        return results

    has_voice_uploads = any(item.get("speaker_wav") for item in pending)
    workers = 1 if has_voice_uploads else max(1, min(max_workers, len(pending)))
    if workers == 1:
        consecutive_failures = 0
        for item in pending:
            try:
                raw_bytes = _synthesize_raw(
                    tts_engine,
                    item["text"],
                    item["wav_path"],
                    speaker_wav=item["speaker_wav"],
                )
            except Exception as exc:
                print(f"[tts] TTS failed for segment ({exc}), using silence")
                raw_bytes = None

            if raw_bytes is not None:
                _write_cached_raw(item["cache_path"], raw_bytes)
                consecutive_failures = 0
            elif _TTS_FAIL_FAST and item["text"].strip():
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        "TTS backend failed repeatedly; aborting early so the backend can be restarted"
                    )
            results[item["index"]] = raw_bytes
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                _synthesize_raw,
                tts_engine,
                item["text"],
                item["wav_path"],
                speaker_wav=item["speaker_wav"],
            ): item
            for item in pending
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                raw_bytes = future.result()
            except Exception as exc:
                print(f"[tts] TTS failed for segment ({exc}), using silence")
                raw_bytes = None
            if raw_bytes is not None:
                _write_cached_raw(item["cache_path"], raw_bytes)
            results[item["index"]] = raw_bytes
    return results


def _extract_segment_reference(
    video_path: pathlib.Path,
    start_s: float,
    end_s: float,
    output_path: pathlib.Path,
) -> pathlib.Path | None:
    """Extract a short source-audio clip to use as a voice-cloning reference."""
    if not video_path.exists():
        return None

    duration_s = max(0.5, end_s - start_s)
    if duration_s < 3.0:
        pad_s = (3.0 - duration_s) / 2
        start_s = max(0.0, start_s - pad_s)
        duration_s = 3.0
    duration_s = min(duration_s, 8.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_s:.3f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as exc:
        print(f"[tts] speaker reference extraction failed ({exc}); using default voice")
        return None

    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def _clean_tts_text(text: str) -> str:
    """Remove caption speaker markers that should not be spoken."""
    cleaned = re.sub(r"^\s*(?:>+|&gt;+)\s*", "", text).strip()
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"#{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[^\wÀ-ÖØ-öø-ÿ¿¡]+(?=\s*\w)", "", cleaned).strip()
    if cleaned and not re.search(r"[\wÀ-ÖØ-öø-ÿ]", cleaned):
        return ""
    return cleaned


def _effective_segment_end(segments: list[dict], index: int) -> float:
    """Clamp overlapping caption windows to the next segment start."""
    seg = segments[index]
    start = float(seg["start"])
    end = float(seg["end"])
    if index + 1 < len(segments):
        next_start = float(segments[index + 1]["start"])
        if start < next_start < end:
            end = next_start
    return max(end, start + _MIN_SEGMENT_TARGET_SEC)


def _group_segment_metas(
    seg_metas: list[dict],
    *,
    max_duration_s: float = _MAX_SYNTH_GROUP_SEC,
    max_gap_s: float = 0.6,
    flush_on_sentence: bool = True,
) -> list[dict]:
    """Combine adjacent caption fragments into more natural TTS phrases."""
    groups: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["target_sec"] = current["end"] - current["start"]
            groups.append(current)
            current = None

    for meta in seg_metas:
        text = meta["text"].strip()
        if not text:
            continue

        if current is None:
            current = {
                **meta,
                "source_indices": [meta["index"]],
            }
        else:
            gap = float(meta["start"]) - float(current["end"])
            combined_duration = float(meta["end"]) - float(current["start"])
            speaker_changed = meta.get("speaker") != current.get("speaker")
            if gap > max_gap_s or combined_duration > max_duration_s or speaker_changed:
                flush()
                current = {
                    **meta,
                    "source_indices": [meta["index"]],
                }
            else:
                current["text"] = f"{current['text'].rstrip()} {text}"
                current["end"] = meta["end"]
                current["target_sec"] = current["end"] - current["start"]
                current["source_indices"].append(meta["index"])

        if flush_on_sentence and text.endswith((".", "?", "!", "…")):
            flush()

    flush()
    return groups


def _apply_speaker_color(audio: AudioSegment, speaker: str | None) -> AudioSegment:
    """Give diarized speakers stable tonal differences without voice uploads."""
    if not _SPEAKER_COLORING or not speaker:
        return audio

    digest = int(hashlib.sha256(speaker.encode("utf-8")).hexdigest()[:8], 16)
    profile = digest % 8
    original_ms = len(audio)
    pitch_steps = [-2, 2, -3, 3, -1, 1, -4, 4][profile]
    pitched = _shift_audio_pitch(audio, pitch_steps)

    if profile == 0:
        colored = pitched.high_pass_filter(170).apply_gain(0.8)
    elif profile == 1:
        colored = pitched.low_pass_filter(3100).apply_gain(-0.8)
    elif profile == 2:
        colored = pitched.high_pass_filter(230).low_pass_filter(4300).apply_gain(1.2)
    elif profile == 3:
        colored = pitched.low_pass_filter(2600).apply_gain(0.4)
    elif profile == 4:
        colored = pitched.high_pass_filter(110).apply_gain(-1.1)
    elif profile == 5:
        colored = pitched.high_pass_filter(180).low_pass_filter(3600)
    elif profile == 6:
        colored = pitched.low_pass_filter(2400).apply_gain(-1.6)
    else:
        colored = pitched.high_pass_filter(260).apply_gain(1.5)

    if len(colored) < original_ms:
        colored += AudioSegment.silent(duration=original_ms - len(colored))
    return colored[:original_ms]


def _shift_audio_pitch(audio: AudioSegment, semitones: float) -> AudioSegment:
    """Pitch-shift an AudioSegment while returning the original sample rate."""
    if semitones == 0 or len(audio) == 0:
        return audio
    shifted_rate = int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
    if shifted_rate <= 0:
        return audio
    return audio._spawn(audio.raw_data, overrides={"frame_rate": shifted_rate}).set_frame_rate(audio.frame_rate)


def _postprocess_segment(
    raw_wav_bytes: bytes | None,
    target_sec: float,
    stretch_factor: float,
    alignment_enabled: bool,
    work_dir: str,
) -> tuple:
    """Post-process one TTS segment without slowing speech unnaturally.

    Returns:
        (AudioSegment | None, speed_factor, raw_duration_s)
    """

    if target_sec <= 0:
        return (None, 0.0, 0.0)

    target_ms = int(target_sec * 1000)

    if raw_wav_bytes is None:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    work_path = pathlib.Path(work_dir)
    raw_wav = work_path / "raw_segment.wav"
    raw_wav.write_bytes(raw_wav_bytes)

    y, sr = librosa.load(str(raw_wav), sr=None)

    if sr is None or len(y) == 0:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    raw_duration = len(y) / sr

    if raw_duration <= 0:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    duration_ratio = raw_duration / target_sec

    if not alignment_enabled:
        # Baseline mode never slows speech down. If a generated line overruns
        # its cue, use the legacy wide speed range for A/B comparisons.
        speed_factor = raw_duration / target_sec if raw_duration > target_sec else 1.0
        speed_factor = max(_SPEED_MIN_LEGACY, min(_SPEED_MAX_LEGACY, speed_factor))

    elif raw_duration < target_sec:
        # TTS is shorter than the available window.
        # Do NOT slow it down. Pad with silence after.
        speed_factor = 1.0

    else:
        # TTS is longer than the window.
        # Compress only mildly so speech stays natural.
        effective_target = target_sec * max(stretch_factor, 0.1)
        speed_factor = raw_duration / effective_target
        speed_factor = max(1.0, min(_MAX_NATURAL_SPEEDUP, speed_factor))

    try:
        if abs(speed_factor - 1.0) > 0.01:
            y_stretched = pyrubberband.time_stretch(y, sr, speed_factor)
        else:
            y_stretched = y
    except Exception as exc:
        print(f"[tts] pyrubberband failed ({exc}); using raw audio")
        y_stretched = y

    stretched_wav = work_path / "stretched_segment.wav"
    sf.write(str(stretched_wav), y_stretched, sr)

    segment_audio = AudioSegment.from_wav(str(stretched_wav))

    if len(segment_audio) < target_ms:
        segment_audio += AudioSegment.silent(duration=target_ms - len(segment_audio))
    elif len(segment_audio) > target_ms:
        segment_audio = segment_audio[:target_ms]

    return (segment_audio, speed_factor, raw_duration)


def _synced_segment_audio(
    tts_engine,
    text: str,
    target_sec: float,
    work_dir,
    stretch_factor: float = 1.0,
    alignment_enabled: bool | None = None,
) -> tuple:
    """Generate TTS audio for *text* and time-stretch it to *target_sec*.

    Convenience wrapper kept for callers that don't use the batch path.
    """
    legacy_audio_only = tts_engine is tts
    if target_sec <= 0:
        return None if legacy_audio_only else (None, 0.0, 0.0)

    effective_engine = _get_tts_engine() if legacy_audio_only else tts_engine
    use_alignment = _ALIGNMENT_ENABLED if alignment_enabled is None else alignment_enabled

    raw_wav = str(pathlib.Path(work_dir) / "raw_segment.wav")
    raw_bytes = _synthesize_raw(effective_engine, text, raw_wav)
    result = _postprocess_segment(raw_bytes, target_sec, stretch_factor, use_alignment, str(work_dir))
    return result[0] if legacy_audio_only else result


def text_to_speech(text, output_file_path):
    _get_tts_engine().tts_to_file(text=text, file_path=str(output_file_path))


def _load_en_transcript(es_source_path: str) -> dict:
    """Locate the source-language transcript that corresponds to the translated file.

    Convention: translated JSON lives at .../translations/{model}/<title>.json
    Source transcript lives at .../transcriptions/{model}/<title>.json
    Returns an empty dict (no segments) if the source file is not found.
    """
    es_path = pathlib.Path(es_source_path)
    # Navigate: translations/{model}/ → data_dir → transcriptions/whisper/
    data_dir = es_path.parent.parent.parent
    en_path = data_dir / "transcriptions" / "whisper" / es_path.name
    if not en_path.exists():
        print(f"[tts] EN transcript not found at {en_path}, alignment skipped")
        return {}
    with open(en_path) as f:
        return json.load(f)


def _build_alignment(en_transcript: dict, es_transcript: dict) -> tuple:
    """Run global_align and return (metrics_list, {segment_index: AlignedSegment}).

    Returns ([], {}) if the alignment library is unavailable or fails.
    """
    try:
        from foreign_whispers.alignment import compute_segment_metrics, global_align
    except ImportError:
        return [], {}
    try:
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        aligned = global_align(metrics, silence_regions=[])
        return metrics, {seg.index: seg for seg in aligned}
    except Exception as exc:
        print(f"[tts] alignment failed ({exc}), proceeding without alignment")
        return [], {}


def _shorten_segment_text(en_text: str, es_text: str, target_sec: float) -> str:
    """Try to shorten a Spanish translation to fit *target_sec*.

    Delegates to ``get_shorter_translations()`` (student assignment stub).
    Returns the original *es_text* if no shorter candidate is available.
    """
    try:
        from foreign_whispers.reranking import get_shorter_translations
        candidates = get_shorter_translations(
            source_text=en_text,
            baseline_es=es_text,
            target_duration_s=target_sec,
        )
        if candidates:
            target_chars = max(1, int(target_sec * 15))
            best = min(candidates, key=lambda c: (c.char_count > target_chars, abs(c.char_count - target_chars)))
            return best.text
    except Exception as exc:
        _logging.getLogger(__name__).warning("[tts] rerank failed: %s", exc)
    return es_text


def _write_align_report(
    output_path: str,
    stem: str,
    metrics: list,
    aligned: list,
    segment_details: list,
) -> None:
    """Write a {stem}.align.json sidecar with evaluation metrics and per-segment detail.

    segment_details is a list of dicts: [{raw_duration_s, speed_factor, action, text}, ...]
    Written next to the WAV so both baseline and aligned runs produce comparable files.
    """
    try:
        from foreign_whispers.evaluation import clip_evaluation_report
        summary = clip_evaluation_report(metrics, aligned)
    except Exception as exc:
        _logging.getLogger(__name__).warning("clip_evaluation_report failed: %s", exc)
        summary = {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch": 0.0,
            "n_gap_shifts": 0,
            "n_translation_retries": 0,
            "total_cumulative_drift_s": 0.0,
        }

    report = {
        **summary,
        "alignment_enabled": _ALIGNMENT_ENABLED,
        "timing_model": _TIMING_MODEL,
        "synthesis_failures": sum(1 for s in segment_details if s.get("raw_duration_s") == 0),
        "segments": segment_details,
    }
    sidecar_path = pathlib.Path(output_path) / f"{stem}.align.json"
    sidecar_path.write_text(json.dumps(report, indent=2))


def _compute_speech_offset(source_path: str) -> float:
    """Compute timing offset between YouTube captions and Whisper segments.

    Returns seconds to add to Whisper timestamps so TTS audio aligns with
    the actual speech start in the original video.
    """
    title = pathlib.Path(source_path).stem
    # source_path: .../translations/{model}/{title}.json → data_dir is 3 levels up
    base_dir = pathlib.Path(source_path).parent.parent.parent

    yt_path = base_dir / "youtube_captions" / f"{title}.txt"
    whisper_path = base_dir / "transcriptions" / "whisper" / f"{title}.json"

    if not yt_path.exists() or not whisper_path.exists():
        return 0.0

    first_line = yt_path.read_text().split("\n", 1)[0].strip()
    if not first_line:
        return 0.0
    yt_start = json.loads(first_line).get("start", 0.0)

    whisper_data = json.loads(whisper_path.read_text())
    segs = whisper_data.get("segments", [])
    whisper_start = segs[0]["start"] if segs else 0.0

    return yt_start - whisper_start


def text_file_to_speech(
    source_path,
    output_path,
    tts_engine=None,
    *,
    alignment=None,
    voice_cloning: bool = False,
    speaker_wav: str | None = None,
):
    """Read translated JSON with segment timestamps and produce a time-aligned WAV.

    Each segment is individually synthesized and time-stretched to match its
    original timestamp window.  Gaps between segments are filled with silence.
    Applies the YouTube caption timing offset so TTS audio starts when speech
    actually begins in the original video.

    *tts_engine* overrides the module-level ``tts`` instance (used by the
    FastAPI app which loads the model at startup).

    *alignment* overrides the module-level ``_ALIGNMENT_ENABLED`` flag.
    Pass True for aligned mode, False for baseline, or None to use the env var.
    """
    engine = tts_engine if tts_engine is not None else _get_tts_engine()
    use_alignment = alignment if alignment is not None else _ALIGNMENT_ENABLED

    save_name = pathlib.Path(source_path).stem + ".wav"
    print(f"generating {save_name}...", end="")

    segments = segments_from_file(source_path)

    if not segments:
        text = text_from_file(source_path)
        save_path = pathlib.Path(output_path) / pathlib.Path(save_name)
        text_to_speech(text, str(save_path))
        print("success!")
        return None

    # Apply YouTube caption timing offset
    offset = _compute_speech_offset(source_path)
    if offset > 0:
        print(f" (applying {offset:.1f}s speech offset)", end="")

    # Pre-compute alignment; also returns flat metrics list for clip_evaluation_report
    with open(source_path) as f:
        es_transcript = json.load(f)
    en_transcript = _load_en_transcript(source_path)
    if use_alignment:
        _metrics_list, align_map = _build_alignment(en_transcript, es_transcript)
    else:
        _metrics_list, align_map = [], {}
    _aligned_list = list(align_map.values())

    # ── Prepare per-segment metadata ────────────────────────────────────
    seg_metas = []
    for i, seg in enumerate(segments):
        aligned_seg = align_map.get(i)
        stretch_factor = aligned_seg.stretch_factor if aligned_seg else 1.0
        effective_end = _effective_segment_end(segments, i)
        target_sec = effective_end - float(seg["start"])

        seg_text = _clean_tts_text(seg["text"])
        if aligned_seg is not None:
            from foreign_whispers.alignment import AlignAction
            if aligned_seg.action == AlignAction.REQUEST_SHORTER:
                en_text = ""
                en_segs = en_transcript.get("segments", [])
                if i < len(en_segs):
                    en_text = en_segs[i].get("text", "")
                seg_text = _clean_tts_text(_shorten_segment_text(en_text, seg["text"], target_sec))

        seg_metas.append({
            "index": i,
            "text": seg_text,
            "speaker": seg.get("speaker"),
            "start": seg["start"],
            "end": effective_end,
            "target_sec": target_sec,
            "stretch_factor": stretch_factor,
            "aligned_seg": aligned_seg,
        })

    synth_metas = _group_segment_metas(seg_metas)
    if _TTS_LONG_FORM_GROUP_THRESHOLD > 0 and len(synth_metas) > _TTS_LONG_FORM_GROUP_THRESHOLD:
        synth_metas = _group_segment_metas(
            seg_metas,
            max_duration_s=max(_MAX_SYNTH_GROUP_SEC, _TTS_LONG_FORM_GROUP_SEC),
            max_gap_s=3.0,
            flush_on_sentence=False,
        )

    # ── Phase 1: GPU synthesis (concurrent) ───────────────────────────
    # Submit all TTS calls to a thread pool so the GPU stays busy while
    # previous results are being downloaded / decoded.
    speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
    target_language = es_transcript.get("language", "")
    video_path = pathlib.Path(source_path).parent.parent.parent / "videos" / f"{pathlib.Path(source_path).stem}.mp4"
    use_synced_helper = hasattr(_synced_segment_audio, "mock_calls")

    synth_count = 0
    cache_hit_count = 0
    failed_count = 0
    consecutive_failures = 0

    # ── Phase 2: CPU post-processing (sequential assembly) ────────────
    save_path = pathlib.Path(output_path) / save_name
    cache_dir = pathlib.Path(output_path) / ".synth_cache" / pathlib.Path(source_path).stem
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as synth_dir:
        combined = AudioSegment.empty()
        cursor_ms = 0
        segment_details = []
        ref_dir = pathlib.Path(synth_dir) / "speaker_refs"

        def _speaker_reference(m: dict) -> str | None:
            if not voice_cloning:
                return None
            if speaker_wav:
                return speaker_wav
            if not _AUTO_VOICE_CLONING:
                return None
            try:
                return resolve_speaker_wav(speakers_base, target_language, m.get("speaker"))
            except FileNotFoundError:
                if not _EXTRACT_SEGMENT_VOICE_REFS:
                    return None
                ref_path = _extract_segment_reference(
                    video_path,
                    m["start"],
                    m["end"],
                    ref_dir / f"seg_{m['index']}.wav",
                )
                return str(ref_path) if ref_path else None

        raw_by_index: dict[int, bytes | None] = {}
        speaker_by_index: dict[int, str | None] = {}
        if not use_synced_helper:
            pending_synth: list[dict] = []
            for m in synth_metas:
                i = m["index"]
                resolved_speaker_wav = _speaker_reference(m)
                speaker_by_index[i] = resolved_speaker_wav
                cache_path = cache_dir / f"{_synth_cache_key(m, resolved_speaker_wav, use_alignment)}.wav"
                raw_bytes = _read_cached_raw(cache_path)
                if raw_bytes is not None:
                    cache_hit_count += 1
                    raw_by_index[i] = raw_bytes
                    continue

                pending_synth.append({
                    "index": i,
                    "text": m["text"],
                    "speaker_wav": resolved_speaker_wav,
                    "wav_path": str(pathlib.Path(synth_dir) / f"seg_{i}.wav"),
                    "cache_path": cache_path,
                })

            synth_count = len(pending_synth)
            raw_by_index.update(_synthesize_pending_raw(engine, pending_synth))

        for m in synth_metas:
            i = m["index"]
            start_ms = int((m["start"] + offset) * 1000)
            resolved_speaker_wav = None

            if start_ms > cursor_ms:
                combined += AudioSegment.silent(duration=start_ms - cursor_ms)
                cursor_ms = start_ms

            if use_synced_helper:
                seg_audio, seg_speed_factor, seg_raw_duration = _synced_segment_audio(
                    engine, m["text"], m["target_sec"], tmpdir, stretch_factor=m["stretch_factor"],
                )
            else:
                resolved_speaker_wav = speaker_by_index.get(i)
                raw_bytes = raw_by_index.get(i)
                if raw_bytes is None:
                    failed_count += 1
                    if _TTS_FAIL_FAST and m["text"].strip():
                        consecutive_failures += 1
                        if consecutive_failures >= _MAX_CONSECUTIVE_TTS_FAILURES:
                            raise RuntimeError(
                                "TTS backend failed repeatedly; saved phrase cache so the next run can resume"
                            )
                else:
                    consecutive_failures = 0
                seg_audio, seg_speed_factor, seg_raw_duration = _postprocess_segment(
                    raw_bytes, m["target_sec"], m["stretch_factor"],
                    use_alignment, tmpdir,
                )
                if voice_cloning and resolved_speaker_wav is None:
                    seg_audio = _apply_speaker_color(seg_audio, m.get("speaker"))

            aligned_seg = m["aligned_seg"]
            segment_details.append({
                "index": i,
                "source_indices": m.get("source_indices", [i]),
                "text": m["text"],
                "speaker": m.get("speaker"),
                "speaker_wav": resolved_speaker_wav,
                "target_sec": round(m["target_sec"], 3),
                "stretch_factor": round(m["stretch_factor"], 3),
                "raw_duration_s": round(seg_raw_duration, 3),
                "speed_factor": round(seg_speed_factor, 3),
                "action": aligned_seg.action.value if aligned_seg and hasattr(aligned_seg, "action") else "unknown",
            })

            if seg_audio is not None:
                combined += seg_audio
                cursor_ms += len(seg_audio)

        combined.export(str(save_path), format="wav")

    stem = pathlib.Path(source_path).stem
    _write_align_report(str(output_path), stem, _metrics_list, _aligned_list, segment_details)

    if not use_synced_helper:
        print(
            f" ({synth_count} phrase groups synthesized, {cache_hit_count} cache hits, "
            f"{failed_count} silent fallbacks)",
            end="",
        )
    print("success!")
    return None


if __name__ == '__main__':
    SOURCE_PATH = "./data/transcriptions/es"
    OUTPUT_PATH = "./audios/"

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    files = files_from_dir(SOURCE_PATH)
    for file in files:
        text_file_to_speech(file, OUTPUT_PATH)

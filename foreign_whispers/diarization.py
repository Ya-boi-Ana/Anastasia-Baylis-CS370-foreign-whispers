"""Speaker diarization using pyannote.audio."""

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _patch_torchaudio_for_pyannote() -> None:
    """Restore torchaudio APIs still referenced by pyannote.audio 3.x."""
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(
                self,
                sample_rate: int,
                num_frames: int,
                num_channels: int,
                bits_per_sample: int = 0,
                encoding: str = "UNKNOWN",
            ):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "info"):
        def info(path, *args, **kwargs):
            import soundfile as sf

            sf_info = sf.info(path)
            return torchaudio.AudioMetaData(
                sample_rate=int(sf_info.samplerate),
                num_frames=int(sf_info.frames),
                num_channels=int(sf_info.channels),
                bits_per_sample=0,
                encoding=str(sf_info.subtype or "UNKNOWN"),
            )

        torchaudio.info = info
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None


@contextmanager
def _pyannote_torch_load_compat():
    """Load trusted pyannote checkpoints with PyTorch 2.6+ compatibility."""
    try:
        import torch
    except Exception:
        yield
        return

    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = compat_load
    try:
        yield
    finally:
        torch.load = original_load


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    if not hf_token or hf_token == "hf_placeholder_token":
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        _patch_torchaudio_for_pyannote()
        from pyannote.audio import Pipeline
    except (ImportError, AttributeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        with _pyannote_torch_load_compat():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        if pipeline is None:
            logger.warning(
                "Could not load pyannote/speaker-diarization-3.1. "
                "Verify the Hugging Face account has accepted gated access "
                "for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0."
            )
            return []

        diarization = pipeline(audio_path)

        return [
            {
                "start_s": float(turn.start),
                "end_s": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    except Exception as exc:
        logger.exception("Diarization failed for %s", audio_path)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    result = []

    for segment in segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for turn in diarization:
            turn_start = float(turn.get("start_s", 0.0))
            turn_end = float(turn.get("end_s", 0.0))

            overlap = max(
                0.0,
                min(seg_end, turn_end) - max(seg_start, turn_start),
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(turn.get("speaker", best_speaker))

        merged_segment = dict(segment)
        merged_segment["speaker"] = best_speaker
        result.append(merged_segment)

    return result

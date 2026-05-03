"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _probe_media_duration_seconds(path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _diarization_too_long(duration_s: float | None) -> bool:
    limit_s = float(getattr(settings, "diarization_max_seconds", 0.0) or 0.0)
    return bool(limit_s > 0 and duration_s is not None and duration_s > limit_s)


async def _extract_audio(
    video_path: Path,
    audio_path: Path,
    *,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    cmd = ["ffmpeg", "-y"]
    if start_s is not None:
        cmd.extend(["-ss", f"{max(0.0, start_s):.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_s is not None:
        cmd.extend(["-t", f"{max(0.001, duration_s):.3f}"])
    cmd.extend([
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ])
    await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)


async def _diarize_video_audio(
    video_path: Path,
    diar_dir: Path,
    title: str,
    duration_s: float | None,
) -> list[dict]:
    if not _diarization_too_long(duration_s):
        audio_path = diar_dir / f"{title}.wav"
        await _extract_audio(video_path, audio_path)
        return _alignment_service.diarize(str(audio_path))

    chunk_s = float(getattr(settings, "diarization_chunk_seconds", 600.0) or 600.0)
    chunk_s = max(60.0, chunk_s)
    all_segments: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="diar_chunks_", dir=diar_dir) as tmp:
        tmp_dir = Path(tmp)
        start_s = 0.0
        chunk_index = 0
        while start_s < float(duration_s or 0.0):
            remaining_s = float(duration_s or 0.0) - start_s
            this_duration_s = min(chunk_s, remaining_s)
            chunk_path = tmp_dir / f"{title}.chunk{chunk_index:04d}.wav"
            await _extract_audio(
                video_path,
                chunk_path,
                start_s=start_s,
                duration_s=this_duration_s,
            )
            chunk_segments = _alignment_service.diarize(str(chunk_path))
            for segment in chunk_segments:
                shifted = dict(segment)
                shifted["start_s"] = float(segment.get("start_s", 0.0)) + start_s
                shifted["end_s"] = float(segment.get("end_s", 0.0)) + start_s
                all_segments.append(shifted)
            start_s += this_duration_s
            chunk_index += 1

    return all_segments


def _speaker_namespace(title: str) -> str:
    return _speaker_filename(title)


def _namespaced_speaker(title: str, speaker: str) -> str:
    return f"{_speaker_namespace(title)}__{_speaker_filename(speaker)}"


def _namespace_diar_segments(title: str, diar_segments: list[dict]) -> list[dict]:
    namespaced = []
    for segment in diar_segments:
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        namespaced.append({**segment, "speaker": _namespaced_speaker(title, speaker)})
    return namespaced


def _merge_speakers_into_json(path, diar_segments: list[dict]) -> None:
    """Persist speaker labels onto transcript-like JSON segments."""
    if not path.exists() or not diar_segments:
        return
    data = json.loads(path.read_text())
    data["segments"] = assign_speakers(data.get("segments", []), diar_segments)
    path.write_text(json.dumps(data))


def _merge_speakers_into_transcript(title: str, diar_segments: list[dict]) -> None:
    """Persist speaker labels onto files used by translation/TTS."""
    namespaced_segments = _namespace_diar_segments(title, diar_segments)
    _merge_speakers_into_json(settings.transcriptions_dir / f"{title}.json", namespaced_segments)
    _merge_speakers_into_json(settings.translations_dir / f"{title}.json", namespaced_segments)


def _speaker_filename(speaker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker).strip("._")
    return safe or "SPEAKER_00"


async def _extract_speaker_references(title: str, diar_segments: list[dict], target_language: str = "es") -> None:
    """Create speakers/{lang}/{speaker}.wav refs for Chatterbox voice cloning."""
    if not diar_segments:
        return

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        return

    speakers_dir = settings.data_dir.parent / "speakers" / target_language
    speakers_dir.mkdir(parents=True, exist_ok=True)

    best_by_speaker: dict[str, dict] = {}
    for segment in diar_segments:
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        duration = float(segment.get("end_s", 0.0)) - float(segment.get("start_s", 0.0))
        if duration <= 0:
            continue
        current = best_by_speaker.get(speaker)
        if current is None or duration > current["duration"]:
            best_by_speaker[speaker] = {**segment, "duration": duration}

    for speaker, segment in best_by_speaker.items():
        output_path = speakers_dir / f"{_namespaced_speaker(title, speaker)}.wav"

        start_s = max(0.0, float(segment.get("start_s", 0.0)))
        duration_s = min(8.0, max(3.0, float(segment["duration"])))
        try:
            await asyncio.to_thread(
                subprocess.run,
                [
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
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            continue


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        diar_segments = data.get("segments", [])
        if diar_segments:
            _merge_speakers_into_transcript(title, diar_segments)
            await _extract_speaker_references(title, diar_segments)
            return DiarizeResponse(
                video_id=video_id,
                speakers=data.get("speakers", []),
                segments=diar_segments,
            )

    # Extract audio from the video.
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file for {video_id} not found")

    try:
        duration_s = await asyncio.to_thread(_probe_media_duration_seconds, video_path)
        diar_segments = await _diarize_video_audio(video_path, diar_dir, title, duration_s)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Diarization audio extraction failed: {exc.stderr.decode(errors='ignore')}"
        )
    speakers = sorted({segment["speaker"] for segment in diar_segments})

    if not diar_segments:
        return DiarizeResponse(
            video_id=video_id,
            speakers=[],
            segments=[],
            skipped=True,
        )

    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))

    _merge_speakers_into_transcript(title, diar_segments)
    await _extract_speaker_references(title, diar_segments)

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
    )

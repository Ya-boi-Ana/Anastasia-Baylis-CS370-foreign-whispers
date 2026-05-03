"""TTS routes for text-to-speech synthesis and audio retrieval."""

import asyncio
import functools
import json
import pathlib
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")
_tts_locks: dict[tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)


def _cached_audio_is_current(wav_path: pathlib.Path, require_speaker_wav: bool = False) -> bool:
    """Return False for cached audio known to use legacy slow-down stretching."""
    report_path = wav_path.with_suffix(".align.json")
    if not report_path.exists():
        return True

    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return True

    if report.get("timing_model") != "non_overlapping_phrase_groups_v1":
        return False

    for segment in report.get("segments", []):
        speed_factor = segment.get("speed_factor")
        if isinstance(speed_factor, (int, float)) and speed_factor < 0.99:
            return False

    segments = report.get("segments", [])
    if segments:
        failures = sum(1 for segment in segments if segment.get("raw_duration_s") == 0)
        if failures / len(segments) > 0.2:
            return False

        if require_speaker_wav:
            speaker_segments = [segment for segment in segments if segment.get("speaker")]
            if speaker_segments and any(not segment.get("speaker_wav") for segment in speaker_segments):
                return False

    return True


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    voice_cloning: bool = Query(False),
    speaker_wav: str | None = Query(None),
):
    """Generate TTS audio for a translated transcript."""
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    wants_voice_cloning = voice_cloning or speaker_wav is not None

    require_speaker_wav = speaker_wav is not None

    if wav_path.exists() and _cached_audio_is_current(wav_path, require_speaker_wav=require_speaker_wav):
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")

    lock = _tts_locks[(video_id, config)]
    async with lock:
        if wav_path.exists() and _cached_audio_is_current(wav_path, require_speaker_wav=require_speaker_wav):
            return {
                "video_id": video_id,
                "audio_path": str(wav_path),
                "config": config,
            }

        await _run_in_threadpool(
            None,
            svc.text_file_to_speech,
            source_path,
            str(audio_dir),
            alignment=alignment,
            voice_cloning=wants_voice_cloning,
            speaker_wav=speaker_wav,
        )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")

"""Speaker diarization using pyannote.audio."""

import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

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
"""Voice resolution for speaker cloning.

Resolves which reference WAV to use for a given language and speaker ID.
If the WAV does not exist but a reference media file does, it auto-extracts
a usable speaker sample via ffmpeg.
"""

from pathlib import Path
import subprocess
import shutil


SUPPORTED_MEDIA = {".wav", ".mp4", ".m4a", ".mp3", ".aac", ".mov"}


def _extract_wav(source: Path, target: Path) -> None:
    """Convert media file into 16kHz mono WAV for cloning."""
    target.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(source),
            "-t", "8",
            "-ar", "16000",
            "-ac", "1",
            str(target),
        ],
        check=True,
    )


def _ensure_reference_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str,
) -> Path | None:
    """
    Ensure speakers/{lang}/{speaker}.wav exists.

    If missing but a matching reference media file exists,
    automatically convert it.
    """

    lang_dir = speakers_dir / target_language
    wav_path = lang_dir / f"{speaker_id}.wav"

    if wav_path.exists():
        return wav_path

    for ext in SUPPORTED_MEDIA:
        candidate = lang_dir / f"{speaker_id}{ext}"
        if candidate.exists():
            if ext == ".wav":
                return candidate

            _extract_wav(candidate, wav_path)
            return wav_path

    return None


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """
    Resolve speaker reference WAV path.

    Resolution order:

    1. speakers/{lang}/{speaker_id}.wav
    2. speakers/{lang}/default.wav
    3. speakers/default.wav

    Automatically converts reference media if needed.
    """

    speakers_dir = Path(speakers_dir)
    target_language = target_language.lower().strip()

    # Speaker-specific voice
    if speaker_id:
        wav_path = _ensure_reference_wav(
            speakers_dir,
            target_language,
            speaker_id,
        )
        if wav_path:
            return f"{target_language}/{speaker_id}.wav"

    # Language default
    lang_default = speakers_dir / target_language / "default.wav"
    if lang_default.exists():
        return f"{target_language}/default.wav"

    # Global default
    global_default = speakers_dir / "default.wav"
    if global_default.exists():
        return "default.wav"

    raise FileNotFoundError(
        "No usable speaker reference found.\n"
        "Expected one of:\n"
        f"- {speakers_dir}/{target_language}/{speaker_id}.wav\n"
        f"- {speakers_dir}/{target_language}/default.wav\n"
        f"- {speakers_dir}/default.wav"
    )
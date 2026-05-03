"""Tests for POST /api/tts/{video_id} endpoint (issue 381)."""

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def ui_dir(tmp_path):
    (tmp_path / "translations" / "argos").mkdir(parents=True)
    (tmp_path / "tts_audio" / "chatterbox").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def client(monkeypatch, ui_dir):
    monkeypatch.setattr("whisper.load_model", lambda *a, **kw: MagicMock())
    
    # Only mock TTS if it's importable; skip if not available in test environment
    try:
        monkeypatch.setattr("TTS.api.TTS", lambda *a, **kw: MagicMock())
    except Exception:
        pass

    from api.src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", ui_dir)

    from api.src.main import app

    with TestClient(app) as c:
        yield c


def _translated_transcript():
    return {
        "text": "Hola mundo",
        "language": "es",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": " Hola mundo"},
        ],
    }


def test_tts_returns_audio_path(client, monkeypatch, ui_dir):
    """POST /api/tts/{video_id}?config=...&alignment=... returns path to generated WAV."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    def fake_tts(source_path, output_path, tts_engine=None, alignment=False, voice_cloning=False, speaker_wav=None):
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.tts_text_file_to_speech", fake_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000&alignment=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "G3Eup4mfJdA"
    assert body["audio_path"].endswith(".wav")
    assert body["config"] == "c-0000000"


def test_tts_passes_voice_cloning_flag(client, monkeypatch, ui_dir):
    """POST /api/tts/{video_id}?voice_cloning=true forwards the flag to the TTS service."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    def fake_tts(source_path, output_path, tts_engine=None, alignment=False, voice_cloning=False, speaker_wav=None):
        assert voice_cloning is True
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.tts_text_file_to_speech", fake_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000&voice_cloning=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "G3Eup4mfJdA"
    assert body["audio_path"].endswith(".wav")
    assert body["config"] == "c-0000000"


def test_tts_passes_speaker_wav(client, monkeypatch, ui_dir):
    """POST /api/tts/{video_id}?speaker_wav=... forwards explicit voice selection."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    def fake_tts(source_path, output_path, tts_engine=None, alignment=False, voice_cloning=False, speaker_wav=None):
        assert voice_cloning is True
        assert speaker_wav == "es/SPEAKER_00.wav"
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.tts_text_file_to_speech", fake_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000&speaker_wav=es/SPEAKER_00.wav")
    assert resp.status_code == 200


def test_tts_skips_if_cached(client, monkeypatch, ui_dir):
    """Skip TTS if WAV already exists in config subdirectory."""
    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    config_dir = ui_dir / "tts_audio" / "chatterbox" / "c-0000000"
    config_dir.mkdir(parents=True)
    wav = config_dir / "Test Title.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    tts_called = {"count": 0}

    def tracking_tts(source_path, output_path, tts_engine=None, alignment=False, voice_cloning=False, speaker_wav=None):
        tts_called["count"] += 1

    monkeypatch.setattr("api.src.services.tts_service.tts_text_file_to_speech", tracking_tts)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000")
    assert resp.status_code == 200
    assert tts_called["count"] == 0


def test_tts_source_not_found(client, monkeypatch, ui_dir):
    """Returns 404 when translated transcript doesn't exist."""
    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: None,
    )

    resp = client.post("/api/tts/NONEXISTENT?config=c-0000000")
    assert resp.status_code == 404


def test_tts_runs_in_threadpool(client, monkeypatch, ui_dir):
    """TTS should run via run_in_executor to avoid blocking the event loop."""
    src = ui_dir / "translations" / "argos" / "Test Title.json"
    src.write_text(json.dumps(_translated_transcript()))

    monkeypatch.setattr(
        "api.src.routers.tts.resolve_title",
        lambda video_id: "Test Title",
    )

    executor_used = {"yes": False}

    def fake_tts(source_path, output_path, tts_engine=None, alignment=False, voice_cloning=False, speaker_wav=None):
        wav = pathlib.Path(output_path) / "Test Title.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

    monkeypatch.setattr("api.src.services.tts_service.tts_text_file_to_speech", fake_tts)

    async def tracking_run(executor, fn, *args, **kwargs):
        executor_used["yes"] = True
        return fn(*args, **kwargs)

    monkeypatch.setattr("api.src.routers.tts._run_in_threadpool", tracking_run)

    resp = client.post("/api/tts/G3Eup4mfJdA?config=c-0000000")
    assert resp.status_code == 200
    assert executor_used["yes"], "TTS should run in a thread pool"


def test_tts_rejects_invalid_config(client, monkeypatch, ui_dir):
    """Config param must match ^c-[0-9a-f]{7}$ to prevent path traversal."""
    resp = client.post("/api/tts/G3Eup4mfJdA?config=../../etc")
    assert resp.status_code == 422


def test_cached_audio_with_legacy_slowdown_is_not_current(tmp_path):
    """Legacy sidecars with speed_factor < 1 should force TTS regeneration."""
    from api.src.routers.tts import _cached_audio_is_current

    wav = tmp_path / "Test Title.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".align.json").write_text(json.dumps({
        "segments": [{"speed_factor": 0.72}],
    }))

    assert _cached_audio_is_current(wav) is False


def test_cached_audio_with_many_silent_fallbacks_is_not_current(tmp_path):
    """Sidecars with many failed syntheses should force TTS regeneration."""
    from api.src.routers.tts import _cached_audio_is_current

    wav = tmp_path / "Test Title.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".align.json").write_text(json.dumps({
        "timing_model": "non_overlapping_phrase_groups_v1",
        "segments": [
            {"speed_factor": 1.0, "raw_duration_s": 0.0},
            {"speed_factor": 1.0, "raw_duration_s": 0.0},
            {"speed_factor": 1.0, "raw_duration_s": 2.0},
        ],
    }))

    assert _cached_audio_is_current(wav) is False


def test_cached_audio_without_speaker_refs_is_current_for_safe_diarized_tts(tmp_path):
    """Safe diarized TTS uses speaker coloring, not uploaded speaker refs."""
    from api.src.routers.tts import _cached_audio_is_current

    wav = tmp_path / "Test Title.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".align.json").write_text(json.dumps({
        "timing_model": "non_overlapping_phrase_groups_v1",
        "segments": [
            {"speaker": "SPEAKER_00", "speaker_wav": None, "raw_duration_s": 1.0, "speed_factor": 1.0},
        ],
    }))

    assert _cached_audio_is_current(wav) is True
    assert _cached_audio_is_current(wav, require_speaker_wav=True) is False
    assert _cached_audio_is_current(wav, require_speaker_profiles=True) is False


def test_cached_audio_with_speaker_profiles_is_current_for_diarized_tts(tmp_path):
    """Diarized safe TTS cache is reusable once speaker profiles are recorded."""
    from api.src.routers.tts import _cached_audio_is_current

    wav = tmp_path / "Test Title.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".align.json").write_text(json.dumps({
        "timing_model": "non_overlapping_phrase_groups_v1",
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "speaker_voice": "speaker-profile-01",
                "speaker_gender": "female",
                "speaker_wav": None,
                "raw_duration_s": 1.0,
                "speed_factor": 1.0,
            },
        ],
    }))

    assert _cached_audio_is_current(wav, require_speaker_profiles=True) is True

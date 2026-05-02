# tests/test_diarization.py
import pytest
from foreign_whispers.diarization import diarize_audio


def test_returns_empty_without_token():
    result = diarize_audio("/any/path.wav", hf_token=None)
    assert result == []


def test_returns_empty_with_empty_token():
    result = diarize_audio("/any/path.wav", hf_token="")
    assert result == []


def test_returns_empty_with_placeholder_token():
    result = diarize_audio("/any/path.wav", hf_token="hf_placeholder_token")
    assert result == []


def test_torchaudio_pyannote_compat_patch(monkeypatch):
    import foreign_whispers.diarization as mod

    class FakeTorchaudio:
        pass

    fake_torchaudio = FakeTorchaudio()
    monkeypatch.setitem(__import__("sys").modules, "torchaudio", fake_torchaudio)

    mod._patch_torchaudio_for_pyannote()

    assert hasattr(fake_torchaudio, "AudioMetaData")
    assert hasattr(fake_torchaudio, "info")
    assert fake_torchaudio.list_audio_backends() == ["soundfile"]
    fake_torchaudio.set_audio_backend("soundfile")


def test_pyannote_torch_load_compat_sets_weights_only_false(monkeypatch):
    import sys
    import types
    import foreign_whispers.diarization as mod

    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        return "checkpoint"

    fake_torch = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with mod._pyannote_torch_load_compat():
        assert fake_torch.load("model.ckpt", weights_only=True) == "checkpoint"

    assert calls == [{"weights_only": False}]
    assert fake_torch.load is fake_load


def test_returns_empty_when_pyannote_absent(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    result = diarize_audio("/any/path.wav", hf_token="fake-token")
    assert result == []


def test_returns_empty_when_pyannote_pipeline_access_denied(monkeypatch):
    import sys
    import types

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, checkpoint_path, use_auth_token=None):
            assert checkpoint_path == "pyannote/speaker-diarization-3.1"
            assert use_auth_token == "fake-token"
            return None

    fake_module = types.SimpleNamespace(Pipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    result = diarize_audio("/any/path.wav", hf_token="fake-token")

    assert result == []


@pytest.mark.requires_pyannote
def test_real_diarization_returns_speaker_labels(tmp_path):
    """Integration test — requires pyannote.audio and FW_HF_TOKEN env var."""
    import os
    token = os.environ.get("FW_HF_TOKEN")
    if not token:
        pytest.skip("FW_HF_TOKEN not set")
    result = diarize_audio("/path/to/sample.wav", hf_token=token)
    assert isinstance(result, list)
    for r in result:
        assert "start_s" in r and "end_s" in r and "speaker" in r


def test_merge_speakers_updates_translation_json(tmp_path, monkeypatch):
    import json
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    transcriptions = data_dir / "transcriptions" / "whisper"
    translations = data_dir / "translations" / "argos"
    transcriptions.mkdir(parents=True)
    translations.mkdir(parents=True)
    payload = {"segments": [{"start": 0.0, "end": 2.0, "text": "Hola"}]}
    (transcriptions / "Demo.json").write_text(json.dumps(payload))
    (translations / "Demo.json").write_text(json.dumps(payload))

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)

    diar = [{"start_s": 0.0, "end_s": 2.0, "speaker": "SPEAKER_01"}]
    mod._merge_speakers_into_transcript("Demo", diar)

    translated = json.loads((translations / "Demo.json").read_text())
    assert translated["segments"][0]["speaker"] == "SPEAKER_01"


def test_extract_speaker_references_writes_one_ref_per_speaker(tmp_path, monkeypatch):
    import asyncio
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "Demo.mp4").write_bytes(b"video")
    calls = []

    def fake_run(cmd, **kwargs):
        from pathlib import Path
        calls.append(cmd)
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"RIFF" + b"\0" * 100)

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    asyncio.run(mod._extract_speaker_references("Demo", [
        {"start_s": 0.0, "end_s": 2.0, "speaker": "SPEAKER_00"},
        {"start_s": 5.0, "end_s": 10.0, "speaker": "SPEAKER_00"},
        {"start_s": 12.0, "end_s": 16.0, "speaker": "SPEAKER_01"},
    ]))

    speakers_dir = tmp_path / "speakers" / "es"
    assert (speakers_dir / "SPEAKER_00.wav").exists()
    assert (speakers_dir / "SPEAKER_01.wav").exists()
    assert len(calls) == 2

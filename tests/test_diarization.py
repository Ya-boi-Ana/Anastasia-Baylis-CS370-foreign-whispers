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
    assert translated["segments"][0]["speaker"] == "Demo__SPEAKER_01"


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
    assert (speakers_dir / "Demo__SPEAKER_00.wav").exists()
    assert (speakers_dir / "Demo__SPEAKER_01.wav").exists()
    assert len(calls) == 2


def test_extract_speaker_references_overwrites_video_namespace_only(tmp_path, monkeypatch):
    import asyncio
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "Demo.mp4").write_bytes(b"video")
    speakers_dir = tmp_path / "speakers" / "es"
    speakers_dir.mkdir(parents=True)
    stale = speakers_dir / "Other__SPEAKER_00.wav"
    stale.write_bytes(b"RIFF" + b"old")
    calls = []

    def fake_run(cmd, **kwargs):
        from pathlib import Path
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"RIFF" + b"new")

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    asyncio.run(mod._extract_speaker_references("Demo", [
        {"start_s": 0.0, "end_s": 4.0, "speaker": "SPEAKER_00"},
    ]))

    assert stale.read_bytes() == b"RIFF" + b"old"
    assert (speakers_dir / "Demo__SPEAKER_00.wav").read_bytes() == b"RIFF" + b"new"
    assert len(calls) == 1


def test_extract_speaker_references_skips_failed_ref(tmp_path, monkeypatch):
    import asyncio
    import subprocess
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "Demo.mp4").write_bytes(b"video")

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    asyncio.run(mod._extract_speaker_references("Demo", [
        {"start_s": 0.0, "end_s": 4.0, "speaker": "SPEAKER_00"},
    ]))

    speakers_dir = tmp_path / "speakers" / "es"
    assert not (speakers_dir / "Demo__SPEAKER_00.wav").exists()


def test_diarize_endpoint_chunks_videos_above_duration_limit(tmp_path, monkeypatch):
    import asyncio
    import json
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "Demo.mp4").write_bytes(b"video")
    transcriptions = data_dir / "transcriptions" / "whisper"
    translations = data_dir / "translations" / "argos"
    transcriptions.mkdir(parents=True)
    translations.mkdir(parents=True)
    payload = {
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "A"},
            {"start": 60.0, "end": 90.0, "text": "B"},
        ]
    }
    (transcriptions / "Demo.json").write_text(json.dumps(payload))
    (translations / "Demo.json").write_text(json.dumps(payload))

    ffmpeg_outputs = []

    def fake_run(cmd, **kwargs):
        from pathlib import Path
        output = Path(cmd[-1])
        ffmpeg_outputs.append(output.name)
        output.write_bytes(b"RIFF" + b"\0" * 100)

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)
    monkeypatch.setattr(mod.settings, "diarization_max_seconds", 10.0)
    monkeypatch.setattr(mod.settings, "diarization_chunk_seconds", 60.0)
    monkeypatch.setattr(mod, "resolve_title", lambda video_id: "Demo")
    monkeypatch.setattr(mod, "_probe_media_duration_seconds", lambda path: 121.0)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    async def fake_extract_refs(*args, **kwargs):
        return None

    monkeypatch.setattr(mod, "_extract_speaker_references", fake_extract_refs)

    def fake_diarize(path):
        if "chunk0001" in path:
            return [{"start_s": 1.0, "end_s": 2.0, "speaker": "SPEAKER_01"}]
        return [{"start_s": 1.0, "end_s": 2.0, "speaker": "SPEAKER_00"}]

    monkeypatch.setattr(mod._alignment_service, "diarize", fake_diarize)

    response = asyncio.run(mod.diarize_endpoint("demo-id"))

    assert response.skipped is False
    assert response.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment.start_s for segment in response.segments] == [1.0, 61.0, 121.0]
    assert any(name.endswith(".chunk0000.wav") for name in ffmpeg_outputs)
    assert (data_dir / "diarizations" / "Demo.json").exists()


def test_diarize_endpoint_uses_single_audio_for_short_videos(tmp_path, monkeypatch):
    import asyncio
    import json
    import api.src.routers.diarize as mod

    data_dir = tmp_path / "api"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "Short.mp4").write_bytes(b"video")
    transcriptions = data_dir / "transcriptions" / "whisper"
    translations = data_dir / "translations" / "argos"
    transcriptions.mkdir(parents=True)
    translations.mkdir(parents=True)
    payload = {"segments": [{"start": 0.0, "end": 5.0, "text": "A"}]}
    (transcriptions / "Short.json").write_text(json.dumps(payload))
    (translations / "Short.json").write_text(json.dumps(payload))

    ffmpeg_outputs = []

    def fake_run(cmd, **kwargs):
        from pathlib import Path
        output = Path(cmd[-1])
        ffmpeg_outputs.append(output.name)
        output.write_bytes(b"RIFF" + b"\0" * 100)

    async def fake_extract_refs(*args, **kwargs):
        return None

    monkeypatch.setattr(mod.settings, "data_dir", data_dir)
    monkeypatch.setattr(mod.settings, "diarization_max_seconds", 600.0)
    monkeypatch.setattr(mod, "resolve_title", lambda video_id: "Short")
    monkeypatch.setattr(mod, "_probe_media_duration_seconds", lambda path: 120.0)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "_extract_speaker_references", fake_extract_refs)
    monkeypatch.setattr(
        mod._alignment_service,
        "diarize",
        lambda path: [{"start_s": 1.0, "end_s": 2.0, "speaker": "SPEAKER_00"}],
    )

    response = asyncio.run(mod.diarize_endpoint("short-id"))

    assert response.skipped is False
    assert response.speakers == ["SPEAKER_00"]
    assert [segment.start_s for segment in response.segments] == [1.0]
    assert ffmpeg_outputs == ["Short.wav"]
    saved = json.loads((data_dir / "translations" / "argos" / "Short.json").read_text())
    assert saved["segments"][0]["speaker"] == "Short__SPEAKER_00"

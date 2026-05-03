"""Tests for alignment wiring in tts.py."""
import json
import pathlib
import tempfile
import pytest
from unittest.mock import MagicMock, patch


def test_synced_segment_stretch_factor_changes_speed(monkeypatch):
    """stretch_factor changes the computed speed ratio: larger factor = lower speed_factor."""
    import api.src.services.tts_engine as tts
    monkeypatch.setattr(tts, "_ALIGNMENT_ENABLED", True)
    from api.src.services.tts_engine import _synced_segment_audio
    import numpy as np
    import soundfile as sf

    sr = 22050
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = pathlib.Path(tmpdir) / "source_15s.wav"
        # 1.5-second raw audio
        sf.write(str(raw_wav), np.zeros(int(sr * 1.5), dtype=np.float32), sr)

        engine = MagicMock()
        def fake_tts(text, file_path, **kwargs):
            import shutil
            shutil.copy(raw_wav, file_path)
        engine.tts_to_file.side_effect = fake_tts

        # stretch_factor=1.0: effective_target=1.0, speed=1.5 → clamped to 1.25
        _, sf_tight, _ = _synced_segment_audio(engine, "hola", target_sec=1.0, work_dir=tmpdir, stretch_factor=1.0)
        # stretch_factor=1.5: effective_target=1.5, speed=1.0 → not clamped
        _, sf_relaxed, _ = _synced_segment_audio(engine, "hola", target_sec=1.0, work_dir=tmpdir, stretch_factor=1.5)

        assert sf_tight == pytest.approx(1.25, abs=0.01)  # hit the clamp
        assert sf_relaxed == pytest.approx(1.0, abs=0.01)  # exactly fits
        assert sf_tight > sf_relaxed  # tighter budget → higher speed


def test_synced_segment_clamp_applied(monkeypatch):
    """Speed factor is clamped to [0.85, 1.25] in alignment-enabled mode."""
    import api.src.services.tts_engine as tts
    monkeypatch.setattr(tts, "_ALIGNMENT_ENABLED", True)
    from api.src.services.tts_engine import _synced_segment_audio
    import numpy as np
    import soundfile as sf

    sr = 22050
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = pathlib.Path(tmpdir) / "source_4s.wav"
        sf.write(str(raw_wav), np.zeros(sr * 4, dtype=np.float32), sr)

        engine = MagicMock()
        def fake_tts(text, file_path, **kwargs):
            import shutil
            shutil.copy(raw_wav, file_path)
        engine.tts_to_file.side_effect = fake_tts

        audio, sf_val, rd = _synced_segment_audio(engine, "test", target_sec=1.0, work_dir=tmpdir, stretch_factor=1.0)
        assert audio is not None
        assert sf_val <= 1.25 + 1e-9
        assert sf_val >= 0.85 - 1e-9  # also within lower bound


def test_baseline_mode_does_not_slow_short_tts(monkeypatch):
    """Baseline mode should pad short TTS audio instead of stretching it slower."""
    import numpy as np
    import soundfile as sf
    from api.src.services.tts_engine import _postprocess_segment

    sr = 22050
    monkeypatch.setattr(
        "api.src.services.tts_engine.pyrubberband.time_stretch",
        lambda y, sr, speed: y,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = pathlib.Path(tmpdir) / "source_1s.wav"
        sf.write(str(raw_wav), np.zeros(sr, dtype=np.float32), sr)

        audio, speed_factor, raw_duration = _postprocess_segment(
            raw_wav.read_bytes(),
            target_sec=3.0,
            stretch_factor=1.0,
            alignment_enabled=False,
            work_dir=tmpdir,
        )

        assert audio is not None
        assert speed_factor == pytest.approx(1.0, abs=0.01)
        assert raw_duration == pytest.approx(1.0, abs=0.01)
        assert len(audio) == pytest.approx(3000, abs=50)


def test_effective_segment_end_clamps_overlap():
    """TTS timing should use the next start when caption windows overlap."""
    from api.src.services.tts_engine import _clean_tts_text, _effective_segment_end

    segments = [
        {"start": 0.0, "end": 3.0, "text": "> Hola"},
        {"start": 1.5, "end": 4.0, "text": "Mundo"},
    ]

    assert _effective_segment_end(segments, 0) == pytest.approx(1.5)
    assert _effective_segment_end(segments, 1) == pytest.approx(4.0)
    assert _clean_tts_text("> Hola") == "Hola"


def test_group_segment_metas_combines_fragments_until_sentence_end():
    """Adjacent fragments should synthesize as one phrase for smoother prosody."""
    from api.src.services.tts_engine import _group_segment_metas

    metas = [
        {"index": 0, "start": 0.0, "end": 1.0, "target_sec": 1.0, "text": "Hola", "speaker": None},
        {"index": 1, "start": 1.0, "end": 2.0, "target_sec": 1.0, "text": "mundo.", "speaker": None},
    ]

    groups = _group_segment_metas(metas)

    assert len(groups) == 1
    assert groups[0]["text"] == "Hola mundo."
    assert groups[0]["source_indices"] == [0, 1]
    assert groups[0]["target_sec"] == pytest.approx(2.0)


def test_text_file_to_speech_calls_alignment(tmp_path):
    """text_file_to_speech calls _build_alignment and passes its stretch_factor."""
    from api.src.services.tts_engine import text_file_to_speech

    es_seg = {"start": 0.0, "end": 3.0, "text": "Hola mundo"}
    en_seg = {"start": 0.0, "end": 3.0, "text": "Hello world"}

    es_dir = tmp_path / "translations" / "argos"
    en_dir = tmp_path / "transcriptions" / "whisper"
    es_dir.mkdir(parents=True)
    en_dir.mkdir(parents=True)

    title = "test_video"
    es_path = es_dir / f"{title}.json"
    en_path = en_dir / f"{title}.json"
    es_path.write_text(json.dumps({"segments": [es_seg], "text": "Hola mundo"}))
    en_path.write_text(json.dumps({"segments": [en_seg], "text": "Hello world"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    called_with_stretch = []

    def fake_synced(engine, text, target_sec, work_dir, stretch_factor=1.0):
        called_with_stretch.append(stretch_factor)
        from pydub import AudioSegment
        return AudioSegment.silent(duration=int(target_sec * 1000)), 1.0, target_sec

    # Patch _build_alignment to return a known stretch_factor so we can verify it propagates
    from foreign_whispers.alignment import AlignAction
    mock_aligned_seg = MagicMock()
    mock_aligned_seg.stretch_factor = 1.2
    mock_aligned_seg.action = AlignAction.MILD_STRETCH

    engine = MagicMock()
    with patch("api.src.services.tts_engine._synced_segment_audio", side_effect=fake_synced), \
         patch("api.src.services.tts_engine._build_alignment", return_value=([], {0: mock_aligned_seg})):
        text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)

    assert len(called_with_stretch) == 1
    assert called_with_stretch[0] == pytest.approx(1.2, abs=0.01)  # propagated from align_map


def test_text_file_to_speech_missing_en_transcript(tmp_path):
    """When EN transcript is absent, alignment is skipped; synthesis still runs."""
    from api.src.services.tts_engine import text_file_to_speech

    # Only ES transcript, no EN counterpart
    es_seg = {"start": 0.0, "end": 2.0, "text": "Hola mundo"}
    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "no_en"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({"segments": [es_seg], "text": "Hola mundo"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    called_with_stretch = []

    def fake_synced(engine, text, target_sec, work_dir, stretch_factor=1.0):
        called_with_stretch.append(stretch_factor)
        from pydub import AudioSegment
        return AudioSegment.silent(duration=int(target_sec * 1000)), 1.0, target_sec

    engine = MagicMock()
    with patch("api.src.services.tts_engine._synced_segment_audio", side_effect=fake_synced):
        text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)

    # Synthesis ran even without EN transcript
    assert len(called_with_stretch) == 1
    # Fallback: stretch_factor = 1.0 (no alignment)
    assert called_with_stretch[0] == pytest.approx(1.0, abs=0.01)
    # WAV was written
    assert (out_dir / f"{title}.wav").exists()


def test_shorten_segment_text_returns_original_when_stub():
    """_shorten_segment_text returns original ES text when stub returns []."""
    from api.src.services.tts_engine import _shorten_segment_text

    result = _shorten_segment_text(
        en_text="This is a long sentence.",
        es_text="Esta es una frase muy larga.",
        target_sec=2.0,
    )
    assert result == "Esta es una frase muy larga."


def test_text_file_to_speech_calls_shorten_for_request_shorter(tmp_path):
    """text_file_to_speech calls _shorten_segment_text for REQUEST_SHORTER segments."""
    from api.src.services.tts_engine import text_file_to_speech
    from foreign_whispers.alignment import AlignAction
    import json

    es_seg = {"start": 0.0, "end": 3.0, "text": "Esta es una oración muy larga que no cabe."}
    en_seg = {"start": 0.0, "end": 3.0, "text": "Hello world"}

    es_dir = tmp_path / "translations" / "argos"
    en_dir = tmp_path / "transcriptions" / "whisper"
    es_dir.mkdir(parents=True, exist_ok=True); en_dir.mkdir(parents=True, exist_ok=True)

    title = "test_shorten"
    (es_dir / f"{title}.json").write_text(json.dumps({"segments": [es_seg], "text": es_seg["text"]}))
    (en_dir / f"{title}.json").write_text(json.dumps({"segments": [en_seg], "text": en_seg["text"]}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    shorten_calls = []

    def fake_shorten(en_text, es_text, target_sec):
        shorten_calls.append((en_text, es_text, target_sec))
        return es_text

    def fake_synced(engine, text, target_sec, work_dir, stretch_factor=1.0):
        from pydub import AudioSegment
        return AudioSegment.silent(duration=int(target_sec * 1000)), 1.0, target_sec

    # Inject REQUEST_SHORTER action directly — don't depend on heuristic thresholds
    mock_aligned_seg = MagicMock()
    mock_aligned_seg.stretch_factor = 1.0
    mock_aligned_seg.action = AlignAction.REQUEST_SHORTER

    engine = MagicMock()
    with patch("api.src.services.tts_engine._shorten_segment_text", side_effect=fake_shorten), \
         patch("api.src.services.tts_engine._synced_segment_audio", side_effect=fake_synced), \
         patch("api.src.services.tts_engine._build_alignment", return_value=([], {0: mock_aligned_seg})):
        text_file_to_speech(str(es_dir / f"{title}.json"), str(out_dir), tts_engine=engine)

    assert len(shorten_calls) == 1, "Expected _shorten_segment_text to be called once"
    assert shorten_calls[0][1] == es_seg["text"]


def test_shorten_segment_text_fallback_on_exception():
    """_shorten_segment_text returns original text when reranking raises."""
    with patch("foreign_whispers.reranking.get_shorter_translations", side_effect=RuntimeError("boom")):
        from api.src.services.tts_engine import _shorten_segment_text
        result = _shorten_segment_text("source", "target", 2.0)
        assert result == "target"


def test_chatterbox_client_retries_transient_failures(monkeypatch):
    """Chatterbox calls should retry so long videos survive transient timeouts."""
    import requests
    import api.src.services.tts_engine as tts
    from api.src.services.tts_engine import ChatterboxClient

    calls = {"count": 0}

    class Response:
        content = b"RIFF-data"

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.Timeout("slow chunk")
        return Response()

    monkeypatch.setattr(tts, "_TTS_RETRIES", 2)
    monkeypatch.setattr(tts, "_TTS_RETRY_BACKOFF_SEC", 0)
    monkeypatch.setattr("api.src.services.tts_engine.requests.post", fake_post)

    data = ChatterboxClient(base_url="http://tts")._synthesize_default("hola")

    assert data == b"RIFF-data"
    assert calls["count"] == 2


def test_clean_tts_text_skips_caption_artifacts():
    from api.src.services.tts_engine import _clean_tts_text

    assert _clean_tts_text("###") == ""
    assert _clean_tts_text("[Aplausos]") == ""
    assert _clean_tts_text("Es ########") == "Es"
    assert _clean_tts_text("que cuando ella rompió [Aplausos]") == "que cuando ella rompió"
    assert _clean_tts_text("±] perfeccionando") == "perfeccionando"


def test_text_file_to_speech_does_not_extract_segment_refs_by_default(tmp_path):
    """Voice cloning should not auto-upload per-segment clips unless explicitly enabled."""
    from api.src.services.tts_engine import text_file_to_speech

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "voice_default"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Hola",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hola"}],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    speaker_wavs = []

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            speaker_wavs.append(kwargs.get("speaker_wav"))
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    with patch("api.src.services.tts_engine.resolve_speaker_wav", side_effect=FileNotFoundError):
        text_file_to_speech(str(es_path), str(out_dir), tts_engine=Engine(), voice_cloning=True)

    assert speaker_wavs == [None]


def test_text_file_to_speech_uses_resolved_refs_for_speaker_matching(tmp_path):
    """Diarized voice cloning should use extracted speaker refs when requested."""
    from api.src.services.tts_engine import text_file_to_speech

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "voice_resolved_default"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Hola",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hola", "speaker": "SPEAKER_00"}],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    speaker_wavs = []

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            speaker_wavs.append(kwargs.get("speaker_wav"))
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    with patch("api.src.services.tts_engine.resolve_speaker_wav", return_value="es/SPEAKER_00.wav"):
        text_file_to_speech(str(es_path), str(out_dir), tts_engine=Engine(), voice_cloning=True)

    assert speaker_wavs == ["es/SPEAKER_00.wav"]


def test_speaker_color_preserves_duration():
    import math
    import struct
    from pydub import AudioSegment
    from api.src.services.tts_engine import _apply_speaker_color

    sample_rate = 16000
    samples = bytearray()
    for i in range(sample_rate):
        value = int(12000 * math.sin(2 * math.pi * 220 * (i / sample_rate)))
        samples.extend(struct.pack("<h", value))
    audio = AudioSegment(
        bytes(samples),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    colored = _apply_speaker_color(audio, "SPEAKER_00")

    assert len(colored) == len(audio)
    assert colored.raw_data != audio.raw_data


def test_text_file_to_speech_uses_explicit_speaker_wav(tmp_path):
    from api.src.services.tts_engine import text_file_to_speech

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "explicit_voice"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Hola",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hola", "speaker": "SPEAKER_00"}],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    speaker_wavs = []

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            speaker_wavs.append(kwargs.get("speaker_wav"))
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    text_file_to_speech(
        str(es_path),
        str(out_dir),
        tts_engine=Engine(),
        voice_cloning=True,
        speaker_wav="es/SPEAKER_00.wav",
    )

    assert speaker_wavs == ["es/SPEAKER_00.wav"]


def test_text_file_to_speech_reuses_raw_phrase_cache(tmp_path):
    from api.src.services.tts_engine import text_file_to_speech

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "cache_resume"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Hola",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hola"}],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    calls = {"count": 0}

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            calls["count"] += 1
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    engine = Engine()
    text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)
    text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)

    assert calls["count"] == 1


def test_text_file_to_speech_groups_long_form_without_silence_cap(tmp_path, monkeypatch):
    from api.src.services import tts_engine
    from api.src.services.tts_engine import text_file_to_speech

    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_THRESHOLD", 2)
    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_SEC", 3.0)

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "long_form_cap"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Uno. Dos. Tres.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Uno."},
            {"start": 1.0, "end": 2.0, "text": "Dos."},
            {"start": 2.0, "end": 3.0, "text": "Tres."},
        ],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    calls = {"count": 0}

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            calls["count"] += 1
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    text_file_to_speech(str(es_path), str(out_dir), tts_engine=Engine())

    assert calls["count"] == 1
    assert (out_dir / f"{title}.wav").exists()


def test_text_file_to_speech_uses_fast_engine_for_long_form_owned_engine(tmp_path, monkeypatch):
    from api.src.services import tts_engine
    from api.src.services.tts_engine import text_file_to_speech

    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_THRESHOLD", 2)
    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_SEC", 3.0)
    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_ENGINE", "flite")
    monkeypatch.setattr(
        tts_engine,
        "_get_tts_engine",
        lambda: (_ for _ in ()).throw(AssertionError("Chatterbox should not load")),
    )

    calls = {"count": 0}

    class FastEngine:
        def tts_to_file(self, text, file_path, **kwargs):
            calls["count"] += 1
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    monkeypatch.setattr(tts_engine, "FfmpegFliteTTSEngine", FastEngine)

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "long_form_fast"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Uno. Dos. Tres.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Uno."},
            {"start": 1.0, "end": 2.0, "text": "Dos."},
            {"start": 2.0, "end": 3.0, "text": "Tres."},
        ],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    text_file_to_speech(str(es_path), str(out_dir))

    assert calls["count"] == 1
    assert (out_dir / f"{title}.wav").exists()


def test_text_file_to_speech_keeps_chatterbox_for_long_form_speaker_matching(tmp_path, monkeypatch):
    from api.src.services import tts_engine
    from api.src.services.tts_engine import text_file_to_speech

    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_THRESHOLD", 2)
    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_GROUP_SEC", 3.0)
    monkeypatch.setattr(tts_engine, "_TTS_LONG_FORM_ENGINE", "flite")
    monkeypatch.setattr(
        tts_engine,
        "FfmpegFliteTTSEngine",
        lambda: (_ for _ in ()).throw(AssertionError("fast fallback should not be used")),
    )

    speaker_wavs = []

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            speaker_wavs.append(kwargs.get("speaker_wav"))
            from pydub import AudioSegment
            AudioSegment.silent(duration=500).export(file_path, format="wav")

    monkeypatch.setattr(tts_engine, "_get_tts_engine", lambda: Engine())
    monkeypatch.setattr(tts_engine, "resolve_speaker_wav", lambda *args: "es/SPEAKER_00.wav")

    es_dir = tmp_path / "translations" / "argos"
    es_dir.mkdir(parents=True)
    title = "long_form_speaker_match"
    es_path = es_dir / f"{title}.json"
    es_path.write_text(json.dumps({
        "language": "es",
        "text": "Uno. Dos. Tres.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Uno.", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "Dos.", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": "Tres.", "speaker": "SPEAKER_00"},
        ],
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    text_file_to_speech(str(es_path), str(out_dir), voice_cloning=True)

    assert speaker_wavs == ["es/SPEAKER_00.wav"]
    assert (out_dir / f"{title}.wav").exists()


def test_parallel_raw_synthesis_serializes_voice_uploads(tmp_path):
    from api.src.services.tts_engine import _synthesize_pending_raw

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            from pydub import AudioSegment
            AudioSegment.silent(duration=100).export(file_path, format="wav")

    pending = [
        {
            "index": 0,
            "text": "Hola",
            "speaker_wav": "es/SPEAKER_00.wav",
            "wav_path": str(tmp_path / "0.wav"),
            "cache_path": tmp_path / "cache0.wav",
        },
        {
            "index": 1,
            "text": "Mundo",
            "speaker_wav": "es/SPEAKER_00.wav",
            "wav_path": str(tmp_path / "1.wav"),
            "cache_path": tmp_path / "cache1.wav",
        },
    ]

    results = _synthesize_pending_raw(Engine(), pending, max_workers=4)

    assert sorted(results) == [0, 1]
    assert all(results.values())


def test_chatterbox_ignores_empty_speaker_wav(tmp_path, monkeypatch):
    from api.src.services.tts_engine import ChatterboxClient

    empty_ref = tmp_path / "empty.wav"
    empty_ref.write_bytes(b"")

    called = {"default": False}

    def fake_default(self, text):
        called["default"] = True
        return b"RIFFfake"

    monkeypatch.setattr(ChatterboxClient, "_synthesize_default", fake_default)

    data = ChatterboxClient()._synthesize_with_voice("hola", str(empty_ref))

    assert data == b"RIFFfake"
    assert called["default"] is True


def test_chatterbox_rewinds_upload_file_on_retry(tmp_path, monkeypatch):
    import requests
    from api.src.services.tts_engine import ChatterboxClient

    ref = tmp_path / "speaker.wav"
    ref.write_bytes(b"RIFF" + b"x" * 100)
    sizes = []

    class Response:
        def __init__(self, ok):
            self.content = b"RIFFok" if ok else b""
            self._ok = ok

        def raise_for_status(self):
            if not self._ok:
                raise requests.HTTPError("temporary")

    def fake_post(url, timeout=None, **kwargs):
        file_obj = kwargs["files"]["voice_file"][1]
        sizes.append(len(file_obj.read()))
        return Response(ok=len(sizes) == 2)

    monkeypatch.setattr("api.src.services.tts_engine.requests.post", fake_post)
    monkeypatch.setattr("api.src.services.tts_engine.time.sleep", lambda *_: None)

    data = ChatterboxClient(base_url="http://tts")._synthesize_with_voice("hola", str(ref))

    assert data == b"RIFFok"
    assert sizes == [104, 104]


def test_chatterbox_falls_back_when_voice_upload_fails(tmp_path, monkeypatch):
    import requests
    from api.src.services.tts_engine import ChatterboxClient

    ref = tmp_path / "speaker.wav"
    ref.write_bytes(b"RIFF" + b"x" * 100)

    def fake_request(self, url, **kwargs):
        if url.endswith("/upload"):
            raise requests.HTTPError("voice failed")
        return b"RIFFdefault"

    monkeypatch.setattr(ChatterboxClient, "_request_with_retries", fake_request)

    data = ChatterboxClient(base_url="http://tts")._synthesize_with_voice("hola", str(ref))

    assert data == b"RIFFdefault"


def test_voice_upload_synthesis_uses_silence_by_default_after_consecutive_errors(tmp_path, monkeypatch):
    from api.src.services import tts_engine
    from api.src.services.tts_engine import _synthesize_pending_raw

    monkeypatch.setattr(tts_engine, "_TTS_FAIL_FAST", False)

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            raise RuntimeError("backend down")

    pending = [
        {
            "index": i,
            "text": f"Phrase {i}",
            "speaker_wav": "es/SPEAKER_00.wav",
            "wav_path": str(tmp_path / f"{i}.wav"),
            "cache_path": tmp_path / f"cache{i}.wav",
        }
        for i in range(10)
    ]

    results = _synthesize_pending_raw(Engine(), pending, max_workers=4, max_consecutive_failures=3)

    assert sorted(results) == list(range(10))
    assert all(value is None for value in results.values())


def test_voice_upload_synthesis_can_fail_fast_after_consecutive_errors(tmp_path, monkeypatch):
    from api.src.services import tts_engine
    import pytest
    from api.src.services.tts_engine import _synthesize_pending_raw

    monkeypatch.setattr(tts_engine, "_TTS_FAIL_FAST", True)

    class Engine:
        def tts_to_file(self, text, file_path, **kwargs):
            raise RuntimeError("backend down")

    pending = [
        {
            "index": i,
            "text": f"Phrase {i}",
            "speaker_wav": "es/SPEAKER_00.wav",
            "wav_path": str(tmp_path / f"{i}.wav"),
            "cache_path": tmp_path / f"cache{i}.wav",
        }
        for i in range(10)
    ]

    with pytest.raises(RuntimeError, match="aborting early"):
        _synthesize_pending_raw(Engine(), pending, max_workers=4, max_consecutive_failures=3)

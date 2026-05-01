"""Shared pytest setup for optional heavy dependencies."""

from __future__ import annotations

import sys
import types


def _install_tts_stub() -> None:
    """Expose ``TTS.api.TTS`` for tests when Coqui TTS is not installed."""
    try:
        __import__("TTS.api")
        return
    except ImportError:
        pass

    tts_pkg = types.ModuleType("TTS")
    api_mod = types.ModuleType("TTS.api")

    class TTS:
        def __init__(self, *args, **kwargs):
            pass

        def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
            from pydub import AudioSegment

            AudioSegment.silent(duration=500).export(file_path, format="wav")

        def to(self, device: str):
            return self

    api_mod.TTS = TTS
    tts_pkg.api = api_mod
    sys.modules["TTS"] = tts_pkg
    sys.modules["TTS.api"] = api_mod


_install_tts_stub()

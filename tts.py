"""Compatibility shim for legacy tests/imports.

The implementation lives in ``api.src.services.tts_engine``.
"""

from importlib import reload

import api.src.services.tts_engine as _engine

_engine = reload(_engine)

_synced_segment_audio = _engine._synced_segment_audio
text_file_to_speech = _engine.text_file_to_speech
text_to_speech = _engine.text_to_speech
tts = _engine.tts


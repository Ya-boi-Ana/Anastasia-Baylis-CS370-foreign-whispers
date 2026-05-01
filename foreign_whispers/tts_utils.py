"""Utilities for natural-speed TTS alignment."""

def safe_duration_hint(
    text: str,
    original_duration_s: float,
    chars_per_second: float = 15.0,
) -> float | None:
    """
    Only return a duration hint when speech would overflow.

    Prevents artificially slow speech caused by stretching
    short text to fill long time windows.
    """

    if not text.strip():
        return None

    estimated_duration = len(text.strip()) / chars_per_second

    # Only constrain when text is too long
    if estimated_duration > original_duration_s:
        return original_duration_s

    return None
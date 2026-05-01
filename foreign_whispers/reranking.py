"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return concise Spanish translation candidates for a duration budget.

    The implementation is local and deterministic: normalize caption markers,
    contract verbose phrases, remove discourse fillers, then fall back to a
    word-boundary trim only if every semantic-preserving candidate is still too
    long.  Candidates closest to the target budget are returned first.
    """
    target_chars = max(8, int(target_duration_s * 15))
    baseline = _normalize_caption_text(baseline_es)
    candidates: list[TranslationCandidate] = []

    def add(text: str, rationale: str) -> None:
        text = _normalize_caption_text(text)
        if text and not any(c.text == text for c in candidates):
            candidates.append(TranslationCandidate(text, len(text), rationale))

    add(baseline, "Normalized caption text")
    contracted = _apply_contractions(baseline)
    add(contracted, "Applied concise Spanish phrasing")
    filler_light = _drop_discourse_fillers(contracted)
    add(filler_light, "Removed discourse fillers")
    compressed = _drop_low_information_phrases(filler_light)
    add(compressed, "Removed low-information phrases")

    if all(c.char_count > target_chars for c in candidates):
        add(_word_boundary_trim(compressed, target_chars), "Trimmed at word boundary as last resort")

    candidates.sort(key=lambda c: (c.char_count > target_chars, abs(c.char_count - target_chars), c.char_count))

    logger.info(
        "get_shorter_translations generated %d candidates for %.1fs budget (%d chars baseline)",
        len(candidates),
        target_duration_s,
        len(baseline),
    )
    return candidates


_CONTRACTIONS = {
    "en este momento": "ahora",
    "en este punto": "ahora",
    "en la actualidad": "ahora",
    "por lo tanto": "así que",
    "debido a": "por",
    "con el fin de": "para",
    "a través de": "por",
    "una cantidad significativa de": "mucho",
    "el resto del mundo": "el mundo",
    "en otras palabras": "o sea",
    "en cierta medida": "algo",
}

_DISCOURSE_FILLERS = {
    "bueno",
    "pues",
    "eh",
    "em",
    "um",
    "este",
    "entonces",
    "básicamente",
    "realmente",
}

_LOW_INFORMATION_PHRASES = [
    r"\bcomo resultado,?\s*",
    r"\bno es que\s+",
    r"\bla realidad es\s+",
]


def _normalize_caption_text(text: str) -> str:
    text = re.sub(r"^\s*(?:>+|&gt;+)\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_contractions(text: str) -> str:
    result = text
    for verbose, concise in _CONTRACTIONS.items():
        result = re.sub(rf"\b{re.escape(verbose)}\b", concise, result, flags=re.IGNORECASE)
    return result


def _drop_discourse_fillers(text: str) -> str:
    kept = []
    for word in text.split():
        normalized = re.sub(r"^[^\wáéíóúüñÁÉÍÓÚÜÑ]+|[^\wáéíóúüñÁÉÍÓÚÜÑ]+$", "", word).lower()
        if normalized not in _DISCOURSE_FILLERS:
            kept.append(word)
    return " ".join(kept)


def _drop_low_information_phrases(text: str) -> str:
    result = text
    for pattern in _LOW_INFORMATION_PHRASES:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result


def _word_boundary_trim(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    kept: list[str] = []
    for word in text.split():
        candidate = " ".join([*kept, word])
        if len(candidate) > target_chars:
            break
        kept.append(word)
    return " ".join(kept) or text[:target_chars].rstrip()

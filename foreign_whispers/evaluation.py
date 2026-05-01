"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict,
) -> dict:
    """Multi-dimensional dubbing quality evaluation.

    Returns normalized scores [0, 1] for different quality dimensions.
    1.0 = perfect, 0.0 = worst possible.

    Dimensions:
        timing_accuracy: How well TTS durations match source windows (inverse of MAE)
        intelligibility: Round-trip STT accuracy if supplied in align_report
        semantic_fidelity: Token overlap proxy unless an embedding score is supplied
        naturalness: Consistency of speaking rates across segments

    Args:
        metrics: Original segment metrics
        aligned: Aligned segments after scheduling
        align_report: From clip_evaluation_report()

    Returns:
        Dict with scores and summary
    """
    if not metrics or not aligned:
        return {
            "timing_accuracy": 0.0,
            "intelligibility": 0.0,
            "semantic_fidelity": 0.0,
            "naturalness": 0.0,
            "overall_score": 0.0,
        }

    # Timing accuracy: inverse of normalized MAE
    mae = align_report.get("mean_abs_duration_error_s", 0)
    avg_duration = _stats.mean(m.source_duration_s for m in metrics)
    normalized_mae = mae / max(avg_duration, 0.1)  # relative error
    timing_accuracy = max(0, 1 - normalized_mae)

    # Intelligibility can be provided by an STT round-trip evaluator. If it is
    # absent, use timing/stretch penalties as a conservative proxy instead of a
    # fake perfect score.
    intelligibility = align_report.get("intelligibility")
    if intelligibility is None:
        retries = align_report.get("n_translation_retries", 0)
        severe = align_report.get("pct_severe_stretch", 0.0) / 100
        intelligibility = max(0.0, 1.0 - severe - 0.03 * retries)

    # Prefer a caller-provided semantic score; otherwise approximate with
    # normalized token overlap between source and target text. This is weak for
    # cross-lingual semantics, but it exposes uncertainty instead of masking it.
    semantic_fidelity = align_report.get("semantic_fidelity")
    if semantic_fidelity is None:
        semantic_fidelity = _token_overlap_proxy(metrics)

    # Naturalness: variance in speaking rates
    rates = []
    for a in aligned:
        if a.scheduled_end > a.scheduled_start:
            rate = len(a.text) / (a.scheduled_end - a.scheduled_start)  # chars per second
            rates.append(rate)
    if rates:
        mean_rate = _stats.mean(rates)
        variance = _stats.variance(rates) if len(rates) > 1 else 0
        # Lower variance = more natural (consistent speaking rate)
        naturalness = max(0, 1 - (variance / (mean_rate ** 2)))  # normalized variance
    else:
        naturalness = 0.0

    overall_score = _stats.mean([timing_accuracy, intelligibility, semantic_fidelity, naturalness])

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "intelligibility": round(float(intelligibility), 3),
        "semantic_fidelity": round(float(semantic_fidelity), 3),
        "naturalness": round(naturalness, 3),
        "overall_score": round(overall_score, 3),
    }


def _token_overlap_proxy(metrics: list[SegmentMetrics]) -> float:
    if not metrics:
        return 0.0
    scores = []
    for m in metrics:
        src = _token_set(m.source_text)
        tgt = _token_set(m.translated_text)
        if not src or not tgt:
            scores.append(0.0)
            continue
        scores.append(len(src & tgt) / len(src | tgt))
    return _stats.mean(scores)


def _token_set(text: str) -> set[str]:
    import re

    return {
        tok
        for tok in re.findall(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", text.lower())
        if len(tok) > 2
    }

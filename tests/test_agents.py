# tests/test_agents.py — renamed module is now foreign_whispers.reranking
from foreign_whispers.reranking import (
    get_shorter_translations,
    analyze_failures,
    TranslationCandidate,
    FailureAnalysis,
)


def test_get_shorter_returns_candidates():
    """Should return list of TranslationCandidate objects."""
    result = get_shorter_translations("hello world", "hola mundo", 1.0)
    assert isinstance(result, list)
    if result:
        assert all(isinstance(c, TranslationCandidate) for c in result)
        # Should be sorted by char_count
        assert all(result[i].char_count <= result[i+1].char_count for i in range(len(result)-1))


def test_analyze_failures_returns_dataclass():
    result = analyze_failures({"mean_abs_duration_error_s": 0.5})
    assert isinstance(result, FailureAnalysis)
    assert result.failure_category == "ok"


def test_analyze_failures_detects_overflow():
    result = analyze_failures({"pct_severe_stretch": 30})
    assert result.failure_category == "duration_overflow"


def test_analyze_failures_detects_drift():
    result = analyze_failures({"total_cumulative_drift_s": 5.0})
    assert result.failure_category == "cumulative_drift"


def test_analyze_failures_detects_stretch_quality():
    result = analyze_failures({"mean_abs_duration_error_s": 1.2})
    assert result.failure_category == "stretch_quality"

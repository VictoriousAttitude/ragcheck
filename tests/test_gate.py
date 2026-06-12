import pytest

from ragcheck.gate import check_gate
from ragcheck.runner import RunResult


def make_result(recall: float, mrr: float = 0.7, fingerprint: str = "abc") -> RunResult:
    return RunResult(
        config={"k": 5, "evalset_fingerprint": fingerprint, "retriever": "x", "n_items": 10},
        summary={"recall@5": recall, "mrr": mrr},
        by_difficulty={},
    )


def test_passes_within_tolerance() -> None:
    outcome = check_gate(make_result(0.80), make_result(0.76), max_drop=0.05)
    assert outcome.passed
    assert "GATE PASSED" in outcome.render()


def test_fails_on_regression_beyond_tolerance() -> None:
    outcome = check_gate(make_result(0.80), make_result(0.70), max_drop=0.05)
    assert not outcome.passed
    assert any("recall@5" in line and "exceeds" in line for line in outcome.lines)


def test_improvement_always_passes() -> None:
    assert check_gate(make_result(0.70), make_result(0.90)).passed


def test_fails_on_different_evalsets() -> None:
    outcome = check_gate(make_result(0.8, fingerprint="a"), make_result(0.8, fingerprint="b"))
    assert not outcome.passed


def test_watched_metric_selection() -> None:
    baseline, current = make_result(0.80, mrr=0.70), make_result(0.60, mrr=0.70)
    assert check_gate(baseline, current, metrics=["mrr"]).passed
    assert not check_gate(baseline, current, metrics=["recall@5"]).passed
    assert not check_gate(baseline, current, metrics=["nonexistent"]).passed


def test_max_drop_validation() -> None:
    with pytest.raises(ValueError):
        check_gate(make_result(0.8), make_result(0.8), max_drop=-0.1)

import dataclasses

import pytest

from ragcheck.gate import check_gate
from ragcheck.metrics import QueryJudgment, recall_at_k
from ragcheck.metrics import mrr as mrr_metric
from ragcheck.runner import RunResult


def make_result(recall: float, mrr: float = 0.7, fingerprint: str = "abc") -> RunResult:
    return RunResult(
        config={"k": 5, "evalset_fingerprint": fingerprint, "retriever": "x", "n_items": 10},
        summary={"recall@5": recall, "mrr": mrr},
        by_difficulty={},
    )


def run_with_hits(hits: list[bool], *, fingerprint: str = "abc") -> RunResult:
    """A run where each query has a single gold span, hit at rank 1 or missed."""
    per_item = []
    judgments = []
    for index, hit in enumerate(hits):
        covered = [[0]] if hit else [[]]
        per_item.append(
            {
                "qid": f"q{index}",
                "query": "x",
                "difficulty": "easy",
                "n_gold": 1,
                "relevant": [hit],
                "covered": covered,
            }
        )
        judgments.append(
            QueryJudgment(n_gold=1, covered=tuple(frozenset(c) for c in covered))
        )
    return RunResult(
        config={"k": 5, "evalset_fingerprint": fingerprint, "retriever": "x", "n_items": len(hits)},
        summary={"recall@5": recall_at_k(judgments, 5), "mrr": mrr_metric(judgments, 5)},
        by_difficulty={},
        per_item=per_item,
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


def test_falls_back_to_point_estimates_without_per_query_data() -> None:
    outcome = check_gate(make_result(0.80), make_result(0.70), max_drop=0.05)
    assert not outcome.passed
    assert any("point estimates" in line for line in outcome.lines)


def test_bootstrap_passes_when_systems_match() -> None:
    baseline = run_with_hits([True, True, False, True, True])
    outcome = check_gate(baseline, run_with_hits([True, True, False, True, True]))
    assert outcome.passed
    assert any("paired bootstrap" in line for line in outcome.lines)
    assert any("CI [" in line for line in outcome.lines)


def test_bootstrap_fails_on_a_large_consistent_regression() -> None:
    baseline = run_with_hits([True] * 8)
    current = run_with_hits([False] * 8)
    outcome = check_gate(baseline, current, max_drop=0.05)
    assert not outcome.passed
    assert any("regression beyond" in line for line in outcome.lines)


def test_bootstrap_tolerates_a_noisy_drop_on_a_tiny_evalset() -> None:
    # One of five queries flips: a -0.2 point drop, but the interval still
    # includes zero, so the build is not tripped by query-sampling noise.
    baseline = run_with_hits([True, True, True, True, True])
    current = run_with_hits([True, True, True, True, False])
    outcome = check_gate(baseline, current, max_drop=0.05)
    assert outcome.passed


def test_no_bootstrap_flag_forces_point_estimates() -> None:
    baseline = run_with_hits([True] * 5)
    # Same per-query data, but the summary claims a regression: only the
    # point-estimate path (which trusts the summary) should fail here.
    current = dataclasses.replace(baseline, summary={"recall@5": 0.0, "mrr": 0.0})
    assert check_gate(baseline, current).passed
    assert not check_gate(baseline, current, bootstrap=False, max_drop=0.05).passed

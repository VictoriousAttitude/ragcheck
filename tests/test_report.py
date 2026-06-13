import pytest

from ragcheck.report import (
    ComparisonRow,
    headline_badge,
    rank_comparison,
    render_comparison,
    render_diff,
    render_markdown,
)
from ragcheck.report.badge import GREEN, RED, YELLOW
from ragcheck.report.diff import diff_results
from ragcheck.runner import RunResult


def make_result(recall: float, fingerprint: str = "abc123") -> RunResult:
    return RunResult(
        config={
            "ragcheck_version": "0.1.0",
            "retriever": "BM25Retriever",
            "k": 5,
            "n_items": 10,
            "evalset_fingerprint": fingerprint,
        },
        summary={
            "hit_rate@5": min(recall + 0.05, 1.0),
            "recall@5": recall,
            "ndcg@5": recall - 0.02,
            "mrr": recall - 0.04,
        },
        by_difficulty={
            "easy": {"n": 6.0, "hit_rate@5": 0.9, "recall@5": 0.88, "ndcg@5": 0.8, "mrr": 0.85},
            "hard": {"n": 4.0, "hit_rate@5": 0.5, "recall@5": 0.45, "ndcg@5": 0.4, "mrr": 0.42},
        },
    )


def test_markdown_contains_config_summary_and_tiers() -> None:
    text = render_markdown(make_result(0.84))
    assert "`BM25Retriever`" in text
    assert "| recall@5 | 0.840 |" in text
    assert "| easy | 6 |" in text
    assert "| hard | 4 |" in text


def test_badge_colors_follow_thresholds() -> None:
    assert GREEN in headline_badge(make_result(0.84))
    assert YELLOW in headline_badge(make_result(0.65))
    assert RED in headline_badge(make_result(0.30))
    assert "84%" in headline_badge(make_result(0.84))
    assert "retrieval recall@5" in headline_badge(make_result(0.84))


def test_diff_entries_and_regression_marker() -> None:
    baseline, current = make_result(0.84), make_result(0.71)
    entries = {e.metric: e for e in diff_results(baseline, current)}
    assert entries["recall@5"].delta == -0.13
    text = render_diff(baseline, current)
    assert "(regression)" in text
    assert "WARNING" not in text


def test_diff_warns_on_different_evalsets() -> None:
    text = render_diff(make_result(0.8, "aaa"), make_result(0.8, "bbb"))
    assert "WARNING" in text


def comparison_rows() -> list[ComparisonRow]:
    return [
        ComparisonRow("bm25", 300, 100, {"recall@5": 0.70, "ndcg@5": 0.60, "mrr": 0.55}),
        ComparisonRow("bm25", 800, 100, {"recall@5": 0.74, "ndcg@5": 0.63, "mrr": 0.60}),
        ComparisonRow("dense", 300, 100, {"recall@5": 0.74, "ndcg@5": 0.58, "mrr": 0.52}),
    ]


def test_rank_orders_by_metric_descending() -> None:
    ranked = rank_comparison(comparison_rows(), "ndcg@5")
    assert [(r.retriever, r.max_chars) for r in ranked][0] == ("bm25", 800)


def test_rank_breaks_ties_deterministically() -> None:
    ranked = rank_comparison(comparison_rows(), "recall@5")
    assert [(r.retriever, r.max_chars) for r in ranked[:2]] == [("bm25", 800), ("dense", 300)]


def test_rank_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        rank_comparison(comparison_rows(), "f1@5")


def test_render_comparison_marks_winner() -> None:
    text = render_comparison(comparison_rows(), "ndcg@5")
    assert "★ bm25" in text
    assert "Best: `bm25` max_chars=800" in text
    assert text.count("★") == 1

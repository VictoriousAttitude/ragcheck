"""Golden tests: every expected value below is computed by hand."""

import pytest

from ragcheck.metrics import QueryJudgment, hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from ragcheck.metrics.core import (
    hit_rate_value,
    mrr_value,
    ndcg_value,
    per_query_resolver,
    recall_value,
)

# Two gold spans; results at ranks 1 and 3 are relevant.
J_TWO_GOLD = QueryJudgment(n_gold=2, covered=(frozenset({0}), frozenset(), frozenset({1})))
# One gold span; nothing relevant.
J_MISS = QueryJudgment(n_gold=1, covered=(frozenset(), frozenset(), frozenset()))
# One gold span; relevant at rank 2.
J_RANK2 = QueryJudgment(n_gold=1, covered=(frozenset(), frozenset({0})))


def test_judgment_requires_gold() -> None:
    with pytest.raises(ValueError):
        QueryJudgment(n_gold=0, covered=())


def test_hit_rate() -> None:
    assert hit_rate_at_k([J_TWO_GOLD], 1) == 1.0
    assert hit_rate_at_k([J_RANK2], 1) == 0.0
    assert hit_rate_at_k([J_RANK2], 2) == 1.0
    assert hit_rate_at_k([J_TWO_GOLD, J_MISS], 3) == 0.5


def test_recall() -> None:
    assert recall_at_k([J_TWO_GOLD], 1) == 0.5
    assert recall_at_k([J_TWO_GOLD], 3) == 1.0
    assert recall_at_k([J_MISS], 3) == 0.0
    assert recall_at_k([J_TWO_GOLD, J_RANK2], 2) == pytest.approx((0.5 + 1.0) / 2)


def test_mrr() -> None:
    assert mrr([J_TWO_GOLD]) == 1.0
    assert mrr([J_RANK2]) == 0.5
    assert mrr([J_MISS]) == 0.0
    assert mrr([J_TWO_GOLD, J_RANK2]) == pytest.approx(0.75)
    assert mrr([J_RANK2], k=1) == 0.0


def test_ndcg() -> None:
    # dcg = 1/log2(2) + 1/log2(4) = 1.5; idcg(2 gold, k=3) = 1 + 1/log2(3)
    expected = 1.5 / (1.0 + 0.6309297535714575)
    assert ndcg_at_k([J_TWO_GOLD], 3) == pytest.approx(expected)
    # single gold at rank 2: dcg = 1/log2(3), idcg = 1
    assert ndcg_at_k([J_RANK2], 2) == pytest.approx(0.6309297535714575)
    assert ndcg_at_k([J_MISS], 3) == 0.0
    # perfect single-result retrieval
    perfect = QueryJudgment(n_gold=1, covered=(frozenset({0}),))
    assert ndcg_at_k([perfect], 1) == 1.0


def test_ndcg_duplicate_coverage_does_not_inflate() -> None:
    # Overlapping chunks hit the same single gold span at every rank.
    duplicated = QueryJudgment(n_gold=1, covered=(frozenset({0}), frozenset({0}), frozenset({0})))
    assert ndcg_at_k([duplicated], 3) == 1.0
    # First coverage at rank 2, duplicated at rank 3: only rank 2 gains.
    late = QueryJudgment(n_gold=1, covered=(frozenset(), frozenset({0}), frozenset({0})))
    assert ndcg_at_k([late], 3) == pytest.approx(0.6309297535714575)


def test_empty_inputs_and_bad_k() -> None:
    assert hit_rate_at_k([], 5) == 0.0
    assert recall_at_k([], 5) == 0.0
    assert mrr([]) == 0.0
    assert ndcg_at_k([], 5) == 0.0
    with pytest.raises(ValueError):
        hit_rate_at_k([J_MISS], 0)


def test_aggregate_is_mean_of_per_query_values() -> None:
    judgments = [J_TWO_GOLD, J_RANK2, J_MISS]
    assert recall_at_k(judgments, 3) == pytest.approx(
        sum(recall_value(j, 3) for j in judgments) / len(judgments)
    )
    assert ndcg_at_k(judgments, 3) == pytest.approx(
        sum(ndcg_value(j, 3) for j in judgments) / len(judgments)
    )
    assert hit_rate_at_k(judgments, 3) == pytest.approx(
        sum(hit_rate_value(j, 3) for j in judgments) / len(judgments)
    )
    assert mrr(judgments, 3) == pytest.approx(
        sum(mrr_value(j, 3) for j in judgments) / len(judgments)
    )


def test_resolver_dispatches_by_name_and_cutoff() -> None:
    assert per_query_resolver("recall@2", 5)(J_TWO_GOLD) == recall_value(J_TWO_GOLD, 2)
    assert per_query_resolver("ndcg@3", 5)(J_TWO_GOLD) == ndcg_value(J_TWO_GOLD, 3)
    assert per_query_resolver("hit_rate@1", 5)(J_RANK2) == hit_rate_value(J_RANK2, 1)


def test_resolver_uses_default_k_for_mrr() -> None:
    assert per_query_resolver("mrr", 1)(J_RANK2) == mrr_value(J_RANK2, 1)
    assert per_query_resolver("mrr", 5)(J_RANK2) == mrr_value(J_RANK2, 5)


def test_resolver_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        per_query_resolver("recall", 5)
    with pytest.raises(ValueError):
        per_query_resolver("f1@5", 5)

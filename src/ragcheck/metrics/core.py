"""Deterministic retrieval metrics over per-query relevance judgments.

All metrics are pure functions of :class:`QueryJudgment` values: same input,
same score, forever. No LLM is involved at any point.

Every aggregate is the mean of a per-query value. The per-query helpers are
public because the regression gate resamples them to put confidence intervals
around a baseline-vs-current change.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryJudgment:
    """Relevance of one query's ranked results against its gold spans.

    ``covered[i]`` holds the indices of the gold spans that the result at rank
    ``i + 1`` overlaps. A result overlapping no gold span has an empty set.
    """

    n_gold: int
    covered: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if self.n_gold < 1:
            raise ValueError("a judgment requires at least one gold span")

    def relevant(self) -> tuple[bool, ...]:
        """Binary relevance per rank."""
        return tuple(bool(c) for c in self.covered)


def hit_rate_value(judgment: QueryJudgment, k: int) -> float:
    """1.0 when at least one of the top *k* results is relevant, else 0.0."""
    return 1.0 if any(judgment.relevant()[:k]) else 0.0


def recall_value(judgment: QueryJudgment, k: int) -> float:
    """Share of this query's gold spans covered by its top *k* results."""
    found: set[int] = set()
    for c in judgment.covered[:k]:
        found |= c
    return len(found) / judgment.n_gold


def mrr_value(judgment: QueryJudgment, k: int | None = None) -> float:
    """Reciprocal rank of the first relevant result, 0.0 when none is found."""
    relevant = judgment.relevant() if k is None else judgment.relevant()[:k]
    for rank, hit in enumerate(relevant, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_value(judgment: QueryJudgment, k: int) -> float:
    """Normalized DCG at *k* with binary, novelty-gated gains.

    A rank gains only when it covers at least one gold span that no earlier
    rank covered. Without the novelty requirement, overlapping chunks hitting
    the same gold span repeatedly would push nDCG above 1.
    """
    dcg = 0.0
    seen: set[int] = set()
    for rank, covered in enumerate(judgment.covered[:k], start=1):
        if covered - seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= covered
    ideal_hits = min(judgment.n_gold, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def hit_rate_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Share of queries with at least one relevant result in the top *k*."""
    _validate(k)
    if not judgments:
        return 0.0
    return sum(hit_rate_value(j, k) for j in judgments) / len(judgments)


def recall_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Mean share of each query's gold spans covered by its top *k* results."""
    _validate(k)
    if not judgments:
        return 0.0
    return sum(recall_value(j, k) for j in judgments) / len(judgments)


def mrr(judgments: Sequence[QueryJudgment], k: int | None = None) -> float:
    """Mean reciprocal rank of the first relevant result, 0 when none is found."""
    if k is not None:
        _validate(k)
    if not judgments:
        return 0.0
    return sum(mrr_value(j, k) for j in judgments) / len(judgments)


def ndcg_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Mean normalized discounted cumulative gain at *k* with binary novel gains."""
    _validate(k)
    if not judgments:
        return 0.0
    return sum(ndcg_value(j, k) for j in judgments) / len(judgments)


def per_query_resolver(name: str, default_k: int) -> Callable[[QueryJudgment], float]:
    """Map a metric name (e.g. ``recall@5``, ``mrr``) to its per-query function.

    ``mrr`` is reported over the full ranking, so it inherits *default_k* (the
    run's k); cutoff metrics carry their k in the name.
    """
    if name == "mrr":

        def mrr_fn(judgment: QueryJudgment) -> float:
            return mrr_value(judgment, default_k)

        return mrr_fn

    base, sep, cut = name.partition("@")
    if not sep or not cut.isdigit():
        raise ValueError(f"cannot resolve metric name {name!r}")
    k = int(cut)
    families: dict[str, Callable[[QueryJudgment, int], float]] = {
        "hit_rate": hit_rate_value,
        "recall": recall_value,
        "ndcg": ndcg_value,
    }
    fn = families.get(base)
    if fn is None:
        raise ValueError(f"unknown metric family {base!r} in {name!r}")

    def at_k(judgment: QueryJudgment) -> float:
        return fn(judgment, k)

    return at_k


def _validate(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

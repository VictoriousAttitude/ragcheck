"""Deterministic retrieval metrics over per-query relevance judgments.

All metrics are pure functions of :class:`QueryJudgment` values: same input,
same score, forever. No LLM is involved at any point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def hit_rate_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Share of queries with at least one relevant result in the top *k*."""
    _validate(judgments, k)
    if not judgments:
        return 0.0
    hits = sum(1 for j in judgments if any(j.relevant()[:k]))
    return hits / len(judgments)


def recall_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Mean share of each query's gold spans covered by its top *k* results."""
    _validate(judgments, k)
    if not judgments:
        return 0.0
    total = 0.0
    for j in judgments:
        found: set[int] = set()
        for c in j.covered[:k]:
            found |= c
        total += len(found) / j.n_gold
    return total / len(judgments)


def mrr(judgments: Sequence[QueryJudgment], k: int | None = None) -> float:
    """Mean reciprocal rank of the first relevant result, 0 when none is found."""
    if k is not None:
        _validate(judgments, k)
    if not judgments:
        return 0.0
    total = 0.0
    for j in judgments:
        relevant = j.relevant() if k is None else j.relevant()[:k]
        for rank, hit in enumerate(relevant, start=1):
            if hit:
                total += 1.0 / rank
                break
    return total / len(judgments)


def ndcg_at_k(judgments: Sequence[QueryJudgment], k: int) -> float:
    """Mean normalized discounted cumulative gain at *k* with binary gains."""
    _validate(judgments, k)
    if not judgments:
        return 0.0
    total = 0.0
    for j in judgments:
        dcg = sum(
            1.0 / math.log2(rank + 1) for rank, hit in enumerate(j.relevant()[:k], start=1) if hit
        )
        ideal_hits = min(j.n_gold, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        total += dcg / idcg
    return total / len(judgments)


def _validate(judgments: Sequence[QueryJudgment], k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    del judgments

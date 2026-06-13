"""Deterministic paired bootstrap for comparing two runs on one query set.

The retrieval metrics are exact given a fixed evalset, so the only uncertainty
in a baseline-vs-current comparison is *which queries* happened to land in the
set. Because both runs answer the same queries, the change in any metric is the
mean of the per-query differences; resampling those differences with
replacement turns query-sampling uncertainty into a confidence interval. The
resampling is seeded, so the interval is reproducible — same input, same gate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """A bootstrap confidence interval around an observed mean difference."""

    point: float
    low: float
    high: float


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> Interval:
    """Percentile-bootstrap CI for the mean of paired per-query differences."""
    n = len(deltas)
    if n == 0:
        raise ValueError("need at least one paired observation")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    point = math.fsum(deltas) / n
    if n == 1 or resamples <= 0:
        return Interval(point, point, point)

    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    alpha = 1.0 - confidence
    return Interval(point, _percentile(means, alpha / 2), _percentile(means, 1.0 - alpha / 2))


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; *values* sorted ascending, *q* in [0, 1]."""
    if len(values) == 1:
        return values[0]
    rank = q * (len(values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] * (high - rank) + values[high] * (rank - low)

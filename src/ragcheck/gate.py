"""Regression gate: fail the build when retrieval quality drops.

Compares a current run against a committed baseline. When both runs carry
per-query data, the gate uses a seeded paired bootstrap and fails only when a
watched metric's whole confidence interval sits below the allowed drop, so
query-sampling noise on a small evalset no longer trips the build. Without
per-query data it falls back to comparing the point estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragcheck.metrics.core import QueryJudgment, per_query_resolver
from ragcheck.report.diff import DiffEntry, diff_results, same_evalset
from ragcheck.runner import RunResult
from ragcheck.stats import Interval, paired_bootstrap_ci

DEFAULT_MAX_DROP = 0.05
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0

PairedJudgments = list[tuple[QueryJudgment, QueryJudgment]]


@dataclass(frozen=True)
class GateOutcome:
    passed: bool
    lines: tuple[str, ...]

    def render(self) -> str:
        verdict = "GATE PASSED" if self.passed else "GATE FAILED"
        return "\n".join([*self.lines, verdict]) + "\n"


def check_gate(
    baseline: RunResult,
    current: RunResult,
    *,
    max_drop: float = DEFAULT_MAX_DROP,
    metrics: Sequence[str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    bootstrap: bool = True,
) -> GateOutcome:
    """Compare *current* to *baseline*; watched metrics default to all shared ones."""
    if max_drop < 0:
        raise ValueError(f"max_drop must be >= 0, got {max_drop}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not same_evalset(baseline, current):
        return GateOutcome(
            passed=False,
            lines=("runs use different evalsets; regenerate the baseline",),
        )

    try:
        entries = _watched_entries(baseline, current, metrics)
    except ValueError as exc:
        return GateOutcome(passed=False, lines=(str(exc),))

    paired = _paired_judgments(baseline, current) if bootstrap else None
    if paired is None:
        note = (
            "bootstrap disabled; comparing point estimates"
            if not bootstrap
            else "per-query data unavailable; comparing point estimates "
            "(rerun without --no-per-item for confidence intervals)"
        )
        return _point_estimate_gate(entries, max_drop, note)
    return _bootstrap_gate(
        entries, paired, baseline, current, max_drop, confidence, resamples, seed
    )


def _watched_entries(
    baseline: RunResult, current: RunResult, metrics: Sequence[str] | None
) -> list[DiffEntry]:
    entries = diff_results(baseline, current)
    if metrics is None:
        return entries
    watched = set(metrics)
    missing = watched - {e.metric for e in entries}
    if missing:
        raise ValueError(f"watched metrics missing from results: {sorted(missing)}")
    return [e for e in entries if e.metric in watched]


def _point_estimate_gate(entries: Sequence[DiffEntry], max_drop: float, note: str) -> GateOutcome:
    lines = [note]
    passed = True
    for entry in entries:
        head = f"{entry.metric}: {entry.baseline:.3f} -> {entry.current:.3f}"
        if entry.delta < -max_drop:
            passed = False
            lines.append(f"{head} ({entry.delta:+.3f}, exceeds allowed drop of {max_drop:.3f})")
        else:
            lines.append(f"{head} ok")
    return GateOutcome(passed=passed, lines=tuple(lines))


def _bootstrap_gate(
    entries: Sequence[DiffEntry],
    paired: PairedJudgments,
    baseline: RunResult,
    current: RunResult,
    max_drop: float,
    confidence: float,
    resamples: int,
    seed: int,
) -> GateOutcome:
    base_k = int(baseline.config["k"])
    cur_k = int(current.config["k"])
    lines = [
        f"paired bootstrap: {len(paired)} queries, {resamples} resamples, "
        f"{round(confidence * 100)}% CI"
    ]
    passed = True
    for entry in entries:
        score_baseline = per_query_resolver(entry.metric, base_k)
        score_current = per_query_resolver(entry.metric, cur_k)
        deltas = [score_current(cur) - score_baseline(base) for base, cur in paired]
        ci = paired_bootstrap_ci(deltas, confidence=confidence, resamples=resamples, seed=seed)
        regressed = ci.high < -max_drop
        passed = passed and not regressed
        lines.append(_bootstrap_line(entry, ci, max_drop, regressed))
    return GateOutcome(passed=passed, lines=tuple(lines))


def _bootstrap_line(entry: DiffEntry, ci: Interval, max_drop: float, regressed: bool) -> str:
    head = (
        f"{entry.metric}: {entry.baseline:.3f} -> {entry.current:.3f} "
        f"({entry.delta:+.3f}, CI [{ci.low:+.3f}, {ci.high:+.3f}])"
    )
    return f"{head} regression beyond {max_drop:.3f}" if regressed else f"{head} ok"


def _paired_judgments(baseline: RunResult, current: RunResult) -> PairedJudgments | None:
    base = _judgments_by_qid(baseline)
    cur = _judgments_by_qid(current)
    if not base or not cur:
        return None
    qids = sorted(base.keys() & cur.keys())
    if not qids:
        return None
    return [(base[qid], cur[qid]) for qid in qids]


def _judgments_by_qid(result: RunResult) -> dict[str, QueryJudgment]:
    judgments: dict[str, QueryJudgment] = {}
    for item in result.per_item:
        covered = tuple(frozenset(ranks) for ranks in item["covered"])
        judgments[item["qid"]] = QueryJudgment(n_gold=item["n_gold"], covered=covered)
    return judgments

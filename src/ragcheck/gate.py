"""Regression gate: fail the build when retrieval quality drops.

The gate compares a current run against a committed baseline and fails when
any watched metric falls more than ``max_drop`` below its baseline value, or
when the two runs were produced from different evalsets (in which case the
comparison would be meaningless).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragcheck.report.diff import diff_results, same_evalset
from ragcheck.runner import RunResult

DEFAULT_MAX_DROP = 0.05


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
) -> GateOutcome:
    """Compare *current* to *baseline*; watched metrics default to all shared ones."""
    if max_drop < 0:
        raise ValueError(f"max_drop must be >= 0, got {max_drop}")
    if not same_evalset(baseline, current):
        return GateOutcome(
            passed=False,
            lines=("runs use different evalsets; regenerate the baseline",),
        )

    entries = diff_results(baseline, current)
    if metrics is not None:
        watched = set(metrics)
        missing = watched - {e.metric for e in entries}
        if missing:
            return GateOutcome(
                passed=False,
                lines=(f"watched metrics missing from results: {sorted(missing)}",),
            )
        entries = [e for e in entries if e.metric in watched]

    lines = []
    passed = True
    for entry in entries:
        if entry.delta < -max_drop:
            passed = False
            lines.append(
                f"{entry.metric}: {entry.baseline:.3f} -> {entry.current:.3f} "
                f"({entry.delta:+.3f}, exceeds allowed drop of {max_drop:.3f})"
            )
        else:
            lines.append(f"{entry.metric}: {entry.baseline:.3f} -> {entry.current:.3f} ok")
    return GateOutcome(passed=passed, lines=tuple(lines))

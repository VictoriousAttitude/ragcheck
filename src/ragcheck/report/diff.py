"""Compare two run results metric by metric."""

from __future__ import annotations

from dataclasses import dataclass

from ragcheck.runner import RunResult


@dataclass(frozen=True)
class DiffEntry:
    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def diff_results(baseline: RunResult, current: RunResult) -> list[DiffEntry]:
    """Entries for every metric present in both summaries, in baseline order."""
    return [
        DiffEntry(metric=name, baseline=value, current=current.summary[name])
        for name, value in baseline.summary.items()
        if name in current.summary
    ]


def same_evalset(baseline: RunResult, current: RunResult) -> bool:
    return bool(baseline.config["evalset_fingerprint"] == current.config["evalset_fingerprint"])


def render_diff(baseline: RunResult, current: RunResult) -> str:
    lines = []
    if not same_evalset(baseline, current):
        lines.append("WARNING: runs use different evalsets; the comparison is not meaningful.")
        lines.append("")
    lines.append(f"{'metric':<14} {'baseline':>9} {'current':>9} {'delta':>8}")
    for entry in diff_results(baseline, current):
        marker = "" if entry.delta >= 0 else "  (regression)"
        lines.append(
            f"{entry.metric:<14} {entry.baseline:>9.3f} {entry.current:>9.3f} "
            f"{entry.delta:>+8.3f}{marker}"
        )
    return "\n".join(lines) + "\n"

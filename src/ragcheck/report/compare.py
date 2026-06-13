"""Rank and render several runs side by side.

``diff`` answers "did this one run regress?"; ``compare`` answers "which of
these configurations is best?" — the half of the measure/compare/gate story
that a pairwise diff cannot express.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonRow:
    """One swept configuration and the metrics it produced."""

    retriever: str
    max_chars: int
    overlap_chars: int
    summary: dict[str, float]


def rank_comparison(rows: Sequence[ComparisonRow], sort_metric: str) -> list[ComparisonRow]:
    """Rows sorted by *sort_metric* descending, ties broken deterministically."""
    if not rows:
        raise ValueError("nothing to compare")
    for row in rows:
        if sort_metric not in row.summary:
            available = ", ".join(sorted(rows[0].summary))
            raise ValueError(f"unknown metric {sort_metric!r}; available: {available}")
    return sorted(
        rows,
        key=lambda r: (-r.summary[sort_metric], r.retriever, r.max_chars, r.overlap_chars),
    )


def render_comparison(rows: Sequence[ComparisonRow], sort_metric: str) -> str:
    """Markdown table of every configuration, best first and starred."""
    ranked = rank_comparison(rows, sort_metric)
    best = ranked[0]
    metrics = list(best.summary)
    header = ["retriever", "max_chars", "overlap", *metrics]
    aligns = ["---", "---:", "---:", *["---:" for _ in metrics]]
    lines = [
        "# ragcheck comparison",
        "",
        f"Ranked by `{sort_metric}`. Best: `{best.retriever}` "
        f"max_chars={best.max_chars} overlap={best.overlap_chars} "
        f"({sort_metric}={best.summary[sort_metric]:.3f}).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for index, row in enumerate(ranked):
        name = f"★ {row.retriever}" if index == 0 else row.retriever
        cells = [name, str(row.max_chars), str(row.overlap_chars)]
        cells += [f"{row.summary[m]:.3f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

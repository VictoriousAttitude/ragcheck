"""Human- and CI-facing views over run results."""

from ragcheck.report.badge import headline_badge, render_badge
from ragcheck.report.compare import ComparisonRow, rank_comparison, render_comparison
from ragcheck.report.diff import render_diff
from ragcheck.report.failures import render_failures
from ragcheck.report.markdown import render_markdown

__all__ = [
    "ComparisonRow",
    "headline_badge",
    "rank_comparison",
    "render_badge",
    "render_comparison",
    "render_diff",
    "render_failures",
    "render_markdown",
]

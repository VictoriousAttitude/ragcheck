"""Human- and CI-facing views over run results."""

from ragcheck.report.badge import headline_badge, render_badge
from ragcheck.report.diff import render_diff
from ragcheck.report.markdown import render_markdown

__all__ = ["headline_badge", "render_badge", "render_diff", "render_markdown"]

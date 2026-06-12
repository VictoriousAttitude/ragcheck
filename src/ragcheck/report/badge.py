"""Flat SVG badge for embedding retrieval quality in a README."""

from __future__ import annotations

from ragcheck.runner import RunResult

GREEN = "#4c1"
YELLOW = "#dfb317"
RED = "#e05d44"

_CHAR_WIDTH = 6.5
_PADDING = 10


def render_badge(label: str, text: str, color: str) -> str:
    label_width = round(_CHAR_WIDTH * len(label)) + _PADDING
    text_width = round(_CHAR_WIDTH * len(text)) + _PADDING
    total = label_width + text_width
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
        f'role="img" aria-label="{label}: {text}">'
        f'<rect width="{label_width}" height="20" fill="#555"/>'
        f'<rect x="{label_width}" width="{text_width}" height="20" fill="{color}"/>'
        f'<g fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
        f'<text x="{label_width / 2}" y="14">{label}</text>'
        f'<text x="{label_width + text_width / 2}" y="14">{text}</text>'
        f"</g></svg>"
    )


def headline_badge(result: RunResult, *, good: float = 0.8, warn: float = 0.6) -> str:
    """Badge for the headline metric: recall at the run's k."""
    k = result.config["k"]
    value = result.summary[f"recall@{k}"]
    color = GREEN if value >= good else YELLOW if value >= warn else RED
    return render_badge(f"retrieval recall@{k}", f"{value:.0%}", color)

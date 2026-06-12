"""Character spans in source documents and how to match against them.

Ground truth in ragcheck is always a :class:`Span` — a half-open character
interval ``[start, end)`` inside a source document. Retrieved chunks are judged
by interval overlap when they carry offsets, or by locating their text inside
the source document when they do not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Span:
    """A half-open character interval ``[start, end)`` in a source document."""

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")


def overlaps(a: Span, b: Span) -> bool:
    """Return True if the two spans share at least one character of one document."""
    return a.doc_id == b.doc_id and a.start < b.end and b.start < a.end


def locate(needle: str, haystack: str) -> tuple[int, int] | None:
    """Find *needle* inside *haystack*, tolerant to whitespace and case differences.

    Returns the character offsets ``(start, end)`` of the match in the original
    *haystack*, or None when the needle does not occur. Matching is performed on
    whitespace-split, casefolded tokens, so chunkers that re-flow line breaks or
    collapse spaces still map back to exact source offsets.
    """
    needle_tokens = [match.group().casefold() for match in _TOKEN.finditer(needle)]
    if not needle_tokens:
        return None
    hay = [(m.group().casefold(), m.start(), m.end()) for m in _TOKEN.finditer(haystack)]
    n = len(needle_tokens)
    for i in range(len(hay) - n + 1):
        if all(hay[i + j][0] == needle_tokens[j] for j in range(n)):
            return hay[i][1], hay[i + n - 1][2]
    return None

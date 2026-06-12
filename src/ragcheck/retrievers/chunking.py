"""Fixed-size chunking that preserves source offsets.

Chunks never split inside a token and always carry the exact half-open
character interval they were cut from, so every chunk can be judged against
span-anchored ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragcheck.corpus.models import Document

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str


def chunk_document(doc: Document, *, max_chars: int = 800, overlap_chars: int = 100) -> list[Chunk]:
    """Cut *doc* into chunks of at most *max_chars*, overlapping by ~*overlap_chars*.

    Boundaries fall on whitespace. A single token longer than *max_chars* becomes
    its own chunk rather than being split.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(f"overlap_chars must be in [0, max_chars), got {overlap_chars}")

    tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(doc.text)]
    if not tokens:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(tokens):
        start = tokens[i][0]
        j = i
        while j + 1 < len(tokens) and tokens[j + 1][1] - start <= max_chars:
            j += 1
        end = tokens[j][1]
        chunks.append(Chunk(doc_id=doc.doc_id, start=start, end=end, text=doc.text[start:end]))
        if j + 1 >= len(tokens):
            break
        next_i = j + 1
        while next_i > i + 1 and tokens[next_i - 1][0] >= end - overlap_chars:
            next_i -= 1
        i = next_i
    return chunks

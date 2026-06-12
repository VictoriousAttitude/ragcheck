"""The adapter contract every retriever implements.

This is the whole integration surface of ragcheck: wrap your existing RAG
stack in an object with a single ``retrieve`` method and it can be evaluated.
Offsets are optional — when a retriever cannot report where a chunk came from,
ragcheck falls back to locating the chunk text inside the source documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked retrieval result.

    ``doc_id``, ``start`` and ``end`` are optional provenance: the source
    document and the half-open character interval the chunk was cut from.
    """

    text: str
    score: float
    doc_id: str | None = None
    start: int | None = None
    end: int | None = None


@runtime_checkable
class Retriever(Protocol):
    """Anything that returns ranked chunks for a query."""

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top *k* chunks for *query*, best first."""
        ...

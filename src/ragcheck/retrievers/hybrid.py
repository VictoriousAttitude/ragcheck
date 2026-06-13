"""Reciprocal-rank-fusion retriever combining several base retrievers.

The fusion is rank-based, so it needs no score calibration between retrievers:
each base contributes ``1 / (rrf_k + rank)`` to every chunk it ranks, and chunks
surfaced by more than one retriever accumulate score from each. The class itself
is dependency-free — it only fuses other retrievers' outputs — so it stays fully
testable even when its bases (e.g. a dense encoder) require the optional extras.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragcheck.retrievers.base import RetrievedChunk, Retriever

DEFAULT_RRF_K = 60
DEFAULT_POOL = 50

Key = tuple[object, ...]


class HybridRetriever:
    """Fuse several retrievers' rankings with reciprocal rank fusion (RRF)."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rrf_k: int = DEFAULT_RRF_K,
        pool: int = DEFAULT_POOL,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retriever needs at least one base retriever")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
        if pool < 1:
            raise ValueError(f"pool must be >= 1, got {pool}")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k
        self._pool = pool

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        depth = max(k, self._pool)
        scores: dict[Key, float] = {}
        # Remember the first chunk seen for each key, with the order it appeared,
        # so ties break deterministically and provenance is preserved.
        first_seen: dict[Key, tuple[int, RetrievedChunk]] = {}
        order = 0
        for retriever in self._retrievers:
            for rank, chunk in enumerate(retriever.retrieve(query, depth)):
                key = _key(chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                if key not in first_seen:
                    first_seen[key] = (order, chunk)
                    order += 1
        ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]][0]))
        results = []
        for key, score in ranked[:k]:
            chunk = first_seen[key][1]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=score,
                    doc_id=chunk.doc_id,
                    start=chunk.start,
                    end=chunk.end,
                )
            )
        return results


def _key(chunk: RetrievedChunk) -> Key:
    """Identity used to fuse the same chunk across retrievers.

    Prefer span provenance (doc + interval); fall back to the text when a
    retriever cannot report where a chunk came from.
    """
    if chunk.doc_id is not None and chunk.start is not None and chunk.end is not None:
        return (chunk.doc_id, chunk.start, chunk.end)
    return (chunk.text,)

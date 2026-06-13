"""Cross-encoder reranking stage layered over any base retriever.

The base retriever supplies a candidate pool; a cross-encoder then re-scores each
``(query, passage)`` pair and the top *k* are returned. The scorer is injected, so
the stage stays dependency-free and testable. The default scorer uses
sentence-transformers and is only imported when requested — install it with
``pip install ragcheck[dense]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragcheck.retrievers.base import RetrievedChunk, Retriever

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 20


class CrossEncoder(Protocol):
    """Scores ``(query, passage)`` pairs for relevance; higher is better."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class SentenceTransformerCrossEncoder:
    """Default cross-encoder backed by sentence-transformers (``ragcheck[dense]``)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder as _CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install it with "
                "'pip install ragcheck[dense]'"
            ) from exc
        self._model = _CrossEncoder(model_name)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(list(pairs))
        return [float(value) for value in scores]


class RerankRetriever:
    """Re-score a base retriever's top candidates with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        scorer: CrossEncoder | None = None,
        *,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        if candidates < 1:
            raise ValueError(f"candidates must be >= 1, got {candidates}")
        self._base = base
        self._scorer = scorer if scorer is not None else SentenceTransformerCrossEncoder()
        self._candidates = candidates

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        pool = self._base.retrieve(query, max(k, self._candidates))
        if not pool:
            return []
        scores = self._scorer.score([(query, chunk.text) for chunk in pool])
        # Sort by score desc; ties keep the base retriever's order (stable).
        ranked = sorted(
            zip(scores, range(len(pool)), pool, strict=True),
            key=lambda item: (-item[0], item[1]),
        )
        results = []
        for score, _index, chunk in ranked[:k]:
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

"""Reference dense retriever with a pluggable sentence encoder.

The encoder is injected, so the retriever itself stays dependency-free and
fully testable. The default encoder uses sentence-transformers and is only
imported when actually requested — install it with ``pip install ragcheck[dense]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from ragcheck.corpus.models import Document
from ragcheck.retrievers.base import RetrievedChunk
from ragcheck.retrievers.chunking import Chunk, chunk_document

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Anything that maps a batch of texts to embedding vectors."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Default encoder backed by sentence-transformers (``ragcheck[dense]``)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install it with "
                "'pip install ragcheck[dense]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        rows = self._model.encode(list(texts))
        return [[float(value) for value in row] for row in rows]


class DenseRetriever:
    """Cosine-similarity retrieval over encoded chunks."""

    def __init__(
        self,
        documents: Iterable[Document],
        encoder: Encoder | None = None,
        *,
        max_chars: int = 800,
        overlap_chars: int = 100,
    ) -> None:
        self._encoder = encoder if encoder is not None else SentenceTransformerEncoder()
        self._chunks: list[Chunk] = [
            chunk
            for doc in documents
            for chunk in chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)
        ]
        vectors = self._encoder.encode([c.text for c in self._chunks]) if self._chunks else []
        self._vectors = [_normalize(v) for v in vectors]

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self._chunks:
            return []
        query_vector = _normalize(self._encoder.encode([query])[0])
        scored = sorted(
            ((_dot(query_vector, vector), index) for index, vector in enumerate(self._vectors)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        results = []
        for score, index in scored[:k]:
            chunk = self._chunks[index]
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


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))

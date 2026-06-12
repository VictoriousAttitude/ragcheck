"""Reference lexical retriever: Okapi BM25 over fixed-size chunks.

Implemented from scratch on the standard library so the core package stays
dependency-free and the scoring is fully transparent.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from ragcheck.corpus.models import Document
from ragcheck.retrievers.base import RetrievedChunk
from ragcheck.retrievers.chunking import Chunk, chunk_document
from ragcheck.text import tokenize


class BM25Retriever:
    """Okapi BM25 (k1/b parametrization) over whitespace-aligned chunks."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        max_chars: int = 800,
        overlap_chars: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = [
            chunk
            for doc in documents
            for chunk in chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)
        ]
        self._term_freqs: list[Counter[str]] = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(self._chunks)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()
        }

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self._term_freqs):
            score = self._score(query_terms, tf, self._lengths[index])
            if score > 0.0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
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

    def _score(self, query_terms: list[str], tf: Counter[str], length: int) -> float:
        score = 0.0
        norm = self._k1 * (1.0 - self._b + self._b * length / self._avg_length)
        for term in query_terms:
            frequency = tf.get(term, 0)
            if frequency == 0:
                continue
            score += self._idf[term] * frequency * (self._k1 + 1.0) / (frequency + norm)
        return score

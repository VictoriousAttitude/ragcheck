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
        term_freqs = [Counter(tokenize(c.text)) for c in self._chunks]
        lengths = [sum(tf.values()) for tf in term_freqs]
        avg_length = (sum(lengths) / len(lengths)) if lengths else 0.0
        # Per-chunk length normalization, precomputed once.
        self._norms = [
            k1 * (1.0 - b + b * length / avg_length) if avg_length else 0.0 for length in lengths
        ]
        # Inverted index: term -> [(chunk index, term frequency)]. Scoring a query
        # touches only chunks that share a term with it instead of the whole corpus.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, tf in enumerate(term_freqs):
            for term, frequency in tf.items():
                self._postings.setdefault(term, []).append((index, frequency))
        n = len(self._chunks)
        self._idf = {
            term: math.log((n - len(postings) + 0.5) / (len(postings) + 0.5) + 1.0)
            for term, postings in self._postings.items()
        }

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scores: dict[int, float] = {}
        for term in tokenize(query):
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self._idf[term]
            for index, frequency in postings:
                gain = idf * frequency * (self._k1 + 1.0) / (frequency + self._norms[index])
                scores[index] = scores.get(index, 0.0) + gain
        scored = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        results = []
        for index, score in scored[:k]:
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

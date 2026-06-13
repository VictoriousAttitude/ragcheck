from collections.abc import Sequence

import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from ragcheck.retrievers.base import RetrievedChunk


class FakeRetriever:
    """Returns a fixed ranking, truncated to k."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._chunks = list(chunks)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._chunks[:k]


def chunk(doc_id: str, start: int, end: int, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(text=text, score=0.0, doc_id=doc_id, start=start, end=end)


ALPHA = chunk("d", 0, 5, "alpha")
BETA = chunk("d", 5, 10, "beta")
GAMMA = chunk("d", 10, 15, "gamma")


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(HybridRetriever([FakeRetriever([ALPHA])]), Retriever)


def test_chunk_found_by_both_retrievers_ranks_first() -> None:
    # BETA is ranked by both bases; ALPHA and GAMMA by only one each.
    hybrid = HybridRetriever([FakeRetriever([ALPHA, BETA]), FakeRetriever([BETA, GAMMA])])
    results = hybrid.retrieve("q", k=3)
    assert [(r.doc_id, r.start, r.end) for r in results] == [
        ("d", 5, 10),
        ("d", 0, 5),
        ("d", 10, 15),
    ]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_same_chunk_is_fused_not_duplicated() -> None:
    hybrid = HybridRetriever([FakeRetriever([BETA]), FakeRetriever([BETA])])
    results = hybrid.retrieve("q", k=5)
    assert len(results) == 1
    # Two rank-0 hits: 1/60 + 1/60.
    assert results[0].score == pytest.approx(2.0 / 60.0)


def test_preserves_provenance_offsets() -> None:
    hybrid = HybridRetriever([FakeRetriever([ALPHA, BETA])])
    for result in hybrid.retrieve("q", k=2):
        assert result.doc_id == "d"
        assert result.start is not None and result.end is not None


def test_k_validation_and_empty_bases() -> None:
    hybrid = HybridRetriever([FakeRetriever([])])
    assert hybrid.retrieve("q", k=3) == []
    with pytest.raises(ValueError):
        hybrid.retrieve("q", k=0)


def test_constructor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever([FakeRetriever([ALPHA])], rrf_k=0)
    with pytest.raises(ValueError):
        HybridRetriever([FakeRetriever([ALPHA])], pool=0)


# --- integration with real reference retrievers (still dependency-free) ---

DOCS = [
    Document.from_text(
        "deploy.md", "Releases ship through a staged rollout controlled by the deploy pipeline."
    ),
    Document.from_text(
        "auth.md", "Signed tokens expire after one hour and are rotated by the identity service."
    ),
    Document.from_text(
        "billing.md", "Invoices are generated monthly and failed charges are retried three times."
    ),
]

VOCAB = ["rollout", "deploy", "tokens", "identity", "expire", "invoices", "charges"]


class FakeEncoder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) for word in VOCAB] for text in texts]


def test_fuses_bm25_and_dense() -> None:
    bm25 = BM25Retriever(DOCS, max_chars=200, overlap_chars=20)
    dense = DenseRetriever(DOCS, FakeEncoder(), max_chars=200, overlap_chars=20)
    results = HybridRetriever([bm25, dense]).retrieve("when do tokens expire", k=3)
    assert results
    assert results[0].doc_id == DOCS[1].doc_id

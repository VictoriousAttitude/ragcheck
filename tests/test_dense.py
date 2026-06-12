from collections.abc import Sequence

import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import DenseRetriever, Retriever

DOCS = [
    Document.from_text(
        "deploy.md",
        "Releases ship through a staged rollout controlled by the deploy pipeline.",
    ),
    Document.from_text(
        "auth.md",
        "Signed tokens expire after one hour and are rotated by the identity service.",
    ),
    Document.from_text(
        "billing.md",
        "Invoices are generated monthly and failed charges are retried three times.",
    ),
]

VOCAB = ["rollout", "deploy", "tokens", "identity", "expire", "invoices", "charges"]


class FakeEncoder:
    """Deterministic bag-of-vocabulary embeddings for testing."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) for word in VOCAB] for text in texts]


def make_retriever() -> DenseRetriever:
    return DenseRetriever(DOCS, FakeEncoder(), max_chars=200, overlap_chars=20)


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(make_retriever(), Retriever)


def test_ranks_semantically_matching_chunk_first() -> None:
    results = make_retriever().retrieve("when do tokens expire", k=3)
    assert results
    assert results[0].doc_id == DOCS[1].doc_id


def test_results_carry_valid_offsets_and_descending_scores() -> None:
    by_id = {d.doc_id: d for d in DOCS}
    results = make_retriever().retrieve("invoices and charges", k=3)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert result.doc_id is not None
        assert result.start is not None and result.end is not None
        assert by_id[result.doc_id].text[result.start : result.end] == result.text


def test_k_validation_and_empty_corpus() -> None:
    retriever = make_retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("anything", k=0)
    assert DenseRetriever([], FakeEncoder()).retrieve("anything", k=3) == []


def test_missing_extra_raises_actionable_error() -> None:
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers installed; error path not reachable")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match=r"ragcheck\[dense\]"):
        DenseRetriever(DOCS)

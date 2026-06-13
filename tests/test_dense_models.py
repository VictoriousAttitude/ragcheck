"""Real-model smoke tests for the optional [dense] extra.

These exercise the actual sentence-transformers encoder and cross-encoder, so
they are skipped unless the extra is installed and only run in CI via
``pytest -m dense`` (the default test run deselects them).
"""

import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.dense

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


def test_dense_real_encoder_ranks_relevant_doc() -> None:
    results = DenseRetriever(DOCS, max_chars=200, overlap_chars=20).retrieve(
        "when do access tokens expire", k=3
    )
    assert results
    assert results[0].doc_id == DOCS[1].doc_id


def test_hybrid_real_models_return_valid_offsets() -> None:
    by_id = {d.doc_id: d for d in DOCS}
    hybrid = HybridRetriever(
        [
            BM25Retriever(DOCS, max_chars=200, overlap_chars=20),
            DenseRetriever(DOCS, max_chars=200, overlap_chars=20),
        ]
    )
    results = hybrid.retrieve("how are failed charges retried", k=3)
    assert results
    for result in results:
        assert result.doc_id is not None
        assert result.start is not None and result.end is not None
        assert by_id[result.doc_id].text[result.start : result.end] == result.text


def test_rerank_real_cross_encoder_orders_by_relevance() -> None:
    rerank = RerankRetriever(BM25Retriever(DOCS, max_chars=200, overlap_chars=20))
    results = rerank.retrieve("when do tokens expire", k=3)
    assert results
    assert results[0].doc_id == DOCS[1].doc_id

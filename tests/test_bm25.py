import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import BM25Retriever, Retriever

DOCS = [
    Document.from_text(
        "deploy.md",
        "Deployment guide. Use the rollout command to deploy the service to production. "
        "Rollbacks are handled by the rollout command as well.",
    ),
    Document.from_text(
        "auth.md",
        "Authentication is enforced through signed tokens. Tokens expire after one hour "
        "and are refreshed by the identity service.",
    ),
    Document.from_text(
        "billing.md",
        "Invoices are generated monthly. The billing service retries failed charges "
        "three times before opening a support ticket.",
    ),
]


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(BM25Retriever(DOCS), Retriever)


def test_ranks_topical_document_first() -> None:
    retriever = BM25Retriever(DOCS)
    results = retriever.retrieve("how do tokens expire", k=3)
    assert results
    assert results[0].doc_id == DOCS[1].doc_id


def test_results_carry_valid_offsets() -> None:
    retriever = BM25Retriever(DOCS)
    by_id = {d.doc_id: d for d in DOCS}
    for result in retriever.retrieve("rollout command", k=3):
        assert result.doc_id is not None
        assert result.start is not None and result.end is not None
        assert by_id[result.doc_id].text[result.start : result.end] == result.text


def test_scores_descend_and_k_is_respected() -> None:
    retriever = BM25Retriever(DOCS, max_chars=60, overlap_chars=10)
    results = retriever.retrieve("service", k=2)
    assert len(results) <= 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unmatched_query_returns_empty() -> None:
    retriever = BM25Retriever(DOCS)
    assert retriever.retrieve("zeppelin chromatography", k=5) == []


def test_k_validation() -> None:
    retriever = BM25Retriever(DOCS)
    with pytest.raises(ValueError):
        retriever.retrieve("anything", k=0)

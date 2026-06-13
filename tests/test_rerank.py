from collections.abc import Sequence

import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import BM25Retriever, RerankRetriever, Retriever
from ragcheck.retrievers.base import RetrievedChunk


class FakeBase:
    """Returns a fixed candidate pool, truncated to k."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._chunks = list(chunks)
        self.last_k: int | None = None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.last_k = k
        return self._chunks[:k]


class TermOverlapScorer:
    """Scores a passage by how many query terms it contains."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [
            float(sum(passage.lower().count(term) for term in query.lower().split()))
            for query, passage in pairs
        ]


def chunk(doc_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, score=0.0, doc_id=doc_id, start=0, end=len(text))


IRRELEVANT = chunk("a", "the cat sat on the mat")
RELEVANT = chunk("b", "tokens expire after one hour")
NOISE = chunk("c", "rollout pipeline deploy")


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(RerankRetriever(FakeBase([RELEVANT]), TermOverlapScorer()), Retriever)


def test_reranks_relevant_candidate_to_the_top() -> None:
    # Base puts the relevant chunk in the middle; the scorer pulls it to rank 0.
    base = FakeBase([IRRELEVANT, RELEVANT, NOISE])
    results = RerankRetriever(base, TermOverlapScorer()).retrieve("when do tokens expire", k=3)
    assert results[0].doc_id == "b"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ties_keep_base_order() -> None:
    base = FakeBase([IRRELEVANT, NOISE])  # both score 0 for this query
    results = RerankRetriever(base, TermOverlapScorer()).retrieve("tokens", k=2)
    assert [r.doc_id for r in results] == ["a", "c"]


def test_over_retrieves_candidates_then_truncates() -> None:
    base = FakeBase([IRRELEVANT, RELEVANT, NOISE])
    results = RerankRetriever(base, TermOverlapScorer(), candidates=10).retrieve("tokens expire", 1)
    assert base.last_k == 10
    assert len(results) == 1
    assert results[0].doc_id == "b"


def test_preserves_provenance_and_uses_scorer_scores() -> None:
    base = FakeBase([RELEVANT, IRRELEVANT])
    results = RerankRetriever(base, TermOverlapScorer()).retrieve("tokens expire", k=2)
    assert results[0].score == pytest.approx(2.0)
    assert results[0].start == 0 and results[0].end == len(RELEVANT.text)


def test_k_validation_and_empty_pool() -> None:
    retriever = RerankRetriever(FakeBase([]), TermOverlapScorer())
    assert retriever.retrieve("anything", k=3) == []
    with pytest.raises(ValueError):
        RerankRetriever(FakeBase([RELEVANT]), TermOverlapScorer()).retrieve("q", k=0)


def test_constructor_rejects_bad_candidates() -> None:
    with pytest.raises(ValueError):
        RerankRetriever(FakeBase([RELEVANT]), TermOverlapScorer(), candidates=0)


def test_missing_extra_raises_actionable_error() -> None:
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers installed; error path not reachable")
    except ImportError:
        pass
    docs = [Document.from_text("a.md", "tokens expire after one hour")]
    with pytest.raises(RuntimeError, match=r"ragcheck\[dense\]"):
        RerankRetriever(BM25Retriever(docs))

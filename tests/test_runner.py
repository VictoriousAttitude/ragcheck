from pathlib import Path

import pytest

from ragcheck.corpus import Document
from ragcheck.dataset import EvalItem
from ragcheck.matching import Span
from ragcheck.retrievers import BM25Retriever, RetrievedChunk
from ragcheck.runner import evaluate, read_results, save_results

DOC = Document.from_text(
    "ops.md",
    "Deployments use a staged rollout with automated health probes. "
    "Signed tokens expire after one hour and are rotated weekly. "
    "Invoices are generated monthly by the billing service.",
)
DOCS = [DOC]

TOKEN_SPAN = Span(DOC.doc_id, 63, 122)  # the signed-tokens sentence


def make_item(query: str, span: Span, difficulty: str = "medium") -> EvalItem:
    qid = EvalItem.derive_qid(query, [span], "test")
    return EvalItem(qid=qid, query=query, answers=(span,), difficulty=difficulty, gen_method="test")


class StaticRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._chunks[:k]


def test_offset_chunks_are_judged_by_interval_overlap() -> None:
    item = make_item("When do tokens expire?", TOKEN_SPAN)
    hit = RetrievedChunk(text="irrelevant", score=1.0, doc_id=DOC.doc_id, start=70, end=90)
    result = evaluate([item], StaticRetriever([hit]), DOCS, k=1)
    assert result.summary["recall@1"] == 1.0
    assert result.summary["mrr"] == 1.0


def test_text_only_chunks_fall_back_to_locating_in_gold_documents() -> None:
    item = make_item("When do tokens expire?", TOKEN_SPAN)
    snippet = DOC.text[TOKEN_SPAN.start : TOKEN_SPAN.end]
    hit = RetrievedChunk(text=snippet.upper(), score=1.0)
    result = evaluate([item], StaticRetriever([hit]), DOCS, k=1)
    assert result.summary["recall@1"] == 1.0


def test_wrong_retrieval_scores_zero() -> None:
    item = make_item("When do tokens expire?", TOKEN_SPAN)
    miss = RetrievedChunk(text="something entirely different", score=1.0)
    result = evaluate([item], StaticRetriever([miss]), DOCS, k=1)
    assert result.summary["recall@1"] == 0.0
    assert result.summary["mrr"] == 0.0


def test_end_to_end_with_bm25() -> None:
    item = make_item("When do signed tokens expire?", TOKEN_SPAN)
    result = evaluate([item], BM25Retriever(DOCS, max_chars=80, overlap_chars=10), DOCS, k=3)
    assert result.summary["hit_rate@3"] == 1.0
    assert result.config["n_items"] == 1
    assert result.config["retriever"] == "BM25Retriever"


def test_by_difficulty_breakdown_counts() -> None:
    easy = make_item("Signed tokens expire after one hour?", TOKEN_SPAN, "easy")
    hard = make_item("How long until credentials lapse?", TOKEN_SPAN, "hard")
    retriever = BM25Retriever(DOCS, max_chars=80, overlap_chars=10)
    result = evaluate([easy, hard], retriever, DOCS, k=3)
    assert result.by_difficulty["easy"]["n"] == 1.0
    assert result.by_difficulty["hard"]["n"] == 1.0
    assert "medium" not in result.by_difficulty


def test_results_roundtrip(tmp_path: Path) -> None:
    item = make_item("When do tokens expire?", TOKEN_SPAN)
    result = evaluate([item], BM25Retriever(DOCS), DOCS, k=2)
    out = tmp_path / "results.json"
    save_results(result, out)
    assert read_results(out) == result


def test_empty_evalset_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate([], BM25Retriever(DOCS), DOCS)

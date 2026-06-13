import pytest

from ragcheck.corpus import Document
from ragcheck.dataset.models import EvalItem
from ragcheck.matching.spans import Span
from ragcheck.report import render_failures
from ragcheck.report.failures import _quote
from ragcheck.retrievers import BM25Retriever
from ragcheck.runner import FailureCase, GoldSpan, RetrievedHit, explain_failures

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


def span_of(doc: Document, substring: str) -> Span:
    start = doc.text.index(substring)
    return Span(doc.doc_id, start, start + len(substring))


def item(query: str, doc: Document, substring: str, difficulty: str = "easy") -> EvalItem:
    answer = span_of(doc, substring)
    return EvalItem(
        qid=EvalItem.derive_qid(query, [answer], "test"),
        query=query,
        answers=(answer,),
        difficulty=difficulty,
        gen_method="test",
    )


def test_orders_misses_before_hits_and_detects_them() -> None:
    hit = item("when do tokens expire", DOCS[1], "tokens expire after one hour")
    miss = item("zeppelin chromatography quasar", DOCS[0], "staged rollout", "hard")
    cases = explain_failures([hit, miss], BM25Retriever(DOCS), DOCS, k=5, limit=5)

    assert [c.qid for c in cases] == [miss.qid, hit.qid]
    assert cases[0].first_hit_rank is None and cases[0].recall == 0.0
    assert cases[0].hits == ()  # nothing matched the query at all
    assert cases[1].first_hit_rank == 1 and cases[1].recall == 1.0


def test_case_carries_gold_text_and_retrieved_chunks() -> None:
    hit = item("when do tokens expire", DOCS[1], "tokens expire after one hour")
    (case,) = explain_failures([hit], BM25Retriever(DOCS), DOCS, k=3, limit=1)
    assert case.gold[0].text == "tokens expire after one hour"
    assert case.hits[0].covered == (0,)
    assert case.hits[0].doc_id == DOCS[1].doc_id


def test_limit_validation() -> None:
    hit = item("tokens", DOCS[1], "tokens expire after one hour")
    with pytest.raises(ValueError):
        explain_failures([hit], BM25Retriever(DOCS), DOCS, limit=0)


def test_render_marks_miss_and_covered_hit() -> None:
    miss = FailureCase(
        qid="aaa",
        query="how often are signing keys rotated",
        difficulty="hard",
        recall=0.0,
        first_hit_rank=None,
        gold=(GoldSpan("auth.md", 40, 76, "are rotated by the identity service."),),
        hits=(RetrievedHit(1, "Signed tokens expire after one hour.", (), "auth.md", 0, 36),),
    )
    found = FailureCase(
        qid="bbb",
        query="when do tokens expire",
        difficulty="easy",
        recall=1.0,
        first_hit_rank=1,
        gold=(GoldSpan("auth.md", 0, 36, "Signed tokens expire after one hour."),),
        hits=(RetrievedHit(1, "Signed tokens expire after one hour.", (0,), "auth.md", 0, 36),),
    )
    out = render_failures([miss, found], k=5, total=10)
    assert "worst 2 of 10 queries by recall@5" in out
    assert "MISS" in out and "none of the top 5 chunks overlap" in out
    assert "first relevant chunk at rank 1" in out
    assert "> 1. auth.md[0:36]" in out  # the covered hit is pointed at
    assert "how often are signing keys rotated" in out


def test_render_handles_no_cases() -> None:
    assert "no queries to explain" in render_failures([], k=5, total=0)


def test_quote_flattens_and_truncates() -> None:
    quoted = _quote("a   b\nc " + "x" * 100)
    assert quoted.startswith('"a b c ')
    assert quoted.endswith('\u2026"')
    assert len(quoted) <= 64 + 2

from pathlib import Path

from ragcheck.corpus import Document
from ragcheck.dataset import EvalItem, generate_evalset, read_evalset, save_evalset

DOC = Document.from_text(
    "handbook.md",
    """# Service handbook

## Deployment

Releases ship through a staged rollout. Each stage bakes for thirty minutes
and an automated health probe decides whether the next stage proceeds.

## Authentication

Signed tokens are required on every request. The identity service rotates
signing keys weekly without downtime.

Token lifetime is one hour.
""",
)


def test_definition_template_extracts_question_and_span() -> None:
    items = generate_evalset([DOC])
    definitions = [i for i in items if i.gen_method == "template:definition"]
    assert any(i.query == "What is Token lifetime?" for i in definitions)
    for item in definitions:
        span = item.answers[0]
        assert span.doc_id == DOC.doc_id
        assert DOC.text[span.start : span.end].rstrip().endswith((".", "!", "?"))


def test_section_template_targets_first_paragraph() -> None:
    items = generate_evalset([DOC])
    sections = {i.query: i for i in items if i.gen_method == "template:section"}
    item = sections["What should I know about Deployment?"]
    span = item.answers[0]
    assert DOC.text[span.start : span.end].startswith("Releases ship")


def test_all_items_have_difficulty_and_metadata() -> None:
    items = generate_evalset([DOC])
    assert items
    for item in items:
        assert item.difficulty in ("easy", "medium", "hard")
        assert 0.0 <= item.meta["anchor_score"] <= 1.0
        assert item.meta["source_path"] == "handbook.md"


def test_leaked_items_are_dropped() -> None:
    # A definition whose subject is the entire answer content leaks fully.
    doc = Document.from_text("tiny.md", "Latency is low.")
    assert generate_evalset([doc], max_leakage=0.3) == []


def test_generation_is_deterministic() -> None:
    assert generate_evalset([DOC]) == generate_evalset([DOC])


def test_evalset_jsonl_roundtrip(tmp_path: Path) -> None:
    items = generate_evalset([DOC])
    out = tmp_path / "evalset.jsonl"
    assert save_evalset(items, out) == len(items)
    assert read_evalset(out) == items


def test_qid_is_content_derived() -> None:
    items = generate_evalset([DOC])
    for item in items:
        assert item.qid == EvalItem.derive_qid(item.query, item.answers, item.gen_method)

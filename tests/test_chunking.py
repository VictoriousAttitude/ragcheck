import pytest

from ragcheck.corpus import Document
from ragcheck.retrievers import chunk_document

WORDS = " ".join(f"word{i:03d}" for i in range(100))  # 100 tokens, 7 chars each


def make_doc(text: str) -> Document:
    return Document.from_text("doc.md", text)


def test_offsets_slice_back_to_chunk_text() -> None:
    doc = make_doc(WORDS)
    for chunk in chunk_document(doc, max_chars=80, overlap_chars=16):
        assert doc.text[chunk.start : chunk.end] == chunk.text
        assert chunk.doc_id == doc.doc_id


def test_chunks_cover_document_and_respect_budget() -> None:
    doc = make_doc(WORDS)
    chunks = chunk_document(doc, max_chars=80, overlap_chars=16)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(doc.text)
    assert all(len(c.text) <= 80 for c in chunks)
    # consecutive chunks overlap
    assert all(b.start < a.end for a, b in zip(chunks, chunks[1:], strict=False))


def test_oversized_token_becomes_own_chunk() -> None:
    doc = make_doc("short " + "x" * 50 + " tail")
    chunks = chunk_document(doc, max_chars=20, overlap_chars=0)
    assert any(c.text == "x" * 50 for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(make_doc("   \n  ")) == []


def test_parameter_validation() -> None:
    doc = make_doc(WORDS)
    with pytest.raises(ValueError):
        chunk_document(doc, max_chars=0)
    with pytest.raises(ValueError):
        chunk_document(doc, max_chars=100, overlap_chars=100)

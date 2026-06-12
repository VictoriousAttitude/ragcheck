import pytest

from ragcheck.matching import Span, locate, overlaps

DOC = "doc-a"


def test_span_rejects_empty_or_negative_intervals() -> None:
    with pytest.raises(ValueError):
        Span(DOC, 5, 5)
    with pytest.raises(ValueError):
        Span(DOC, -1, 4)


def test_overlap_requires_same_document() -> None:
    assert not overlaps(Span("doc-a", 0, 10), Span("doc-b", 0, 10))


def test_overlap_half_open_semantics() -> None:
    assert overlaps(Span(DOC, 0, 10), Span(DOC, 9, 20))
    assert not overlaps(Span(DOC, 0, 10), Span(DOC, 10, 20))
    assert overlaps(Span(DOC, 5, 6), Span(DOC, 0, 100))


def test_locate_exact_text() -> None:
    haystack = "Alpha beta gamma delta."
    assert locate("beta gamma", haystack) == (6, 16)


def test_locate_tolerates_whitespace_and_case() -> None:
    haystack = "Configure the   server\nbefore starting it."
    assert locate("configure THE server", haystack) == (0, 22)


def test_locate_returns_original_offsets_for_reflowed_chunk() -> None:
    haystack = "First line of text.\nSecond line of text.\n"
    found = locate("text. Second line", haystack)
    assert found is not None
    start, end = found
    assert haystack[start:end] == "text.\nSecond line"


def test_locate_missing_and_empty() -> None:
    assert locate("not present", "some haystack") is None
    assert locate("   ", "some haystack") is None

"""Template-based evaluation set generation.

Deterministic and dependency-free: questions are derived from structural
patterns in the documents (definition sentences, section headings) and every
answer is anchored to the exact character span the question came from. Items
whose query leaks too much of the answer wording are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from ragcheck.corpus.models import Document
from ragcheck.dataset.leakage import (
    DEFAULT_MAX_LEAKAGE,
    anchor_score,
    classify_difficulty,
    leakage_score,
)
from ragcheck.dataset.models import EvalItem
from ragcheck.matching.spans import Span
from ragcheck.text import STOPWORDS, content_tokens

_SENTENCE = re.compile(r"[^\s.!?][^.!?\n]*[.!?]")
_DEFINITION = re.compile(
    r"^(?:The\s+|A\s+|An\s+)?(?P<subject>[A-Z][\w()-]*(?:\s+[\w()-]+){0,4}?)\s+(?P<verb>is|are)\s+\S"
)
_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_PARAGRAPH = re.compile(r"(?:[ \t]*\S[^\n]*\n?)+")

_MIN_SECTION_PARAGRAPH = 30
_MAX_HEADING_LENGTH = 60


def generate_evalset(
    documents: Iterable[Document], *, max_leakage: float = DEFAULT_MAX_LEAKAGE
) -> list[EvalItem]:
    """Generate evaluation items for *documents*, filtered for leakage."""
    items: dict[str, EvalItem] = {}
    for doc in documents:
        for query, span, gen_method in _candidates(doc):
            answer_text = doc.text[span.start : span.end]
            if leakage_score(query, answer_text) > max_leakage:
                continue
            anchor = anchor_score(query, answer_text)
            qid = EvalItem.derive_qid(query, [span], gen_method)
            items[qid] = EvalItem(
                qid=qid,
                query=query,
                answers=(span,),
                difficulty=classify_difficulty(anchor),
                gen_method=gen_method,
                meta={"anchor_score": round(anchor, 3), "source_path": doc.path},
            )
    return sorted(items.values(), key=lambda item: item.qid)


def _candidates(doc: Document) -> Iterator[tuple[str, Span, str]]:
    yield from _definition_candidates(doc)
    yield from _heading_candidates(doc)


def _definition_candidates(doc: Document) -> Iterator[tuple[str, Span, str]]:
    """Sentences of the form "X is/are ..." become "What is/are X?"."""
    for match in _SENTENCE.finditer(doc.text):
        sentence = match.group()
        definition = _DEFINITION.match(sentence)
        if definition is None:
            continue
        subject = definition.group("subject").strip()
        if not _is_valid_subject(subject):
            continue
        verb = definition.group("verb")
        query = f"What {verb} {subject}?"
        yield query, Span(doc.doc_id, match.start(), match.end()), "template:definition"


def _heading_candidates(doc: Document) -> Iterator[tuple[str, Span, str]]:
    """Each heading with a real first paragraph becomes a section question."""
    headings = list(_HEADING.finditer(doc.text))
    for index, match in enumerate(headings):
        title = re.sub(r"[`*#_]", "", match.group("title")).strip()
        if not title or len(title) > _MAX_HEADING_LENGTH or not content_tokens(title):
            continue
        body_start = match.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(doc.text)
        paragraph = _first_paragraph(doc.text, body_start, body_end)
        if paragraph is None:
            continue
        query = f"What should I know about {title}?"
        yield query, Span(doc.doc_id, *paragraph), "template:section"


def _first_paragraph(text: str, start: int, end: int) -> tuple[int, int] | None:
    for match in _PARAGRAPH.finditer(text, start, end):
        block = match.group()
        stripped = block.strip()
        if len(stripped) < _MIN_SECTION_PARAGRAPH:
            continue
        leading = len(block) - len(block.lstrip())
        return match.start() + leading, match.start() + leading + len(stripped)
    return None


def _is_valid_subject(subject: str) -> bool:
    tokens = subject.casefold().split()
    return bool(tokens) and any(token not in STOPWORDS for token in tokens)

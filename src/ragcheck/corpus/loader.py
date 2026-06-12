"""Load source documents from disk and persist corpora as JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ragcheck.corpus.models import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents for supported files under *root* in stable path order.

    *root* may also point at a single file. Empty files are skipped. Document
    paths are stored relative to *root* so a corpus is reproducible no matter
    where the source tree is checked out.
    """
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        stored = path.name if root.is_file() else path.relative_to(root).as_posix()
        yield Document.from_text(path=stored, text=text)


def load_corpus(root: Path) -> list[Document]:
    return list(iter_documents(root))


def save_corpus(documents: Iterable[Document], out_path: Path) -> int:
    """Write *documents* to *out_path* as JSONL and return the count written."""
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_corpus(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                documents.append(Document.from_dict(json.loads(line)))
    return documents

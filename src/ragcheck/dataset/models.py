"""Evaluation items: a query plus span-anchored ground truth."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragcheck.matching.spans import Span

DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class EvalItem:
    """One evaluation query with its gold answer spans.

    Answers are character intervals in source documents, never chunk IDs, so an
    evaluation set survives any re-chunking of the corpus.
    """

    qid: str
    query: str
    answers: tuple[Span, ...]
    difficulty: str
    gen_method: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("an eval item requires at least one answer span")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {self.difficulty!r}")

    @staticmethod
    def derive_qid(query: str, answers: Iterable[Span], gen_method: str) -> str:
        key = "|".join([gen_method, query, *(f"{s.doc_id}:{s.start}:{s.end}" for s in answers)])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "query": self.query,
            "answers": [{"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in self.answers],
            "difficulty": self.difficulty,
            "gen_method": self.gen_method,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalItem:
        return cls(
            qid=data["qid"],
            query=data["query"],
            answers=tuple(Span(a["doc_id"], a["start"], a["end"]) for a in data["answers"]),
            difficulty=data["difficulty"],
            gen_method=data["gen_method"],
            meta=data.get("meta", {}),
        )


def save_evalset(items: Iterable[EvalItem], out_path: Path) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_evalset(path: Path) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(EvalItem.from_dict(json.loads(line)))
    return items

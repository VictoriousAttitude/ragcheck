"""Evaluation runner: judge a retriever against an evalset and report metrics.

Judging strategy per retrieved chunk, in order of preference:

1. The chunk carries ``doc_id`` + offsets — exact interval overlap with gold spans.
2. The chunk carries text only — its text is located inside the gold documents
   (whitespace/case tolerant) and the recovered offsets are compared.

Results are a plain dictionary serialized to JSON: diffable, committable, and
consumed by the reporting and gating layers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragcheck import __version__
from ragcheck.corpus.models import Document
from ragcheck.dataset.models import DIFFICULTIES, EvalItem
from ragcheck.matching.spans import Span, locate, overlaps
from ragcheck.metrics.core import QueryJudgment, hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from ragcheck.retrievers.base import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RunResult:
    config: dict[str, Any]
    summary: dict[str, float]
    by_difficulty: dict[str, dict[str, float]]
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "summary": self.summary,
            "by_difficulty": self.by_difficulty,
            "per_item": self.per_item,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            config=data["config"],
            summary=data["summary"],
            by_difficulty=data["by_difficulty"],
            per_item=data.get("per_item", []),
        )


def evaluate(
    items: Sequence[EvalItem],
    retriever: Retriever,
    documents: Sequence[Document],
    *,
    k: int = 5,
    retriever_name: str = "",
) -> RunResult:
    if not items:
        raise ValueError("cannot evaluate an empty evalset")
    texts = {doc.doc_id: doc.text for doc in documents}

    judgments: list[QueryJudgment] = []
    per_item: list[dict[str, Any]] = []
    for item in items:
        retrieved = retriever.retrieve(item.query, k)[:k]
        covered = tuple(_covered_gold(chunk, item.answers, texts) for chunk in retrieved)
        judgment = QueryJudgment(n_gold=len(item.answers), covered=covered)
        judgments.append(judgment)
        per_item.append(
            {
                "qid": item.qid,
                "query": item.query,
                "difficulty": item.difficulty,
                "n_gold": judgment.n_gold,
                "relevant": list(judgment.relevant()),
                "covered": [sorted(c) for c in covered],
            }
        )

    by_difficulty: dict[str, dict[str, float]] = {}
    for tier in DIFFICULTIES:
        tier_judgments = [
            j for item, j in zip(items, judgments, strict=True) if item.difficulty == tier
        ]
        if tier_judgments:
            by_difficulty[tier] = {"n": float(len(tier_judgments)), **_metrics(tier_judgments, k)}

    config = {
        "ragcheck_version": __version__,
        "retriever": retriever_name or type(retriever).__name__,
        "k": k,
        "n_items": len(items),
        "evalset_fingerprint": _fingerprint(items),
    }
    return RunResult(
        config=config,
        summary=_metrics(judgments, k),
        by_difficulty=by_difficulty,
        per_item=per_item,
    )


def save_results(result: RunResult, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_results(path: Path) -> RunResult:
    return RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _covered_gold(
    chunk: RetrievedChunk, answers: tuple[Span, ...], texts: dict[str, str]
) -> frozenset[int]:
    if chunk.doc_id is not None and chunk.start is not None and chunk.end is not None:
        if chunk.end <= chunk.start:
            return frozenset()
        chunk_span = Span(chunk.doc_id, chunk.start, chunk.end)
        return frozenset(i for i, gold in enumerate(answers) if overlaps(chunk_span, gold))

    covered = set()
    located: dict[str, tuple[int, int] | None] = {}
    for i, gold in enumerate(answers):
        text = texts.get(gold.doc_id)
        if text is None:
            continue
        if gold.doc_id not in located:
            located[gold.doc_id] = locate(chunk.text, text)
        offsets = located[gold.doc_id]
        if offsets is not None and overlaps(Span(gold.doc_id, *offsets), gold):
            covered.add(i)
    return frozenset(covered)


def _metrics(judgments: Sequence[QueryJudgment], k: int) -> dict[str, float]:
    ks = sorted({1, min(5, k), k})
    metrics: dict[str, float] = {}
    for cutoff in ks:
        metrics[f"hit_rate@{cutoff}"] = hit_rate_at_k(judgments, cutoff)
        metrics[f"recall@{cutoff}"] = recall_at_k(judgments, cutoff)
        metrics[f"ndcg@{cutoff}"] = ndcg_at_k(judgments, cutoff)
    metrics["mrr"] = mrr(judgments, k=k)
    return metrics


def _fingerprint(items: Sequence[EvalItem]) -> str:
    digest = hashlib.sha256("|".join(item.qid for item in items).encode("utf-8"))
    return digest.hexdigest()[:16]

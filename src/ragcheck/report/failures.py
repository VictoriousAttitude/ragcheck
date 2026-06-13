"""Plain-text drill-down into the worst-scoring queries.

Turns the aggregate score into something actionable: for each failing query it
shows the gold span and the chunks the retriever returned instead, marking which
retrieved chunk (if any) actually overlapped the answer. This is what the
span-anchored ground truth buys you — failures you can read, not just a number.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragcheck.runner import FailureCase, RetrievedHit

_PREVIEW = 64
_LOC_WIDTH = 18


def render_failures(cases: Sequence[FailureCase], *, k: int, total: int) -> str:
    if not cases:
        return f"no queries to explain (0 of {total}).\n"
    lines = [f"worst {len(cases)} of {total} queries by recall@{k}", ""]
    for position, case in enumerate(cases, start=1):
        if case.first_hit_rank is None:
            status, verdict = "MISS", f"none of the top {k} chunks overlap the gold span"
        else:
            status, verdict = "OK  ", f"first relevant chunk at rank {case.first_hit_rank}"
        lines.append(
            f"[{position}] {status}  recall@{k}={case.recall:.2f}  {case.difficulty}  q={case.qid}"
        )
        lines.append(f"    query  {case.query}")
        for span in case.gold:
            loc = f"{span.doc_id}[{span.start}:{span.end}]"
            lines.append(f"    gold   {loc:<{_LOC_WIDTH}} {_quote(span.text)}")
        lines.append("    got")
        for hit in case.hits:
            pointer = ">" if hit.covered else " "
            lines.append(f"    {pointer} {hit.rank}. {_loc(hit):<{_LOC_WIDTH}} {_quote(hit.text)}")
        lines.append(f"    => {verdict}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _loc(hit: RetrievedHit) -> str:
    if hit.doc_id is not None and hit.start is not None and hit.end is not None:
        return f"{hit.doc_id}[{hit.start}:{hit.end}]"
    return "(text-only)"


def _quote(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > _PREVIEW:
        flat = flat[: _PREVIEW - 1] + "\u2026"
    return f'"{flat}"'

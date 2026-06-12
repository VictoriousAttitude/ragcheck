"""Leakage scoring and difficulty tiers for generated queries.

Synthetic queries derived from a passage tend to reuse its exact wording, which
makes retrieval artificially easy and inflates every metric. Two scores guard
against that:

- ``leakage_score``: how much of the *answer's* content the query gives away.
  A query that quotes most of the answer is discarded — it measures nothing.
- ``anchor_score``: how much of the *query's* content appears verbatim in the
  answer. High anchoring means plain lexical search will find the span, so the
  item is easy; low anchoring requires semantic matching, so it is hard.
  Metrics are always reported per tier.
"""

from __future__ import annotations

from ragcheck.text import content_tokens

DEFAULT_MAX_LEAKAGE = 0.6
EASY_ANCHOR = 0.75
MEDIUM_ANCHOR = 0.4


def leakage_score(query: str, answer_text: str) -> float:
    """Share of the answer's content tokens that the query reveals."""
    answer = content_tokens(answer_text)
    if not answer:
        return 1.0
    return len(content_tokens(query) & answer) / len(answer)


def anchor_score(query: str, answer_text: str) -> float:
    """Share of the query's content tokens found verbatim in the answer."""
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 1.0
    return len(query_tokens & content_tokens(answer_text)) / len(query_tokens)


def classify_difficulty(anchor: float) -> str:
    if anchor >= EASY_ANCHOR:
        return "easy"
    if anchor >= MEDIUM_ANCHOR:
        return "medium"
    return "hard"

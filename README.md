# ragcheck

![retrieval recall](assets/badge.svg)
[![ci](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/ci.yml)
[![dogfood](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/dogfood.yml/badge.svg)](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/dogfood.yml)

> pytest for retrieval — measure your RAG search quality.

Most RAG failures are retrieval failures: the wrong text gets fetched and the model
confidently answers from it. Teams ship this step with zero measurement. ragcheck turns
retrieval quality into a number you can track, diff, and gate in CI — a test suite for search.

The recall badge above is live: this repository runs ragcheck on its own example corpus
on every push and fails the build if retrieval quality regresses.

## Quickstart

```bash
pip install ragcheck   # not yet on PyPI; for now: pip install git+https://github.com/VictoriousAttitude/ragcheck

ragcheck ingest ./docs -o corpus.jsonl                  # load your documentation
ragcheck generate corpus.jsonl -o evalset.jsonl         # build a leakage-filtered evalset
ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
ragcheck report results.json --md report.md --badge badge.svg
```

Lock in a baseline and gate every change:

```bash
cp results.json baseline.json   # commit this
ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
ragcheck gate results.json --baseline baseline.json --max-drop 0.05  # exit 1 on regression
```

The gate is noise-aware: because both runs answer the same queries, it bootstraps the
per-query differences and fails only when it is confident (default 95%) that the drop is
real and beyond `--max-drop`, not a fluke of a small evalset. Tune with `--confidence` and
`--resamples`, or `--no-bootstrap` to compare raw point estimates.

## Find the best configuration

`run` scores one setup; `compare` sweeps a grid of retrievers and chunking
parameters over the same evalset and ranks them, so "setup A vs B" becomes a
single command instead of bookkeeping result files by hand:

```bash
ragcheck compare evalset.jsonl --corpus corpus.jsonl \
  --retriever bm25 --max-chars 300 --max-chars 800 --overlap-chars 50 --overlap-chars 150
```

```
# ragcheck comparison

Ranked by `ndcg@5`. Best: `bm25` max_chars=800 overlap=50 (ndcg@5=0.629).

| retriever | max_chars | overlap | hit_rate@5 | recall@5 | ndcg@5 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ★ bm25 | 800 | 50 | ...
```

Rank by any metric with `--sort recall@5`, and add `-o compare.json` for a
committable record. Progress prints to stderr, so the table pipes cleanly.

## Test your own RAG stack

ragcheck evaluates *any* retriever through a five-line adapter. Point `--adapter` at an
instance, a class, or a factory that accepts the corpus documents:

```python
# myadapter.py
from ragcheck.retrievers import RetrievedChunk

class MyRetriever:
    def __init__(self, documents):
        self.index = build_my_existing_index(documents)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = self.index.search(query, k)
        return [RetrievedChunk(text=h.text, score=h.score) for h in hits]
```

```bash
ragcheck run evalset.jsonl --corpus corpus.jsonl --adapter myadapter:MyRetriever
```

Returning `doc_id`/`start`/`end` provenance gives exact judging; text-only chunks are
located inside the source documents automatically (whitespace- and case-tolerant).

## How it works

1. **Ingest** — markdown/text documents get stable content-hashed IDs.
2. **Generate** — template-derived questions whose answers are *character spans in the
   source documents*, never chunk IDs. Re-chunking your corpus never invalidates an
   evalset, so chunking strategies compare on equal footing.
3. **Leakage filter** — synthetic questions that quote their answer's wording measure
   nothing; they are dropped, and the rest are tiered easy/medium/hard by how much of
   the query appears verbatim in the answer. Metrics are reported per tier.
4. **Run** — retrieved chunks are judged by span overlap; deterministic metrics:
   hit\_rate@k, recall@k, MRR, nDCG. No LLM judge anywhere in the core.
5. **Gate** — compare against a committed baseline in CI. When per-query data is present
   the gate runs a *seeded paired bootstrap* and fails only when a watched metric's whole
   confidence interval falls below tolerance, so query-sampling noise on a small evalset
   never trips the build; without it, it falls back to comparing point estimates.

## CI integration

```yaml
- run: pip install ragcheck
- run: ragcheck ingest ./docs -o corpus.jsonl
- run: ragcheck generate corpus.jsonl -o evalset.jsonl
- run: ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
- run: ragcheck gate results.json --baseline baseline.json --max-drop 0.05
```

See [`.github/workflows/dogfood.yml`](.github/workflows/dogfood.yml) — this repo gates itself.

## Design principles

1. **Span-anchored ground truth.** Eval sets survive any re-chunking.
2. **Deterministic first.** Same input, same score, forever.
3. **Bring your own stack.** A tiny protocol, no framework lock-in, no heavy core deps.
4. **Everything is a file.** JSONL in, JSON out — diffable, committable, CI-native.

## Why trust the numbers?

The metrics are golden-tested against hand-computed values. The repo dogfoods itself in
CI — which already caught a real bug (overlapping chunks double-counting a gold span
pushed nDCG above 1.0; now impossible by construction and regression-tested). The
harness is validated against published BEIR results.

## Benchmarks

The harness is validated against [BEIR](https://github.com/beir-cellar/beir) datasets
(document-level relevance mapped to whole-document gold spans). Reference BM25
retriever, k=10:

| dataset | docs | queries | chunk size | hit_rate@10 | recall@10 | ndcg@10 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scifact | 5183 | 300 | 300 | 0.753 | 0.729 | 0.590 | 0.556 |
| scifact | 5183 | 300 | 800 | 0.767 | 0.744 | 0.629 | 0.600 |
| nfcorpus | 3633 | 323 | 300 | 0.644 | 0.131 | 0.252 | 0.475 |
| nfcorpus | 3633 | 323 | 800 | 0.656 | 0.138 | 0.259 | 0.470 |
| fiqa | 57600 | 648 | 300 | 0.384 | 0.228 | 0.180 | 0.235 |
| fiqa | 57600 | 648 | 800 | 0.429 | 0.262 | 0.210 | 0.268 |
| arguana | 8674 | 1401 | 300 | 0.484 | 0.484 | 0.171 | 0.080 |
| arguana | 8674 | 1401 | 800 | 0.693 | 0.693 | 0.293 | 0.171 |
| scidocs | 25657 | 1000 | 300 | 0.438 | 0.132 | 0.129 | 0.242 |
| scidocs | 25657 | 1000 | 800 | 0.464 | 0.149 | 0.141 | 0.253 |

Chunk size alone moves nDCG@10 by 3–4 points on the same corpus and retriever — exactly
the kind of difference these evals exist to catch. (NFCorpus recall@10 is structurally
low: its queries average dozens of relevant documents, more than fit in the top 10.
ArguAna has exactly one relevant document per query, so its recall@10 equals hit_rate@10;
its low nDCG/MRR show BM25 finding the counter-argument but ranking it poorly — the
ranking-vs-recall split these metrics exist to separate.) Reproduce with
[`benchmarks/beir_runner.py`](benchmarks/beir_runner.py).

## Status

Early development: CLI shape is settling. The harness is now validated across five BEIR
datasets (scifact, nfcorpus, fiqa, arguana, scidocs). Next: PDF loading, an optional
LLM-assisted question generator, and a PyPI release.

## License

MIT

# ragcheck

![retrieval recall](assets/badge.svg)
[![ci](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/ci.yml)
[![dogfood](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/dogfood.yml/badge.svg)](https://github.com/VictoriousAttitude/ragcheck/actions/workflows/dogfood.yml)

> pytest for retrieval — measure your RAG search quality.

Most RAG failures are retrieval failures: the wrong text gets fetched and the model
answers from it. ragcheck turns retrieval quality into a number you can track, diff, and
gate in CI. There is no LLM in the scoring path, so the same input always produces the
same score.

What that buys you:

- **Deterministic, judge-free metrics.** `recall@k`, `nDCG`, `MRR`, and `hit_rate@k`
  computed from span overlap. No model call to score a run, so results are byte-reproducible,
  free to run, and fast.
- **Span-anchored ground truth.** Answers are character intervals in the source documents,
  not chunk IDs. Re-chunking never invalidates an eval set, so two chunking strategies are
  judged on equal footing.
- **A statistical regression gate.** CI fails only when a metric drop is significant under a
  seeded paired bootstrap, so query-sampling noise on a small eval set will not redden the build.
- **Validated against BEIR.** The harness reproduces published BM25 results on SciFact,
  NFCorpus, FiQA, ArguAna, and SciDocs ([table below](#benchmarks)).

## Quickstart

```bash
pip install ragcheck   # not yet on PyPI; for now: pip install git+https://github.com/VictoriousAttitude/ragcheck
# optional dense/hybrid/rerank retrievers: pip install "ragcheck[dense]"

ragcheck ingest ./docs -o corpus.jsonl                  # load your documentation
ragcheck generate corpus.jsonl -o evalset.jsonl         # build a leakage-filtered evalset
ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
ragcheck report results.json --md report.md --badge badge.svg
```

The recall badge above is live: this repository runs ragcheck on its own example corpus on
every push and fails the build if retrieval quality regresses.

## See why a query failed

An aggregate score tells you *that* retrieval is imperfect, not *which* queries broke or
*why*. `explain` lists the worst queries and, for each, prints the gold span next to the
chunks the retriever actually returned — `>` marks a chunk that overlapped the answer:

```bash
ragcheck explain evalset.jsonl --corpus corpus.jsonl -k 1 --worst 2
```

```
worst 2 of 31 queries by recall@1

[1] MISS  recall@1=0.00  easy  q=b87e7bfdd828
    query  What are Incidents?
    gold   dc96b14269a0aac7[21:95] "Incidents are coordinated in a dedicated channel by the on-call…"
    got
      1. 7eb242d28fcec701[679:846] "is active during the last week of each quarter and during compa…"
    => none of the top 1 chunks overlap the gold span

[2] OK    recall@1=1.00  hard  q=07f01346c44c
    query  What should I know about Rollbacks?
    gold   7eb242d28fcec701[387:640] "## Rollbacks Rollbacks reuse the same pipeline in reverse and c…"
    got
    > 1. 7eb242d28fcec701[339:736] "halts the rollout and pages the release owner. ## Rollbacks Rol…"
    => first relevant chunk at rank 1
```

The first query is a lexical trap: BM25 ranks an unrelated billing chunk ("last week of each
quarter") above the incidents document. Span anchoring is what makes this readable — the
report points at exact source offsets, not opaque chunk IDs. (Doc IDs are content hashes; see
[How it works](#how-it-works).)

## Find the best configuration

`run` scores one setup; `compare` sweeps a grid of retrievers and chunking parameters over the
same evalset and ranks them, so "setup A vs. B" is one command instead of bookkeeping result
files by hand:

```bash
ragcheck compare evalset.jsonl --corpus corpus.jsonl \
  --max-chars 300 --max-chars 800 --overlap-chars 50 --overlap-chars 150
```

```
Ranked by `ndcg@5`. Best: `bm25` max_chars=300 overlap=150 (ndcg@5=1.000).

| retriever | max_chars | overlap | recall@1 | ndcg@5 | mrr |
| --- | ---: | ---: | ---: | ---: | ---: |
| ★ bm25 | 300 | 150 | 1.000 | 1.000 | 1.000 |
| bm25 | 300 | 50 | 0.968 | 0.988 | 0.984 |
| bm25 | 800 | 50 | 0.968 | 0.984 | 0.978 |
| bm25 | 800 | 150 | 0.968 | 0.984 | 0.978 |
```

Rank by any metric with `--sort recall@5`, and add `-o compare.json` for a committable record.
Progress prints to stderr, so the table pipes cleanly.

`--retriever` sweeps strategies, not just chunk sizes:

- `bm25` — lexical Okapi BM25 over an inverted index, the zero-dependency default.
- `dense` — embedding cosine similarity.
- `hybrid` — reciprocal-rank fusion of `bm25` and `dense`.
- `rerank` — a cross-encoder re-scoring `bm25`'s top candidates.

The last three need encoder models (`pip install "ragcheck[dense]"`) and are deliberately kept
out of the core, so the default install stays pure-Python and the base metrics stay exactly
reproducible.

## Gate regressions in CI

```bash
cp results.json baseline.json   # commit this
ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
ragcheck gate results.json --baseline baseline.json --max-drop 0.05  # exit 1 on regression
```

Because both runs answer the same queries, the change in any metric is the mean of the
per-query differences. The gate resamples those paired differences with a seeded bootstrap and
fails only when it is confident (default 95%) that the whole confidence interval sits below
`-max-drop` — a real drop, not a fluke of a small evalset. Tune with `--confidence` and
`--resamples`, or use `--no-bootstrap` to compare raw point estimates.

## Bring your own retriever

ragcheck evaluates *any* retriever through a small adapter. Point `--adapter` at an instance, a
class, or a factory that accepts the corpus documents:

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

Returning `doc_id`/`start`/`end` provenance gives exact judging; text-only chunks are located
inside the source documents automatically (whitespace- and case-tolerant).

## How it works

1. **Ingest** — markdown/text documents get stable content-hashed IDs.
2. **Generate** — template-derived questions whose answers are *character spans in the source
   documents*, never chunk IDs, so re-chunking the corpus never invalidates an evalset.
3. **Leakage filter** — synthetic questions that quote their answer's wording measure nothing;
   they are dropped, and the rest are tiered easy/medium/hard by how much of the query appears
   verbatim in the answer. Metrics are reported per tier.
4. **Run** — retrieved chunks are judged by span overlap. Metrics: `hit_rate@k`, `recall@k`,
   `MRR`, `nDCG`. No LLM judge anywhere in the core.
5. **Gate** — compare against a committed baseline in CI via the seeded paired bootstrap above.

```yaml
- run: pip install ragcheck
- run: ragcheck ingest ./docs -o corpus.jsonl
- run: ragcheck generate corpus.jsonl -o evalset.jsonl
- run: ragcheck run evalset.jsonl --corpus corpus.jsonl -o results.json
- run: ragcheck gate results.json --baseline baseline.json --max-drop 0.05
```

See [`.github/workflows/dogfood.yml`](.github/workflows/dogfood.yml) — this repo gates itself.

## Benchmarks

Validated against [BEIR](https://github.com/beir-cellar/beir) (document-level relevance mapped
to whole-document gold spans). Reference BM25 retriever, k=10:

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

Chunk size alone moves nDCG@10 by 3–4 points on the same corpus and retriever, which is exactly
the kind of difference these evals exist to catch. Two reads worth knowing: NFCorpus recall@10
is structurally low because its queries average dozens of relevant documents, more than fit in
the top 10; ArguAna has exactly one relevant document per query, so its recall@10 equals
hit_rate@10, and its low nDCG/MRR show BM25 finding the counter-argument but ranking it poorly.
Reproduce with [`benchmarks/beir_runner.py`](benchmarks/beir_runner.py).

## Design principles

1. **Span-anchored ground truth.** Eval sets survive any re-chunking.
2. **Deterministic first.** Same input, same score, forever. No LLM judge in the core.
3. **Bring your own stack.** A tiny protocol, no framework lock-in, no heavy core dependencies.
4. **Everything is a file.** JSONL in, JSON out — diffable, committable, CI-native.

Metrics are golden-tested against hand-computed values, and the repo dogfoods itself in CI —
which already caught a real bug (overlapping chunks double-counting a gold span pushed nDCG
above 1.0; now impossible by construction and regression-tested).

## Status

Early development; the CLI shape is settling. Implemented: ingest, generate, run, compare,
explain, report, diff, and a statistical gate; four reference retrievers (bm25, dense, hybrid,
rerank); BEIR validation across five datasets. Next: PDF loading, an optional LLM-assisted
question generator, and a PyPI release.

## License

MIT

# ragcheck

> pytest for retrieval — measure your RAG search quality, gate it in CI, ship with confidence.

**Status: early development.** API and CLI are not stable yet.

## Why

Most RAG failures are retrieval failures: the wrong text gets fetched and the model
confidently answers from it. Teams ship this step with zero measurement. ragcheck turns
retrieval quality into a number you can track, diff, and gate in CI — a test suite for search.

## What it does

- Ingest a documentation corpus (`ragcheck ingest`)
- Generate an evaluation set with span-anchored ground truth (`ragcheck generate`)
- Run it against any retriever through a five-line adapter (`ragcheck run`)
- Score with deterministic metrics: hit@k, recall@k, MRR, nDCG — no LLM judge in the core
- Emit JSON/Markdown reports, an SVG badge, and a CI gate that fails the build on regression

## Design principles

1. **Span-anchored ground truth.** Answers are character offsets in source documents,
   never chunk IDs — re-chunking your corpus never invalidates an eval set, so chunking
   strategies can be compared on equal footing.
2. **Deterministic first.** Same input, same score, forever. LLM-as-judge is an optional
   extra, always reported separately.
3. **Bring your own stack.** A tiny adapter protocol, no framework lock-in, no heavy
   dependencies in the core.
4. **Everything is a file.** JSONL in, JSON out — diffable, committable, CI-native.

## License

MIT

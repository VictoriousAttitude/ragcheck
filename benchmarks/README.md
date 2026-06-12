# Benchmarks

Runs ragcheck's harness against [BEIR](https://github.com/beir-cellar/beir) datasets to
validate that its judging and metrics behave sanely on established ground truth, and to
publish reproducible numbers for the reference retrievers.

BEIR relevance is document-level; it maps onto ragcheck's span model as a gold span
covering the whole document, so any chunk cut from a relevant document counts as a hit.

## Reproduce

```bash
pip install -e .
python benchmarks/beir_runner.py --dataset scifact --chunk-size 300 --chunk-size 800
```

Datasets are downloaded on demand into `benchmarks/data/` (gitignored). Results land in
`benchmarks/results/` as JSON plus a markdown table per dataset.

Add `--retriever dense` (requires `pip install -e ".[dense]"`) to include the dense
reference retriever.

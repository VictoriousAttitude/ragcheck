"""Run ragcheck against a BEIR dataset and publish a results table.

BEIR (https://github.com/beir-cellar/beir) is the standard zero-shot retrieval
benchmark suite. This runner parses the dataset format directly — corpus.jsonl,
queries.jsonl, and qrels/test.tsv — so no extra dependencies are needed.

Relevance in BEIR is document-level, which maps onto ragcheck's span model as a
gold span covering the whole document: any retrieved chunk cut from a relevant
document counts as a hit.

Usage:
    python benchmarks/beir_runner.py --dataset scifact --chunk-size 300 --chunk-size 800
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import click

from ragcheck.corpus.models import Document
from ragcheck.dataset.leakage import anchor_score, classify_difficulty
from ragcheck.dataset.models import EvalItem
from ragcheck.matching.spans import Span
from ragcheck.retrievers import BM25Retriever, DenseRetriever, Retriever
from ragcheck.runner import evaluate, save_results

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def ensure_dataset(name: str, data_dir: Path) -> Path:
    target = data_dir / name
    if target.is_dir():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BEIR_BASE_URL}/{name}.zip"
    click.echo(f"downloading {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(data_dir)
    return target


def load_beir(path: Path, max_queries: int) -> tuple[list[Document], list[EvalItem]]:
    documents: list[Document] = []
    by_beir_id: dict[str, Document] = {}
    with (path / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            text = "\n\n".join(part for part in (entry.get("title", ""), entry["text"]) if part)
            doc = Document.from_text(path=str(entry["_id"]), text=text)
            documents.append(doc)
            by_beir_id[str(entry["_id"])] = doc

    queries: dict[str, str] = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            queries[str(entry["_id"])] = entry["text"]

    relevant: dict[str, list[str]] = {}
    with (path / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                relevant.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    items: list[EvalItem] = []
    for query_id, corpus_ids in sorted(relevant.items()):
        query = queries.get(query_id)
        docs = [by_beir_id[c] for c in corpus_ids if c in by_beir_id]
        if query is None or not docs:
            continue
        answers = tuple(Span(doc.doc_id, 0, len(doc.text)) for doc in docs)
        anchor = max(anchor_score(query, doc.text) for doc in docs)
        items.append(
            EvalItem(
                qid=f"beir-{path.name}-{query_id}",
                query=query,
                answers=answers,
                difficulty=classify_difficulty(anchor),
                gen_method=f"beir:{path.name}",
            )
        )
        if max_queries and len(items) >= max_queries:
            break
    return documents, items


def build_retriever(name: str, documents: list[Document], size: int) -> Retriever:
    overlap = size // 8
    if name == "dense":
        return DenseRetriever(documents, max_chars=size, overlap_chars=overlap)
    return BM25Retriever(documents, max_chars=size, overlap_chars=overlap)


@click.command()
@click.option("--dataset", default="scifact", show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=Path("benchmarks/data"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("benchmarks/results"))
@click.option("-k", type=int, default=10, show_default=True)
@click.option("--chunk-size", "chunk_sizes", multiple=True, type=int, default=(300, 800))
@click.option(
    "--retriever",
    "retrievers",
    multiple=True,
    type=click.Choice(["bm25", "dense"]),
    default=("bm25",),
)
@click.option("--max-queries", type=int, default=0, help="Limit queries (0 = all).")
def main(
    dataset: str,
    data_dir: Path,
    out_dir: Path,
    k: int,
    chunk_sizes: tuple[int, ...],
    retrievers: tuple[str, ...],
    max_queries: int,
) -> None:
    path = ensure_dataset(dataset, data_dir)
    documents, items = load_beir(path, max_queries)
    click.echo(f"{dataset}: {len(documents)} documents, {len(items)} queries")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, dict[str, float]]] = []
    for retriever_name in retrievers:
        for size in chunk_sizes:
            label = f"{retriever_name}@{size}"
            click.echo(f"running {label} ...")
            retriever = build_retriever(retriever_name, documents, size)
            result = evaluate(items, retriever, documents, k=k, retriever_name=label)
            result = dataclasses.replace(result, per_item=[])
            save_results(result, out_dir / f"{dataset}-{retriever_name}-{size}.json")
            rows.append((retriever_name, size, result.summary))
            click.echo(f"  recall@{k}: {result.summary[f'recall@{k}']:.3f}")

    table = _markdown_table(dataset, len(documents), len(items), k, rows)
    (out_dir / f"{dataset}.md").write_text(table, encoding="utf-8")
    click.echo(f"wrote {out_dir / f'{dataset}.md'}")


def _markdown_table(
    dataset: str, n_docs: int, n_queries: int, k: int, rows: list[tuple[str, int, dict[str, float]]]
) -> str:
    metrics = [f"hit_rate@{k}", f"recall@{k}", f"ndcg@{k}", "mrr"]
    lines = [
        f"# BEIR {dataset}",
        "",
        f"{n_docs} documents, {n_queries} test queries, k={k}.",
        "",
        "| retriever | chunk size | " + " | ".join(metrics) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in metrics) + " |",
    ]
    for name, size, summary in rows:
        cells = " | ".join(f"{summary[m]:.3f}" for m in metrics)
        lines.append(f"| {name} | {size} | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

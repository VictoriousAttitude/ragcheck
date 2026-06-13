"""Command-line interface: ingest, generate, run, compare, report, diff, gate."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import click

from ragcheck import __version__
from ragcheck.corpus import load_corpus, read_corpus, save_corpus
from ragcheck.corpus.models import Document
from ragcheck.dataset import generate_evalset, read_evalset, save_evalset
from ragcheck.dataset.leakage import DEFAULT_MAX_LEAKAGE
from ragcheck.gate import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_DROP,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    check_gate,
)
from ragcheck.report import (
    ComparisonRow,
    headline_badge,
    rank_comparison,
    render_comparison,
    render_diff,
    render_markdown,
)
from ragcheck.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    Retriever,
)
from ragcheck.runner import evaluate, read_results, save_results


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """pytest for retrieval: measure RAG search quality and gate it in CI."""


@main.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
def ingest(source: Path, output: Path) -> None:
    """Load markdown/text documents from SOURCE into a corpus file."""
    documents = load_corpus(source)
    if not documents:
        raise click.ClickException(f"no supported documents found under {source}")
    _ensure_parent(output)
    count = save_corpus(documents, output)
    click.echo(f"wrote {count} documents to {output}")


@main.command()
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("evalset.jsonl"))
@click.option("--max-leakage", type=float, default=DEFAULT_MAX_LEAKAGE, show_default=True)
def generate(corpus: Path, output: Path, max_leakage: float) -> None:
    """Generate a leakage-filtered evaluation set from CORPUS."""
    items = generate_evalset(read_corpus(corpus), max_leakage=max_leakage)
    if not items:
        raise click.ClickException("no eval items could be generated from this corpus")
    _ensure_parent(output)
    count = save_evalset(items, output)
    tiers = Counter(item.difficulty for item in items)
    breakdown = ", ".join(f"{tier}={tiers.get(tier, 0)}" for tier in ("easy", "medium", "hard"))
    click.echo(f"wrote {count} items to {output} ({breakdown})")


@main.command()
@click.argument("evalset", type=click.Path(exists=True, path_type=Path))
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--retriever",
    "retriever_name",
    type=click.Choice(["bm25", "dense", "hybrid", "rerank"]),
    default="bm25",
)
@click.option("--adapter", default=None, help="Custom retriever as 'module.path:attribute'.")
@click.option("-k", type=int, default=5, show_default=True)
@click.option("--max-chars", type=int, default=800, show_default=True, help="bm25 chunk size.")
@click.option("--overlap-chars", type=int, default=100, show_default=True, help="bm25 overlap.")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("results.json"))
@click.option("--per-item/--no-per-item", default=True, show_default=True)
def run(
    evalset: Path,
    corpus: Path,
    retriever_name: str,
    adapter: str | None,
    k: int,
    max_chars: int,
    overlap_chars: int,
    output: Path,
    per_item: bool,
) -> None:
    """Evaluate a retriever against EVALSET and write results."""
    documents = read_corpus(corpus)
    items = read_evalset(evalset)
    if adapter is not None:
        retriever, name = _load_adapter(adapter, documents)
    else:
        retriever, name = _build_reference_retriever(
            retriever_name, documents, max_chars, overlap_chars
        )
    result = evaluate(items, retriever, documents, k=k, retriever_name=name)
    if not per_item:
        result = dataclasses.replace(result, per_item=[])
    _ensure_parent(output)
    save_results(result, output)
    for metric, value in result.summary.items():
        click.echo(f"{metric}: {value:.3f}")
    click.echo(f"wrote {output}")


@main.command()
@click.argument("evalset", type=click.Path(exists=True, path_type=Path))
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--retriever",
    "retriever_names",
    multiple=True,
    type=click.Choice(["bm25", "dense", "hybrid", "rerank"]),
    default=("bm25",),
    show_default=True,
)
@click.option("-k", type=int, default=5, show_default=True)
@click.option(
    "--max-chars",
    "max_chars_values",
    multiple=True,
    type=int,
    default=(300, 800),
    show_default=True,
    help="Chunk sizes to sweep.",
)
@click.option(
    "--overlap-chars",
    "overlap_values",
    multiple=True,
    type=int,
    default=(100,),
    show_default=True,
    help="Chunk overlaps to sweep.",
)
@click.option("--sort", "sort_metric", default=None, help="Metric to rank by (default ndcg@k).")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
def compare(
    evalset: Path,
    corpus: Path,
    retriever_names: tuple[str, ...],
    k: int,
    max_chars_values: tuple[int, ...],
    overlap_values: tuple[int, ...],
    sort_metric: str | None,
    output: Path | None,
) -> None:
    """Sweep retriever and chunking configs over EVALSET and rank them."""
    documents = read_corpus(corpus)
    items = read_evalset(evalset)
    sort_metric = sort_metric or f"ndcg@{k}"

    rows: list[ComparisonRow] = []
    fingerprint = ""
    for retriever_name in retriever_names:
        for max_chars in max_chars_values:
            for overlap_chars in overlap_values:
                retriever, label = _build_reference_retriever(
                    retriever_name, documents, max_chars, overlap_chars
                )
                click.echo(f"running {label} ...", err=True)
                result = evaluate(items, retriever, documents, k=k, retriever_name=label)
                fingerprint = result.config["evalset_fingerprint"]
                rows.append(
                    ComparisonRow(
                        retriever=retriever_name,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                        summary=result.summary,
                    )
                )

    try:
        ranked = rank_comparison(rows, sort_metric)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if output is not None:
        _ensure_parent(output)
        record = {
            "sort_metric": sort_metric,
            "k": k,
            "evalset_fingerprint": fingerprint,
            "rows": [dataclasses.asdict(row) for row in ranked],
        }
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        click.echo(f"wrote {output}", err=True)
    click.echo(render_comparison(ranked, sort_metric), nl=False)


@main.command()
@click.argument("results", type=click.Path(exists=True, path_type=Path))
@click.option("--md", "md_path", type=click.Path(path_type=Path), default=None)
@click.option("--badge", "badge_path", type=click.Path(path_type=Path), default=None)
def report(results: Path, md_path: Path | None, badge_path: Path | None) -> None:
    """Render RESULTS as markdown and/or an SVG badge (stdout when no path given)."""
    result = read_results(results)
    if md_path is None and badge_path is None:
        click.echo(render_markdown(result), nl=False)
        return
    if md_path is not None:
        _ensure_parent(md_path)
        md_path.write_text(render_markdown(result), encoding="utf-8")
        click.echo(f"wrote {md_path}")
    if badge_path is not None:
        _ensure_parent(badge_path)
        badge_path.write_text(headline_badge(result), encoding="utf-8")
        click.echo(f"wrote {badge_path}")


@main.command()
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.argument("current", type=click.Path(exists=True, path_type=Path))
def diff(baseline: Path, current: Path) -> None:
    """Compare two results files metric by metric."""
    click.echo(render_diff(read_results(baseline), read_results(current)), nl=False)


@main.command()
@click.argument("results", type=click.Path(exists=True, path_type=Path))
@click.option("--baseline", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--max-drop", type=float, default=DEFAULT_MAX_DROP, show_default=True)
@click.option("--metric", "metrics", multiple=True, help="Watch only these metrics.")
@click.option("--confidence", type=float, default=DEFAULT_CONFIDENCE, show_default=True)
@click.option("--resamples", type=int, default=DEFAULT_RESAMPLES, show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option(
    "--bootstrap/--no-bootstrap",
    default=True,
    show_default=True,
    help="Use confidence intervals when per-query data is present.",
)
def gate(
    results: Path,
    baseline: Path,
    max_drop: float,
    metrics: tuple[str, ...],
    confidence: float,
    resamples: int,
    seed: int,
    bootstrap: bool,
) -> None:
    """Fail (exit 1) when RESULTS regress beyond --max-drop against --baseline."""
    try:
        outcome = check_gate(
            read_results(baseline),
            read_results(results),
            max_drop=max_drop,
            metrics=list(metrics) or None,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
            bootstrap=bootstrap,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(outcome.render(), nl=False)
    if not outcome.passed:
        sys.exit(1)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_reference_retriever(
    name: str, documents: Sequence[Document], max_chars: int, overlap_chars: int
) -> tuple[Retriever, str]:
    """Construct a built-in retriever and the label recorded in results.

    ``dense``, ``hybrid`` and ``rerank`` need the optional ``[dense]`` extra; a
    missing install surfaces as an actionable command error rather than a stack
    trace.
    """
    label = f"{name}(max_chars={max_chars},overlap_chars={overlap_chars})"
    try:
        if name == "bm25":
            retriever: Retriever = BM25Retriever(
                documents, max_chars=max_chars, overlap_chars=overlap_chars
            )
        elif name == "dense":
            retriever = DenseRetriever(documents, max_chars=max_chars, overlap_chars=overlap_chars)
        elif name == "hybrid":
            retriever = HybridRetriever(
                [
                    BM25Retriever(documents, max_chars=max_chars, overlap_chars=overlap_chars),
                    DenseRetriever(documents, max_chars=max_chars, overlap_chars=overlap_chars),
                ]
            )
        elif name == "rerank":
            retriever = RerankRetriever(
                BM25Retriever(documents, max_chars=max_chars, overlap_chars=overlap_chars)
            )
        else:  # pragma: no cover - click.Choice guards the allowed names
            raise click.ClickException(f"unknown retriever {name!r}")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    return retriever, label


def _load_adapter(spec: str, documents: Sequence[Document]) -> tuple[Retriever, str]:
    """Resolve 'module.path:attribute' to a retriever.

    The attribute may be a ready :class:`Retriever` instance or a callable that
    accepts the corpus documents and returns one (a class or factory function).
    """
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise click.ClickException(f"adapter must look like 'module.path:attribute', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise click.ClickException(f"cannot import adapter module {module_name!r}: {exc}") from exc
    try:
        candidate = getattr(module, attribute)
    except AttributeError as exc:
        raise click.ClickException(f"{module_name!r} has no attribute {attribute!r}") from exc

    if isinstance(candidate, type) or not isinstance(candidate, Retriever):
        if not callable(candidate):
            raise click.ClickException(f"adapter {spec!r} is not a retriever or a factory")
        candidate = candidate(documents)
    if isinstance(candidate, type) or not isinstance(candidate, Retriever):
        raise click.ClickException(f"adapter {spec!r} did not produce a retriever")
    return candidate, spec

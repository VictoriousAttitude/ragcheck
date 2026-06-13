import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragcheck.cli import main

HANDBOOK = """# Service handbook

## Deployment

Releases ship through a staged rollout. Each stage bakes for thirty minutes
and an automated health probe decides whether the next stage proceeds.

## Authentication

Signed tokens are required on every request. The identity service rotates
signing keys weekly without downtime.

Token lifetime is one hour.
"""

ADAPTER_MODULE = """
from ragcheck.retrievers import BM25Retriever


def build(documents):
    return BM25Retriever(documents, max_chars=120, overlap_chars=20)
"""


@pytest.fixture()
def workspace(tmp_path: Path) -> dict[str, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text(HANDBOOK, encoding="utf-8")
    return {
        "docs": docs,
        "corpus": tmp_path / "corpus.jsonl",
        "evalset": tmp_path / "evalset.jsonl",
        "results": tmp_path / "results.json",
        "tmp": tmp_path,
    }


def invoke(*args: str) -> str:
    runner = CliRunner()
    outcome = runner.invoke(main, list(args))
    assert outcome.exit_code == 0, outcome.output
    return outcome.output


def test_full_pipeline(workspace: dict[str, Path]) -> None:
    out = invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    assert "wrote 1 documents" in out

    out = invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    assert "wrote" in out and "easy=" in out

    out = invoke(
        "run",
        str(workspace["evalset"]),
        "--corpus",
        str(workspace["corpus"]),
        "-o",
        str(workspace["results"]),
    )
    assert "recall@5:" in out

    md = workspace["tmp"] / "report.md"
    badge = workspace["tmp"] / "badge.svg"
    invoke("report", str(workspace["results"]), "--md", str(md), "--badge", str(badge))
    assert "# ragcheck report" in md.read_text(encoding="utf-8")
    assert badge.read_text(encoding="utf-8").startswith("<svg")

    out = invoke("diff", str(workspace["results"]), str(workspace["results"]))
    assert "metric" in out

    out = invoke("gate", str(workspace["results"]), "--baseline", str(workspace["results"]))
    assert "GATE PASSED" in out


def test_gate_exits_nonzero_on_regression(workspace: dict[str, Path]) -> None:
    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    invoke(
        "run",
        str(workspace["evalset"]),
        "--corpus",
        str(workspace["corpus"]),
        "-o",
        str(workspace["results"]),
    )
    degraded = json.loads(workspace["results"].read_text(encoding="utf-8"))
    degraded["summary"] = {name: 0.0 for name in degraded["summary"]}
    for item in degraded["per_item"]:
        item["covered"] = [[] for _ in item["covered"]]
        item["relevant"] = [False for _ in item["relevant"]]
    current = workspace["tmp"] / "degraded.json"
    current.write_text(json.dumps(degraded), encoding="utf-8")

    runner = CliRunner()
    outcome = runner.invoke(
        main,
        ["gate", str(current), "--baseline", str(workspace["results"]), "--max-drop", "0.01"],
    )
    assert outcome.exit_code == 1
    assert "GATE FAILED" in outcome.output


def test_explain_lists_worst_queries(workspace: dict[str, Path]) -> None:
    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    out = invoke(
        "explain",
        str(workspace["evalset"]),
        "--corpus",
        str(workspace["corpus"]),
        "--worst",
        "3",
    )
    assert "queries by recall@5" in out
    assert "query  " in out and "gold   " in out


def test_compare_ranks_configs(workspace: dict[str, Path]) -> None:
    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    record_path = workspace["tmp"] / "compare.json"
    out = invoke(
        "compare",
        str(workspace["evalset"]),
        "--corpus",
        str(workspace["corpus"]),
        "--max-chars",
        "200",
        "--max-chars",
        "800",
        "-o",
        str(record_path),
    )
    assert "# ragcheck comparison" in out
    assert "★" in out

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["sort_metric"] == "ndcg@5"
    assert len(record["rows"]) == 2
    sort_values = [row["summary"]["ndcg@5"] for row in record["rows"]]
    assert sort_values == sorted(sort_values, reverse=True)


def test_compare_rejects_unknown_sort_metric(workspace: dict[str, Path]) -> None:
    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    runner = CliRunner()
    outcome = runner.invoke(
        main,
        [
            "compare",
            str(workspace["evalset"]),
            "--corpus",
            str(workspace["corpus"]),
            "--sort",
            "f1@5",
        ],
    )
    assert outcome.exit_code != 0
    assert "unknown metric" in outcome.output


def test_custom_adapter(workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    (workspace["tmp"] / "myadapter.py").write_text(ADAPTER_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(workspace["tmp"]))

    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    out = invoke(
        "run",
        str(workspace["evalset"]),
        "--corpus",
        str(workspace["corpus"]),
        "--adapter",
        "myadapter:build",
        "-o",
        str(workspace["results"]),
    )
    assert "recall@5:" in out
    saved = json.loads(workspace["results"].read_text(encoding="utf-8"))
    assert saved["config"]["retriever"] == "myadapter:build"


def test_bad_adapter_spec_is_rejected(workspace: dict[str, Path]) -> None:
    invoke("ingest", str(workspace["docs"]), "-o", str(workspace["corpus"]))
    invoke("generate", str(workspace["corpus"]), "-o", str(workspace["evalset"]))
    runner = CliRunner()
    outcome = runner.invoke(
        main,
        [
            "run",
            str(workspace["evalset"]),
            "--corpus",
            str(workspace["corpus"]),
            "--adapter",
            "nonexistent.module:thing",
        ],
    )
    assert outcome.exit_code != 0
    assert "cannot import adapter module" in outcome.output

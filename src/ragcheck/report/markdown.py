"""Markdown rendering of a run result."""

from __future__ import annotations

from ragcheck.runner import RunResult


def render_markdown(result: RunResult) -> str:
    config = result.config
    lines = [
        "# ragcheck report",
        "",
        f"- retriever: `{config['retriever']}`",
        f"- k: {config['k']}",
        f"- items: {config['n_items']}",
        f"- evalset: `{config['evalset_fingerprint']}`",
        f"- ragcheck: {config['ragcheck_version']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value:.3f} |" for name, value in result.summary.items())

    if result.by_difficulty:
        k = config["k"]
        columns = [f"hit_rate@{k}", f"recall@{k}", f"ndcg@{k}", "mrr"]
        lines += [
            "",
            "## By difficulty",
            "",
            "| tier | n | " + " | ".join(columns) + " |",
            "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |",
        ]
        for tier, metrics in result.by_difficulty.items():
            cells = " | ".join(f"{metrics[c]:.3f}" for c in columns)
            lines.append(f"| {tier} | {int(metrics['n'])} | {cells} |")

    return "\n".join(lines) + "\n"

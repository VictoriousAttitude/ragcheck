from pathlib import Path

from ragcheck.corpus import Document, load_corpus, read_corpus, save_corpus


def make_tree(root: Path) -> None:
    (root / "guide").mkdir()
    (root / "intro.md").write_text("# Intro\n\nWelcome to the project.\n", encoding="utf-8")
    (root / "guide" / "setup.txt").write_text("Install it with pip.\n", encoding="utf-8")
    (root / "guide" / "script.py").write_text("print('ignored')\n", encoding="utf-8")
    (root / "empty.md").write_text("", encoding="utf-8")


def test_load_corpus_filters_and_orders(tmp_path: Path) -> None:
    make_tree(tmp_path)
    docs = load_corpus(tmp_path)
    assert [d.path for d in docs] == ["guide/setup.txt", "intro.md"]


def test_doc_id_is_stable_and_content_derived(tmp_path: Path) -> None:
    make_tree(tmp_path)
    first = load_corpus(tmp_path)
    second = load_corpus(tmp_path)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    assert Document.from_text("a.md", "same text").doc_id == (
        Document.from_text("b.md", "same text").doc_id
    )
    assert Document.from_text("a.md", "one").doc_id != Document.from_text("a.md", "two").doc_id


def test_single_file_root(tmp_path: Path) -> None:
    target = tmp_path / "single.md"
    target.write_text("Only one document.\n", encoding="utf-8")
    docs = load_corpus(target)
    assert len(docs) == 1
    assert docs[0].path == "single.md"


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    make_tree(tmp_path)
    docs = load_corpus(tmp_path)
    out = tmp_path / "corpus.jsonl"
    written = save_corpus(docs, out)
    assert written == len(docs)
    assert read_corpus(out) == docs

"""Core corpus data model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document with a content-derived stable identifier.

    The identifier is a hash of the document text, so the same content always
    maps to the same ``doc_id`` regardless of where or when it was ingested.
    Ground-truth answer spans reference ``doc_id`` plus character offsets into
    ``text``, which keeps evaluation sets valid across re-chunking.
    """

    doc_id: str
    path: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def derive_id(text: str) -> str:
        """Return the stable content-derived identifier for *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_text(cls, path: str, text: str, meta: dict[str, Any] | None = None) -> Document:
        return cls(doc_id=cls.derive_id(text), path=path, text=text, meta=meta or {})

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "path": self.path, "text": self.text, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            doc_id=data["doc_id"],
            path=data["path"],
            text=data["text"],
            meta=data.get("meta", {}),
        )

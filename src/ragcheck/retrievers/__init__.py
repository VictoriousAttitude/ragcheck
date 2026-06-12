"""Retriever protocol, reference retrievers, and chunking utilities."""

from ragcheck.retrievers.base import RetrievedChunk, Retriever
from ragcheck.retrievers.bm25 import BM25Retriever
from ragcheck.retrievers.chunking import Chunk, chunk_document
from ragcheck.retrievers.dense import DenseRetriever, Encoder

__all__ = [
    "BM25Retriever",
    "Chunk",
    "DenseRetriever",
    "Encoder",
    "RetrievedChunk",
    "Retriever",
    "chunk_document",
]

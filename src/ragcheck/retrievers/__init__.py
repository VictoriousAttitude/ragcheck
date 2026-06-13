"""Retriever protocol, reference retrievers, and chunking utilities."""

from ragcheck.retrievers.base import RetrievedChunk, Retriever
from ragcheck.retrievers.bm25 import BM25Retriever
from ragcheck.retrievers.chunking import Chunk, chunk_document
from ragcheck.retrievers.dense import DenseRetriever, Encoder
from ragcheck.retrievers.hybrid import HybridRetriever
from ragcheck.retrievers.rerank import CrossEncoder, RerankRetriever

__all__ = [
    "BM25Retriever",
    "Chunk",
    "CrossEncoder",
    "DenseRetriever",
    "Encoder",
    "HybridRetriever",
    "RerankRetriever",
    "RetrievedChunk",
    "Retriever",
    "chunk_document",
]

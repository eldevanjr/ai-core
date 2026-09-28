"""Abstrações de RAG do ai-core."""

from __future__ import annotations

from ai_core.rag.chunking import ChunkingConfig, TextChunker
from ai_core.rag.models import SearchRequest, SearchResult, VectorMatch
from ai_core.rag.repository import (
    InMemoryKnowledgeRepository,
    InMemoryVectorStore,
    KnowledgeRepository,
    VectorStore,
)
from ai_core.rag.search import KnowledgeSearch, RepositoryKnowledgeSearch

__all__ = [
    "ChunkingConfig",
    "InMemoryKnowledgeRepository",
    "InMemoryVectorStore",
    "KnowledgeRepository",
    "KnowledgeSearch",
    "RepositoryKnowledgeSearch",
    "SearchRequest",
    "SearchResult",
    "TextChunker",
    "VectorMatch",
    "VectorStore",
]

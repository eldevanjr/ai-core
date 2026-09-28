"""Modelos de RAG: busca, resultados e correspondências vetoriais."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.knowledge import KnowledgeChunk
from ai_core.models.source import Source


class VectorMatch(CoreModel):
    """Correspondência retornada por um VectorStore."""

    chunk_id: str
    workspace_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(CoreModel):
    """Pedido de busca. Toda busca é obrigatoriamente escopada a um workspace."""

    workspace_id: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, gt=0)


class SearchResult(CoreModel):
    """Resultado de busca com o chunk, a pontuação e a origem verificável."""

    chunk: KnowledgeChunk
    score: float
    source: Source | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

"""Knowledge: documentos e chunks indexáveis para RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id, utc_now


class KnowledgeDocument(CoreModel):
    """Documento de conhecimento pertencente a um workspace."""

    id: str = Field(default_factory=new_id)
    workspace_id: str
    source_id: str | None = None
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeChunk(CoreModel):
    """Fragmento de um documento, unidade básica de indexação e busca.

    O core não armazena obrigatoriamente o vetor; ``embedding_reference`` é a
    abstração que liga o chunk ao armazenamento vetorial concreto.
    """

    id: str = Field(default_factory=new_id)
    document_id: str
    workspace_id: str
    content: str
    position: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_reference: str | None = None

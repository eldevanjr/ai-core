"""Busca de conhecimento (RAG) sobre repositório + vector store.

Regras obrigatórias:

- toda busca respeita ``workspace_id``; conhecimento de outro workspace nunca é
  retornado;
- a consulta só é enviada ao provider de embeddings se a política de privacidade
  permitir a localização dele (``LOCAL_ONLY`` nunca envia texto para cloud).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_core.embeddings.provider import EmbeddingProvider
from ai_core.models.knowledge import KnowledgeChunk
from ai_core.models.source import Source
from ai_core.policies.privacy import PrivacyPolicy
from ai_core.rag.models import SearchRequest, SearchResult
from ai_core.rag.repository import KnowledgeRepository, VectorStore


class KnowledgeSearch(ABC):
    """Contrato de busca semântica de conhecimento."""

    @abstractmethod
    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Retorna os trechos mais relevantes para a consulta dentro do workspace."""
        raise NotImplementedError


class RepositoryKnowledgeSearch(KnowledgeSearch):
    """Busca compondo :class:`EmbeddingProvider`, :class:`VectorStore` e repositório."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        privacy: PrivacyPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._privacy = privacy or PrivacyPolicy()
        self._privacy.assert_allowed(embeddings.location)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        self._privacy.assert_allowed(self._embeddings.location)
        embedding = self._embeddings.embed(request.query)
        matches = self._vector_store.query(
            workspace_id=request.workspace_id,
            vector=embedding.vector,
            limit=request.limit,
            filters=request.filters,
        )
        results: list[SearchResult] = []
        for match in matches:
            chunk = self._repository.get_chunk(match.chunk_id, workspace_id=request.workspace_id)
            if chunk is None:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=match.score,
                    source=self._source_of(chunk),
                    metadata=match.metadata,
                )
            )
        return results

    def _source_of(self, chunk: KnowledgeChunk) -> Source | None:
        document = self._repository.get_document(chunk.document_id, workspace_id=chunk.workspace_id)
        if document is None or document.source_id is None:
            return None
        return self._repository.get_source(document.source_id)

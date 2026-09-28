"""Repositório de conhecimento e VectorStore.

O core define as interfaces; aplicações fornecem implementações concretas
(PostgreSQL/pgvector, etc.). Aqui também vivem implementações em memória usadas
como referência e em testes — sem dependência obrigatória de banco.

Toda leitura/remoção por id exige ``workspace_id``: um recurso de outro
workspace se comporta como inexistente, sem revelar que existe.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from ai_core.errors import ConfigurationError, WorkspaceMismatchError
from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument
from ai_core.models.source import Source
from ai_core.rag.models import VectorMatch


class KnowledgeRepository(ABC):
    """Armazena documentos, chunks e as fontes que os originaram."""

    @abstractmethod
    def add_source(self, source: Source) -> Source:
        """Persiste uma fonte verificável e a retorna."""
        raise NotImplementedError

    @abstractmethod
    def get_source(self, source_id: str) -> Source | None:
        """Busca uma fonte pelo id.

        ``Source`` não possui workspace próprio; o acesso a ela deve partir de um
        documento já escopado (ver :meth:`get_document`).
        """
        raise NotImplementedError

    @abstractmethod
    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Persiste um documento e o retorna."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str, *, workspace_id: str) -> KnowledgeDocument | None:
        """Busca um documento do workspace pelo id."""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self, workspace_id: str) -> list[KnowledgeDocument]:
        """Lista documentos de um workspace."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str, *, workspace_id: str) -> None:
        """Remove um documento do workspace e seus chunks."""
        raise NotImplementedError

    @abstractmethod
    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """Persiste chunks.

        Levanta :class:`WorkspaceMismatchError` se o documento de um chunk existir
        em outro workspace.
        """
        raise NotImplementedError

    @abstractmethod
    def get_chunks(self, document_id: str, *, workspace_id: str) -> list[KnowledgeChunk]:
        """Lista os chunks de um documento do workspace, ordenados por posição."""
        raise NotImplementedError

    @abstractmethod
    def get_chunk(self, chunk_id: str, *, workspace_id: str) -> KnowledgeChunk | None:
        """Busca um chunk do workspace pelo id."""
        raise NotImplementedError


class VectorStore(ABC):
    """Armazena vetores e realiza busca por similaridade escopada por workspace.

    Implementações devem rejeitar vetores com dimensão diferente da já indexada
    no workspace (ex.: troca do modelo de embedding sem reindexar).
    """

    @abstractmethod
    def upsert(
        self,
        *,
        chunk_id: str,
        workspace_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insere ou atualiza o vetor de um chunk."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        *,
        workspace_id: str,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Retorna as correspondências mais similares dentro do workspace."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, *, workspace_id: str, chunk_id: str) -> None:
        """Remove o vetor de um chunk."""
        raise NotImplementedError

    @abstractmethod
    def delete_workspace(self, workspace_id: str) -> None:
        """Remove todos os vetores de um workspace."""
        raise NotImplementedError


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Implementação em memória de :class:`KnowledgeRepository` (referência/testes)."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}

    def add_source(self, source: Source) -> Source:
        self._sources[source.id] = source
        return source

    def get_source(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        existing = self._documents.get(document.id)
        if existing is not None and existing.workspace_id != document.workspace_id:
            raise WorkspaceMismatchError(f"Documento '{document.id}' pertence a outro workspace.")
        self._documents[document.id] = document
        return document

    def get_document(self, document_id: str, *, workspace_id: str) -> KnowledgeDocument | None:
        document = self._documents.get(document_id)
        return document if document and document.workspace_id == workspace_id else None

    def list_documents(self, workspace_id: str) -> list[KnowledgeDocument]:
        return [d for d in self._documents.values() if d.workspace_id == workspace_id]

    def delete_document(self, document_id: str, *, workspace_id: str) -> None:
        if self.get_document(document_id, workspace_id=workspace_id) is None:
            return
        del self._documents[document_id]
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.document_id != document_id}

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        for chunk in chunks:
            document = self._documents.get(chunk.document_id)
            if document is not None and document.workspace_id != chunk.workspace_id:
                raise WorkspaceMismatchError(
                    f"Chunk '{chunk.id}' e documento '{document.id}' em workspaces diferentes."
                )
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get_chunks(self, document_id: str, *, workspace_id: str) -> list[KnowledgeChunk]:
        chunks = [
            c
            for c in self._chunks.values()
            if c.document_id == document_id and c.workspace_id == workspace_id
        ]
        return sorted(chunks, key=lambda c: c.position)

    def get_chunk(self, chunk_id: str, *, workspace_id: str) -> KnowledgeChunk | None:
        chunk = self._chunks.get(chunk_id)
        return chunk if chunk and chunk.workspace_id == workspace_id else None


class InMemoryVectorStore(VectorStore):
    """Implementação em memória de :class:`VectorStore` com similaridade de cosseno."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str], tuple[list[float], dict[str, Any]]] = {}

    def upsert(
        self,
        *,
        chunk_id: str,
        workspace_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check_dimension(workspace_id, vector, ignore=chunk_id)
        self._vectors[(workspace_id, chunk_id)] = (list(vector), dict(metadata or {}))

    def query(
        self,
        *,
        workspace_id: str,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if limit <= 0:
            raise ConfigurationError("limit deve ser maior que zero.")
        self._check_dimension(workspace_id, vector)
        filters = filters or {}
        matches: list[VectorMatch] = []
        for (ws, chunk_id), (stored, metadata) in self._vectors.items():
            if ws != workspace_id:
                continue
            if any(metadata.get(key) != value for key, value in filters.items()):
                continue
            matches.append(
                VectorMatch(
                    chunk_id=chunk_id,
                    workspace_id=ws,
                    score=_cosine_similarity(vector, stored),
                    metadata=metadata,
                )
            )
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def delete(self, *, workspace_id: str, chunk_id: str) -> None:
        self._vectors.pop((workspace_id, chunk_id), None)

    def delete_workspace(self, workspace_id: str) -> None:
        self._vectors = {k: v for k, v in self._vectors.items() if k[0] != workspace_id}

    def _check_dimension(
        self, workspace_id: str, vector: list[float], *, ignore: str | None = None
    ) -> None:
        if not vector:
            raise ConfigurationError("Vetor vazio não pode ser indexado nem consultado.")
        indexed = next(
            (
                len(stored)
                for (ws, chunk_id), (stored, _) in self._vectors.items()
                if ws == workspace_id and chunk_id != ignore
            ),
            None,
        )
        if indexed is not None and indexed != len(vector):
            raise ConfigurationError(
                f"Dimensão {len(vector)} difere da dimensão indexada ({indexed}) no "
                f"workspace '{workspace_id}'. O modelo de embedding mudou? Reindexe."
            )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)

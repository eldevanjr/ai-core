from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument


def test_document_defaults() -> None:
    document = KnowledgeDocument(workspace_id="ws", title="Doc", content="texto")
    assert document.id
    assert document.source_id is None
    assert document.metadata == {}
    assert document.created_at.tzinfo is not None


def test_document_full_and_roundtrip() -> None:
    document = KnowledgeDocument(
        id="doc-1",
        workspace_id="ws",
        source_id="src-1",
        title="Doc",
        content="texto",
        metadata={"lang": "pt"},
    )
    assert KnowledgeDocument.model_validate(document.model_dump()) == document


def test_chunk_defaults() -> None:
    chunk = KnowledgeChunk(document_id="doc-1", workspace_id="ws", content="trecho")
    assert chunk.id
    assert chunk.position == 0
    assert chunk.metadata == {}
    assert chunk.embedding_reference is None


def test_chunk_full() -> None:
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="doc-1",
        workspace_id="ws",
        content="trecho",
        position=3,
        metadata={"tipo": "topic"},
        embedding_reference="vec-1",
    )
    assert chunk.position == 3
    assert chunk.embedding_reference == "vec-1"


def test_chunk_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunk.model_validate(
            {"document_id": "d", "workspace_id": "w", "content": "c", "x": 1}
        )

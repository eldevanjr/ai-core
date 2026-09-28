from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.errors import ConfigurationError, WorkspaceMismatchError
from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument
from ai_core.models.source import Source, SourceKind
from ai_core.rag.models import SearchRequest
from ai_core.rag.repository import (
    InMemoryKnowledgeRepository,
    InMemoryVectorStore,
    _cosine_similarity,
)
from ai_core.rag.search import RepositoryKnowledgeSearch
from tests.fakes import FakeEmbeddingProvider


def test_repository_document_lifecycle() -> None:
    repository = InMemoryKnowledgeRepository()
    document = repository.add_document(
        KnowledgeDocument(id="doc-1", workspace_id="ws-a", title="Doc", content="texto")
    )
    assert document.id == "doc-1"
    assert repository.get_document("doc-1", workspace_id="ws-a") is document
    assert repository.get_document("missing", workspace_id="ws-a") is None
    assert repository.list_documents("ws-a") == [document]
    assert repository.list_documents("ws-b") == []
    repository.add_document(
        KnowledgeDocument(id="doc-2", workspace_id="ws-b", title="Outro", content="x")
    )
    assert len(repository.list_documents("ws-b")) == 1
    repository.delete_document("doc-1", workspace_id="ws-a")
    assert repository.get_document("doc-1", workspace_id="ws-a") is None
    repository.delete_document("inexistente", workspace_id="ws-a")


def test_repository_chunks() -> None:
    repository = InMemoryKnowledgeRepository()
    chunks = [
        KnowledgeChunk(id="c2", document_id="doc", workspace_id="ws", content="b", position=1),
        KnowledgeChunk(id="c1", document_id="doc", workspace_id="ws", content="a", position=0),
    ]
    repository.add_chunks(chunks)
    repository.add_document(
        KnowledgeDocument(id="doc", workspace_id="ws", title="Doc", content="a b")
    )
    ordered = repository.get_chunks("doc", workspace_id="ws")
    assert [chunk.id for chunk in ordered] == ["c1", "c2"]
    assert repository.get_chunk("c1", workspace_id="ws") is not None
    assert repository.get_chunk("missing", workspace_id="ws") is None
    repository.delete_document("doc", workspace_id="ws")
    assert repository.get_chunks("doc", workspace_id="ws") == []
    assert repository.get_chunk("c1", workspace_id="ws") is None


def test_vector_store_query_and_limit() -> None:
    store = InMemoryVectorStore()
    store.upsert(chunk_id="a", workspace_id="ws", vector=[1.0, 0.0], metadata={"type": "topic"})
    store.upsert(chunk_id="b", workspace_id="ws", vector=[0.0, 1.0], metadata={"type": "task"})
    matches = store.query(workspace_id="ws", vector=[1.0, 0.0], limit=10)
    assert [match.chunk_id for match in matches] == ["a", "b"]
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(0.0)
    assert len(store.query(workspace_id="ws", vector=[1.0, 0.0], limit=1)) == 1


def test_vector_store_workspace_isolation_and_filters() -> None:
    store = InMemoryVectorStore()
    store.upsert(chunk_id="a", workspace_id="ws-a", vector=[1.0], metadata={"type": "task"})
    store.upsert(chunk_id="b", workspace_id="ws-b", vector=[1.0], metadata={"type": "task"})
    only_a = store.query(workspace_id="ws-a", vector=[1.0], limit=10)
    assert [match.chunk_id for match in only_a] == ["a"]
    filtered = store.query(workspace_id="ws-a", vector=[1.0], limit=10, filters={"type": "missing"})
    assert filtered == []
    matched = store.query(workspace_id="ws-a", vector=[1.0], limit=10, filters={"type": "task"})
    assert [match.chunk_id for match in matched] == ["a"]


def test_vector_store_delete() -> None:
    store = InMemoryVectorStore()
    store.upsert(chunk_id="a", workspace_id="ws", vector=[1.0])
    store.delete(workspace_id="ws", chunk_id="a")
    assert store.query(workspace_id="ws", vector=[1.0], limit=10) == []
    store.upsert(chunk_id="b", workspace_id="ws", vector=[1.0])
    store.delete(workspace_id="ws", chunk_id="missing")
    store.delete_workspace("ws")
    assert store.query(workspace_id="ws", vector=[1.0], limit=10) == []


def test_cosine_similarity_edges() -> None:
    assert _cosine_similarity([0.0], [1.0]) == 0.0
    assert _cosine_similarity([0.0], [0.0]) == 0.0
    assert _cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_search_returns_ranked_results() -> None:
    repository = InMemoryKnowledgeRepository()
    repository.add_chunks(
        [
            KnowledgeChunk(
                id="c1", document_id="d", workspace_id="ws", content="sincronização offline"
            ),
            KnowledgeChunk(id="c2", document_id="d", workspace_id="ws", content="outro assunto"),
        ]
    )
    embeddings = FakeEmbeddingProvider(dimension=2)
    store = InMemoryVectorStore()
    store.upsert(chunk_id="c1", workspace_id="ws", vector=[1.0, 0.0], metadata={"kind": "topic"})
    store.upsert(chunk_id="c2", workspace_id="ws", vector=[0.0, 1.0], metadata={"kind": "task"})
    search = RepositoryKnowledgeSearch(
        repository=repository, embeddings=embeddings, vector_store=store
    )
    results = search.search(SearchRequest(workspace_id="ws", query="sincronização"))
    assert [result.chunk.id for result in results] == ["c1", "c2"]
    assert results[0].metadata == {"kind": "topic"}


def test_search_respects_workspace_and_missing_chunks() -> None:
    repository = InMemoryKnowledgeRepository()
    repository.add_chunks(
        [KnowledgeChunk(id="foreign", document_id="d", workspace_id="ws-b", content="x")]
    )
    store = InMemoryVectorStore()
    store.upsert(chunk_id="foreign", workspace_id="ws-a", vector=[1.0])
    store.upsert(chunk_id="ghost", workspace_id="ws-a", vector=[1.0])
    search = RepositoryKnowledgeSearch(
        repository=repository, embeddings=FakeEmbeddingProvider(1), vector_store=store
    )
    assert search.search(SearchRequest(workspace_id="ws-a", query="q")) == []


def test_search_request_limit_validation() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(workspace_id="ws", query="q", limit=0)


def test_repository_hides_other_workspace_resources() -> None:
    repository = InMemoryKnowledgeRepository()
    repository.add_document(KnowledgeDocument(id="d", workspace_id="ws-a", title="t", content="c"))
    repository.add_chunks(
        [KnowledgeChunk(id="c", document_id="d", workspace_id="ws-a", content="c")]
    )
    assert repository.get_document("d", workspace_id="ws-b") is None
    assert repository.get_chunk("c", workspace_id="ws-b") is None
    assert repository.get_chunks("d", workspace_id="ws-b") == []
    repository.delete_document("d", workspace_id="ws-b")
    assert repository.get_document("d", workspace_id="ws-a") is not None
    assert repository.get_chunk("c", workspace_id="ws-a") is not None


def test_repository_rejects_cross_workspace_writes() -> None:
    repository = InMemoryKnowledgeRepository()
    repository.add_document(KnowledgeDocument(id="d", workspace_id="ws-a", title="t", content="c"))
    with pytest.raises(WorkspaceMismatchError):
        repository.add_document(
            KnowledgeDocument(id="d", workspace_id="ws-b", title="t", content="c")
        )
    valid = KnowledgeChunk(id="ok", document_id="d", workspace_id="ws-a", content="c")
    foreign = KnowledgeChunk(id="bad", document_id="d", workspace_id="ws-b", content="c")
    with pytest.raises(WorkspaceMismatchError):
        repository.add_chunks([valid, foreign])
    assert repository.get_chunk("ok", workspace_id="ws-a") is None


def test_repository_sources() -> None:
    repository = InMemoryKnowledgeRepository()
    source = repository.add_source(Source(id="s", type=SourceKind.FILE, location="a.md"))
    assert repository.get_source("s") is source
    assert repository.get_source("missing") is None


def test_vector_store_rejects_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.upsert(chunk_id="a", workspace_id="ws", vector=[1.0, 0.0])
    store.upsert(chunk_id="a", workspace_id="ws", vector=[0.0, 1.0, 0.0])
    with pytest.raises(ConfigurationError, match="Reindexe"):
        store.upsert(chunk_id="b", workspace_id="ws", vector=[1.0, 0.0])
    with pytest.raises(ConfigurationError):
        store.query(workspace_id="ws", vector=[1.0], limit=1)
    store.upsert(chunk_id="x", workspace_id="other", vector=[1.0])
    assert [m.chunk_id for m in store.query(workspace_id="other", vector=[1.0], limit=1)] == ["x"]


def test_vector_store_rejects_empty_vector_and_invalid_limit() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ConfigurationError):
        store.upsert(chunk_id="a", workspace_id="ws", vector=[])
    with pytest.raises(ConfigurationError):
        store.query(workspace_id="ws", vector=[], limit=1)
    with pytest.raises(ConfigurationError):
        store.query(workspace_id="ws", vector=[1.0], limit=0)


def test_search_fills_verifiable_source() -> None:
    repository = InMemoryKnowledgeRepository()
    source = repository.add_source(Source(id="s", type=SourceKind.FILE, location="spec.md"))
    repository.add_document(
        KnowledgeDocument(id="d1", workspace_id="ws", source_id="s", title="t", content="c")
    )
    repository.add_document(KnowledgeDocument(id="d2", workspace_id="ws", title="t", content="c"))
    repository.add_chunks(
        [
            KnowledgeChunk(id="with", document_id="d1", workspace_id="ws", content="c"),
            KnowledgeChunk(id="without", document_id="d2", workspace_id="ws", content="c"),
            KnowledgeChunk(id="orphan", document_id="gone", workspace_id="ws", content="c"),
        ]
    )
    store = InMemoryVectorStore()
    store.upsert(chunk_id="with", workspace_id="ws", vector=[1.0, 0.0])
    store.upsert(chunk_id="without", workspace_id="ws", vector=[0.9, 0.1])
    store.upsert(chunk_id="orphan", workspace_id="ws", vector=[0.5, 0.5])
    search = RepositoryKnowledgeSearch(
        repository=repository, embeddings=FakeEmbeddingProvider(2), vector_store=store
    )
    results = {r.chunk.id: r for r in search.search(SearchRequest(workspace_id="ws", query="q"))}
    assert results["with"].source == source
    assert results["without"].source is None
    assert results["orphan"].source is None

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.embeddings.models import Embedding
from ai_core.embeddings.provider import EmbeddingProvider
from ai_core.models.common import ProviderLocation
from tests.fakes import FakeEmbeddingProvider


def test_embedding_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


def test_embedding_model_defaults() -> None:
    embedding = Embedding(text="oi")
    assert embedding.vector == []
    assert embedding.model is None
    assert embedding.dimension is None


def test_fake_embedding_provider() -> None:
    provider = FakeEmbeddingProvider(dimension=4)
    assert provider.dimension == 4
    assert provider.location is ProviderLocation.LOCAL
    embedding = provider.embed("abc")
    assert embedding.vector == [3.0, 3.0, 3.0, 3.0]
    assert embedding.dimension == 4
    assert embedding.model == "fake-embed"


def test_embed_many_default_implementation() -> None:
    provider = FakeEmbeddingProvider(dimension=2)
    embeddings = provider.embed_many(["a", "bb"])
    assert [item.text for item in embeddings] == ["a", "bb"]
    assert embeddings[1].vector == [2.0, 2.0]


def test_embedding_rejects_vector_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="dimension=768"):
        Embedding(text="x", vector=[1.0, 2.0], dimension=768)
    with pytest.raises(ValidationError):
        Embedding(text="x", dimension=0)
    assert Embedding(text="x", vector=[1.0, 2.0], dimension=2).dimension == 2
    assert Embedding(text="x", dimension=768).vector == []

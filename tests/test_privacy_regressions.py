"""Regressões das regras de privacidade (plano §10 e §17).

LOCAL_ONLY nunca envia conteúdo para provider cloud, e LOCAL_FIRST só usa cloud
com confirmação explícita.
"""

from __future__ import annotations

import pytest

from ai_core.errors import NoProviderAvailableError, PolicyViolationError
from ai_core.llm.router import ModelRouter, RouterMode, RoutingCriteria, TaskComplexity
from ai_core.models.common import ProviderLocation
from ai_core.policies.privacy import PrivacyMode, PrivacyPolicy
from ai_core.rag.models import SearchRequest
from ai_core.rag.repository import InMemoryKnowledgeRepository, InMemoryVectorStore
from ai_core.rag.search import RepositoryKnowledgeSearch
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


def _cloud() -> FakeLLMProvider:
    return FakeLLMProvider("cloud", ProviderLocation.CLOUD)


def test_router_mode_given_as_string_is_enforced() -> None:
    router = ModelRouter(mode="LOCAL_ONLY", providers=[_cloud()])  # type: ignore[arg-type]
    assert router.mode is RouterMode.LOCAL_ONLY
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_router_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="LOCALONLY"):
        ModelRouter(mode="LOCALONLY")  # type: ignore[arg-type]


def test_router_provider_location_given_as_string_is_enforced() -> None:
    cloud = FakeLLMProvider("cloud", "CLOUD")  # type: ignore[arg-type]
    router = ModelRouter(privacy=PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY), providers=[cloud])
    with pytest.raises(NoProviderAvailableError):
        router.select()


@pytest.mark.parametrize(
    "criteria",
    [
        RoutingCriteria(complexity=TaskComplexity.HIGH),
        RoutingCriteria(complexity=TaskComplexity.HIGH, multimodal=True),
        RoutingCriteria(),
    ],
)
def test_local_first_auto_never_picks_cloud_without_confirmation(
    criteria: RoutingCriteria,
) -> None:
    local = FakeLLMProvider("local", supports_vision=True)
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD, supports_vision=True)
    router = ModelRouter(providers=[cloud, local])
    assert router.privacy.mode is PrivacyMode.LOCAL_FIRST
    assert router.select(criteria) is local


def test_local_first_auto_without_local_requires_confirmation() -> None:
    router = ModelRouter(providers=[_cloud()])
    with pytest.raises(NoProviderAvailableError):
        router.select(RoutingCriteria(complexity=TaskComplexity.HIGH))


def test_local_first_auto_uses_cloud_when_confirmed() -> None:
    cloud = _cloud()
    router = ModelRouter(providers=[FakeLLMProvider("local"), cloud])
    criteria = RoutingCriteria(complexity=TaskComplexity.HIGH, allow_cloud=True)
    assert router.select(criteria) is cloud


def test_search_blocks_cloud_embeddings_under_local_only() -> None:
    embeddings = FakeEmbeddingProvider(location=ProviderLocation.CLOUD)
    with pytest.raises(PolicyViolationError):
        RepositoryKnowledgeSearch(
            repository=InMemoryKnowledgeRepository(),
            embeddings=embeddings,
            vector_store=InMemoryVectorStore(),
            privacy=PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY),
        )


def test_search_rechecks_privacy_before_embedding_query() -> None:
    embeddings = FakeEmbeddingProvider(location=ProviderLocation.CLOUD)
    privacy = PrivacyPolicy(mode=PrivacyMode.ALLOW_CLOUD)
    search = RepositoryKnowledgeSearch(
        repository=InMemoryKnowledgeRepository(),
        embeddings=embeddings,
        vector_store=InMemoryVectorStore(),
        privacy=privacy,
    )
    privacy.mode = PrivacyMode.LOCAL_ONLY
    with pytest.raises(PolicyViolationError):
        search.search(SearchRequest(workspace_id="ws", query="segredo"))
    assert embeddings.calls == []

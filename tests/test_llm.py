from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.errors import NoProviderAvailableError, PolicyViolationError
from ai_core.llm.provider import LLMProvider
from ai_core.llm.request import LLMRequest
from ai_core.llm.response import LLMResponse, LLMUsage
from ai_core.llm.router import ModelRouter, RouterMode, RoutingCriteria, TaskComplexity
from ai_core.models.common import ProviderLocation
from ai_core.policies.privacy import PrivacyMode, PrivacyPolicy
from tests.fakes import FakeLLMProvider


def test_request_defaults() -> None:
    request = LLMRequest(user_prompt="oi")
    assert request.model is None
    assert request.system_prompt is None
    assert request.context == []
    assert request.temperature == 0.0
    assert request.max_tokens is None
    assert request.metadata == {}
    assert request.requires_vision is False


def test_request_validation() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="oi", temperature=3.0)
    with pytest.raises(ValidationError):
        LLMRequest(user_prompt="oi", max_tokens=0)


def test_response_defaults() -> None:
    response = LLMResponse(text="ok")
    assert response.model is None
    assert response.provider is None
    assert response.usage == LLMUsage()
    assert response.duration_ms is None
    assert response.metadata == {}


def test_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_fake_provider_generates() -> None:
    provider = FakeLLMProvider("local")
    response = provider.generate(LLMRequest(user_prompt="teste"))
    assert response.text == "echo:teste"
    assert response.provider == "local"


def test_register_replaces_and_get() -> None:
    router = ModelRouter()
    first = FakeLLMProvider("local")
    router.register(first)
    assert router.get("local") is first
    replacement = FakeLLMProvider("local")
    router.register(replacement)
    assert router.get("local") is replacement
    assert len(router.providers) == 1
    assert router.get("missing") is None


def test_constructor_registers_providers() -> None:
    local = FakeLLMProvider("local")
    router = ModelRouter(providers=[local])
    assert router.providers == [local]


def test_auto_defaults_to_local() -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(providers=[cloud, local])
    assert router.select() is local


ALLOW_CLOUD = PrivacyPolicy(mode=PrivacyMode.ALLOW_CLOUD)


def test_auto_high_complexity_prefers_cloud_when_cloud_allowed() -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(privacy=ALLOW_CLOUD, providers=[local, cloud])
    criteria = RoutingCriteria(complexity=TaskComplexity.HIGH)
    assert router.select(criteria) is cloud


@pytest.mark.parametrize("flag", ["latency_sensitive", "cost_sensitive"])
def test_auto_high_complexity_stays_local_when_latency_or_cost_sensitive(flag: str) -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(privacy=ALLOW_CLOUD, providers=[cloud, local])
    criteria = RoutingCriteria.model_validate({"complexity": "HIGH", flag: True})
    assert router.select(criteria) is local


def test_auto_multimodal_requires_vision_provider() -> None:
    local = FakeLLMProvider("local")
    vision = FakeLLMProvider("vision", ProviderLocation.CLOUD, supports_vision=True)
    router = ModelRouter(privacy=ALLOW_CLOUD, providers=[local, vision])
    assert router.select(RoutingCriteria(multimodal=True)) is vision
    router_without_vision = ModelRouter(privacy=ALLOW_CLOUD, providers=[local])
    with pytest.raises(NoProviderAvailableError):
        router_without_vision.select(RoutingCriteria(multimodal=True))


def test_auto_multimodal_prefers_local_vision_provider() -> None:
    local = FakeLLMProvider("llava", supports_vision=True)
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD, supports_vision=True)
    router = ModelRouter(privacy=ALLOW_CLOUD, providers=[cloud, local])
    assert router.select(RoutingCriteria(multimodal=True)) is local


def test_auto_only_cloud_when_cloud_allowed() -> None:
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(privacy=ALLOW_CLOUD, providers=[cloud])
    assert router.select() is cloud


def test_requested_model_prefers_declaring_provider() -> None:
    generic = FakeLLMProvider("generic")
    qwen = FakeLLMProvider("ollama-qwen", models=("qwen2.5",))
    llama = FakeLLMProvider("ollama-llama", models=("llama3.1",))
    router = ModelRouter(providers=[generic, llama, qwen])
    assert router.select(RoutingCriteria(requested_model="qwen2.5")) is qwen
    assert router.select(RoutingCriteria(requested_model="mistral")) is generic


def test_requested_model_unavailable_raises() -> None:
    router = ModelRouter(providers=[FakeLLMProvider("ollama", models=("llama3.1",))])
    with pytest.raises(NoProviderAvailableError):
        router.select(RoutingCriteria(requested_model="qwen2.5"))


def test_criteria_for_request() -> None:
    request = LLMRequest(user_prompt="descreva", model="llava", requires_vision=True)
    criteria = RoutingCriteria.for_request(request, allow_cloud=True)
    assert criteria.multimodal is True
    assert criteria.requested_model == "llava"
    assert criteria.allow_cloud is True
    assert RoutingCriteria.for_request(LLMRequest(user_prompt="x")).multimodal is False


def test_auto_high_complexity_without_cloud_uses_local() -> None:
    local = FakeLLMProvider("local")
    router = ModelRouter(providers=[local])
    assert router.select(RoutingCriteria(complexity=TaskComplexity.HIGH)) is local


def test_local_only_selects_local() -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(mode=RouterMode.LOCAL_ONLY, providers=[cloud, local])
    assert router.select() is local


def test_local_only_without_local_raises() -> None:
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(mode=RouterMode.LOCAL_ONLY, providers=[cloud])
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_cloud_only_selects_cloud() -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(mode=RouterMode.CLOUD_ONLY, providers=[local, cloud])
    assert router.select() is cloud


def test_cloud_only_without_cloud_raises() -> None:
    router = ModelRouter(mode=RouterMode.CLOUD_ONLY, providers=[FakeLLMProvider("local")])
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_cloud_only_blocked_by_local_only_privacy() -> None:
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(
        mode=RouterMode.CLOUD_ONLY,
        privacy=PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY),
        providers=[cloud],
    )
    with pytest.raises(PolicyViolationError):
        router.select()


def test_no_providers_raises() -> None:
    router = ModelRouter()
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_no_compatible_providers_raises() -> None:
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(privacy=PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY), providers=[cloud])
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_manual_selects_named_provider() -> None:
    local = FakeLLMProvider("local")
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(mode=RouterMode.MANUAL, providers=[local, cloud])
    criteria = RoutingCriteria(manual_provider="cloud")
    assert router.select(criteria) is cloud


def test_manual_ignores_requested_model() -> None:
    local = FakeLLMProvider("local")
    router = ModelRouter(mode=RouterMode.MANUAL, providers=[local])
    with pytest.raises(NoProviderAvailableError):
        router.select(RoutingCriteria(requested_model="local"))


def test_manual_multimodal_requires_vision() -> None:
    router = ModelRouter(mode=RouterMode.MANUAL, providers=[FakeLLMProvider("local")])
    with pytest.raises(NoProviderAvailableError):
        router.select(RoutingCriteria(manual_provider="local", multimodal=True))


def test_manual_requires_name() -> None:
    router = ModelRouter(mode=RouterMode.MANUAL, providers=[FakeLLMProvider("local")])
    with pytest.raises(NoProviderAvailableError):
        router.select()


def test_manual_unknown_provider() -> None:
    router = ModelRouter(mode=RouterMode.MANUAL, providers=[FakeLLMProvider("local")])
    with pytest.raises(NoProviderAvailableError):
        router.select(RoutingCriteria(manual_provider="nope"))


def test_manual_cloud_blocked_by_privacy() -> None:
    cloud = FakeLLMProvider("cloud", ProviderLocation.CLOUD)
    router = ModelRouter(
        mode=RouterMode.MANUAL,
        privacy=PrivacyPolicy(mode=PrivacyMode.LOCAL_ONLY),
        providers=[cloud],
    )
    with pytest.raises(PolicyViolationError):
        router.select(RoutingCriteria(manual_provider="cloud"))

"""Model Router: escolhe qual LLMProvider atenderá uma demanda."""

from __future__ import annotations

from enum import StrEnum

from ai_core.errors import NoProviderAvailableError
from ai_core.llm.provider import LLMProvider
from ai_core.llm.request import LLMRequest
from ai_core.models.base import CoreModel
from ai_core.models.common import ProviderLocation
from ai_core.policies.privacy import PrivacyPolicy


class RouterMode(StrEnum):
    """Modos de roteamento de modelo."""

    AUTO = "AUTO"
    LOCAL_ONLY = "LOCAL_ONLY"
    CLOUD_ONLY = "CLOUD_ONLY"
    MANUAL = "MANUAL"


class TaskComplexity(StrEnum):
    """Complexidade estimada da demanda, usada pelo modo AUTO."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RoutingCriteria(CoreModel):
    """Critérios observáveis para seleção de provider.

    - ``complexity``: demandas ``HIGH`` preferem cloud no modo AUTO.
    - ``multimodal``: restringe a providers com ``supports_vision``.
    - ``requested_model``: modelo configurado; prefere providers que o declaram.
    - ``manual_provider``: nome do provider no modo MANUAL.
    - ``allow_cloud``: confirmação explícita exigida por ``LOCAL_FIRST`` para cloud.
    - ``latency_sensitive`` / ``cost_sensitive``: mantêm a demanda no provider local.
    """

    complexity: TaskComplexity = TaskComplexity.MEDIUM
    multimodal: bool = False
    requested_model: str | None = None
    manual_provider: str | None = None
    allow_cloud: bool = False
    latency_sensitive: bool = False
    cost_sensitive: bool = False

    @classmethod
    def for_request(cls, request: LLMRequest, **overrides: object) -> RoutingCriteria:
        """Deriva critérios de uma requisição (modelo e necessidade de visão)."""
        values: dict[str, object] = {
            "multimodal": request.requires_vision,
            "requested_model": request.model,
        }
        values.update(overrides)
        return cls.model_validate(values)


class ModelRouter:
    """Seleciona providers respeitando modo e política de privacidade.

    Regras centrais:

    - em ``LOCAL_ONLY`` (modo ou política de privacidade), nenhum provider cloud
      pode ser escolhido;
    - em política ``LOCAL_FIRST``, o modo AUTO só escolhe cloud com
      ``RoutingCriteria.allow_cloud=True``. ``CLOUD_ONLY`` e ``MANUAL`` já são
      escolhas explícitas de configuração.
    """

    def __init__(
        self,
        *,
        mode: RouterMode = RouterMode.AUTO,
        privacy: PrivacyPolicy | None = None,
        providers: list[LLMProvider] | None = None,
    ) -> None:
        self.mode = RouterMode(mode)
        self.privacy = privacy or PrivacyPolicy()
        self.providers: list[LLMProvider] = []
        for provider in providers or []:
            self.register(provider)

    def register(self, provider: LLMProvider) -> None:
        """Registra um provider, substituindo outro de mesmo nome."""
        self.providers = [p for p in self.providers if p.name != provider.name] + [provider]

    def get(self, name: str) -> LLMProvider | None:
        """Retorna um provider registrado pelo nome, se existir."""
        return next((p for p in self.providers if p.name == name), None)

    def select(self, criteria: RoutingCriteria | None = None) -> LLMProvider:
        """Escolhe o provider adequado ou levanta ``NoProviderAvailableError``."""
        criteria = criteria or RoutingCriteria()
        if self.mode == RouterMode.MANUAL:
            return self._select_manual(criteria)
        candidates = self._candidates(criteria)
        if not candidates:
            raise NoProviderAvailableError(
                f"Nenhum provider compatível com o modo {self.mode}, a política de "
                f"privacidade {self.privacy.mode} e os critérios informados."
            )
        if self.mode == RouterMode.AUTO:
            return self._select_auto(candidates, criteria)
        return candidates[0]

    def _candidates(self, criteria: RoutingCriteria) -> list[LLMProvider]:
        if self.mode == RouterMode.CLOUD_ONLY:
            self.privacy.assert_allowed(ProviderLocation.CLOUD)
        candidates = [p for p in self.providers if self._permitted(p, criteria)]
        if criteria.multimodal:
            candidates = [p for p in candidates if p.supports_vision]
        return _prefer_model(candidates, criteria.requested_model)

    def _permitted(self, provider: LLMProvider, criteria: RoutingCriteria) -> bool:
        location = provider.location
        if not self.privacy.allows(location):
            return False
        if self.mode == RouterMode.LOCAL_ONLY:
            return location == ProviderLocation.LOCAL
        if self.mode == RouterMode.CLOUD_ONLY:
            return location == ProviderLocation.CLOUD
        return criteria.allow_cloud or not self.privacy.requires_confirmation(location)

    def _select_manual(self, criteria: RoutingCriteria) -> LLMProvider:
        name = criteria.manual_provider
        if name is None:
            raise NoProviderAvailableError("Modo MANUAL exige 'manual_provider' nos critérios.")
        provider = self.get(name)
        if provider is None:
            raise NoProviderAvailableError(f"Provider '{name}' não registrado.")
        self.privacy.assert_allowed(provider.location)
        if criteria.multimodal and not provider.supports_vision:
            raise NoProviderAvailableError(f"Provider '{name}' não suporta entrada multimodal.")
        return provider

    @staticmethod
    def _select_auto(candidates: list[LLMProvider], criteria: RoutingCriteria) -> LLMProvider:
        local = [p for p in candidates if p.location == ProviderLocation.LOCAL]
        cloud = [p for p in candidates if p.location != ProviderLocation.LOCAL]
        stay_local = criteria.latency_sensitive or criteria.cost_sensitive
        if criteria.complexity == TaskComplexity.HIGH and cloud and not stay_local:
            return cloud[0]
        return local[0] if local else candidates[0]


def _prefer_model(providers: list[LLMProvider], model: str | None) -> list[LLMProvider]:
    """Prioriza quem declara o modelo; providers sem declaração aceitam qualquer um."""
    if model is None:
        return providers
    declared = [p for p in providers if model in p.models]
    return declared or [p for p in providers if not p.models]

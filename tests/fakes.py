"""Fakes compartilhados pelos testes do ai-core."""

from __future__ import annotations

from ai_core.embeddings.models import Embedding
from ai_core.embeddings.provider import EmbeddingProvider
from ai_core.llm.provider import LLMProvider
from ai_core.llm.request import LLMRequest
from ai_core.llm.response import LLMResponse
from ai_core.models.common import ProviderLocation


class FakeLLMProvider(LLMProvider):
    """Provider de LLM determinístico, só para testes."""

    def __init__(
        self,
        name: str,
        location: ProviderLocation = ProviderLocation.LOCAL,
        *,
        models: tuple[str, ...] = (),
        supports_vision: bool = False,
    ) -> None:
        self.name = name
        self.location = location
        self.models = models
        self.supports_vision = supports_vision

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text=f"echo:{request.user_prompt}", provider=self.name)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Provider de embeddings determinístico baseado no tamanho do texto."""

    def __init__(
        self,
        dimension: int = 3,
        name: str = "fake-embed",
        location: ProviderLocation = ProviderLocation.LOCAL,
    ) -> None:
        self.name = name
        self.location = location
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Embedding:
        self.calls.append(text)
        vector = [float(len(text))] * self._dimension
        return Embedding(text=text, vector=vector, model=self.name, dimension=self._dimension)

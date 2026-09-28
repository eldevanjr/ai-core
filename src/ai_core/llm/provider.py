"""Interface de provider de LLM.

Implementações concretas (Ollama, OpenAI, Anthropic) vivem fora do core; o core
conhece apenas esta abstração e a localização do provider (local/cloud).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_core.llm.request import LLMRequest
from ai_core.llm.response import LLMResponse
from ai_core.models.common import ProviderLocation


class LLMProvider(ABC):
    """Contrato abstrato de um provider de linguagem.

    - ``location``: onde o conteúdo é processado; decide o que a política de
      privacidade permite.
    - ``models``: modelos atendidos pelo provider. Vazio significa "não declarado"
      (o provider aceita qualquer ``requested_model``).
    - ``supports_vision``: se o provider aceita entrada multimodal (imagens).
    """

    name: str
    location: ProviderLocation = ProviderLocation.LOCAL
    models: tuple[str, ...] = ()
    supports_vision: bool = False

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Gera texto a partir da requisição."""
        raise NotImplementedError

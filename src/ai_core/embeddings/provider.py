"""Interface de provider de embeddings.

O core não assume dimensão fixa: cada implementação descobre ou configura a sua
própria dimensão e a expõe via :attr:`EmbeddingProvider.dimension`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_core.embeddings.models import Embedding
from ai_core.models.common import ProviderLocation


class EmbeddingProvider(ABC):
    """Contrato abstrato para geração de embeddings."""

    name: str
    location: ProviderLocation = ProviderLocation.LOCAL

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensão dos vetores produzidos por este provider."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Gera o embedding de um único texto."""
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[Embedding]:
        """Gera embeddings para múltiplos textos (implementação padrão sequencial)."""
        return [self.embed(text) for text in texts]

"""Modelos do subsistema de embeddings."""

from __future__ import annotations

from pydantic import Field, model_validator

from ai_core.models.base import CoreModel


class Embedding(CoreModel):
    """Vetor de embedding associado a um texto de origem.

    Quando ``dimension`` é informada e o vetor não está vazio, ambos precisam
    concordar; isso detecta cedo um provider mal configurado.
    """

    text: str
    vector: list[float] = Field(default_factory=list)
    model: str | None = None
    dimension: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_dimension(self) -> Embedding:
        if self.dimension is not None and self.vector and len(self.vector) != self.dimension:
            raise ValueError(
                f"vector possui {len(self.vector)} posições, mas dimension={self.dimension}"
            )
        return self

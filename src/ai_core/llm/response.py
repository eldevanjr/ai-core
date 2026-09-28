"""Resposta de geração retornada por um LLMProvider."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel


class LLMUsage(CoreModel):
    """Contagem de tokens reportada pelo provider, quando disponível."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(CoreModel):
    """Resultado de uma chamada de geração."""

    text: str
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

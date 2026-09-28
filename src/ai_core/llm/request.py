"""Requisição de geração enviada a um LLMProvider."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel


class LLMRequest(CoreModel):
    """Parâmetros de uma chamada de geração."""

    model: str | None = None
    system_prompt: str | None = None
    user_prompt: str
    context: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    requires_vision: bool = False

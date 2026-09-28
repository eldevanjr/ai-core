"""Abstrações de LLM do ai-core."""

from __future__ import annotations

from ai_core.llm.provider import LLMProvider
from ai_core.llm.request import LLMRequest
from ai_core.llm.response import LLMResponse, LLMUsage
from ai_core.llm.router import ModelRouter, RouterMode, RoutingCriteria, TaskComplexity

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "ModelRouter",
    "RouterMode",
    "RoutingCriteria",
    "TaskComplexity",
]

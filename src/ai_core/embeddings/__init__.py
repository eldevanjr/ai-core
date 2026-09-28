"""Abstrações de embeddings do ai-core."""

from __future__ import annotations

from ai_core.embeddings.models import Embedding
from ai_core.embeddings.provider import EmbeddingProvider

__all__ = ["Embedding", "EmbeddingProvider"]

"""Chunker genérico de texto em tokens aproximados."""

from __future__ import annotations

import re

from pydantic import Field, model_validator

from ai_core.models.base import CoreModel

_TOKEN = re.compile(r"\S+")


class ChunkingConfig(CoreModel):
    """Configuração de chunking.

    Valores iniciais recomendados pelo plano: 500-800 tokens por chunk e 50-100
    tokens de overlap. Aqui "token" é uma palavra (sequência sem espaço); em
    modelos de LLM uma palavra costuma equivaler a ~1,3 token.
    """

    chunk_size: int = Field(default=600, ge=1)
    overlap: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkingConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap deve ser menor que chunk_size")
        return self


class TextChunker:
    """Divide texto em janelas de tokens com overlap.

    O texto de cada chunk é recortado do original, preservando quebras de linha e
    indentação (importante para logs, código e diffs). Implementações
    especializadas podem substituir este comportamento.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, text: str) -> list[str]:
        """Retorna a lista de chunks (possivelmente vazia para texto vazio)."""
        spans = [match.span() for match in _TOKEN.finditer(text)]
        step = self.config.chunk_size - self.config.overlap
        chunks: list[str] = []
        for start in range(0, len(spans), step):
            end = min(start + self.config.chunk_size, len(spans))
            chunks.append(text[spans[start][0] : spans[end - 1][1]])
            if end == len(spans):
                break
        return chunks

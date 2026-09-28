"""Erros e exceções comuns do ai-core."""

from __future__ import annotations


class AiCoreError(Exception):
    """Erro base de todos os erros do ai-core."""


class NoProviderAvailableError(AiCoreError):
    """Nenhum provider compatível com a política/critério foi encontrado."""


class PolicyViolationError(AiCoreError):
    """Uma operação violaria uma política configurada (privacidade/aprovação)."""


class WorkspaceMismatchError(AiCoreError):
    """Um recurso acessado não pertence ao workspace informado."""


class ConfigurationError(AiCoreError):
    """Configuração inválida de um componente do core."""

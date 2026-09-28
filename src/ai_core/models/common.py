"""Tipos e helpers compartilhados pelos modelos do ai-core."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def new_id() -> str:
    """Gera um identificador único (hex, sem hífens) para novas entidades."""
    return uuid4().hex


def utc_now() -> datetime:
    """Retorna o instante atual em UTC com timezone explícito."""
    return datetime.now(UTC)


class ProviderLocation(StrEnum):
    """Localização de execução de um provider de IA."""

    LOCAL = "LOCAL"
    CLOUD = "CLOUD"


class ApprovalMode(StrEnum):
    """Modos de aprovação suportados (ver :class:`ai_core.policies.ApprovalPolicy`).

    Vive em ``models`` para que os modelos (ex.: ``ApprovalRecord``) não dependam
    da camada de políticas.
    """

    ALWAYS_REQUIRED = "ALWAYS_REQUIRED"
    REQUIRED_FOR_EXTERNAL_ACTION = "REQUIRED_FOR_EXTERNAL_ACTION"
    AUTO_APPROVED = "AUTO_APPROVED"
    MANUAL_ONLY = "MANUAL_ONLY"

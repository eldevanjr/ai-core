"""Estruturas comuns de auditoria compartilhadas pelo ecossistema."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id, utc_now


class AuditEvent(CoreModel):
    """Registro auditável de uma operação relevante (incluindo ações externas).

    Nunca deve conter secrets (tokens, senhas, credenciais).
    """

    id: str = Field(default_factory=new_id)
    workspace_id: str
    action: str
    actor: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    proposal_id: str | None = None
    model: str | None = None
    provider: str | None = None
    retrieved_source_ids: list[str] = Field(default_factory=list)
    destination: str | None = None
    approval: str | None = None
    execution: str | None = None
    external_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)

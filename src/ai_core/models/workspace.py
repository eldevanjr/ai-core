"""Workspace: contexto lógico de trabalho, independente de ferramenta externa."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id, utc_now


class WorkspaceStatus(StrEnum):
    """Estado de ciclo de vida de um workspace."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DISABLED = "DISABLED"


class Workspace(CoreModel):
    """Contexto lógico que isola conhecimento, evidências, tarefas e políticas.

    Um workspace não representa diretamente um projeto Jira ou um repositório
    GitHub; esses são apenas destinos externos que podem ser associados a ele.
    """

    id: str = Field(default_factory=new_id)
    name: str
    description: str | None = None
    slug: str
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

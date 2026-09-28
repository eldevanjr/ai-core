"""Task: representação neutra de trabalho, independente de ferramenta externa."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id, utc_now
from ai_core.models.source import Source


class TaskType(StrEnum):
    """Categorias neutras de trabalho reconhecidas pelo Task Engine."""

    BUG = "BUG"
    TASK = "TASK"
    FEATURE = "FEATURE"
    IMPROVEMENT = "IMPROVEMENT"
    TECH_DEBT = "TECH_DEBT"
    QUESTION = "QUESTION"
    INVESTIGATION = "INVESTIGATION"
    DOCUMENTATION = "DOCUMENTATION"


class TaskPriority(StrEnum):
    """Prioridades neutras utilizadas pelo ecossistema."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Task(CoreModel):
    """Tarefa neutra. Não contém campos específicos de Jira/GitHub/ClickUp/Linear."""

    id: str = Field(default_factory=new_id)
    workspace_id: str
    type: TaskType
    title: str
    description: str | None = None
    priority: TaskPriority = TaskPriority.MEDIUM
    labels: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

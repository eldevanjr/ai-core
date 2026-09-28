"""Destino genérico: onde uma tarefa pode ser criada em uma ferramenta externa.

O core não conhece Jira/GitHub/ClickUp/Linear. Ele apenas descreve o destino de
forma neutra; cada provider concreto traduz este modelo para a sua hierarquia.
"""

from __future__ import annotations

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id
from ai_core.models.task import TaskPriority


class DestinationRef(CoreModel):
    """Referência neutra a um destino externo de tarefa.

    Nem todos os campos se aplicam a todos os providers (ex.: GitHub não possui
    ``issue_type``; ClickUp usa uma hierarquia diferente).
    """

    id: str = Field(default_factory=new_id)
    provider: str
    connection_id: str | None = None
    external_workspace_id: str | None = None
    external_project_id: str | None = None
    external_container_id: str | None = None
    issue_type: str | None = None
    default_priority: TaskPriority | None = None
    enabled: bool = True

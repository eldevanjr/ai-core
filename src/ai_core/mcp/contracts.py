"""Contratos comuns das ferramentas MCP.

O core define os contratos/modelos; cada aplicação implementa seu servidor MCP.
Operações relacionadas a conhecimento ou tarefas exigem ``workspace_id``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.destination import DestinationRef
from ai_core.models.knowledge import KnowledgeDocument
from ai_core.models.proposal import ProposalStatus, TaskProposal
from ai_core.models.task import Task
from ai_core.rag.models import SearchResult


class ToolName(StrEnum):
    """Ferramentas MCP comuns ao ecossistema."""

    SEARCH_KNOWLEDGE = "search_knowledge"
    GET_KNOWLEDGE = "get_knowledge"
    SEARCH_TASKS = "search_tasks"
    GENERATE_TASK = "generate_task"
    LIST_TASK_PROPOSALS = "list_task_proposals"
    APPROVE_TASK = "approve_task"


class SearchKnowledgeRequest(CoreModel):
    """Entrada de ``search_knowledge``."""

    workspace_id: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, gt=0)


class SearchKnowledgeResponse(CoreModel):
    """Saída de ``search_knowledge``."""

    results: list[SearchResult] = Field(default_factory=list)


class GetKnowledgeRequest(CoreModel):
    """Entrada de ``get_knowledge``."""

    workspace_id: str
    document_id: str


class GetKnowledgeResponse(CoreModel):
    """Saída de ``get_knowledge``."""

    document: KnowledgeDocument | None = None


class SearchTasksRequest(CoreModel):
    """Entrada de ``search_tasks``."""

    workspace_id: str
    query: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, gt=0)
    offset: int = Field(default=0, ge=0)


class SearchTasksResponse(CoreModel):
    """Saída de ``search_tasks``."""

    tasks: list[Task] = Field(default_factory=list)
    total: int | None = Field(default=None, ge=0)


class GenerateTaskRequest(CoreModel):
    """Entrada de ``generate_task`` (gera proposta, não cria tarefa externa)."""

    workspace_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    instructions: str | None = None


class GenerateTaskResponse(CoreModel):
    """Saída de ``generate_task``."""

    proposal: TaskProposal


class ListTaskProposalsRequest(CoreModel):
    """Entrada de ``list_task_proposals``."""

    workspace_id: str
    status: ProposalStatus | None = None
    limit: int = Field(default=50, gt=0)
    offset: int = Field(default=0, ge=0)


class ListTaskProposalsResponse(CoreModel):
    """Saída de ``list_task_proposals``."""

    proposals: list[TaskProposal] = Field(default_factory=list)
    total: int | None = Field(default=None, ge=0)


class ApproveTaskRequest(CoreModel):
    """Entrada de ``approve_task``."""

    workspace_id: str
    proposal_id: str
    approved: bool = True
    destination: DestinationRef | None = None
    approved_by: str | None = None


class ApproveTaskResponse(CoreModel):
    """Saída de ``approve_task``."""

    proposal: TaskProposal


TOOL_CONTRACTS: dict[ToolName, tuple[type[CoreModel], type[CoreModel]]] = {
    ToolName.SEARCH_KNOWLEDGE: (SearchKnowledgeRequest, SearchKnowledgeResponse),
    ToolName.GET_KNOWLEDGE: (GetKnowledgeRequest, GetKnowledgeResponse),
    ToolName.SEARCH_TASKS: (SearchTasksRequest, SearchTasksResponse),
    ToolName.GENERATE_TASK: (GenerateTaskRequest, GenerateTaskResponse),
    ToolName.LIST_TASK_PROPOSALS: (ListTaskProposalsRequest, ListTaskProposalsResponse),
    ToolName.APPROVE_TASK: (ApproveTaskRequest, ApproveTaskResponse),
}
"""Entrada e saída de cada ferramenta MCP, para registrar schemas nos servidores."""

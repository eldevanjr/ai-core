"""Modelos de domínio compartilhados do ai-core."""

from __future__ import annotations

from ai_core.models.audit import AuditEvent
from ai_core.models.base import CoreModel
from ai_core.models.common import ApprovalMode, ProviderLocation, new_id, utc_now
from ai_core.models.destination import DestinationRef
from ai_core.models.evidence import Evidence, EvidenceType
from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument
from ai_core.models.proposal import (
    ALLOWED_TRANSITIONS,
    ApprovalRecord,
    ContextReference,
    ExecutionResult,
    ProposalStatus,
    TaskExplanation,
    TaskProposal,
)
from ai_core.models.source import Source, SourceKind
from ai_core.models.task import Task, TaskPriority, TaskType
from ai_core.models.workspace import Workspace, WorkspaceStatus

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ApprovalMode",
    "ApprovalRecord",
    "AuditEvent",
    "ContextReference",
    "CoreModel",
    "DestinationRef",
    "Evidence",
    "EvidenceType",
    "ExecutionResult",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "ProposalStatus",
    "ProviderLocation",
    "Source",
    "SourceKind",
    "Task",
    "TaskExplanation",
    "TaskPriority",
    "TaskProposal",
    "TaskType",
    "Workspace",
    "WorkspaceStatus",
    "new_id",
    "utc_now",
]

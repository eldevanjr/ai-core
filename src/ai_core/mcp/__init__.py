"""Contratos MCP do ai-core."""

from __future__ import annotations

from ai_core.mcp.contracts import (
    TOOL_CONTRACTS,
    ApproveTaskRequest,
    ApproveTaskResponse,
    GenerateTaskRequest,
    GenerateTaskResponse,
    GetKnowledgeRequest,
    GetKnowledgeResponse,
    ListTaskProposalsRequest,
    ListTaskProposalsResponse,
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    SearchTasksRequest,
    SearchTasksResponse,
    ToolName,
)

__all__ = [
    "TOOL_CONTRACTS",
    "ApproveTaskRequest",
    "ApproveTaskResponse",
    "GenerateTaskRequest",
    "GenerateTaskResponse",
    "GetKnowledgeRequest",
    "GetKnowledgeResponse",
    "ListTaskProposalsRequest",
    "ListTaskProposalsResponse",
    "SearchKnowledgeRequest",
    "SearchKnowledgeResponse",
    "SearchTasksRequest",
    "SearchTasksResponse",
    "ToolName",
]

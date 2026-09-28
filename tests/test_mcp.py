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
from ai_core.models.destination import DestinationRef
from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument
from ai_core.models.proposal import ProposalStatus, TaskProposal
from ai_core.models.task import Task, TaskType
from ai_core.rag.models import SearchResult


def test_tool_names() -> None:
    assert {tool.value for tool in ToolName} == {
        "search_knowledge",
        "get_knowledge",
        "search_tasks",
        "generate_task",
        "list_task_proposals",
        "approve_task",
    }


def test_search_knowledge_contract() -> None:
    request = SearchKnowledgeRequest(workspace_id="ws", query="sync", limit=5)
    assert request.filters == {}
    result = SearchResult(
        chunk=KnowledgeChunk(document_id="d", workspace_id="ws", content="c"), score=0.9
    )
    response = SearchKnowledgeResponse(results=[result])
    assert SearchKnowledgeResponse.model_validate(response.model_dump()) == response


def test_get_knowledge_contract() -> None:
    request = GetKnowledgeRequest(workspace_id="ws", document_id="doc-1")
    assert request.document_id == "doc-1"
    assert GetKnowledgeResponse().document is None
    document = KnowledgeDocument(workspace_id="ws", title="T", content="c")
    assert GetKnowledgeResponse(document=document).document is document


def test_search_tasks_contract() -> None:
    request = SearchTasksRequest(workspace_id="ws")
    assert request.query is None
    assert request.limit == 10
    assert request.offset == 0
    task = Task(workspace_id="ws", type=TaskType.BUG, title="t")
    assert SearchTasksResponse(tasks=[task]).tasks == [task]


def test_generate_task_contract() -> None:
    request = GenerateTaskRequest(workspace_id="ws", evidence_ids=["e1"], instructions="foque")
    assert request.evidence_ids == ["e1"]
    response = GenerateTaskResponse(proposal=TaskProposal(workspace_id="ws"))
    assert response.proposal.workspace_id == "ws"


def test_list_task_proposals_contract() -> None:
    request = ListTaskProposalsRequest(workspace_id="ws", status=ProposalStatus.NEEDS_REVIEW)
    assert request.status is ProposalStatus.NEEDS_REVIEW
    task = Task(workspace_id="ws", type=TaskType.BUG, title="t")
    proposals = ListTaskProposalsResponse(
        proposals=[TaskProposal(workspace_id="ws", task=task, status=ProposalStatus.NEEDS_REVIEW)]
    )
    assert len(proposals.proposals) == 1
    default = ListTaskProposalsRequest(workspace_id="ws")
    assert default.status is None
    assert (default.limit, default.offset) == (50, 0)
    assert proposals.total is None


def test_tool_contracts_cover_every_tool() -> None:
    assert set(TOOL_CONTRACTS) == set(ToolName)
    assert TOOL_CONTRACTS[ToolName.APPROVE_TASK] == (ApproveTaskRequest, ApproveTaskResponse)


def test_approve_task_contract() -> None:
    request = ApproveTaskRequest(
        workspace_id="ws",
        proposal_id="p1",
        destination=DestinationRef(provider="JIRA"),
        approved_by="user",
    )
    assert request.approved is True
    assert request.approved_by == "user"
    assert request.destination is not None
    response = ApproveTaskResponse(proposal=TaskProposal(workspace_id="ws"))
    assert response.proposal.workspace_id == "ws"
    assert ApproveTaskRequest(workspace_id="ws", proposal_id="p").destination is None

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.models.common import ApprovalMode
from ai_core.models.destination import DestinationRef
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
from ai_core.models.task import Task, TaskType


def test_proposal_status_members() -> None:
    assert {member.value for member in ProposalStatus} == {
        "DRAFT",
        "NEEDS_REVIEW",
        "APPROVED",
        "REJECTED",
        "CREATED",
        "FAILED",
    }


def test_proposal_defaults() -> None:
    proposal = TaskProposal(workspace_id="ws")
    assert proposal.id
    assert proposal.task is None
    assert proposal.sources == []
    assert proposal.explanation is None
    assert proposal.status is ProposalStatus.DRAFT
    assert proposal.destination_reference is None
    assert proposal.created_at.tzinfo is not None
    assert proposal.updated_at.tzinfo is not None


def test_proposal_status_cannot_be_assigned_directly() -> None:
    proposal = TaskProposal(workspace_id="ws")
    with pytest.raises(ValidationError):
        proposal.status = ProposalStatus.CREATED  # type: ignore[misc]
    assert proposal.status is ProposalStatus.DRAFT


def test_explanation_full_roundtrip() -> None:
    source = Source(type=SourceKind.SCREENSHOT, location="screenshot.png")
    destination = DestinationRef(provider="JIRA", external_project_id="APP")
    explanation = TaskExplanation(
        sources=[source],
        retrieved_context=[ContextReference(source=source, excerpt="trecho", relevance="alta")],
        analysis_summary="Screenshot indica erro 500.",
        reasoning_summary="Relacionado à sincronização offline.",
        field_justifications={"priority": "Erro bloqueia sincronização"},
        decisions=["Marcar como BUG"],
        rejected_alternatives=["FEATURE"],
        routing_decision="BUG -> Jira/APP",
        destination=destination,
        approval=ApprovalRecord(mode=ApprovalMode.ALWAYS_REQUIRED, approved=True, approved_by="u"),
        execution_result=ExecutionResult(
            success=True,
            provider="JIRA",
            external_id="193",
            external_url="https://jira/193",
        ),
    )
    assert TaskExplanation.model_validate(explanation.model_dump()) == explanation
    assert explanation.approval is not None
    assert explanation.approval.approved is True
    assert explanation.execution_result is not None
    assert explanation.execution_result.external_id == "193"


def test_explanation_defaults() -> None:
    explanation = TaskExplanation()
    assert explanation.sources == []
    assert explanation.retrieved_context == []
    assert explanation.analysis_summary is None
    assert explanation.reasoning_summary is None
    assert explanation.field_justifications == {}
    assert explanation.decisions == []
    assert explanation.rejected_alternatives == []
    assert explanation.routing_decision is None
    assert explanation.destination is None
    assert explanation.approval is None
    assert explanation.execution_result is None


def test_proposal_with_task_and_explanation() -> None:
    task = Task(workspace_id="ws", type=TaskType.BUG, title="Falha")
    proposal = TaskProposal(
        workspace_id="ws",
        task=task,
        sources=[Source(type=SourceKind.LOG_LINE, location="error.log:842")],
        explanation=TaskExplanation(analysis_summary="resumo"),
        status=ProposalStatus.NEEDS_REVIEW,
        destination_reference=DestinationRef(provider="JIRA", external_project_id="APP"),
    )
    assert proposal.task is not None
    assert proposal.task.title == "Falha"
    assert proposal.explanation is not None
    assert proposal.destination_reference is not None
    assert proposal.destination_reference.external_project_id == "APP"


def test_approval_record_defaults() -> None:
    record = ApprovalRecord(mode=ApprovalMode.AUTO_APPROVED, approved=True)
    assert record.approved_by is None
    assert record.decided_at.tzinfo is not None


def test_execution_result_defaults() -> None:
    result = ExecutionResult(success=False, error="timeout")
    assert result.provider is None
    assert result.external_id is None
    assert result.external_url is None
    assert result.error == "timeout"


def _proposal(status: ProposalStatus = ProposalStatus.DRAFT) -> TaskProposal:
    task = Task(workspace_id="ws", type=TaskType.BUG, title="Falha")
    return TaskProposal(workspace_id="ws", task=task, status=status)


def _approval(approved: bool = True) -> ApprovalRecord:
    return ApprovalRecord(mode=ApprovalMode.ALWAYS_REQUIRED, approved=approved, approved_by="u")


def test_happy_path_transitions_update_timestamp_and_record_approval() -> None:
    draft = _proposal()
    review = draft.transition_to(ProposalStatus.NEEDS_REVIEW)
    assert draft.status is ProposalStatus.DRAFT
    assert review.status is ProposalStatus.NEEDS_REVIEW
    assert review.updated_at >= draft.updated_at
    assert review.id == draft.id
    approved = review.transition_to(ProposalStatus.APPROVED, approval=_approval())
    assert approved.explanation is not None
    assert approved.explanation.approval == _approval().model_copy(
        update={"decided_at": approved.explanation.approval.decided_at}  # type: ignore[union-attr]
    )
    created = approved.transition_to(ProposalStatus.CREATED)
    assert created.status is ProposalStatus.CREATED
    assert created.can_transition_to(ProposalStatus.DRAFT) is False


def test_approval_keeps_existing_explanation() -> None:
    review = _proposal(ProposalStatus.NEEDS_REVIEW).model_copy(
        update={"explanation": TaskExplanation(analysis_summary="resumo")}
    )
    approved = review.transition_to(ProposalStatus.APPROVED, approval=_approval())
    assert approved.explanation is not None
    assert approved.explanation.analysis_summary == "resumo"
    assert approved.explanation.approval is not None


def test_rejection_can_record_negative_approval() -> None:
    rejected = _proposal(ProposalStatus.NEEDS_REVIEW).transition_to(
        ProposalStatus.REJECTED, approval=_approval(approved=False)
    )
    assert rejected.explanation is not None
    assert rejected.explanation.approval is not None
    assert rejected.explanation.approval.approved is False


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (ProposalStatus.DRAFT, ProposalStatus.CREATED),
        (ProposalStatus.DRAFT, ProposalStatus.APPROVED),
        (ProposalStatus.NEEDS_REVIEW, ProposalStatus.CREATED),
        (ProposalStatus.CREATED, ProposalStatus.FAILED),
    ],
)
def test_invalid_transitions_are_rejected(start: ProposalStatus, target: ProposalStatus) -> None:
    with pytest.raises(ValueError, match="transição inválida"):
        _proposal(start).transition_to(target, approval=_approval())


def test_approved_requires_positive_approval_record() -> None:
    review = _proposal(ProposalStatus.NEEDS_REVIEW)
    with pytest.raises(ValueError, match="ApprovalRecord"):
        review.transition_to(ProposalStatus.APPROVED)
    with pytest.raises(ValueError, match="ApprovalRecord"):
        review.transition_to(ProposalStatus.APPROVED, approval=_approval(approved=False))


def test_failed_can_be_retried() -> None:
    failed = _proposal(ProposalStatus.FAILED)
    assert failed.transition_to(ProposalStatus.APPROVED, approval=_approval()).status is (
        ProposalStatus.APPROVED
    )


def test_transitions_graph_covers_every_status() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(ProposalStatus)
    reachable_created = [
        s for s, targets in ALLOWED_TRANSITIONS.items() if ProposalStatus.CREATED in targets
    ]
    assert reachable_created == [ProposalStatus.APPROVED]


def test_proposal_requires_task_beyond_draft() -> None:
    with pytest.raises(ValidationError, match="exige uma task"):
        TaskProposal(workspace_id="ws", status=ProposalStatus.NEEDS_REVIEW)
    with pytest.raises(ValidationError, match="exige uma task"):
        TaskProposal(workspace_id="ws").transition_to(ProposalStatus.NEEDS_REVIEW)
    assert TaskProposal(workspace_id="ws", status=ProposalStatus.REJECTED).task is None


def test_proposal_rejects_task_from_other_workspace() -> None:
    task = Task(workspace_id="other", type=TaskType.BUG, title="t")
    with pytest.raises(ValidationError, match="workspace"):
        TaskProposal(workspace_id="ws", task=task)


def test_manual_only_approval_requires_person() -> None:
    with pytest.raises(ValidationError, match="approved_by"):
        ApprovalRecord(mode=ApprovalMode.MANUAL_ONLY, approved=True)
    rejection = ApprovalRecord(mode=ApprovalMode.MANUAL_ONLY, approved=False)
    assert rejection.approved_by is None

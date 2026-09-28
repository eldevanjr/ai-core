from __future__ import annotations

from ai_core.models.audit import AuditEvent


def test_audit_event_defaults() -> None:
    event = AuditEvent(workspace_id="ws", action="analyze")
    assert event.id
    assert event.actor is None
    assert event.evidence_ids == []
    assert event.proposal_id is None
    assert event.model is None
    assert event.provider is None
    assert event.retrieved_source_ids == []
    assert event.destination is None
    assert event.approval is None
    assert event.execution is None
    assert event.external_id is None
    assert event.metadata == {}
    assert event.timestamp.tzinfo is not None


def test_audit_event_full_roundtrip() -> None:
    event = AuditEvent(
        id="audit-1",
        workspace_id="ws",
        action="create_task",
        actor="user",
        evidence_ids=["e1", "e2"],
        proposal_id="p1",
        model="qwen3",
        provider="OLLAMA",
        retrieved_source_ids=["s1"],
        destination="JIRA/APP/BUG",
        approval="APPROVED",
        execution="CREATED",
        external_id="JIRA-193",
        metadata={"latency_ms": 42},
    )
    assert AuditEvent.model_validate(event.model_dump()) == event

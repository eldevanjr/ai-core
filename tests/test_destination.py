from __future__ import annotations

from ai_core.models.destination import DestinationRef
from ai_core.models.task import TaskPriority


def test_destination_ref_defaults() -> None:
    destination = DestinationRef(provider="JIRA")
    assert destination.id
    assert destination.enabled is True
    assert destination.connection_id is None
    assert destination.external_workspace_id is None
    assert destination.external_project_id is None
    assert destination.external_container_id is None
    assert destination.issue_type is None
    assert destination.default_priority is None


def test_destination_ref_full() -> None:
    destination = DestinationRef(
        id="dest-1",
        provider="JIRA",
        connection_id="conn-1",
        external_workspace_id="empresa",
        external_project_id="APP",
        external_container_id="container",
        issue_type="BUG",
        default_priority=TaskPriority.HIGH,
        enabled=False,
    )
    assert destination.provider == "JIRA"
    assert destination.default_priority is TaskPriority.HIGH
    assert destination.enabled is False

from __future__ import annotations

from ai_core.models.source import Source, SourceKind
from ai_core.models.task import Task, TaskPriority, TaskType


def test_task_type_members() -> None:
    assert {member.value for member in TaskType} == {
        "BUG",
        "TASK",
        "FEATURE",
        "IMPROVEMENT",
        "TECH_DEBT",
        "QUESTION",
        "INVESTIGATION",
        "DOCUMENTATION",
    }


def test_task_priority_members() -> None:
    assert {member.value for member in TaskPriority} == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_task_defaults() -> None:
    task = Task(workspace_id="ws", type=TaskType.BUG, title="Falha offline")
    assert task.id
    assert task.priority is TaskPriority.MEDIUM
    assert task.description is None
    assert task.labels == []
    assert task.acceptance_criteria == []
    assert task.sources == []
    assert task.related_entities == []
    assert task.created_at.tzinfo is not None


def test_task_full_roundtrip() -> None:
    task = Task(
        id="task-1",
        workspace_id="ws",
        type=TaskType.FEATURE,
        title="Sincronização",
        description="desc",
        priority=TaskPriority.HIGH,
        labels=["mobile", "sync"],
        acceptance_criteria=["funciona offline"],
        sources=[Source(type=SourceKind.TRANSCRIPT, location="00:43:12")],
        related_entities=["ServicoAbastecimento"],
    )
    assert Task.model_validate(task.model_dump()) == task
    assert task.sources[0].location == "00:43:12"

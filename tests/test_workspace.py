from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from ai_core.models.workspace import Workspace, WorkspaceStatus


def test_defaults() -> None:
    workspace = Workspace(name="Mobile", slug="mobile")
    assert workspace.id
    assert workspace.status is WorkspaceStatus.ACTIVE
    assert workspace.description is None
    assert workspace.metadata == {}
    assert workspace.created_at.tzinfo is UTC


def test_explicit_fields() -> None:
    workspace = Workspace(
        id="ws-1",
        name="Mobile App",
        description="Aplicativo mobile",
        slug="mobile-app",
        status=WorkspaceStatus.ARCHIVED,
        metadata={"tier": "gold"},
    )
    assert workspace.id == "ws-1"
    assert workspace.description == "Aplicativo mobile"
    assert workspace.status is WorkspaceStatus.ARCHIVED
    assert workspace.metadata == {"tier": "gold"}


def test_serialization_roundtrip() -> None:
    workspace = Workspace(id="ws-1", name="Mobile", slug="mobile")
    assert Workspace.model_validate(workspace.model_dump()) == workspace


def test_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        Workspace.model_validate({"name": "x", "slug": "x", "unknown": 1})


def test_assignment_is_validated() -> None:
    workspace = Workspace(name="x", slug="x")
    workspace.status = WorkspaceStatus.DISABLED
    assert workspace.status is WorkspaceStatus.DISABLED
    with pytest.raises(ValidationError):
        workspace.status = "nope"  # type: ignore[assignment]

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.models.evidence import Evidence, EvidenceType


def test_evidence_type_members() -> None:
    assert {member.value for member in EvidenceType} == {
        "TEXT",
        "TRANSCRIPT",
        "AUDIO",
        "VIDEO",
        "IMAGE",
        "SCREENSHOT",
        "LOG",
        "ERROR",
        "PDF",
        "DOCUMENT",
        "URL",
        "CODE",
        "DIFF",
    }


@pytest.mark.parametrize("source_type", list(EvidenceType))
def test_evidence_accepts_every_type(source_type: EvidenceType) -> None:
    evidence = Evidence(
        workspace_id="ws-1",
        source_type=source_type,
        source_name="arquivo.txt",
        mime_type="text/plain",
        content="conteúdo",
        metadata={"linhas": 10},
    )
    assert evidence.source_type is source_type
    assert evidence.workspace_id == "ws-1"
    assert evidence.id
    assert evidence.mime_type == "text/plain"


def test_evidence_optional_fields() -> None:
    evidence = Evidence(workspace_id="ws", source_type=EvidenceType.TEXT, source_name="t")
    assert evidence.mime_type is None
    assert evidence.content is None
    assert evidence.metadata == {}


def test_evidence_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate(
            {"workspace_id": "ws", "source_type": "TEXT", "source_name": "t", "x": 1}
        )

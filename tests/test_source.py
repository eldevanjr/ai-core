from __future__ import annotations

from ai_core.models.source import Source, SourceKind


def test_source_kind_members() -> None:
    assert {member.value for member in SourceKind} == {
        "FILE",
        "TRANSCRIPT",
        "TIMESTAMP",
        "LOG_LINE",
        "URL",
        "ISSUE",
        "COMMIT",
        "SCREENSHOT",
        "OTHER",
    }


def test_source_full() -> None:
    source = Source(
        id="src-1",
        evidence_id="ev-1",
        type=SourceKind.TIMESTAMP,
        location="00:30:34",
        title="Reunião 26/09",
        metadata={"speaker": "João"},
    )
    assert source.id == "src-1"
    assert source.evidence_id == "ev-1"
    assert source.type is SourceKind.TIMESTAMP
    assert source.location == "00:30:34"
    assert source.metadata == {"speaker": "João"}


def test_source_defaults() -> None:
    source = Source(type=SourceKind.URL)
    assert source.id
    assert source.evidence_id is None
    assert source.location is None
    assert source.title is None
    assert source.metadata == {}

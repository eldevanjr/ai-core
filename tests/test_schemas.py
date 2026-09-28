from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_core.mcp.contracts import TOOL_CONTRACTS
from ai_core.schemas import SCHEMA_MODELS, export_json_schemas, main, write_json_schemas


def test_export_covers_entities_policies_and_mcp_contracts() -> None:
    schemas = export_json_schemas()
    assert len(schemas) == len(SCHEMA_MODELS)
    for name in ("Workspace", "Evidence", "TaskProposal", "PrivacyPolicy", "SearchResult"):
        assert name in schemas
    for request, response in TOOL_CONTRACTS.values():
        assert request.__name__ in schemas
        assert response.__name__ in schemas


def test_schema_reflects_enums_and_forbids_extra_fields() -> None:
    proposal = export_json_schemas()["TaskProposal"]
    assert proposal["additionalProperties"] is False
    status = proposal["$defs"]["ProposalStatus"]
    assert set(status["enum"]) == {
        "DRAFT",
        "NEEDS_REVIEW",
        "APPROVED",
        "REJECTED",
        "CREATED",
        "FAILED",
    }


def test_write_json_schemas(tmp_path: Path) -> None:
    written = write_json_schemas(tmp_path / "out")
    assert len(written) == len(SCHEMA_MODELS)
    workspace = json.loads((tmp_path / "out" / "Workspace.schema.json").read_text("utf-8"))
    assert workspace["title"] == "Workspace"


def test_main_writes_schemas(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 0
    assert "Workspace.schema.json" in capsys.readouterr().out


def test_main_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "uso:" in capsys.readouterr().err


def test_main_reads_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["ai_core.schemas", str(tmp_path)])
    assert main() == 0
    assert capsys.readouterr().out

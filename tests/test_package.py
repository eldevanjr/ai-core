from __future__ import annotations

from pathlib import Path

import ai_core
import ai_core.models
from ai_core import (
    ApprovalPolicy,
    EmbeddingProvider,
    Evidence,
    KnowledgeChunk,
    KnowledgeDocument,
    LLMProvider,
    ModelRouter,
    PrivacyPolicy,
    Source,
    Task,
    TaskProposal,
    Workspace,
)


def test_version() -> None:
    assert ai_core.__version__ == "0.2.0"


def test_public_exports() -> None:
    assert ai_core.__all__
    for symbol in (
        ApprovalPolicy,
        EmbeddingProvider,
        Evidence,
        KnowledgeChunk,
        KnowledgeDocument,
        LLMProvider,
        ModelRouter,
        PrivacyPolicy,
        Source,
        Task,
        TaskProposal,
        Workspace,
    ):
        assert symbol is not None


def test_star_import_symbols_are_unique() -> None:
    assert len(ai_core.__all__) == len(set(ai_core.__all__))


def test_mcp_and_schema_exports_at_root() -> None:
    for name in ("ToolName", "TOOL_CONTRACTS", "ApproveTaskRequest", "ALLOWED_TRANSITIONS"):
        assert name in ai_core.__all__
        assert getattr(ai_core, name) is not None


def test_models_do_not_depend_on_policies() -> None:
    models_dir = Path(ai_core.__file__).parent / "models"
    for module in models_dir.glob("*.py"):
        source = module.read_text("utf-8")
        assert "from ai_core.policies" not in source, module.name
        assert "import ai_core.policies" not in source, module.name
    assert ai_core.models.ApprovalMode is ai_core.ApprovalMode

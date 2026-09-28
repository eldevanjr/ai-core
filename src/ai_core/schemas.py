"""Exportação de JSON Schema dos contratos do ai-core.

Os schemas gerados aqui são a fonte única para o pacote ``ai-core-types``
(TypeScript), evitando duplicação manual dos modelos (plano §22 / CA-17).

Uso::

    uv run python -m ai_core.schemas <diretório-de-saída>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ai_core.embeddings.models import Embedding
from ai_core.llm.request import LLMRequest
from ai_core.llm.response import LLMResponse
from ai_core.mcp.contracts import TOOL_CONTRACTS
from ai_core.models.audit import AuditEvent
from ai_core.models.base import CoreModel
from ai_core.models.destination import DestinationRef
from ai_core.models.evidence import Evidence
from ai_core.models.knowledge import KnowledgeChunk, KnowledgeDocument
from ai_core.models.proposal import TaskExplanation, TaskProposal
from ai_core.models.source import Source
from ai_core.models.task import Task
from ai_core.models.workspace import Workspace
from ai_core.policies.approval import ApprovalPolicy
from ai_core.policies.privacy import PrivacyPolicy
from ai_core.policies.routing import RoutingPolicy
from ai_core.rag.models import SearchRequest, SearchResult

SCHEMA_MODELS: tuple[type[CoreModel], ...] = (
    Workspace,
    Evidence,
    Source,
    KnowledgeDocument,
    KnowledgeChunk,
    Task,
    TaskProposal,
    TaskExplanation,
    DestinationRef,
    AuditEvent,
    LLMRequest,
    LLMResponse,
    Embedding,
    SearchRequest,
    SearchResult,
    PrivacyPolicy,
    ApprovalPolicy,
    RoutingPolicy,
    *(model for pair in TOOL_CONTRACTS.values() for model in pair),
)
"""Modelos publicados como contrato (entidades, políticas e MCP)."""


def export_json_schemas() -> dict[str, dict[str, Any]]:
    """Retorna o JSON Schema (modo serialização) de cada modelo, pelo nome."""
    return {
        model.__name__: model.model_json_schema(mode="serialization") for model in SCHEMA_MODELS
    }


def write_json_schemas(directory: Path) -> list[Path]:
    """Grava um ``<Modelo>.schema.json`` por modelo e retorna os caminhos gerados."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in sorted(export_json_schemas().items()):
        path = directory / f"{name}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("uso: python -m ai_core.schemas <diretório-de-saída>", file=sys.stderr)
        return 2
    for path in write_json_schemas(Path(args[0])):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

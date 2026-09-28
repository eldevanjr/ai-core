"""Source: origem verificável e rastreável de uma informação."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id


class SourceKind(StrEnum):
    """Natureza da origem de uma informação."""

    FILE = "FILE"
    TRANSCRIPT = "TRANSCRIPT"
    TIMESTAMP = "TIMESTAMP"
    LOG_LINE = "LOG_LINE"
    URL = "URL"
    ISSUE = "ISSUE"
    COMMIT = "COMMIT"
    SCREENSHOT = "SCREENSHOT"
    OTHER = "OTHER"


class Source(CoreModel):
    """Referência navegável para a origem de um trecho de conhecimento ou tarefa."""

    id: str = Field(default_factory=new_id)
    evidence_id: str | None = None
    type: SourceKind
    location: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

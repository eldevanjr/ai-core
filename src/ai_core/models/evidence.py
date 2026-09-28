"""Evidence: matéria-prima genérica a partir da qual conhecimento é derivado."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.common import new_id, utc_now


class EvidenceType(StrEnum):
    """Tipos de evidência aceitos pelo ecossistema."""

    TEXT = "TEXT"
    TRANSCRIPT = "TRANSCRIPT"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"
    SCREENSHOT = "SCREENSHOT"
    LOG = "LOG"
    ERROR = "ERROR"
    PDF = "PDF"
    DOCUMENT = "DOCUMENT"
    URL = "URL"
    CODE = "CODE"
    DIFF = "DIFF"


class Evidence(CoreModel):
    """Evidência multimodal associada a um workspace.

    O ai-core fornece apenas o modelo comum; o processamento específico de cada
    formato é responsabilidade das aplicações.
    """

    id: str = Field(default_factory=new_id)
    workspace_id: str
    source_type: EvidenceType
    source_name: str
    mime_type: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

"""Modelo base compartilhado por todas as entidades do ai-core."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CoreModel(BaseModel):
    """Base de todos os modelos do core.

    Configuração intencional:

    - ``extra="forbid"``: contratos são explícitos; campos desconhecidos falham cedo.
    - ``validate_assignment=True``: mudanças de estado (ex.: TaskProposal) são validadas.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

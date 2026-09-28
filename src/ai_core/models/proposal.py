"""TaskProposal e TaskExplanation: proposta auditável antes de qualquer ação externa."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from ai_core.models.base import CoreModel
from ai_core.models.common import ApprovalMode, new_id, utc_now
from ai_core.models.destination import DestinationRef
from ai_core.models.source import Source
from ai_core.models.task import Task


class ProposalStatus(StrEnum):
    """Estados do ciclo de vida de uma proposta de tarefa."""

    DRAFT = "DRAFT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CREATED = "CREATED"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    ProposalStatus.DRAFT: frozenset({ProposalStatus.NEEDS_REVIEW, ProposalStatus.REJECTED}),
    ProposalStatus.NEEDS_REVIEW: frozenset(
        {ProposalStatus.APPROVED, ProposalStatus.REJECTED, ProposalStatus.DRAFT}
    ),
    ProposalStatus.APPROVED: frozenset({ProposalStatus.CREATED, ProposalStatus.FAILED}),
    ProposalStatus.FAILED: frozenset({ProposalStatus.APPROVED, ProposalStatus.NEEDS_REVIEW}),
    ProposalStatus.REJECTED: frozenset({ProposalStatus.DRAFT}),
    ProposalStatus.CREATED: frozenset(),
}
"""Grafo de estados: nenhuma proposta chega a CREATED sem passar por APPROVED."""

_REQUIRES_TASK = frozenset(
    {
        ProposalStatus.NEEDS_REVIEW,
        ProposalStatus.APPROVED,
        ProposalStatus.CREATED,
        ProposalStatus.FAILED,
    }
)


class ContextReference(CoreModel):
    """Item de contexto recuperado, com justificativa descritiva da relevância.

    Não expõe raciocínio interno privado do modelo.
    """

    source: Source
    excerpt: str | None = None
    relevance: str | None = None


class ApprovalRecord(CoreModel):
    """Registro factual da aprovação humana (ou automática) de uma proposta.

    Em ``MANUAL_ONLY`` uma aprovação precisa identificar a pessoa (``approved_by``).
    """

    mode: ApprovalMode
    approved: bool
    approved_by: str | None = None
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _manual_requires_person(self) -> ApprovalRecord:
        if self.mode == ApprovalMode.MANUAL_ONLY and self.approved and not self.approved_by:
            raise ValueError("aprovação MANUAL_ONLY exige 'approved_by'")
        return self


class ExecutionResult(CoreModel):
    """Resultado da execução de uma proposta em um destino externo."""

    success: bool
    provider: str | None = None
    external_id: str | None = None
    external_url: str | None = None
    error: str | None = None
    executed_at: datetime = Field(default_factory=utc_now)


class TaskExplanation(CoreModel):
    """Resumo factual e auditável de como uma tarefa foi proposta.

    Propositalmente não armazena chain-of-thought privado do modelo: apenas
    evidências, contexto, justificativas resumidas e o resultado observável.
    """

    sources: list[Source] = Field(default_factory=list)
    retrieved_context: list[ContextReference] = Field(default_factory=list)
    analysis_summary: str | None = None
    reasoning_summary: str | None = None
    field_justifications: dict[str, str] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)
    routing_decision: str | None = None
    destination: DestinationRef | None = None
    approval: ApprovalRecord | None = None
    execution_result: ExecutionResult | None = None
    created_at: datetime = Field(default_factory=utc_now)


class TaskProposal(CoreModel):
    """Tarefa proposta, ainda não executada, com fontes e explicação auditáveis.

    ``status`` não pode ser atribuído diretamente: use :meth:`transition_to`, que
    valida o grafo :data:`ALLOWED_TRANSITIONS` e atualiza ``updated_at``.
    """

    id: str = Field(default_factory=new_id)
    workspace_id: str
    task: Task | None = None
    sources: list[Source] = Field(default_factory=list)
    explanation: TaskExplanation | None = None
    status: ProposalStatus = Field(default=ProposalStatus.DRAFT, frozen=True)
    destination_reference: DestinationRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_consistency(self) -> TaskProposal:
        if self.task is not None and self.task.workspace_id != self.workspace_id:
            raise ValueError("task.workspace_id difere do workspace da proposta")
        if self.task is None and self.status in _REQUIRES_TASK:
            raise ValueError(f"status {self.status} exige uma task")
        return self

    def can_transition_to(self, status: ProposalStatus) -> bool:
        """Indica se a transição para ``status`` é permitida a partir do atual."""
        return status in ALLOWED_TRANSITIONS[self.status]

    def transition_to(
        self, status: ProposalStatus, *, approval: ApprovalRecord | None = None
    ) -> TaskProposal:
        """Retorna uma nova proposta no estado ``status``.

        Ir para ``APPROVED`` exige um ``ApprovalRecord`` aprovado, que fica
        registrado em ``explanation.approval``.
        """
        status = ProposalStatus(status)
        if not self.can_transition_to(status):
            raise ValueError(f"transição inválida: {self.status} -> {status}")
        explanation = self.explanation
        if status == ProposalStatus.APPROVED and (approval is None or not approval.approved):
            raise ValueError("transição para APPROVED exige um ApprovalRecord aprovado")
        if approval is not None:
            base = explanation or TaskExplanation()
            explanation = base.model_copy(update={"approval": approval})
        return type(self).model_validate(
            {**dict(self), "status": status, "explanation": explanation, "updated_at": utc_now()}
        )

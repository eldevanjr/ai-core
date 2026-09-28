"""Política de aprovação humana para ações (especialmente externas)."""

from __future__ import annotations

from ai_core.models.base import CoreModel
from ai_core.models.common import ApprovalMode
from ai_core.models.proposal import ApprovalRecord

__all__ = ["ApprovalMode", "ApprovalPolicy"]


class ApprovalPolicy(CoreModel):
    """Define quando uma ação exige aprovação explícita.

    O padrão do ecossistema é ``ALWAYS_REQUIRED``: por padrão nenhuma ação
    externa ocorre sem aprovação humana.

    - ``ALWAYS_REQUIRED``: toda ação exige aprovação.
    - ``REQUIRED_FOR_EXTERNAL_ACTION``: só ações externas exigem aprovação.
    - ``AUTO_APPROVED``: execução automática permitida.
    - ``MANUAL_ONLY``: toda ação exige aprovação de uma pessoa identificada
      (``ApprovalRecord.approved_by``); fluxos automáticos nunca aprovam.
    """

    mode: ApprovalMode = ApprovalMode.ALWAYS_REQUIRED

    def requires_approval(self, *, external: bool = True) -> bool:
        """Indica se a ação descrita exige aprovação humana."""
        if self.mode == ApprovalMode.AUTO_APPROVED:
            return False
        if self.mode == ApprovalMode.REQUIRED_FOR_EXTERNAL_ACTION:
            return external
        return True

    def permits_automatic_execution(self) -> bool:
        """Indica se execução automática (sem interação) é permitida pela política."""
        return self.mode == ApprovalMode.AUTO_APPROVED

    def is_satisfied_by(self, record: ApprovalRecord) -> bool:
        """Indica se o registro de aprovação atende a esta política."""
        if not record.approved:
            return False
        return self.mode != ApprovalMode.MANUAL_ONLY or bool(record.approved_by)

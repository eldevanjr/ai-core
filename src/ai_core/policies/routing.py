"""Política de roteamento: decide para qual destino externo uma tarefa vai."""

from __future__ import annotations

from pydantic import Field

from ai_core.models.base import CoreModel
from ai_core.models.destination import DestinationRef
from ai_core.models.source import SourceKind
from ai_core.models.task import Task, TaskPriority, TaskType

NO_DESTINATION_MESSAGE = "Não existe destino configurado para este tipo de tarefa."


class RoutingRule(CoreModel):
    """Regra do workspace que associa condições de uma tarefa a um destino.

    Cada condição preenchida deve casar com a tarefa. Quanto mais condições,
    mais específica a regra (e maior a prioridade na resolução).
    """

    name: str | None = None
    task_type: TaskType | None = None
    priority: TaskPriority | None = None
    labels: list[str] = Field(default_factory=list)
    source_kind: SourceKind | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    destination: DestinationRef

    def matches(self, task: Task) -> bool:
        """Indica se a tarefa satisfaz todas as condições desta regra."""
        if not self.destination.enabled:
            return False
        if self.task_type is not None and self.task_type != task.type:
            return False
        if self.priority is not None and self.priority != task.priority:
            return False
        if any(label not in task.labels for label in self.labels):
            return False
        if self.source_kind is not None and not any(
            source.type == self.source_kind for source in task.sources
        ):
            return False
        return all(task.metadata.get(key) == value for key, value in self.metadata.items())

    def specificity(self) -> int:
        """Número de condições ativas; regras mais específicas vencem."""
        return (
            (1 if self.task_type is not None else 0)
            + (1 if self.priority is not None else 0)
            + len(self.labels)
            + (1 if self.source_kind is not None else 0)
            + len(self.metadata)
        )


class RoutingPolicy(CoreModel):
    """Conjunto de regras de roteamento e destino padrão de um workspace.

    Hierarquia de resolução (da mais para a menos prioritária):

    1. destino escolhido manualmente na aprovação;
    2. regra mais específica que casar;
    3. destino padrão do workspace;
    4. nenhum destino (não executar ação externa).

    Em empate de especificidade vence a regra declarada primeiro em ``rules``.
    """

    rules: list[RoutingRule] = Field(default_factory=list)
    default_destination: DestinationRef | None = None

    def resolve(self, task: Task, *, manual: DestinationRef | None = None) -> DestinationRef | None:
        """Resolve o destino de uma tarefa seguindo a hierarquia de prioridade."""
        if manual is not None and manual.enabled:
            return manual
        matching = sorted(
            (rule for rule in self.rules if rule.matches(task)),
            key=lambda rule: rule.specificity(),
            reverse=True,
        )
        if matching:
            return matching[0].destination
        if self.default_destination is not None and self.default_destination.enabled:
            return self.default_destination
        return None

# ai-core

Biblioteca comum de contratos e abstrações de IA do ecossistema composto pelo
**Whisper Transcriber** e pelo **AI Work Analyzer**.

O `ai-core` não é uma aplicação: é uma biblioteca Python compartilhada. Ele não
conhece Whisper, Jira, GitHub, ClickUp, Linear, Electron, React, UI ou captura de
áudio. Ele fornece os modelos, interfaces e políticas genéricas que os produtos
implementam.

## Princípio

```
        ai-core
           |
   +-------+-------+
   |               |
   v               v
Whisper       AI Work
Transcriber   Analyzer
```

- O `ai-core` não depende dos produtos.
- Whisper e AI Work Analyzer podem depender do `ai-core`.
- A comunicação entre os produtos acontece via MCP, nunca por banco compartilhado.

## Escopo

| Área | Conteúdo |
| --- | --- |
| Modelos | `Workspace`, `Evidence`, `Source`, `KnowledgeDocument`, `KnowledgeChunk`, `Task`, `TaskProposal`, `TaskExplanation`, `AuditEvent`, `DestinationRef` |
| LLM | `LLMProvider`, `LLMRequest`, `LLMResponse`, `ModelRouter` (`AUTO`, `LOCAL_ONLY`, `CLOUD_ONLY`, `MANUAL`) |
| Embeddings | `EmbeddingProvider`, `Embedding` (sem dimensão fixa) |
| RAG | `KnowledgeRepository`, `VectorStore`, `KnowledgeSearch`, `SearchRequest`, `SearchResult`, `TextChunker` |
| Políticas | `PrivacyPolicy`, `ApprovalPolicy`, `RoutingPolicy` |
| MCP | contratos de `search_knowledge`, `get_knowledge`, `search_tasks`, `generate_task`, `list_task_proposals`, `approve_task` |

Regras centrais:

- Toda busca e todo acesso por id ao repositório de conhecimento respeitam
  `workspace_id`; recurso de outro workspace se comporta como inexistente.
- `LOCAL_ONLY` nunca envia conteúdo para provider cloud — nem para LLM, nem para
  embeddings (a `RepositoryKnowledgeSearch` valida a política antes de cada busca).
- `LOCAL_FIRST` (padrão) só usa cloud no modo AUTO com confirmação explícita
  (`RoutingCriteria(allow_cloud=True)`).
- Aprovação humana é `ALWAYS_REQUIRED` por padrão; `TaskProposal` só muda de
  estado via `transition_to`, e só chega a `CREATED` passando por `APPROVED`.
- Nenhuma dependência obrigatória de PostgreSQL (as interfaces permitem
  implementações externas; há implementações em memória para referência/testes).

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) para gerenciamento
- Pydantic v2
- pytest + coverage
- ruff
- mypy (strict)

## Uso

```bash
uv sync
uv run pytest
uv run ruff check
uv run mypy
```

```python
from ai_core import (
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRecord,
    InMemoryKnowledgeRepository,
    InMemoryVectorStore,
    ModelRouter,
    PrivacyPolicy,
    ProposalStatus,
    Task,
    TaskProposal,
    TaskType,
    Workspace,
)

workspace = Workspace(name="Mobile App", slug="mobile-app")
task = Task(workspace_id=workspace.id, type=TaskType.BUG, title="Falha na sincronização offline")

router = ModelRouter(privacy=PrivacyPolicy())
assert ApprovalPolicy().requires_approval(external=True)

proposal = TaskProposal(workspace_id=workspace.id, task=task)
proposal = proposal.transition_to(ProposalStatus.NEEDS_REVIEW)
proposal = proposal.transition_to(
    ProposalStatus.APPROVED,
    approval=ApprovalRecord(mode=ApprovalMode.ALWAYS_REQUIRED, approved=True, approved_by="ana"),
)
```

### JSON Schema (fonte única para `ai-core-types`)

```bash
uv run python -m ai_core.schemas schemas/
```

Gera um `<Modelo>.schema.json` por entidade, política e contrato MCP, para gerar
os tipos TypeScript sem duplicar os modelos manualmente.

## Estrutura

```
src/ai_core/
  models/     # entidades de domínio
  llm/        # provider, request/response e model router
  embeddings/ # provider e modelos de embedding
  rag/        # repository, vector store, busca e chunking
  policies/   # privacidade, aprovação e roteamento
  mcp/        # contratos das ferramentas MCP
  schemas.py  # exportação de JSON Schema
tests/
```

## Licença

Apache-2.0. Veja [LICENSE](LICENSE).

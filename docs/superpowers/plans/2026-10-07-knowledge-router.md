# Knowledge Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un flag `is_knowledge_router` que convierte un agente en router de especialistas (sub-agentes con su propio silo LightRAG). Debe delegar en uno (respuesta tal cual) o en varios (síntesis), y conservar citas que abren el PDF correcto y subgrafos agrupados por silo, también al recargar la conversación.

**Architecture:** En modo router, cada `IACTTool` devuelve `(texto, artefacto)` con los chunks de su sub-agente, etiquetados con `silo_id`/`silo_name` y renumerados sobre un contador de citas del turno compartido (celda + `asyncio.Lock`, el patrón existente). Además lleva `return_direct=True`. Una tool nueva, `consultar_varios`, ejecuta varios especialistas en paralelo y reclama las citas en el orden de la lista. El resto del pipeline (merge en vivo, historial) no cambia porque el artefacto tiene la misma forma que el de una llamada LightRAG directa. Se añaden dos arreglos para los turnos que terminan en `ToolMessage` (streaming e historial), y el frontend resuelve el silo por chunk y parte el bubble por silo.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Alembic, LangChain 1.2.10 / LangGraph, pytest + pytest-asyncio, React 18 + TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-07-knowledge-router-design.md`

## Global Constraints

- **Trabajar en un worktree** creado desde el `HEAD` de `lightrag`, rama `feature/knowledge-router`. El árbol principal tiene trabajo sin commitear de System One (`backend/tools/agentTools.py` importa el `backend/tools/system_one.py` no trackeado) que no debe entrar en estos commits.
- `/ponytail-review` sobre el diff de cada tarea **antes** de su commit, aplicando o declinando (con motivo) lo que encuentre (CLAUDE.md).
- `git add` solo de los ficheros listados en cada tarea.
- Fuera del modo router, `IACTTool` devuelve exactamente lo mismo que hoy (texto plano). Los orquestadores clásicos no cambian.
- Marcadores de cita: `[N](cite://N)`, con N resolviendo a `chunks[N-1]` del payload fusionado del turno.
- El router decide con el `description` de cada sub-agente; no se usa `AgentTool.description`.
- Con el flag activo y al menos un especialista, el silo propio del agente no se usa.
- Alembic toma la BD de las variables `DATABASE_*`, **no** de `SQLALCHEMY_DATABASE_URI`. Para la prueba de migración, apuntar `DATABASE_NAME` a una BD desechable.
- Migración nueva: `revision = 'kr001'`, `down_revision = 'srclabel001'`.
- Columna: `Agent.is_knowledge_router BOOLEAN NOT NULL DEFAULT false`.
- Nombre de la tool multi-especialista: `consultar_varios(especialistas: list[str], pregunta: str)`.

## Review Focus

1. **Un especialista falla dentro de `consultar_varios`.** Se espera que la respuesta del otro llegue igual y la numeración siga siendo correcta. Test en la Task 5.
2. **Nombres desconocidos o repetidos en `especialistas`.** Se espera que se ignoren o se deduplican, sin ejecutar dos veces al mismo especialista. Test en la Task 5.
3. **Especialista sin LightRAG (sin artefacto) junto a uno con LightRAG.** Se espera texto de ambos y citas solo del segundo, numeradas desde el offset correcto. Test en la Task 5.
4. **Recarga de una conversación con dos turnos pass-through seguidos.** Cada respuesta debe salir en su turno, con su propio grafo y sin arrastrar el del turno anterior. Test en la Task 6.
5. **Misma entidad ("Quemador") en dos silos.** Se espera que siga habiendo dos nodos tras el merge (Task 3, y comprobado vía `consultar_varios` en la Task 5), y que en el frontend cada uno salga en el bubble de su silo (verificación manual en la Task 8).

---

### Task 1: Validar `app_id` al asociar agentes como tools

**Files:**
- Modify: `backend/repositories/agent_repository.py:196-202`
- Modify: `backend/services/agent_service.py:589`
- Test: `tests/integration/repositories/test_agent_repository_tool_ids.py` (nuevo)

**Interfaces:**
- Produces: `AgentRepository.get_valid_tool_ids(db: Session, tool_ids: List[int], app_id: int) -> List[int]`

- [ ] **Step 0: Crear el worktree** (solo la primera vez, antes de cualquier tarea)

```bash
cd /home/miniguezdeonzono/LKS/ai-core-tools
git worktree add -b feature/knowledge-router ../ai-core-tools-knowledge-router lightrag
cp docs/superpowers/specs/2026-10-07-knowledge-router-design.md ../ai-core-tools-knowledge-router/docs/superpowers/specs/
cp docs/superpowers/plans/2026-10-07-knowledge-router.md ../ai-core-tools-knowledge-router/docs/superpowers/plans/
cp docs/ai/rfc-multilingual-silo-routing.md ../ai-core-tools-knowledge-router/docs/ai/
cd ../ai-core-tools-knowledge-router
git add docs/superpowers/specs/2026-10-07-knowledge-router-design.md docs/superpowers/plans/2026-10-07-knowledge-router.md docs/ai/rfc-multilingual-silo-routing.md
git commit -m "docs: knowledge router design, plan and multilingual routing RFC"
docker compose -f docker/docker-compose.yaml --profile test up -d db_test
```

Desde aquí, todos los comandos se ejecutan en `../ai-core-tools-knowledge-router`.

- [ ] **Step 1: Escribir el test que falla**

```python
"""get_valid_tool_ids must only accept agents of the caller's own App.

The UI only lists same-app tool agents, but the API trusted whatever ids it
got: a crafted request could attach another App's agent as a tool.
"""
from models.agent import Agent
from models.app import App
from repositories.agent_repository import AgentRepository


def _tool_agent(db, app_id, name):
    agent = Agent(name=name, description="", system_prompt="", app_id=app_id, is_tool=True)
    db.add(agent)
    db.flush()
    return agent


def test_rejects_tool_agents_from_another_app(db, fake_app, fake_user):
    other_app = App(
        name="Other Workspace", slug="other-workspace-tool-ids", owner_id=fake_user.user_id,
        agent_rate_limit=0, max_file_size_mb=10,
    )
    db.add(other_app)
    db.flush()
    own = _tool_agent(db, fake_app.app_id, "own")
    foreign = _tool_agent(db, other_app.app_id, "foreign")

    valid = AgentRepository.get_valid_tool_ids(db, [own.agent_id, foreign.agent_id], fake_app.app_id)

    assert valid == [own.agent_id]
```

- [ ] **Step 2: Ejecutarlo y ver que falla**

Run: `poetry run pytest tests/integration/repositories/test_agent_repository_tool_ids.py -v`
Expected: FAIL con `TypeError: get_valid_tool_ids() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Implementar**

`backend/repositories/agent_repository.py`:

```python
    @staticmethod
    def get_valid_tool_ids(db: Session, tool_ids: List[int], app_id: int) -> List[int]:
        """Get valid tool IDs: agents marked as tools that belong to *app_id*."""
        tools_query = db.query(Agent.agent_id).filter(
            Agent.agent_id.in_(tool_ids),
            Agent.is_tool == True,
            Agent.app_id == app_id,
        )
        return [id for (id,) in tools_query]
```

(Conserva el decorador que tenga hoy el método.)

`backend/services/agent_service.py`, en `update_agent_tools`:

```python
        valid_tool_ids = set(AgentRepository.get_valid_tool_ids(db, [int(id) for id in tool_ids if id], agent.app_id))
```

- [ ] **Step 4: Ejecutar los tests**

Run: `poetry run pytest tests/integration/repositories/test_agent_repository_tool_ids.py tests/unit/services/test_agent_service.py -v`
Expected: PASS. Los tests de `test_agent_service.py` mockean `get_valid_tool_ids` y aceptan el argumento extra.

- [ ] **Step 5: Detectar asociaciones cross-app ya existentes** (solo informar, no borrar)

```bash
docker compose -f docker/docker-compose.yaml exec db psql -U "$DATABASE_USER" -d "$DATABASE_NAME" -c \
'SELECT at.agent_id, a.app_id AS agent_app, at.tool_id, t.app_id AS tool_app
 FROM agent_tools at JOIN "Agent" a ON a.agent_id = at.agent_id JOIN "Agent" t ON t.agent_id = at.tool_id
 WHERE a.app_id <> t.app_id;'
```

Anota el resultado (lo esperado es 0 filas) para el informe final.

- [ ] **Step 6: `/ponytail-review` del diff y commit**

```bash
git add backend/repositories/agent_repository.py backend/services/agent_service.py tests/integration/repositories/test_agent_repository_tool_ids.py
git commit -m "fix(security): only attach tool agents from the agent's own app"
```

---

### Task 2: Flag `is_knowledge_router` (BD → API)

**Files:**
- Create: `alembic/versions/kr001_agent_is_knowledge_router.py`
- Modify: `backend/models/agent.py` (junto a `enable_code_interpreter`, ~línea 81)
- Modify: `backend/schemas/agent_schemas.py` (`AgentDetailSchema` ~línea 161, `CreateUpdateAgentSchema` ~línea 232)
- Modify: `backend/services/agent_service.py` (detalle ~línea 269, actualización ~línea 494)
- Modify: `backend/routers/internal/agents.py` (`agent_dict` ~línea 381)
- Test: `tests/integration/routers/internal/test_agents.py`

**Interfaces:**
- Produces: `Agent.is_knowledge_router: bool`, y el campo `is_knowledge_router` en el JSON de detalle y en el payload de create/update.

- [ ] **Step 1: Escribir el test que falla**

Añade a `tests/integration/routers/internal/test_agents.py`, dentro de la clase de tests de creación (junto a `test_create_agent_as_tool`):

```python
    def test_create_agent_as_knowledge_router(
        self, client, fake_app, fake_ai_service, owner_headers, db
    ):
        """The knowledge-router flag round-trips through create and detail."""
        db.flush()
        payload = agent_payload(name="Router", service_id=fake_ai_service.service_id)
        payload["is_knowledge_router"] = True
        response = client.post(
            f"/internal/apps/{fake_app.app_id}/agents/0",
            json=payload,
            headers=owner_headers,
        )
        assert response.status_code in (200, 201)
        assert response.json()["is_knowledge_router"] is True

        agent_id = response.json()["agent_id"]
        detail = client.get(f"/internal/apps/{fake_app.app_id}/agents/{agent_id}", headers=owner_headers)
        assert detail.json()["is_knowledge_router"] is True
```

- [ ] **Step 2: Ejecutarlo y ver que falla**

Run: `poetry run pytest tests/integration/routers/internal/test_agents.py -k knowledge_router -v`
Expected: FAIL con `KeyError: 'is_knowledge_router'`

- [ ] **Step 3: Implementar**

`alembic/versions/kr001_agent_is_knowledge_router.py`:

```python
"""add Agent.is_knowledge_router

Opt-in mode where the agent routes each question to its agent-tools
(specialists) and passes a single specialist's answer through unchanged.
See docs/superpowers/specs/2026-10-07-knowledge-router-design.md.

Revision ID: kr001
Revises: srclabel001
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa

revision = 'kr001'
down_revision = 'srclabel001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Agent',
        sa.Column('is_knowledge_router', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade():
    op.drop_column('Agent', 'is_knowledge_router')
```

`backend/models/agent.py`, debajo de `enable_code_interpreter`:

```python
    # Knowledge router: agent-tools become pass-through specialists (see tools/knowledge_router.py)
    is_knowledge_router = Column(Boolean, default=False, nullable=False, server_default='false')
```

`backend/schemas/agent_schemas.py`. En `AgentDetailSchema` y en `CreateUpdateAgentSchema`, debajo de `enable_code_interpreter: bool = False`:

```python
    is_knowledge_router: bool = False
```

`backend/services/agent_service.py`. En la construcción de `AgentDetailSchema`, debajo de `enable_code_interpreter=...`:

```python
            is_knowledge_router=getattr(agent, 'is_knowledge_router', False) or False,
```

En la actualización, debajo de `agent.enable_code_interpreter = bool(enable_ci_value)`:

```python
        agent.is_knowledge_router = bool(data.get('is_knowledge_router', False))
```

`backend/routers/internal/agents.py`, en `agent_dict`, debajo de `'enable_code_interpreter': ...`:

```python
        'is_knowledge_router': agent_data.is_knowledge_router,
```

- [ ] **Step 4: Ejecutar los tests**

Run: `poetry run pytest tests/integration/routers/internal/test_agents.py -v`
Expected: PASS (todo el fichero)

- [ ] **Step 5: Probar la migración con upgrade, downgrade y upgrade en una BD desechable**

```bash
docker compose -f docker/docker-compose.yaml exec db_test createdb -U test_user kr_mig
export DATABASE_HOST=localhost DATABASE_PORT=5433 DATABASE_USER=test_user DATABASE_PASSWORD=test_pass DATABASE_NAME=kr_mig
poetry run alembic upgrade head && poetry run alembic downgrade -1 && poetry run alembic upgrade head
poetry run alembic current
docker compose -f docker/docker-compose.yaml exec db_test dropdb -U test_user kr_mig
```

Expected: `alembic current` muestra `kr001 (head)`. Si la cadena completa desde cero falla en una migración anterior sin relación con esta, anótalo y repite la prueba sobre un `pg_dump` de la BD de dev restaurado en `kr_mig`.

- [ ] **Step 6: `/ponytail-review` del diff y commit**

```bash
git add alembic/versions/kr001_agent_is_knowledge_router.py backend/models/agent.py backend/schemas/agent_schemas.py backend/services/agent_service.py backend/routers/internal/agents.py tests/integration/routers/internal/test_agents.py
git commit -m "feat(agents): add is_knowledge_router flag"
```

---

### Task 3: Helpers puros del router y dedup de entidades por silo

**Files:**
- Create: `backend/tools/knowledge_router.py`
- Modify: `backend/tools/streaming_utils.py:291-292` (`merge_lightrag_graph`)
- Test: `tests/unit/tools/test_knowledge_router.py` (nuevo)
- Test: `tests/unit/tools/test_streaming_utils.py`

**Interfaces:**
- Produces (en `tools/knowledge_router.py`):
  - `rebase_citations(text: str, delta: int) -> str`
  - `tag_graph_with_silo(graph: dict, silo_id: Optional[int], silo_name: Optional[str]) -> dict`
  - `graph_artifact(graph: Optional[dict]) -> Optional[List[Document]]`
  - `chunk_count(graph: Optional[dict]) -> int`
  - `KNOWLEDGE_ROUTER_INSTRUCTION: str`, `KNOWLEDGE_ROUTER_TOOL_NAME = "consultar_varios"`
- Produces: `merge_lightrag_graph` deduplica entidades por `(silo_id, id)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/unit/tools/test_knowledge_router.py`:

```python
"""Knowledge router: specialists' citations renumbered onto one turn-wide
counter so cite://N keeps resolving to chunks[N-1] of the merged payload.
See docs/superpowers/specs/2026-10-07-knowledge-router-design.md."""
from tools.knowledge_router import chunk_count, graph_artifact, rebase_citations, tag_graph_with_silo


def test_rebase_citations_shifts_number_and_link():
    assert rebase_citations("a [1](cite://1) b [2](cite://2)", 3) == "a [4](cite://4) b [5](cite://5)"


def test_rebase_citations_normalizes_cjk_brackets():
    assert rebase_citations("a 【2】(cite://2)", 1) == "a [3](cite://3)"


def test_rebase_citations_zero_delta_is_identity():
    text = "a [1](cite://1) and [7] plain"
    assert rebase_citations(text, 0) == text


def test_rebase_citations_leaves_bare_numbers_alone():
    assert rebase_citations("step [1] then [1](cite://1)", 2) == "step [1] then [3](cite://3)"


def test_tag_graph_with_silo_stamps_every_element():
    graph = {"data": {"entities": [{"id": "E"}], "relationships": [{"id": "r"}], "chunks": [{"id": "c"}]}}
    tagged = tag_graph_with_silo(graph, 7, "Manuales ES")
    for key in ("entities", "relationships", "chunks"):
        assert tagged["data"][key][0]["silo_id"] == 7
        assert tagged["data"][key][0]["silo_name"] == "Manuales ES"
    assert "silo_id" not in graph["data"]["chunks"][0]  # input not mutated


def test_graph_artifact_shape_and_none():
    assert graph_artifact(None) is None
    graph = {"data": {"chunks": [{"id": "c"}]}}
    assert graph_artifact(graph)[0].metadata["lightrag_raw_data"] == graph


def test_chunk_count():
    assert chunk_count(None) == 0
    assert chunk_count({"data": {"chunks": [{"id": "a"}, {"id": "b"}]}}) == 2
```

Añade a `tests/unit/tools/test_streaming_utils.py`:

```python
def test_merge_keeps_same_entity_name_from_two_silos_apart():
    from tools.streaming_utils import merge_lightrag_graph

    es = {"data": {"entities": [{"id": "Domusa", "silo_id": 7}], "relationships": [], "chunks": []}}
    en = {"data": {"entities": [{"id": "Domusa", "silo_id": 8}], "relationships": [], "chunks": []}}

    merged = merge_lightrag_graph(es, en)

    assert [(e["id"], e["silo_id"]) for e in merged["data"]["entities"]] == [("Domusa", 7), ("Domusa", 8)]


def test_merge_still_dedups_untagged_entities_by_id():
    from tools.streaming_utils import merge_lightrag_graph

    a = {"data": {"entities": [{"id": "X", "v": 1}], "relationships": [], "chunks": []}}
    b = {"data": {"entities": [{"id": "X", "v": 2}], "relationships": [], "chunks": []}}

    assert merge_lightrag_graph(a, b)["data"]["entities"] == [{"id": "X", "v": 2}]
```

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `poetry run pytest tests/unit/tools/test_knowledge_router.py tests/unit/tools/test_streaming_utils.py -v`
Expected: FAIL. `ModuleNotFoundError: No module named 'tools.knowledge_router'`, y el test de Domusa devuelve un solo nodo.

- [ ] **Step 3: Implementar**

`backend/tools/knowledge_router.py`:

```python
"""Knowledge router: an agent that delegates each question to specialist
sub-agents (agent-as-tool), each answering from its own LightRAG silo.

The turn's citations must keep resolving as cite://N -> chunks[N-1] of the
merged LightRAG payload, so every specialist's local [n](cite://n) markers are
renumbered onto one turn-wide counter (see IACTTool.claim_citations).
Design: docs/superpowers/specs/2026-10-07-knowledge-router-design.md
"""
import re
from typing import List, Optional

from langchain_core.documents import Document

KNOWLEDGE_ROUTER_TOOL_NAME = "consultar_varios"

KNOWLEDGE_ROUTER_INSTRUCTION = """<knowledge_router>
You answer by delegating to specialist agents; you have no knowledge base of your own.
Each specialist tool's description says which knowledge it covers.
- If the question is only about ONE specialist's knowledge, call that specialist's tool once with the user's question. Its answer is shown to the user as-is.
- Otherwise (a general question, or one that spans several specialists), call `consultar_varios` ONCE with the names of every relevant specialist (include all of them when unsure), then write the final answer from their replies.
- Never call two specialist tools one after another for the same question; use `consultar_varios` instead.
- Keep every [N](cite://N) citation exactly as the specialists wrote it, next to the claim it supports. Never invent, renumber or drop citations.
</knowledge_router>"""

# [n](cite://n), also with the full-width brackets some models emit (【n】).
_CITE_RE = re.compile(r"[\[【](\d+)[\]】]\(cite://(\d+)\)")


def rebase_citations(text: str, delta: int) -> str:
    """Shift every cite marker by *delta*; the cite:// number is authoritative."""
    if delta == 0:
        return text

    def _shift(m: re.Match) -> str:
        n = int(m.group(2)) + delta
        return f"[{n}](cite://{n})"

    return _CITE_RE.sub(_shift, text)


def tag_graph_with_silo(graph: dict, silo_id: Optional[int], silo_name: Optional[str]) -> dict:
    """Stamp silo_id/silo_name on every entity, relationship and chunk (new dict)."""
    data = graph.get("data") or {}
    tag = {"silo_id": silo_id, "silo_name": silo_name}
    return {
        **graph,
        "data": {
            **data,
            **{k: [{**item, **tag} for item in data.get(k) or []] for k in ("entities", "relationships", "chunks")},
        },
    }


def graph_artifact(graph: Optional[dict]) -> Optional[List[Document]]:
    """Wrap a payload like a direct LightRAG tool's artifact, so the streaming
    accumulator and history reconstruction pick it up unchanged."""
    if not graph:
        return None
    return [Document(page_content="", metadata={"lightrag_raw_data": graph})]


def chunk_count(graph: Optional[dict]) -> int:
    return len(((graph or {}).get("data") or {}).get("chunks") or [])
```

`backend/tools/streaming_utils.py`, en `merge_lightrag_graph`. Sustituye las dos líneas de `entities_by_id` y amplía el docstring:

```python
    Entities are deduped by (silo_id, id): id is the entity name, and silo_id
    (set only on knowledge-router payloads, None otherwise) keeps the same name
    from two silos as two nodes instead of collapsing them.
```

```python
    entities_by_id = {(e.get("silo_id"), e.get("id")): e for e in acc_data.get("entities", [])}
    entities_by_id.update({(e.get("silo_id"), e.get("id")): e for e in new_data.get("entities", [])})
```

- [ ] **Step 4: Ejecutar los tests**

Run: `poetry run pytest tests/unit/tools/test_knowledge_router.py tests/unit/tools/test_streaming_utils.py tests/unit/services/test_agent_cache_service_history.py -v`
Expected: PASS

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/knowledge_router.py backend/tools/streaming_utils.py tests/unit/tools/test_knowledge_router.py tests/unit/tools/test_streaming_utils.py
git commit -m "feat(knowledge-router): citation rebase/silo tagging helpers, dedup entities per silo"
```

---

### Task 4: `IACTTool` en modo router

**Files:**
- Modify: `backend/tools/agentTools.py`: clase `IACTTool` (~1478-1826)
- Test: `tests/unit/tools/test_knowledge_router.py`

**Interfaces:**
- Consumes: `rebase_citations`, `tag_graph_with_silo`, `graph_artifact`, `chunk_count` (Task 3).
- Produces (en `IACTTool`):
  - campos `silo_name: Optional[str]`, `citation_offset: Optional[List[int]]`, `citation_lock: Any`, `chunks_seen: int`
  - `enable_router_mode(self, offset: List[int], lock: asyncio.Lock) -> None`: activa `return_direct`, `response_format="content_and_artifact"`, el contador y el lock
  - `async run_subagent(self, query: str) -> Tuple[str, Optional[dict]]`: texto final con numeración local del sub-agente, y payload LightRAG fusionado de esta ejecución
  - `claim_citations(self, text: str, graph: Optional[dict]) -> Tuple[str, Optional[dict]]`: renumera sobre el contador del turno y etiqueta por silo; quien la llama debe tener el lock
  - `_arun` devuelve `str` en modo clásico y `(str, Optional[List[Document]])` en modo router

- [ ] **Step 1: Escribir los tests que fallan**

Añade a `tests/unit/tools/test_knowledge_router.py`:

```python
import asyncio
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from models.agent import Agent
from tools import agentTools


class _CitingReactAgent:
    """Fake sub-agent: one LightRAG retrieval of *chunk_ids*, then an answer citing
    each chunk with the sub-agent's OWN counter, which keeps running across calls
    in the same turn (its tools are built once per turn, like the real ones)."""

    def __init__(self, chunk_ids, delay=0.0):
        self.chunk_ids = list(chunk_ids)
        self.delay = delay
        self.counter = 0

    def _turn(self):
        raw = {"data": {
            "entities": [{"id": "Quemador", "source_id": self.chunk_ids[0]}],
            "relationships": [],
            "chunks": [{"id": c, "content": c} for c in self.chunk_ids],
        }}
        tool_msg = ToolMessage(
            content="context", name="retrieve", tool_call_id="call-r",
            artifact=[Document(page_content="context", metadata={"lightrag_raw_data": raw})],
        )
        first = self.counter + 1
        self.counter += len(self.chunk_ids)
        cites = "".join(f"[{n}](cite://{n})" for n in range(first, self.counter + 1))
        return tool_msg, AIMessage(content=f"Answer {cites}")

    async def ainvoke(self, payload):
        await asyncio.sleep(self.delay)
        tool_msg, final = self._turn()
        return {"messages": [*payload["messages"], tool_msg, final]}

    async def astream(self, _payload, stream_mode=None):
        await asyncio.sleep(self.delay)
        tool_msg, final = self._turn()
        yield ("updates", {"tools": {"messages": [tool_msg]}})
        yield ("updates", {"model": {"messages": [final]}})


def _router_tool(name, react_agent, offset, lock, silo_id=7, silo_name="Manuales ES"):
    agent = Agent(name=name, description=f"{name} manuals", system_prompt="")
    agent.agent_id = silo_id
    agent.silo_id = silo_id
    with patch("tools.agentTools.get_llm", return_value=object()):
        tool = agentTools.IACTTool(agent)
    tool.silo_name = silo_name
    tool.react_agent = react_agent
    tool.enable_router_mode(offset, lock)
    return tool


@pytest.fixture
def no_writer():
    with patch.object(agentTools.IACTTool, "_get_stream_writer_or_none", return_value=None):
        yield


def _chunks(artifact):
    return artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]


@pytest.mark.asyncio
async def test_router_mode_returns_tagged_artifact(no_writer):
    offset, lock = [0], asyncio.Lock()
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, lock)

    text, artifact = await tool._arun("q")

    assert text == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(artifact)] == ["a", "b"]
    assert {(c["silo_id"], c["silo_name"]) for c in _chunks(artifact)} == {(7, "Manuales ES")}
    assert offset == [2]
    assert tool.return_direct is True


@pytest.mark.asyncio
async def test_router_mode_streaming_keeps_graph_instead_of_forwarding_it():
    emitted = []
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), [0], asyncio.Lock())

    with patch("langgraph.config.get_stream_writer", return_value=emitted.append):
        text, artifact = await tool._arun("q")

    assert text == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(artifact)] == ["a", "b"]
    assert all(e.get("type") != "_lightrag_graph" for e in emitted)


@pytest.mark.asyncio
async def test_router_mode_rebases_onto_turn_offset(no_writer):
    offset = [3]
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, asyncio.Lock())

    text, _ = await tool._arun("q")

    assert text == "Answer [4](cite://4)[5](cite://5)"
    assert offset == [5]


@pytest.mark.asyncio
async def test_same_specialist_twice_in_a_turn_keeps_numbering(no_writer):
    offset, lock = [0], asyncio.Lock()
    es = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, lock, silo_id=7)
    en = _router_tool("EN", _CitingReactAgent(["c"]), offset, lock, silo_id=8, silo_name="Manuals EN")

    t1, _ = await es._arun("q1")  # ES local 1,2 -> 1,2
    t2, _ = await en._arun("q2")  # EN local 1   -> 3
    t3, _ = await es._arun("q3")  # ES local 3,4 (its counter kept running) -> 4,5

    assert (t1, t2, t3) == (
        "Answer [1](cite://1)[2](cite://2)",
        "Answer [3](cite://3)",
        "Answer [4](cite://4)[5](cite://5)",
    )
    assert offset == [5]


@pytest.mark.asyncio
async def test_classic_mode_still_returns_plain_text(no_writer):
    agent = Agent(name="Classic", description="", system_prompt="")
    with patch("tools.agentTools.get_llm", return_value=object()):
        tool = agentTools.IACTTool(agent)
    tool.react_agent = _CitingReactAgent(["a"])

    assert await tool._arun("q") == "Answer [1](cite://1)"
    assert tool.return_direct is False
```

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `poetry run pytest tests/unit/tools/test_knowledge_router.py -v`
Expected: FAIL con `AttributeError: 'IACTTool' object has no attribute 'enable_router_mode'` (los tests de la Task 3 siguen en PASS)

- [ ] **Step 3: Implementar**

En `backend/tools/agentTools.py`, import a nivel de módulo (junto a los demás `from tools...`):

```python
from tools.knowledge_router import chunk_count, graph_artifact, rebase_citations, tag_graph_with_silo
```

Campos nuevos en `class IACTTool`, debajo de `attached_files: Optional[List[Dict]] = None`:

```python
    silo_name: Optional[str] = None
    # Knowledge-router mode (enable_router_mode): the parent turn's citation
    # counter + lock, shared by all its router tools, and how many chunks this
    # tool has already handed back this turn.
    citation_offset: Optional[List[int]] = None
    citation_lock: Any = None
    chunks_seen: int = 0
```

En `IACTTool.create`, justo después de `instance = cls(...)`:

```python
        if agent.silo_id is not None:
            # Off the event loop: lazy-loads the Silo row.
            instance.silo_name = await asyncio.to_thread(lambda: agent.silo.name if agent.silo else None)
```

Sustituye el `_arun` actual (líneas ~1782-1826) por estos cuatro métodos:

```python
    def enable_router_mode(self, offset: List[int], lock: asyncio.Lock) -> None:
        """Knowledge-router specialist: its answer goes straight to the user
        (return_direct) and it hands back its sub-agent's LightRAG chunks as an
        artifact, renumbered onto the parent turn's citation counter."""
        self.citation_offset = offset
        self.citation_lock = lock
        self.response_format = "content_and_artifact"
        self.return_direct = True

    async def run_subagent(self, query: str) -> Tuple[str, Optional[dict]]:
        """Run the sub-agent once: (final answer, merged LightRAG payload of this run).

        Cite markers in the answer use the sub-agent's own numbering — see
        claim_citations."""
        from tools.streaming_utils import (
            extract_lightrag_graph_from_artifact,
            map_stream_event,
            merge_lightrag_graph,
        )

        messages = [HumanMessage(content=self._format_tool_query(query))]
        stream_writer = self._get_stream_writer_or_none()
        graph: Optional[dict] = None

        if stream_writer is None or not hasattr(self.react_agent, "astream"):
            result = await self.react_agent.ainvoke({"messages": messages})
            for msg in (result.get("messages") or []) if isinstance(result, dict) else []:
                raw = extract_lightrag_graph_from_artifact(getattr(msg, "artifact", None))
                if raw:
                    graph = merge_lightrag_graph(graph, raw)
            return self._extract_last_message_content(result), graph

        latest_state: Any = None
        async for mode, chunk in self.react_agent.astream(
            {"messages": messages},
            stream_mode=["updates", "custom"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for state_delta in chunk.values():
                    if isinstance(state_delta, dict) and "messages" in state_delta:
                        latest_state = state_delta

            for event in map_stream_event(mode, chunk) or []:
                if event.get("type") == "_lightrag_graph":
                    graph = merge_lightrag_graph(graph, event["data"])
                else:
                    self._emit_subagent_stream_event(stream_writer, event)

        text = self._extract_last_message_content(latest_state) if latest_state is not None else ""
        return text, graph

    def claim_citations(self, text: str, graph: Optional[dict]) -> Tuple[str, Optional[dict]]:
        """Renumber this run's cite markers onto the parent turn's counter and tag
        the payload with its silo. Caller must hold citation_lock.

        The sub-agent's own counter keeps running across calls in the same turn
        (its tools are built once per turn), so this run starts at chunks_seen + 1,
        not necessarily at 1: n -> n + offset - chunks_seen.
        ponytail: a sub-agent run that raises leaves chunks_seen behind its real
        counter; only matters if the same specialist is called again that turn.
        """
        count = chunk_count(graph)
        delta = self.citation_offset[0] - self.chunks_seen
        self.chunks_seen += count
        self.citation_offset[0] += count
        if graph:
            graph = tag_graph_with_silo(graph, self.agent.silo_id, self.silo_name)
        return rebase_citations(text, delta), graph

    async def _arun(self, query: str, *args, **kwargs):
        """Asynchronous execution of the agent tool"""
        if self.react_agent is None:
            raise RuntimeError(
                "IACTTool must be built via 'await IACTTool.create(...)' before use."
            )
        try:
            if self.citation_offset is None:
                text, _ = await self.run_subagent(query)
                return text
            async with self.citation_lock:
                text, graph = await self.run_subagent(query)
                text, graph = self.claim_citations(text, graph)
            return text, graph_artifact(graph)
        except Exception as e:
            logger.error(f"Error executing agent tool {self.name} (async): {str(e)}")
            error = f"Error executing agent tool: {str(e)}"
            return error if self.citation_offset is None else (error, None)
```

Comprueba que `Tuple` ya está importado desde `typing` (lo está, línea 18).

- [ ] **Step 4: Ejecutar los tests**

Run: `poetry run pytest tests/unit/tools/test_knowledge_router.py tests/unit/tools/test_agent_tools.py tests/unit/tools/test_agent_tools_sandbox_lifecycle.py -v`
Expected: PASS. `test_iact_tool_arun_forwards_subagent_tool_events` cubre que el modo clásico sigue reenviando los eventos.

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/agentTools.py tests/unit/tools/test_knowledge_router.py
git commit -m "feat(knowledge-router): IACTTool router mode returns renumbered, silo-tagged chunks"
```

---

### Task 5: `consultar_varios` y cableado en `create_agent`

**Files:**
- Modify: `backend/tools/knowledge_router.py`
- Modify: `backend/tools/agentTools.py`: `create_agent` (bucle de agent-tools ~932-943, silo propio ~949)
- Test: `tests/unit/tools/test_knowledge_router.py`

**Interfaces:**
- Consumes: `IACTTool.enable_router_mode`, `run_subagent`, `claim_citations` (Task 4); `graph_artifact`, `KNOWLEDGE_ROUTER_INSTRUCTION`, `KNOWLEDGE_ROUTER_TOOL_NAME` (Task 3).
- Produces: `create_consultar_varios_tool(specialists: List[Any], lock: asyncio.Lock) -> StructuredTool`, cuya coroutine `(especialistas: List[str], pregunta: str) -> Tuple[str, Optional[List[Document]]]`.

- [ ] **Step 1: Escribir los tests que fallan**

Añade a `tests/unit/tools/test_knowledge_router.py`:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.knowledge_router import create_consultar_varios_tool


class _BrokenReactAgent:
    async def ainvoke(self, _payload):
        raise RuntimeError("down")


class _PlainReactAgent:
    """Specialist without LightRAG: answers, no artifact."""

    async def ainvoke(self, payload):
        return {"messages": [*payload["messages"], AIMessage(content="plain answer")]}


def _es_en(es_agent, en_agent):
    offset, lock = [0], asyncio.Lock()
    es = _router_tool("ES", es_agent, offset, lock, silo_id=7)
    en = _router_tool("EN", en_agent, offset, lock, silo_id=8, silo_name="Manuals EN")
    return create_consultar_varios_tool([es, en], lock), offset


@pytest.mark.asyncio
async def test_consultar_varios_numbers_in_list_order_not_finish_order(no_writer):
    # ES finishes last but is first in the list: its chunks must still be 1,2.
    tool, offset = _es_en(_CitingReactAgent(["a", "b"], delay=0.05), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert text == "### ES\nAnswer [1](cite://1)[2](cite://2)\n\n### EN\nAnswer [3](cite://3)"
    assert [(c["id"], c["silo_id"]) for c in _chunks(artifact)] == [("a", 7), ("b", 7), ("c", 8)]
    entities = artifact[0].metadata["lightrag_raw_data"]["data"]["entities"]
    assert sorted((e["id"], e["silo_id"]) for e in entities) == [("Quemador", 7), ("Quemador", 8)]
    assert offset == [3]


@pytest.mark.asyncio
async def test_consultar_varios_survives_one_specialist_failing(no_writer):
    tool, offset = _es_en(_BrokenReactAgent(), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert "### ES\n(error: down)" in text
    assert "### EN\nAnswer [1](cite://1)" in text
    assert [c["id"] for c in _chunks(artifact)] == ["c"]
    assert offset == [1]


@pytest.mark.asyncio
async def test_consultar_varios_ignores_unknown_and_duplicate_names(no_writer):
    tool, _ = _es_en(_CitingReactAgent(["a"]), _CitingReactAgent(["c"]))

    text, _ = await tool.coroutine(especialistas=["EN", "EN", "Nope"], pregunta="q")

    assert text == "### EN\nAnswer [1](cite://1)"


@pytest.mark.asyncio
async def test_consultar_varios_with_no_valid_names_lists_valid_ones(no_writer):
    tool, _ = _es_en(_CitingReactAgent(["a"]), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["Nope"], pregunta="q")

    assert artifact is None
    assert "ES" in text and "EN" in text


@pytest.mark.asyncio
async def test_consultar_varios_mixes_plain_and_lightrag_specialists(no_writer):
    tool, offset = _es_en(_PlainReactAgent(), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert text == "### ES\nplain answer\n\n### EN\nAnswer [1](cite://1)"
    assert [c["id"] for c in _chunks(artifact)] == ["c"]
    assert offset == [1]


def _router_agent(is_router: bool, silo_id):
    return SimpleNamespace(
        agent_id=1, type="agent", ai_service=None, enable_code_interpreter=False,
        has_memory=False, memory_max_tokens=None, memory_max_messages=None,
        memory_summarize_threshold=None, output_parser_id=None, server_tools=[],
        tool_associations=[SimpleNamespace(tool=SimpleNamespace(type="agent"))],
        silo_id=silo_id, silo=None, skill_associations=[], mcp_associations=[],
        system_prompt="You route.", is_knowledge_router=is_router,
    )


async def _build(agent, specialist):
    with (
        patch("tools.agentTools.get_llm", return_value=MagicMock()),
        patch("tools.agentTools.get_output_parser", return_value=None),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as build,
        patch("tools.agentTools.MCPClientManager.get_client", new=AsyncMock(return_value=None)),
        patch("tools.agentTools.IACTTool.create", new=AsyncMock(return_value=specialist)),
        patch("tools.agentTools._resolve_and_build_retriever_tool", return_value=None) as own_silo,
    ):
        await agentTools.create_agent(agent)
    return build.call_args.kwargs, own_silo


def _specialist():
    specialist = MagicMock(spec=agentTools.IACTTool)
    specialist.name, specialist.description = "ES", "Spanish manuals"
    return specialist


@pytest.mark.asyncio
async def test_create_agent_wires_knowledge_router():
    specialist = _specialist()

    kwargs, own_silo = await _build(_router_agent(True, silo_id=5), specialist)

    specialist.enable_router_mode.assert_called_once()
    assert "consultar_varios" in [getattr(t, "name", None) for t in kwargs["tools"]]
    assert "<knowledge_router>" in kwargs["system_prompt"]
    own_silo.assert_not_called()


@pytest.mark.asyncio
async def test_create_agent_without_flag_leaves_agent_tools_alone():
    specialist = _specialist()

    kwargs, _ = await _build(_router_agent(False, silo_id=None), specialist)

    specialist.enable_router_mode.assert_not_called()
    assert "consultar_varios" not in [getattr(t, "name", None) for t in kwargs["tools"]]
    assert "<knowledge_router>" not in kwargs["system_prompt"]
```

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `poetry run pytest tests/unit/tools/test_knowledge_router.py -v`
Expected: FAIL con `ImportError: cannot import name 'create_consultar_varios_tool'`

- [ ] **Step 3: Implementar `create_consultar_varios_tool`**

En `backend/tools/knowledge_router.py`, amplía los imports:

```python
import asyncio
import re
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.streaming_utils import merge_lightrag_graph
from utils.logger import get_logger

logger = get_logger(__name__)
```

Y añade al final:

```python
class _ConsultarVariosInput(BaseModel):
    especialistas: List[str] = Field(description="Names of the specialist tools to ask")
    pregunta: str = Field(description="The user's question, as asked")


def create_consultar_varios_tool(specialists: List[Any], lock: asyncio.Lock) -> StructuredTool:
    """Ask several specialists the same question in parallel; merge in list order.

    *specialists* are router-mode IACTTools. Their sub-agents run concurrently,
    but citations are claimed afterwards in the order requested, so numbering
    never depends on which one finished first.
    """
    by_name = {t.name: t for t in specialists}
    catalog = "\n".join(f"- {t.name}: {t.description}" for t in specialists)

    async def _consultar(especialistas: List[str], pregunta: str) -> Tuple[str, Optional[List[Document]]]:
        chosen = [by_name[n] for n in dict.fromkeys(especialistas) if n in by_name]
        if not chosen:
            return f"Unknown specialists {especialistas}. Valid names: {', '.join(by_name)}", None
        async with lock:
            runs = await asyncio.gather(*(t.run_subagent(pregunta) for t in chosen), return_exceptions=True)
            parts: List[str] = []
            merged: Optional[dict] = None
            for t, run in zip(chosen, runs):
                if isinstance(run, BaseException):
                    logger.error("Knowledge router specialist %s failed: %s", t.name, run)
                    parts.append(f"### {t.name}\n(error: {run})")
                    continue
                text, graph = t.claim_citations(*run)
                parts.append(f"### {t.name}\n{text}")
                if graph:
                    merged = merge_lightrag_graph(merged, graph)
        return "\n\n".join(parts), graph_artifact(merged)

    return StructuredTool.from_function(
        coroutine=_consultar,
        name=KNOWLEDGE_ROUTER_TOOL_NAME,
        description=(
            "Ask several specialists the same question at once and get all their answers.\n"
            "Specialists:\n" + catalog
        ),
        args_schema=_ConsultarVariosInput,
        response_format="content_and_artifact",
    )
```

Si importar `tools.streaming_utils` desde aquí provoca un import circular (por ejemplo, porque `streaming_utils` acaba importando `agentTools`), mueve el `from tools.streaming_utils import merge_lightrag_graph` dentro de `_consultar`.

- [ ] **Step 4: Cablear en `create_agent`**

En `backend/tools/agentTools.py`, amplía el import de la Task 4:

```python
from tools.knowledge_router import (
    KNOWLEDGE_ROUTER_INSTRUCTION,
    chunk_count,
    create_consultar_varios_tool,
    graph_artifact,
    rebase_citations,
    tag_graph_with_silo,
)
```

Sustituye el bucle `for tool in agent.tool_associations:` de `create_agent` (~932-943) por:

```python
    # Knowledge router: every agent-tool becomes a pass-through specialist sharing
    # one turn-wide citation counter, plus consultar_varios for questions that
    # span several of them. See tools/knowledge_router.py.
    router_offset: Optional[List[int]] = [0] if getattr(agent, "is_knowledge_router", False) else None
    router_lock = asyncio.Lock()
    specialists: List[IACTTool] = []
    for tool in agent.tool_associations:
        sub_agent = tool.tool
        built = await discover_tool(
            sub_agent,
            user_context=user_context,
            working_dir=working_dir,
            sandbox_handle=sandbox_handle,
            sandbox_provider=sandbox_provider,
            sandbox_session_key=sandbox_session_key,
            sandbox_session_service=sandbox_session_service,
            attached_files=attached_files,
        )
        if router_offset is not None and isinstance(built, IACTTool):
            built.enable_router_mode(router_offset, router_lock)
            specialists.append(built)
        tools.append(built)
    if specialists:
        tools.append(create_consultar_varios_tool(specialists, router_lock))
        system_prompt_content += "\n\n" + KNOWLEDGE_ROUTER_INSTRUCTION
```

En la línea siguiente al bucle, la del silo propio:

```python
    # A knowledge router's knowledge lives in its specialists; its own silo
    # would number citations on a separate counter, so it is not used.
    if agent.silo_id is not None and not specialists:
```

`IACTTool` se define más abajo en el mismo módulo. La anotación `List[IACTTool]` de una variable local no se evalúa en ejecución, y el `isinstance` se ejecuta cuando la clase ya existe, así que no hay problema.

- [ ] **Step 5: Ejecutar los tests**

Run: `poetry run pytest tests/unit/tools/ -v`
Expected: PASS (todo `tests/unit/tools/`)

- [ ] **Step 6: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/knowledge_router.py backend/tools/agentTools.py tests/unit/tools/test_knowledge_router.py
git commit -m "feat(knowledge-router): consultar_varios tool and create_agent wiring"
```

---

### Task 6: Respuesta de los turnos que terminan en `ToolMessage` (pass-through)

**Files:**
- Modify: `backend/tools/streaming_utils.py` (función nueva)
- Modify: `backend/services/agent_streaming_service.py` (`stream_agent_chat`: init ~línea 292, bucle de updates ~355-362, tras el flush ~398-403)
- Modify: `backend/services/agent_cache_service.py` (`get_conversation_history_async`, bucle ~335-390)
- Test: `tests/unit/tools/test_streaming_utils.py`
- Test: `tests/unit/services/test_agent_cache_service_history.py`

**Interfaces:**
- Produces: `final_tool_answer(last_msg: Any, streamed_text: str) -> Optional[str]` en `tools/streaming_utils.py`.

- [ ] **Step 1: Escribir los tests que fallan**

Añade a `tests/unit/tools/test_streaming_utils.py`:

```python
def test_final_tool_answer_when_turn_ends_on_tool_message():
    from langchain_core.messages import ToolMessage
    from tools.streaming_utils import final_tool_answer

    msg = ToolMessage(content="ES answer [1](cite://1)", tool_call_id="c1")
    assert final_tool_answer(msg, "") == "ES answer [1](cite://1)"


def test_final_tool_answer_ignored_when_model_answered():
    from langchain_core.messages import AIMessage, ToolMessage
    from tools.streaming_utils import final_tool_answer

    assert final_tool_answer(ToolMessage(content="x", tool_call_id="c1"), "model text") is None
    assert final_tool_answer(AIMessage(content="final"), "") is None
    assert final_tool_answer(None, "") is None
```

Añade a `tests/unit/services/test_agent_cache_service_history.py`:

```python
@pytest.mark.asyncio
async def test_turns_ending_on_tool_message_show_it_as_the_answer():
    """Knowledge-router pass-through (return_direct): the specialist's ToolMessage
    IS the answer — no AI message follows it."""
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "ES", "args": {}, "id": "call_1"}]),
        _lightrag_tool_message("call_1"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "EN", "args": {}, "id": "call_2"}]),
        ToolMessage(content="EN answer", tool_call_id="call_2"),
    ]

    history = await _history_for(messages)

    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "q1"), ("agent", "tool text"), ("user", "q2"), ("agent", "EN answer"),
    ]
    assert history[1]["lightrag_graph"]["data"]["chunks"] == [{"id": "c1"}]
    assert "lightrag_graph" not in history[3]
```

- [ ] **Step 2: Ejecutarlos y ver que fallan**

Run: `poetry run pytest tests/unit/tools/test_streaming_utils.py tests/unit/services/test_agent_cache_service_history.py -v`
Expected: FAIL. `ImportError: cannot import name 'final_tool_answer'`, y el historial devuelve solo las dos entradas de usuario.

- [ ] **Step 3: Implementar el helper**

En `backend/tools/streaming_utils.py`, debajo de `merge_lightrag_graph` (y asegúrate de que `ToolMessage` se importa de `langchain_core.messages` a nivel de módulo; si no está, añádelo):

```python
def final_tool_answer(last_msg: Any, streamed_text: str) -> Optional[str]:
    """Answer of a turn that ended on a ToolMessage (a return_direct tool, e.g.
    a knowledge-router specialist): no model tokens were streamed for it.
    None when the model wrote the answer itself."""
    if streamed_text.strip() or not isinstance(last_msg, ToolMessage):
        return None
    content = last_msg.content
    return content if isinstance(content, str) else str(content)
```

(Si `Optional` no está importado en el módulo, añádelo al import de `typing`.)

- [ ] **Step 4: Usarlo en streaming**

En `backend/services/agent_streaming_service.py`, añade `final_tool_answer` al import de `tools.streaming_utils` (líneas 21-27). En `stream_agent_chat`, debajo de `structured_response = None` (~línea 292):

```python
                last_update_msg = None
```

En el bucle de updates, justo después de `if msg is None: continue` (~línea 358):

```python
                                            last_update_msg = msg
```

Tras el bucle de flush `for _mid, texts in _token_buf.items(): ...` y **antes** del `break` (~línea 403):

```python
                    # Knowledge-router pass-through: the turn ended on the
                    # specialist's ToolMessage (return_direct), so no model
                    # tokens carried the answer.
                    tool_answer = None if structured_response is not None else final_tool_answer(last_update_msg, accumulated_content)
                    if tool_answer:
                        accumulated_content = tool_answer
                        yield format_sse_event(SSE_TOKEN, {"content": tool_answer})
```

- [ ] **Step 5: Usarlo en el historial**

En `backend/services/agent_cache_service.py`, a nivel de módulo:

```python
def _tool_answer_entry(content: str, lightrag_graph) -> dict:
    """History entry for a turn whose answer is a ToolMessage (return_direct)."""
    entry = {"role": "agent", "content": content}
    if lightrag_graph:
        entry["lightrag_graph"] = lightrag_graph
    return entry
```

En `get_conversation_history_async`, debajo de `pending_lightrag_graph = None`:

```python
                # Answer of a turn that ended on a ToolMessage (knowledge-router
                # pass-through): no AI message follows it.
                pending_tool_answer = None
```

En la rama `if msg_type in ['human', 'user']:`, antes de `role = 'user'`:

```python
                            if pending_tool_answer:
                                history.append(_tool_answer_entry(pending_tool_answer, pending_lightrag_graph))
                                pending_tool_answer = pending_lightrag_graph = None
```

En la rama `elif msg_type == 'tool':`, antes de su `continue`:

```python
                            tool_content = msg.content
                            tool_text = _content_blocks_to_str(tool_content) if isinstance(tool_content, list) else str(tool_content or "")
                            pending_tool_answer = tool_text if tool_text.strip() else None
```

Junto al `pending_lightrag_graph = None` que va antes de `history.append(entry)`:

```python
                        pending_tool_answer = None
```

Y después del `for msg in messages:`, antes de `logger.info(f"Retrieved ...")`:

```python
                if pending_tool_answer:
                    history.append(_tool_answer_entry(pending_tool_answer, pending_lightrag_graph))
```

- [ ] **Step 6: Ejecutar los tests**

Run: `poetry run pytest tests/unit/tools/test_streaming_utils.py tests/unit/services/ -v`
Expected: PASS. Los cuatro tests de historial anteriores siguen en verde: un `ToolMessage` seguido de una respuesta AI no se muestra.

- [ ] **Step 7: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/streaming_utils.py backend/services/agent_streaming_service.py backend/services/agent_cache_service.py tests/unit/tools/test_streaming_utils.py tests/unit/services/test_agent_cache_service_history.py
git commit -m "feat(knowledge-router): show pass-through specialist answers live and on reload"
```

---

### Task 7: Frontend (silo por chunk, bubble por silo, checkbox)

**Files:**
- Modify: `frontend/src/types/streaming.ts:52-79`
- Create: `frontend/src/utils/splitGraphBySilo.ts`
- Modify: `frontend/src/components/playground/CitationBadge.tsx:110`
- Modify: `frontend/src/components/playground/LightRAGGraphBubble.tsx`
- Modify: `frontend/src/components/playground/ChatInterface.tsx:1131-1133`
- Modify: `frontend/src/pages/AgentFormPage.tsx` (interfaces ~28/~85, defaults ~239, carga ~362, envío ~681, checkbox tras Code Interpreter ~1238)

**Interfaces:**
- Consumes: `silo_id` y `silo_name` en entidades, relaciones y chunks (Task 4); `is_knowledge_router` en la API (Task 2).
- Produces: `splitGraphBySilo(graph: LightRAGGraphData): SiloGraph[]`, y las props nuevas `title?: string`, `chunkNumbers?: number[]` en `LightRAGGraphBubble`.

No hay framework de tests en el frontend. La verificación es `lint` + `build` aquí y la comprobación manual de la Task 8.

- [ ] **Step 1: Tipos**

En `frontend/src/types/streaming.ts`, añade a `LightRAGEntity`, `LightRAGRelationship` y `LightRAGChunk`, antes de su `[key: string]: unknown;`:

```ts
  /** Source silo — only on knowledge-router payloads (one per specialist). */
  silo_id?: number;
  silo_name?: string;
```

- [ ] **Step 2: `splitGraphBySilo`**

`frontend/src/utils/splitGraphBySilo.ts`:

```ts
import type { LightRAGGraphData } from '../types/streaming';

export interface SiloGraph {
  siloId?: number;
  siloName?: string;
  graph: LightRAGGraphData;
  /** 1-based citation numbers (cite://N) of graph.data.chunks in the full payload. */
  chunkNumbers: number[];
}

/** Split a turn's merged LightRAG payload into one graph per source silo
 * (knowledge-router turns). Untagged payloads come back as a single group. */
export function splitGraphBySilo(graph: LightRAGGraphData): SiloGraph[] {
  const data = graph.data ?? {};
  const entities = data.entities ?? [];
  const relationships = data.relationships ?? [];
  const chunks = data.chunks ?? [];
  const all = [...entities, ...relationships, ...chunks];
  const siloIds = [...new Set(all.map((x) => x.silo_id))];

  if (siloIds.length <= 1) {
    return [{ siloId: siloIds[0], siloName: all[0]?.silo_name, graph, chunkNumbers: chunks.map((_, i) => i + 1) }];
  }
  return siloIds.map((siloId) => ({
    siloId,
    siloName: all.find((x) => x.silo_id === siloId)?.silo_name,
    graph: {
      ...graph,
      data: {
        ...data,
        entities: entities.filter((e) => e.silo_id === siloId),
        relationships: relationships.filter((r) => r.silo_id === siloId),
        chunks: chunks.filter((c) => c.silo_id === siloId),
      },
    },
    chunkNumbers: chunks.flatMap((c, i) => (c.silo_id === siloId ? [i + 1] : [])),
  }));
}
```

- [ ] **Step 3: El PDF se abre con el silo del chunk**

`frontend/src/components/playground/CitationBadge.tsx:110`:

```tsx
            <OpenPdfButton appId={appId} siloId={chunk.silo_id ?? siloId} resourceId={chunk.resource_id} page={chunk.page} />
```

- [ ] **Step 4: `LightRAGGraphBubble`: título y número de chunk global**

En `Props`, añade:

```ts
  /** Silo name shown before the counts when the turn spans several silos. */
  title?: string;
  /** Global citation number of each chunk (defaults to its position + 1). */
  chunkNumbers?: number[];
```

La firma pasa a ser `export default function LightRAGGraphBubble({ graphData, citedChunkIds, appId, siloId, title, chunkNumbers }: Props)`.

En el `<span>` de la cabecera (línea ~184), antepón el título:

```tsx
          <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300 flex-1">
            {title ? `${title} · ` : ''}
            {hasGraph
```

(el resto de la expresión no cambia). En el popover (línea ~273):

```tsx
                <span className="ml-1 font-normal text-gray-400">· Chunk {chunkNumbers?.[openChunk] ?? openChunk + 1}</span>
```

- [ ] **Step 5: `ChatInterface`: un bubble por silo**

Importa `import { splitGraphBySilo } from '../../utils/splitGraphBySilo';` y sustituye las líneas ~1131-1133:

```tsx
                        {message.lightragGraph && splitGraphBySilo(message.lightragGraph).map((g) => (
                          <LightRAGGraphBubble
                            key={g.siloId ?? 'all'}
                            graphData={g.graph}
                            citedChunkIds={citedChunkIds}
                            appId={appId}
                            siloId={g.siloId ?? siloId}
                            title={g.siloName}
                            chunkNumbers={g.chunkNumbers}
                          />
                        ))}
```

Si `splitGraphBySilo` se recalcula en cada render y eso se nota (el grafo se vuelve a maquetar), envuélvelo en un `useMemo` por mensaje. Solo si se observa.

- [ ] **Step 6: Checkbox en el formulario**

En `frontend/src/pages/AgentFormPage.tsx`:
- En la interfaz de la respuesta de la API (~línea 30), debajo de `enable_code_interpreter: boolean;`: `is_knowledge_router?: boolean;`
- En la interfaz del estado del formulario (~línea 87), debajo de `enable_code_interpreter: boolean;`: `is_knowledge_router: boolean;`
- En los valores por defecto (~línea 241): `is_knowledge_router: false,`
- En la carga (~línea 364): `is_knowledge_router: response.is_knowledge_router || false,`
- En el envío (~línea 683): `is_knowledge_router: formData.is_knowledge_router,`
- Después del bloque del checkbox `enable_code_interpreter` (~línea 1238):

```tsx
                      {formData.tool_ids.length > 0 && (
                        <div className="flex items-center p-4 bg-gray-50 rounded-xl">
                          <input
                            id="is_knowledge_router"
                            type="checkbox"
                            checked={formData.is_knowledge_router}
                            onChange={(e) => handleInputChange('is_knowledge_router', e.target.checked)}
                            className="w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <div className="ml-3">
                            <label htmlFor="is_knowledge_router" className="text-sm font-medium text-gray-900">Knowledge router</label>
                            <p className="text-xs text-gray-500">
                              Routes each question to the specialist agent(s) whose description matches; a single
                              specialist's answer is shown as-is. This agent's own silo is not used.
                            </p>
                          </div>
                        </div>
                      )}
```

- [ ] **Step 7: Lint y build**

Run: `cd frontend && npm run lint && npm run build`
Expected: sin errores nuevos. Compara con `git stash`/rama base si hay warnings que ya existían.

- [ ] **Step 8: `/ponytail-review` del diff y commit**

```bash
git add frontend/src/types/streaming.ts frontend/src/utils/splitGraphBySilo.ts frontend/src/components/playground/CitationBadge.tsx frontend/src/components/playground/LightRAGGraphBubble.tsx frontend/src/components/playground/ChatInterface.tsx frontend/src/pages/AgentFormPage.tsx
git commit -m "feat(frontend): knowledge router flag, per-chunk silo for PDFs, one graph bubble per silo"
```

---

### Task 8: Documentación y verificación end-to-end

**Files:**
- Modify: `docs/ai/rfc-multilingual-silo-routing.md` (línea de estado y sección Decision)

- [ ] **Step 1: Actualizar el RFC**

En `docs/ai/rfc-multilingual-silo-routing.md`, sustituye la línea `> **Status**: ...` por:

```markdown
> **Status**: Implemented as option C, generalised to any specialist (not only languages): the `is_knowledge_router` flag — see `docs/superpowers/specs/2026-10-07-knowledge-router-design.md` · Recorded September 30, 2026
```

Y añade al final de la sección `## Decision`:

```markdown
**Update 2026-10-07:** went with C, generalised as a "knowledge router" flag. Citation
numbering uses rebase onto a turn-wide counter (not namespaced keys): specialists are
claimed in list order inside `consultar_varios`, so finish order never matters and the
frontend keeps resolving `cite://N` by position.
```

- [ ] **Step 2: Suite completa**

Run: `poetry run pytest tests/unit/ -q && poetry run pytest tests/integration/routers/internal/test_agents.py tests/integration/repositories/ -q`
Expected: PASS. Si algo falla y no tiene relación con este cambio, compruébalo contra `lightrag` antes de seguir y anótalo.

- [ ] **Step 3: Levantar el stack con el código del worktree**

El compose se llama `docker` en ambos árboles, así que comparte contenedores y volúmenes con el árbol principal: rebuild desde aquí sustituye el stack de dev. La migración `kr001` se aplica sobre la BD de dev; para deshacerla, `alembic downgrade -1` antes de volver a la rama principal.

```bash
cp ../ai-core-tools/.env .env
cd docker && docker compose up -d --build && docker compose logs -f backend
```

(`--build frontend` también reconstruye el backend por `depends_on`.)

- [ ] **Step 4: Comprobación manual en el playground**

1. Edita los dos especialistas (agente ES del silo Domusa y agente EN del silo 39): `is_tool` activo y una `description` que diga qué cubren ("Manuales Domusa en español…" / "Domusa manuals in English…").
2. Crea un agente "Router Domusa" con ambos como tools y el checkbox **Knowledge router** activado.
3. Pregunta general (por ejemplo "¿En qué manuales aparece el parámetro P20?"). Esperado: un bloque de herramientas `consultar_varios` con los dos sub-agentes, respuesta sintetizada, cada `[N]` abre el PDF correcto (comprueba al menos una cita de cada silo) y **dos bubbles**, uno por silo, cada uno con sus chunks y su subgrafo. Una entidad presente en ambos silos aparece en los dos bubbles.
4. Pregunta solo de los manuales en inglés. Esperado: llamada única al especialista EN, respuesta idéntica a la suya, citas que abren PDFs del silo 39 y un único bubble.
5. Recarga la página: las dos respuestas, sus citas y sus bubbles siguen ahí.
6. Una tercera pregunta en la misma conversación funciona: el turno anterior terminó en `ToolMessage`, así que esto valida el orden `ToolMessage → HumanMessage` con el proveedor real.

Anota cualquier desviación con capturas o texto.

- [ ] **Step 5: Eval multilingüe**

Con el set de `scripts/generar_qa_multiling.py` (árbol principal), lanza la evaluación contra:
- el router
- el agente ES solo
- el agente EN solo

El criterio de éxito del spec: la calidad de las citas del router no baja más que el ruido de fondo (~±8 preguntas por ejecución completa) respecto a la mejor de las dos bases, en particular en las preguntas generales.

- [ ] **Step 6: `/ponytail-review` del diff de la tarea y commit**

```bash
git add docs/ai/rfc-multilingual-silo-routing.md
git commit -m "docs(rfc): multilingual routing implemented as the knowledge router"
```

- [ ] **Step 7: Revisión final de la rama**

Lanza `/ponytail-review` y `/code-review` sobre `git diff lightrag...feature/knowledge-router`, y aplica o declina cada hallazgo con su motivo. Después, superpowers:finishing-a-development-branch.

# Cobertura Exhaustiva en el Router de Skills — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar a los agentes `skill-routed` de LightRAG una vía de recuperación exhaustiva (sin `chunk_top_k`) para preguntas de membership/enumeración ("¿en qué documentos aparece X?", "lista todos los X del manual Y") y para resolver referencias a productos por nombre comercial antes de retrieval.

**Architecture:** Dos funciones de datos nuevas que consultan directamente el almacenamiento (SQL crudo contra `lightrag_doc_chunks` para contenido; ORM contra `Resource.extra_metadata` para nombre comercial), una tool nueva de LangChain que las envuelve para el agente, y una extensión del prompt del router (`ROUTER_SKILL_CONTENT`) que le dice al LLM cuándo usarla en vez de `retrieve_from_knowledge_base`. Ningún cambio toca el pipeline de indexación ni las tools existentes.

**Tech Stack:** Python 3.11, SQLAlchemy Core (`text()`) contra Postgres, LangChain `@tool`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-coverage-router-design.md`

## Global Constraints

- Sin tope de *documentos*, con tope defensivo de *filas* — máx. 3 snippets por documento, máx. 2000 filas totales por consulta (del spec, sección "Límites").
- Sin coincidencias → mensaje de abstención explícito, nunca alucinar (del spec, sección "Errores").
- Misma convención de citas que `retrieve_from_knowledge_base` (`cite://N` resuelto por el frontend vía `_append_lightrag_citation_sources`).
- La tool nueva solo se expone para silos LightRAG en modo `skill-routed` — mismo gate que `_create_dynamic_lightrag_tool` hoy.
- Búsqueda literal (ILIKE), sin expansión de sinónimos — explícitamente fuera de v1.

---

### Task 1: `find_chunks_mentioning` — búsqueda exhaustiva de contenido

**Files:**
- Modify: `backend/tools/vector_stores/lightrag_store.py` (añadir método a `LightRAGStore`, cerca de `_cleanup_postgres`)
- Test: `tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py`

**Interfaces:**
- Produces: `LightRAGStore.find_chunks_mentioning(self, collection_name: str, term: str, doc_filter: Optional[str] = None) -> dict[str, list[tuple[str, str]]]` — clave = `resource_id` en string, valor = lista de `(file_path, snippet)` (máx. 3 por doc). `file_path` es la etiqueta con página ya incluida (p.ej. `"CDOC004043.pdf p.63"`) — Task 3 la necesita tal cual para las citas.
- Produces (helper puro, testeable sin BD): `_group_chunk_rows(rows: list[tuple[str, str, str]], per_doc_cap: int = 3) -> dict[str, list[tuple[str, str]]]` — cada tupla de entrada es `(full_doc_id, file_path, content)`.

- [ ] **Step 1: Write the failing test for `_group_chunk_rows`**

```python
# tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py
"""Pins the exhaustive-coverage query for cobertura/G07/G09-style questions.

No chunk_top_k here on purpose: these questions are membership ("does X
appear in doc D, yes or no"), not ranking — a document with one incidental
mention must count exactly the same as one with five relevant ones, which a
similarity top-k cannot guarantee.
"""

from tools.vector_stores.lightrag_store import _group_chunk_rows
# find_chunks_mentioning is a thin wrapper around this (it just runs the SQL
# and calls _group_chunk_rows) — exercised indirectly in Task 3's tool tests,
# which mock it at the LightRAGStore instance level.


def test_groups_snippets_by_resource_id():
    rows = [
        ("res271-p10", "CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí."),
        ("res271-p42", "CDOC004043.pdf p.42", "Revisar la válvula de seguridad cada año."),
        ("res275-p3", "CDOC001961.pdf p.3", "La válvula de seguridad debe purgarse."),
    ]
    grouped = _group_chunk_rows(rows)
    assert set(grouped.keys()) == {"271", "275"}
    assert len(grouped["271"]) == 2
    assert len(grouped["275"]) == 1
    # file_path travels with the snippet — Task 3's citations need the real
    # "file.pdf p.N" label, not just the resource id.
    assert grouped["271"][0] == ("CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí.")


def test_caps_snippets_per_document():
    rows = [
        (f"res271-p{i}", f"CDOC004043.pdf p.{i}", f"mención {i}")
        for i in range(10)
    ]
    grouped = _group_chunk_rows(rows, per_doc_cap=3)
    assert len(grouped["271"]) == 3


def test_ignores_rows_with_unparseable_doc_id():
    """Non-PDF sources (crawled Domain pages) get a content-hash id, not
    res{N}-p{N} — see _lightrag_doc_id's fallback. Skip them rather than crash."""
    rows = [("doc-abc123hash", "https://example.com/page", "algo de contenido")]
    grouped = _group_chunk_rows(rows)
    assert grouped == {}


def test_empty_rows_returns_empty_dict():
    assert _group_chunk_rows([]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py -v`
Expected: FAIL with `ImportError: cannot import name '_group_chunk_rows'`

- [ ] **Step 3: Implement `_group_chunk_rows` and `find_chunks_mentioning`**

Add to `backend/tools/vector_stores/lightrag_store.py`, near `_cleanup_postgres` (same file already has `re` and `text` imported at module level — verify `from sqlalchemy import text` exists at top of file; `_cleanup_postgres` imports it locally, follow that same local-import style for consistency):

```python
# Matches the PARENT doc id (no "-chunk-N" suffix) — see _lightrag_doc_id and
# CHUNK_ID_RESOURCE_PAGE_RE just above, which matches the CHUNK id instead.
_FULL_DOC_ID_RESOURCE_RE = re.compile(r"^res(\d+)-p\d+$")


def _group_chunk_rows(
    rows: list[tuple[str, str, str]], per_doc_cap: int = 3,
) -> dict[str, list[tuple[str, str]]]:
    """Group (full_doc_id, file_path, content) rows by resource id.

    Rows whose full_doc_id doesn't match the res{N}-p{N} shape (non-PDF
    sources, e.g. crawled Domain pages — see _lightrag_doc_id's content-hash
    fallback) are skipped: there is no resource_id to group them under.

    per_doc_cap: at most this many snippets per document — the FIRST match
    per document already answers "does it appear here", which is the whole
    point; the rest is round-tripped for content, not to be exhaustive here.

    file_path travels alongside each snippet (not just the content) because
    Task 3's citation block needs the real "file.pdf p.N" label, not a
    synthetic one.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for full_doc_id, file_path, content in rows:
        match = _FULL_DOC_ID_RESOURCE_RE.match(full_doc_id)
        if not match:
            continue
        resource_id = match.group(1)
        snippets = grouped.setdefault(resource_id, [])
        if len(snippets) < per_doc_cap:
            snippets.append((file_path, content.strip()))
    return grouped
```

Add as a method on `LightRAGStore` (same class as `_cleanup_postgres`):

```python
    # Defensive ceiling on TOTAL rows scanned per call — not a ranking cut
    # (there is none here), just a circuit breaker against a pathologically
    # generic term matching most of the corpus. Far above chunk_top_k=30,
    # which is the limit this method exists to route around.
    _COVERAGE_QUERY_ROW_CAP = 2000

    def find_chunks_mentioning(
        self, collection_name: str, term: str, doc_filter: Optional[str] = None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Every document whose chunks literally contain *term* — no chunk_top_k,
        no similarity ranking. See _group_chunk_rows for the per-document cap.

        doc_filter: a resource_id (as string or int) to scope the search to one
        document — used for "list all X in manual Y" questions (e.g. G07).
        """
        from sqlalchemy import text  # noqa: WPS433

        sql = "SELECT full_doc_id, file_path, content FROM lightrag_doc_chunks " \
              "WHERE workspace = :ws AND content ILIKE :pattern"
        params = {"ws": collection_name, "pattern": f"%{term}%"}
        if doc_filter is not None:
            sql += " AND full_doc_id LIKE :doc_pattern"
            params["doc_pattern"] = f"res{doc_filter}-p%"
        sql += f" ORDER BY full_doc_id LIMIT {self._COVERAGE_QUERY_ROW_CAP}"

        with self.db.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return _group_chunk_rows([(r[0], r[1], r[2]) for r in rows])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/tools/vector_stores/lightrag_store.py tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py
git commit -m "feat(lightrag): add find_chunks_mentioning for exhaustive coverage queries

Cobertura questions ('en que documentos aparece X') are membership
questions, not ranking questions - chunk_top_k answers the wrong one,
which is why the eval's cobertura block stalls around 50% PASS even
with reranking. This bypasses chunk_top_k entirely: a literal SQL scan
of lightrag_doc_chunks, grouped by document, with a generous row cap
as a circuit breaker (not a ranking cut)."
```

---

### Task 2: `resolve_document_by_name` — resolución de nombre comercial

**Files:**
- Modify: `backend/services/silo_service.py` (nuevo staticmethod en `SiloService`)
- Test: `tests/unit/services/test_resolve_document_by_name.py`

**Interfaces:**
- Consumes: `models.resource.Resource` (columna `extra_metadata`, JSON), `models.repository.Repository` (columna `app_id`).
- Produces: `SiloService.resolve_document_by_name(app_id: int, name: str, db: Session) -> list[int]` — lista de `resource_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_resolve_document_by_name.py
"""Resolves a product's commercial name (e.g. 'TERMAT') to its resource_id(s)
via Resource.extra_metadata->>'DescArticulo' — a deterministic metadata match,
not a reranker. See the design doc's "reranker por nombre de documento"
section for why a semantic reranker doesn't apply here: file codes like
CDOC004043.pdf carry no semantic signal, but this metadata field does.
"""

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from services.silo_service import SiloService


@pytest.fixture()
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[
        Silo.__table__, Repository.__table__, Resource.__table__,
    ])
    Maker = sessionmaker(bind=engine)
    s = Maker()
    yield s
    s.close()
    engine.dispose()


def _seed(session, app_id=1):
    silo = Silo(silo_id=1, app_id=app_id, name="test")
    repo = Repository(repository_id=1, app_id=app_id, silo_id=1, name="repo")
    session.add_all([silo, repo])
    session.add(Resource(
        resource_id=271, repository_id=1, uri="CDOC004043.pdf",
        extra_metadata={"DescArticulo": "TERMAT 25", "document": "CDOC004043"},
    ))
    session.add(Resource(
        resource_id=275, repository_id=1, uri="CDOC004425.pdf",
        extra_metadata={"DescArticulo": "NANOCLIMA 4 HDX 135", "document": "CDOC004425"},
    ))
    session.commit()


def test_matches_by_commercial_name(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "TERMAT", session) == [271]


def test_case_insensitive_and_partial(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "nanoclima", session) == [275]


def test_no_match_returns_empty_list(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "THERMAPRO", session) == []


def test_scoped_to_app_id(session):
    _seed(session, app_id=1)
    assert SiloService.resolve_document_by_name(2, "TERMAT", session) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/services/test_resolve_document_by_name.py -v`
Expected: FAIL with `AttributeError: type object 'SiloService' has no attribute 'resolve_document_by_name'`

- [ ] **Step 3: Implement `resolve_document_by_name`**

Add as a `@staticmethod` on `SiloService` in `backend/services/silo_service.py`, near `get_silos_by_app_id`:

```python
    @staticmethod
    def resolve_document_by_name(app_id: int, name: str, db: Session) -> List[int]:
        """resource_id(s) whose extra_metadata.DescArticulo matches *name* (ILIKE).

        Deterministic metadata match, not a reranker: DescArticulo is the
        commercial product name (e.g. "TERMAT 25"), unlike the opaque file
        code (CDOC004043.pdf) that carries no semantic signal a reranker
        could use. Used by the skill router to resolve a question naming a
        product (cross_documento comparisons) to a document BEFORE
        retrieval, instead of relying on similarity search to find it.
        """
        from models.repository import Repository  # noqa: PLC0415 — avoids import cycle

        rows = (
            db.query(Resource.resource_id)
            .join(Repository, Resource.repository_id == Repository.repository_id)
            .filter(Repository.app_id == app_id)
            .filter(Resource.extra_metadata["DescArticulo"].astext.ilike(f"%{name}%"))
            .all()
        )
        return [r[0] for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/services/test_resolve_document_by_name.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/services/silo_service.py tests/unit/services/test_resolve_document_by_name.py
git commit -m "feat(silo): add resolve_document_by_name for product-name resolution

Resource.extra_metadata already carries a DescArticulo commercial name
per document (confirmed live on the DOMUSA corpus: 'FUSION TRIO
200/50 (t)', 'DUAL CLIMA 12R NE DOMUSA iCONNECT'...) that nothing
reads today. cross_documento questions that name two products by
their commercial name ('TERMAT' vs 'THERMAPRO HT 290') can resolve
them to a resource_id deterministically instead of hoping similarity
search finds the right document."
```

---

### Task 3: `list_documents_mentioning` — la tool y su wiring

**Files:**
- Modify: `backend/tools/agentTools.py` (nueva tool junto a `_create_dynamic_lightrag_tool`; wiring en `_resolve_and_build_retriever_tool` y en su caller)
- Test: `tests/unit/tools/test_list_documents_mentioning_tool.py`

**Interfaces:**
- Consumes: `LightRAGStore.find_chunks_mentioning` (Task 1), `SiloService.resolve_document_by_name` (Task 2), `_append_lightrag_citation_sources` (ya existe, línea 1055 de `agentTools.py`).
- Produces: `_create_coverage_tool(silo: Silo, app_id: int) -> BaseTool` — la tool `list_documents_mentioning(term: str, doc: Optional[str] = None) -> tuple[str, list[Document]]`.
- Modifica el contrato de `_resolve_and_build_retriever_tool`: pasa de devolver `Optional[BaseTool]` a `Optional[list[BaseTool]]` cuando el modo es `skill-routed` (una o dos tools); sigue devolviendo una tool suelta (no lista) para el resto de modos, así que el caller debe manejar ambos casos.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tools/test_list_documents_mentioning_tool.py
"""Pins list_documents_mentioning: the tool must return every matching
document (via find_chunks_mentioning, no chunk_top_k) and must resolve a
named product to its resource_id (via resolve_document_by_name) before
searching, when the caller passes `doc`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools.agentTools import _create_coverage_tool


def _fake_silo(silo_id=37):
    return SimpleNamespace(silo_id=silo_id, vector_db_type="LIGHTRAG")


@pytest.mark.asyncio
async def test_returns_abstention_when_nothing_matches():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with patch("services.silo_service._get_vector_store") as get_store:
        get_store.return_value.find_chunks_mentioning.return_value = {}
        content, artifact = await tool.ainvoke({"term": "algo inexistente"})
    assert "No se encontró ningún documento" in content
    assert artifact == []


@pytest.mark.asyncio
async def test_returns_one_citation_per_matched_document():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {
        "271": [("CDOC004043.pdf p.10", "aparece SG Ready aquí")],
        "275": [("CDOC004425.pdf p.5", "también menciona SG Ready")],
    }
    with patch("services.silo_service._get_vector_store") as get_store:
        get_store.return_value.find_chunks_mentioning.return_value = grouped
        content, artifact = await tool.ainvoke({"term": "SG Ready"})
    assert "2" in content  # "2 documento(s) mencionan..."
    assert len(artifact) == 1  # one wrapper Document carrying both citations
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert len(chunks) == 2
    assert {"CDOC004043.pdf p.10", "CDOC004425.pdf p.5"} == {c["file_path"] for c in chunks}


@pytest.mark.asyncio
async def test_resolves_doc_param_by_commercial_name_before_searching():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service._get_vector_store") as get_store,
        patch("services.silo_service.SiloService.resolve_document_by_name",
              return_value=[271]) as resolve_mock,
    ):
        get_store.return_value.find_chunks_mentioning.return_value = {}
        await tool.ainvoke({"term": "P20", "doc": "TERMAT"})
    resolve_mock.assert_called_once()
    assert resolve_mock.call_args.args[0:2] == (1, "TERMAT")
    # the resolved resource_id (271), not the raw name "TERMAT", must reach
    # find_chunks_mentioning's doc_filter
    _, kwargs = get_store.return_value.find_chunks_mentioning.call_args
    assert get_store.return_value.find_chunks_mentioning.call_args.args[2] == 271


@pytest.mark.asyncio
async def test_falls_back_to_whole_corpus_when_doc_name_unresolved():
    """An unresolvable `doc` (typo, or a name not in extra_metadata) must not
    fail the call — search the whole corpus rather than return nothing."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service._get_vector_store") as get_store,
        patch("services.silo_service.SiloService.resolve_document_by_name",
              return_value=[]),
    ):
        get_store.return_value.find_chunks_mentioning.return_value = {}
        await tool.ainvoke({"term": "P20", "doc": "Modelo Inexistente"})
    assert get_store.return_value.find_chunks_mentioning.call_args.args[2] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/tools/test_list_documents_mentioning_tool.py -v`
Expected: FAIL with `ImportError: cannot import name '_create_coverage_tool'`

- [ ] **Step 3: Implement `_create_coverage_tool`**

Add to `backend/tools/agentTools.py`, right after `_create_dynamic_lightrag_tool` (~line 84):

```python
def _create_coverage_tool(silo: Silo, app_id: int):
    """Return a list_documents_mentioning(term, doc) tool for skill-routed agents.

    Bypasses chunk_top_k entirely (see find_chunks_mentioning) — for
    membership/enumeration questions ("in which documents does X appear",
    "list every X in manual Y"), not for explaining or reasoning about X.
    """
    silo_id = silo.silo_id
    _citation_offset: List[int] = [0]

    @tool(response_format="content_and_artifact")
    async def list_documents_mentioning(term: str, doc: Optional[str] = None) -> tuple:
        """Search EVERY document for literal mentions of a term — use this
        instead of retrieve_from_knowledge_base when the question asks to
        enumerate all documents where something appears, or list everything
        of some kind within one document. Do not use it to explain, compare,
        or reason about X — only to find where it is present.

        Args:
            term: the word or phrase to search for, verbatim.
            doc: optional — a document code, or a product's commercial name
                 (e.g. "TERMAT"), to scope the search to just that document.
        """
        from services.silo_service import SiloService, _get_vector_store, COLLECTION_PREFIX
        from db.database import SessionLocal

        resolved_doc_filter = None
        if doc:
            db = SessionLocal()
            try:
                resource_ids = await asyncio.to_thread(
                    SiloService.resolve_document_by_name, app_id, doc, db,
                )
            finally:
                db.close()
            if resource_ids:
                resolved_doc_filter = resource_ids[0]
            elif doc.isdigit():
                resolved_doc_filter = int(doc)
            # else: doc named neither a known product nor a resource_id —
            # search the whole corpus rather than fail the call.

        vector_store = _get_vector_store(silo)
        collection_name = COLLECTION_PREFIX + str(silo_id)
        grouped = await asyncio.to_thread(
            vector_store.find_chunks_mentioning, collection_name, term, resolved_doc_filter,
        )

        if not grouped:
            return f"No se encontró ningún documento que mencione '{term}'.", []

        chunks = [
            {"file_path": file_path, "content": content}
            for snippets in grouped.values()
            for file_path, content in snippets
        ]
        wrapper_doc = Document(
            page_content=f"{len(grouped)} documento(s) mencionan '{term}'.",
            metadata={"lightrag_raw_data": {"data": {"chunks": chunks}}},
        )
        content = _append_lightrag_citation_sources(
            wrapper_doc.page_content, [wrapper_doc], _citation_offset,
        )
        return content, [wrapper_doc]

    return list_documents_mentioning
```

Wire it in `_resolve_and_build_retriever_tool` (~line 642) — change the `skill-routed` branch:

```python
    if lightrag_mode == "skill-routed":
        skill_assocs = getattr(agent, "skill_associations", None) or []
        if any(getattr(a.skill, "name", None) == LIGHTRAG_ROUTER_SKILL_NAME
               for a in skill_assocs if a.skill):
            # The dynamic tool sets lightrag_query_mode itself, per call.
            return [
                _create_dynamic_lightrag_tool(silo, resolved_sp),
                _create_coverage_tool(silo, agent.app_id),
            ]
        # ponytail: router skill toggled off — fall back transparently to hybrid
        lightrag_mode = "hybrid"
```

And update the caller (~line 519-523) to accept either shape:

```python
        retriever_tool = await asyncio.to_thread(
            _resolve_and_build_retriever_tool, agent, search_params
        )
        if retriever_tool is not None:
            if isinstance(retriever_tool, list):
                tools.extend(retriever_tool)
            else:
                tools.append(retriever_tool)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/tools/test_list_documents_mentioning_tool.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full agentTools test suite to check nothing else broke**

Run: `poetry run pytest tests/unit/tools/ -v`
Expected: all passed (in particular any existing test touching `_resolve_and_build_retriever_tool`'s return shape)

- [ ] **Step 6: Commit**

```bash
git add backend/tools/agentTools.py tests/unit/tools/test_list_documents_mentioning_tool.py
git commit -m "feat(agents): wire list_documents_mentioning for skill-routed agents

Skill-routed LightRAG agents now get a second tool alongside
retrieve_from_knowledge_base, for membership/enumeration questions
that chunk_top_k structurally cannot answer. A doc= hint resolves
through resolve_document_by_name first (commercial name -> resource_id)
before falling back to treating it as a literal document code, and
falls back to searching the whole corpus rather than failing when
neither resolves."
```

---

### Task 4: Extender el router para que use la tool nueva

**Files:**
- Modify: `backend/services/agent_service.py` (añadir Step 0 a `ROUTER_SKILL_CONTENT`)
- Test: `tests/unit/services/test_router_skill_content.py`

**Interfaces:**
- Consumes: nada de código — solo el string `ROUTER_SKILL_CONTENT` ya existente.
- Produces: nada nuevo — modifica el contenido del prompt.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_router_skill_content.py
"""The router prompt is what actually decides whether
list_documents_mentioning ever gets called — a typo or an accidental
deletion here is invisible until someone notices cobertura questions are
still routed to `local`/`mix`. Pin the phrases the routing logic depends on.
"""

from services.agent_service import ROUTER_SKILL_CONTENT


def test_mentions_the_new_tool_by_name():
    assert "list_documents_mentioning" in ROUTER_SKILL_CONTENT


def test_step_0_precedes_the_existing_flowchart_steps():
    content = ROUTER_SKILL_CONTENT
    step0 = content.index("Step 0")
    step1 = content.index("Step 1")
    assert step0 < step1


def test_covers_both_enumeration_shapes():
    """Both "which documents mention X" (cobertura) and "list all X within
    one document" (G07-style) must be recognized — losing either regresses
    a whole eval-set block silently."""
    content = ROUTER_SKILL_CONTENT.lower()
    assert "enumerar" in content or "list all" in content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/services/test_router_skill_content.py -v`
Expected: FAIL — `list_documents_mentioning` not yet mentioned, no "Step 0"

- [ ] **Step 3: Add Step 0 to `ROUTER_SKILL_CONTENT`**

In `backend/services/agent_service.py`, insert right after the `## Decision flowchart` heading and its "Evaluate in order" line, BEFORE the existing `**Step 1 — ...**`:

```
**Step 0 — Does the question ask to enumerate EVERY document where something
appears, or list EVERY instance of something within ONE named document —
with no request to explain, compare, or synthesize?**
  → Yes → call `list_documents_mentioning` instead of
    `retrieve_from_knowledge_base`. Examples: "In which manuals does X
    appear?", "Where is Y mentioned?", "List all the parameters documented
    in manual Z". If the question names a product by its commercial name
    (not a document code) to scope the search, pass it as `doc`.
  → No → continue to Step 1 below.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/services/test_router_skill_content.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/services/agent_service.py tests/unit/services/test_router_skill_content.py
git commit -m "feat(agents): route enumeration questions to list_documents_mentioning

Without this the new tool from the previous commit is wired but
unreachable: the LLM only calls a tool the router prompt tells it
about. Step 0 runs before the existing local/global/hybrid/mix/naive
flowchart so an enumeration question never falls through to a top-k
mode first."
```

---

### Task 5: Flag de re-evaluación selectiva y verificación del criterio de aceptación

**Files:**
- Modify: `scripts/run_eval_rag.py` (nuevo flag `--only-ids`)
- Test: manual — este task no añade tests unitarios nuevos, es la verificación end-to-end del spec.

**Interfaces:**
- Consumes: `run_questions` (ya existe, ver `scripts/run_eval_rag.py`, usado por `retry_failed`).
- Produces: flag CLI `--only-ids ID1,ID2,...` en `main()`, mismo patrón que `--retry-failed` pero sin exigir que las preguntas hayan fallado antes — sirve para re-medir un subconjunto tras un cambio de código, no solo para reintentar errores.

- [ ] **Step 1: Añadir el flag**

En `scripts/run_eval_rag.py`, junto a los demás `parser.add_argument`:

```python
    parser.add_argument(
        "--only-ids", default=None,
        help="Comma-separated question ids to run against an EXISTING result "
             "file, overwriting just those entries (e.g. after a code change "
             "you want to re-measure without paying for a full 97-question run)",
    )
```

Y en el bloque de despacho, junto al de `--retry-failed`:

```python
    if args.only_ids:
        eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        vocab = corpus_vocabulary(eval_set)
        ids = [i.strip() for i in args.only_ids.split(",") if i.strip()]
        return retry_failed(
            SimpleNamespace(**{**vars(args), "retry_failed": args.retry_failed}),
            eval_set, vocab, override_ids=ids,
        )
```

`retry_failed` necesita aceptar `override_ids` opcional — usarlo en vez de derivar `failed_ids` de los errores del fichero cuando se pasa:

```python
def retry_failed(args, eval_set: dict, vocab: list[str], override_ids: Optional[list[str]] = None) -> int:
    out_path = Path(args.retry_failed)
    existing = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in eval_set["preguntas"]}
    if override_ids is not None:
        failed_ids = override_ids
    else:
        failed_ids = [
            r["id"] for r in existing["resultados"]
            if r.get("error") or not r.get("respuesta_agente")
        ]
    if not failed_ids:
        print("Nada que reintentar: ninguna pregunta tiene error/respuesta vacia.", file=sys.stderr)
        return 0
    # ... resto sin cambios
```

Añadir `from types import SimpleNamespace` y `Optional` (ya importado) a los imports de cabecera si no están.

- [ ] **Step 2: Verificar manualmente que no rompe el modo existente**

Run: `poetry run pytest tests/unit -k eval_rag -v` (si no hay tests unitarios de este script, confirmar que no hay ninguno roto por el cambio de firma: `grep -rn "retry_failed(" tests/`)
Expected: sin fallos: no hay callers de `retry_failed` fuera de `main()` que pasen posicionalmente un cuarto argumento distinto.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_eval_rag.py
git commit -m "feat(eval-rag): add --only-ids to re-measure a subset after a code change

retry_failed's mechanism (overwrite matching entries in an existing
result file in place) is exactly what's needed to re-measure cobertura
+ G07 + G09 + cross_documento after this session's router change,
without re-paying for a full 97-question run or requiring those
questions to have errored first."
```

- [ ] **Step 4: Ejecutar el criterio de aceptación del spec**

Requiere un silo LightRAG indexado con el corpus DOMUSA disponible (recrear si el docker se reinició — ver el registro de la sesión: silo `test_new_embedding`, agente `test_pdf`, `rag_k`/`rag_chunk_top_k` a sus valores por defecto).

```bash
python scripts/run_eval_rag.py --api-key <KEY> --timeout 240 \
  --only-ids Q1a,Q2a,Q3a,Q5a,Q6a,Q7a,Q8a,A01,A02,A03,A04,A05,A06,A07,A08,A09,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,G07,G09,D01,D02,D03,D04,D05,D06,D07,D08 \
  --retry-failed benchmark/resultados/resultado_chunk2000_DEFAULT_skill-routed_chunktopk30_norerank_gpt-oss-120b_withprompt_20260831.json
```

Comparar el PASS/PARTIAL/FAIL resultante (evaluación manual, mismo criterio que los informes previos) contra el baseline: `cobertura` 11/6/5, `cross_documento` 2/3/3 (`benchmark/informes/informe_eval_gpt-oss-120b_test_new_embedding_20260831_retry.md`). Documentar el resultado en un nuevo `benchmark/informes/informe_eval_coverage_router_<fecha>.md`.

---

## Self-Review

**Cobertura del spec**: Context/causa raíz → Task 1+3 (bypass de chunk_top_k) y Task 4 (el router deja de mandar estas preguntas a `local`). Non-goals → ninguna task los toca (Q4a/G10 ya fuera del corpus real desde antes de este plan). `list_documents_mentioning` → Task 3. `resolve_document_by_name` (antes fast-follow, ahora en el build) → Task 2, consumida por Task 3. Router Step 0 → Task 4. Límites (tope por doc / tope total) → Task 1 (`per_doc_cap`, `_COVERAGE_QUERY_ROW_CAP`). Errores (abstención, `doc` no resuelto) → Task 3. Testing/criterio de aceptación → Task 5.

**Placeholders**: ninguno — cada step lleva código real, sin "TODO"/"similar a".

**Consistencia de tipos**: `find_chunks_mentioning` devuelve `dict[str, list[tuple[str, str]]]` en Task 1 y Task 3 lo consume exactamente así (`for file_path, content in snippets`). `resolve_document_by_name` devuelve `list[int]` en Task 2 y Task 3 lo indexa como tal (`resource_ids[0]`). `_create_coverage_tool(silo, app_id)` firma igual en Task 3's implementación y en su wiring dentro de `_resolve_and_build_retriever_tool`.

**Alcance**: un solo subsistema (retrieval de cobertura), sin dependencias externas nuevas. No requiere descomponerse en planes separados.

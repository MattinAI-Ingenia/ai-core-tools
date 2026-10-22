# Coverage Chunk Ids — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que los chunks que devuelve la herramienta de cobertura (`list_documents_mentioning`) lleven el `id` de LightRAG, para que el playground pinte en verde los chunks citados y las entidades extraídas de ellos.

**Architecture:** La consulta SQL de cobertura selecciona también `lightrag_doc_chunks.id`; el id viaja como cuarto elemento de cada snippet (`file_path, content, page, chunk_id`) por `_group_chunk_rows` y `augment_with_semantic` hasta un único constructor de chunks en `agentTools.py`, que lo publica como `id` cuando existe. El frontend solo cambia la clave de React de los chips, porque un chunk de cobertura y su gemelo de la búsqueda normal comparten `id`.

**Tech Stack:** Python 3.11, SQLAlchemy (SQL textual), LangChain, pytest + pytest-asyncio, React 18 + TypeScript (eslint).

**Spec:** `docs/superpowers/specs/2026-10-14-coverage-chunk-ids-design.md`

## Global Constraints

- **Rama `fix/coverage-chunk-ids`**, creada desde `lightrag` tras el merge de `fix/lightrag-store-connection-leak`. El árbol tiene ficheros sin trackear que no son de esta tarea (`capturas/`, `pass1_prompt.txt`, `presentacion_mattinai.html`, `resultado.json`) y `.claude/settings.json` modificado: `git add` **solo** de los ficheros listados en cada tarea.
- Entorno de tests: `PY=~/.cache/pypoetry/virtualenvs/ai-core-tools-OqwuKQTr-py3.12/bin/python` (FastAPI 0.142; los otros entornos no importan todo `tests/unit/routers`). Comando: `cd /home/miniguezdeonzono/LKS/ai-core-tools && $PY -m pytest <ruta> -q -p no:cacheprovider -o addopts=""`.
- `/ponytail-review` sobre el diff de cada tarea **antes** de su commit, aplicando o declinando (con motivo) lo que encuentre (CLAUDE.md).
- La numeración de citas es posicional: `cite://N` resuelve a `chunks[N-1]` del payload fusionado del turno. `merge_lightrag_graph` concatena chunks **sin deduplicar** y no se toca.
- El `id` es el de LightRAG de **ese** chunk (`lightrag_doc_chunks.id`, p. ej. `res319-p71-chunk-000`). No se deduce de documento y página (`silo_39` tiene una página con más de un chunk).
- Un snippet sin id conocido **omite la clave `id`** del chunk; nunca `None` ni `""`.
- Los snippets de 3 elementos `(file_path, content, page)` siguen aceptándose en `agentTools.py`: muchos tests existentes parchean `find_chunks_mentioning` con ellos y no se reescriben.
- `_group_chunk_rows` es estricto: sus filas tienen 4 columnas `(full_doc_id, file_path, content, chunk_id)`.
- No se cambia el frontend salvo la clave de React de los chips. No se añade un respaldo en el frontend que busque el gemelo por documento y página.
- Backend desplegado desde imagen: tras el código hay que `docker compose build backend` y recrear el contenedor para probar en `http://localhost/`. `.env` ya tiene `UVICORN_WORKERS=4`, `DB_POOL_SIZE=5`, `DB_MAX_OVERFLOW=5`.

## Review Focus

1. **Una página con varios chunks.** El segundo chunk (`res5-p3-chunk-001`) debe conservar su propio id, no el `-chunk-000` de la página. Test en la Task 1.
2. **Chunk semántico sin `id` en el payload.** El chunk de cobertura debe omitir la clave, no emitir `None`/`""`. Tests en la Task 2: el snippet semántico sin id (`coverage_union`) y el chunk publicado por el tool (parametrizado).
3. **La misma página llega por las dos herramientas.** Dos chunks con el mismo `id` en un payload: se conservan ambos y en su posición (la numeración no se mueve), y los chips no pueden duplicar clave de React. Test en la Task 2 (merge) y comprobación en la Task 3.
4. **Modo "solo documento" (sin `term`, decenas de páginas).** Cada snippet lleva su id y no se recorta. Test en la Task 2.
5. **Snippets heredados de 3 elementos.** No fallan y no inventan un `id`. Test parametrizado en la Task 2, y toda la suite existente de `test_list_documents_mentioning_tool.py` sin tocar.

---

### Task 1: El id del chunk sale de la consulta SQL

**Files:**
- Modify: `backend/tools/vector_stores/lightrag_store.py` (`_group_chunk_rows`, líneas 210-238; SQL y retorno de `find_chunks_mentioning`, líneas ~2545-2612)
- Test: `tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py`

**Interfaces:**
- Consumes: nada.
- Produces: `_group_chunk_rows(rows: list[tuple[str, str, str, str]], per_doc_cap: int = 3) -> dict[str, list[tuple[str, str, int, str]]]`. Fila = `(full_doc_id, file_path, content, chunk_id)`; snippet = `(file_path, content, page, chunk_id)`. `LightRAGStore.find_chunks_mentioning` devuelve ese mismo `grouped`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py`, sustituye estos cuatro tests por las versiones con id y añade los dos nuevos.

```python
def test_groups_snippets_by_resource_id():
    rows = [
        ("res271-p10", "CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí.", "res271-p10-chunk-000"),
        ("res271-p42", "CDOC004043.pdf p.42", "Revisar la válvula de seguridad cada año.", "res271-p42-chunk-000"),
        ("res275-p3", "CDOC001961.pdf p.3", "La válvula de seguridad debe purgarse.", "res275-p3-chunk-000"),
    ]
    grouped = _group_chunk_rows(rows)
    assert set(grouped.keys()) == {"271", "275"}
    assert len(grouped["271"]) == 2
    assert len(grouped["275"]) == 1
    # file_path and page travel with the snippet — Task 3's citations need the
    # real "file.pdf p.N" label, and the frontend's "Open PDF" button needs
    # the page number, not just the resource id. The chunk id travels too: the
    # playground marks a cited chunk by it.
    assert grouped["271"][0] == (
        "CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí.", 10, "res271-p10-chunk-000",
    )


def test_caps_snippets_per_document():
    rows = [
        (f"res271-p{i}", f"CDOC004043.pdf p.{i}", f"mención {i}", f"res271-p{i}-chunk-000")
        for i in range(10)
    ]
    grouped = _group_chunk_rows(rows, per_doc_cap=3)
    assert len(grouped["271"]) == 3


def test_ignores_rows_with_unparseable_doc_id():
    """Non-PDF sources (crawled Domain pages) get a content-hash id, not
    res{N}-p{N} — see _lightrag_doc_id's fallback. Skip them rather than crash."""
    rows = [("doc-abc123hash", "https://example.com/page", "algo de contenido", "doc-abc123hash-chunk-000")]
    grouped = _group_chunk_rows(rows)
    assert grouped == {}


def test_a_page_with_several_chunks_keeps_each_chunks_own_id():
    """The id cannot be derived from document and page: a page can be split in
    several chunks, and the second one is res5-p3-chunk-001, not -chunk-000."""
    rows = [
        ("res5-p3", "D.pdf (p. 3)", "primera parte", "res5-p3-chunk-000"),
        ("res5-p3", "D.pdf (p. 3)", "segunda parte", "res5-p3-chunk-001"),
    ]
    grouped = _group_chunk_rows(rows, per_doc_cap=15)
    assert [s[3] for s in grouped["5"]] == ["res5-p3-chunk-000", "res5-p3-chunk-001"]
```

Y, más abajo, en `test_term_none_does_not_cap_snippets_to_three`, las filas del mock pasan a 4 columnas; después de él añade el test de la consulta:

```python
def test_term_none_does_not_cap_snippets_to_three():
    """The whole point of the doc-only mode is the WHOLE document — the
    default per_doc_cap=3 would silently defeat it."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        (f"res271-p{i}", f"CDOC004043.pdf p.{i}", f"contenido {i}", f"res271-p{i}-chunk-000")
        for i in range(10)
    ]
    grouped, _ = store.find_chunks_mentioning("silo_1", term=None, doc_filter="271")
    assert len(grouped["271"]) == 10
    assert grouped["271"][3][3] == "res271-p3-chunk-000"


def test_the_query_selects_the_lightrag_chunk_id():
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = []
    store.find_chunks_mentioning("silo_1", term="P20", doc_filter=None)
    sql_arg = str(conn.execute.call_args[0][0])
    assert "SELECT full_doc_id, file_path, content, id FROM lightrag_doc_chunks" in sql_arg
```

- [ ] **Step 2: Ejecutar y comprobar que fallan**

Run: `$PY -m pytest tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py -q -p no:cacheprovider -o addopts=""`
Expected: FAIL (`ValueError: too many values to unpack (expected 3)` en los tests de `_group_chunk_rows`, y `AssertionError` en `test_the_query_selects_the_lightrag_chunk_id`).

- [ ] **Step 3: Implementar**

En `backend/tools/vector_stores/lightrag_store.py`, `_group_chunk_rows` pasa a:

```python
def _group_chunk_rows(
    rows: list[tuple[str, str, str, str]], per_doc_cap: int = 3,
) -> dict[str, list[tuple[str, str, int, str]]]:
    """Group (full_doc_id, file_path, content, chunk_id) rows by resource id.

    Rows whose full_doc_id doesn't match the res{N}-p{N} shape (non-PDF
    sources, e.g. crawled Domain pages — see _lightrag_doc_id's content-hash
    fallback) are skipped: there is no resource_id to group them under.

    per_doc_cap: at most this many snippets per document — the FIRST match
    per document already answers "does it appear here", which is the whole
    point; the rest is round-tripped for content, not to be exhaustive here.

    file_path travels alongside each snippet (not just the content) because
    Task 3's citation block needs the real "file.pdf p.N" label, not a
    synthetic one. page travels too — the frontend's "Open PDF at page N"
    button needs it (see agentTools.py's chunk dicts), and it was otherwise
    discarded here despite being right there in full_doc_id.

    chunk_id is LightRAG's own id for the chunk ("res319-p71-chunk-000"), the
    string the graph's source_id fields use: the playground marks a cited chunk
    and the entities extracted from it by matching it. It is read from the row,
    never rebuilt from the page — a page can hold more than one chunk.
    """
    grouped: dict[str, list[tuple[str, str, int, str]]] = {}
    for full_doc_id, file_path, content, chunk_id in rows:
        match = _FULL_DOC_ID_RESOURCE_RE.match(full_doc_id)
        if not match:
            continue
        resource_id, page = match.group(1), int(match.group(2))
        snippets = grouped.setdefault(resource_id, [])
        if len(snippets) < per_doc_cap:
            snippets.append((file_path, content.strip(), page, chunk_id))
    return grouped
```

En `find_chunks_mentioning`: el tipo de retorno pasa a `-> tuple[dict[str, list[tuple[str, str, int, str]]], bool]:`, la consulta a

```python
        sql = "SELECT full_doc_id, file_path, content, id FROM lightrag_doc_chunks WHERE workspace = :ws"
```

y la última línea a

```python
        return _group_chunk_rows([(r[0], r[1], r[2], r[3]) for r in rows], per_doc_cap=per_doc_cap), cap_hit
```

- [ ] **Step 4: Ejecutar y comprobar que pasan**

Run: `$PY -m pytest tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py -q -p no:cacheprovider -o addopts=""`
Expected: PASS (todos).

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/vector_stores/lightrag_store.py tests/unit/tools/vector_stores/lightrag/test_find_chunks_mentioning.py
git commit -m "feat(lightrag): coverage query returns each chunk's own LightRAG id"
```

---

### Task 2: Los chunks de cobertura publican su `id`

**Files:**
- Modify: `backend/tools/coverage_union.py` (tipo `Snippet`, línea 33; candidato semántico, línea ~130)
- Modify: `backend/tools/agentTools.py` (nuevo `_coverage_chunk` justo antes de `_create_coverage_tool`, línea ~154; los tres `chunks.append` de las líneas ~277-316)
- Test: `tests/unit/tools/test_coverage_union.py`, `tests/unit/tools/test_list_documents_mentioning_tool.py`, `tests/unit/tools/test_streaming_utils.py`

**Interfaces:**
- Consumes: snippets `(file_path, content, page, chunk_id)` de la Task 1 (y, por compatibilidad, `(file_path, content, page)`).
- Produces: `Snippet = Tuple[str, str, Optional[int], Optional[str]]` en `coverage_union.py`; `_coverage_chunk(resource_id, snippet) -> dict` en `agentTools.py`, con claves `file_path`, `content`, `resource_id` (int), `page` y, solo si el snippet trae un id no vacío, `id`.

- [ ] **Step 1: Escribir los tests que fallan**

Añade al final de `tests/unit/tools/test_coverage_union.py` (usa los helpers `_run`, `_chunk`, `_semantic`, `_literal`, `_scores` del propio fichero):

```python
def test_a_semantic_only_snippet_carries_the_id_of_its_chunk():
    chunk = {**_chunk(3, "Boiler class (according to EN 303- 5)"), "id": "res3-p1-chunk-000"}
    grouped, _ = _run(augment_with_semantic(
        _literal(), "EN 303-5",
        semantic_chunks_fn=_semantic([chunk]),
        rerank_fn=_scores({"Boiler class (according to EN 303- 5)": 0.9}),
    ))
    assert grouped[3][0] == (
        "DOC3.pdf (p. 1)", "Boiler class (according to EN 303- 5)", 1, "res3-p1-chunk-000",
    )


def test_a_semantic_chunk_without_id_gives_a_snippet_with_no_chunk_id():
    grouped, _ = _run(augment_with_semantic(
        _literal(), "EN 303-5",
        semantic_chunks_fn=_semantic([_chunk(3, "Boiler class (according to EN 303- 5)")]),
        rerank_fn=_scores({"Boiler class (according to EN 303- 5)": 0.9}),
    ))
    assert grouped[3][0][3] is None
```

Añade al final de `tests/unit/tools/test_list_documents_mentioning_tool.py`:

```python
@pytest.mark.asyncio
async def test_every_coverage_chunk_carries_the_lightrag_chunk_id():
    """The playground marks a cited chunk, and the entities whose source_id lists
    its id, by matching that id: a chunk without one can never turn green."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {
        "271": [
            ("CDOC004043.pdf (p. 10)", "aparece SG Ready aquí", 10, "res271-p10-chunk-000"),
            ("CDOC004043.pdf (p. 42)", "otra mención de SG Ready", 42, "res271-p42-chunk-001"),
        ],
        "275": [("CDOC004425.pdf (p. 5)", "también menciona SG Ready", 5, "res275-p5-chunk-000")],
    }
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)),
        patch("services.silo_service.SiloService.resolve_term_variants", return_value=[]),
        patch("tools.agentTools.augment_with_semantic", new=AsyncMock(side_effect=lambda g, *a, **k: (g, {}))),
    ):
        _, artifact = await tool.coroutine(term="SG Ready")
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    # first snippet of each document (the ones the numbered list cites), then the rest
    assert [c["id"] for c in chunks] == ["res271-p10-chunk-000", "res275-p5-chunk-000", "res271-p42-chunk-001"]


@pytest.mark.asyncio
async def test_whole_document_mode_carries_the_chunk_id_of_every_page():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {"271": [
        ("CDOC004043.pdf p.9", "P00...", 9, "res271-p9-chunk-000"),
        ("CDOC004043.pdf p.15", "P81...", 15, "res271-p15-chunk-000"),
    ]}
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)),
        patch("services.silo_service.SiloService.resolve_document_by_name", return_value=[271]),
    ):
        _, artifact = await tool.coroutine(doc="Dual Clima R")
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert [c["id"] for c in chunks] == ["res271-p9-chunk-000", "res271-p15-chunk-000"]


@pytest.mark.asyncio
@pytest.mark.parametrize("snippet", [
    ("CDOC004043.pdf p.10", "aparece SG Ready aquí", 10),                # legacy 3-tuple
    ("CDOC004043.pdf p.10", "aparece SG Ready aquí", 10, None),          # semantic chunk with no id
    ("CDOC004043.pdf p.10", "aparece SG Ready aquí", 10, ""),
])
async def test_a_snippet_without_a_chunk_id_omits_the_id_key(snippet):
    """Never None or "": the frontend builds its set of cited ids from this key."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=({"271": [snippet]}, False)),
        patch("services.silo_service.SiloService.resolve_term_variants", return_value=[]),
        patch("tools.agentTools.augment_with_semantic", new=AsyncMock(side_effect=lambda g, *a, **k: (g, {}))),
    ):
        _, artifact = await tool.coroutine(term="SG Ready")
    chunk = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"][0]
    assert "id" not in chunk
    assert chunk["resource_id"] == 271 and chunk["page"] == 10
```

Añade al final de `tests/unit/tools/test_streaming_utils.py`:

```python
def test_twin_chunks_with_the_same_id_keep_their_positions():
    """cite://N is positional (chunks[N-1]). The coverage tool and the normal
    search can return the same page, hence two chunks with one id in a turn:
    both must stay, in order, or every later citation would point elsewhere."""
    from tools.streaming_utils import merge_lightrag_graph

    first = {"data": {"chunks": [{"id": "res271-p10-chunk-000", "content": "cobertura"}]}}
    second = {"data": {"chunks": [{"id": "res271-p10-chunk-000", "content": "búsqueda"}]}}
    merged = merge_lightrag_graph(first, second)
    assert [c["content"] for c in merged["data"]["chunks"]] == ["cobertura", "búsqueda"]
```

- [ ] **Step 2: Ejecutar y comprobar que fallan**

Run: `$PY -m pytest tests/unit/tools/test_coverage_union.py tests/unit/tools/test_list_documents_mentioning_tool.py tests/unit/tools/test_streaming_utils.py -q -p no:cacheprovider -o addopts=""`
Expected: FAIL en los tests de `coverage_union` (`IndexError`/tupla de 3), en los de `id` del tool (`KeyError: 'id'` y `too many values to unpack` con los snippets de 4); `test_twin_chunks_…` ya pasa (fija el comportamiento actual) y `test_a_snippet_without_a_chunk_id_omits_the_id_key` también.

- [ ] **Step 3: Implementar**

En `backend/tools/coverage_union.py`:

```python
Snippet = Tuple[str, str, Optional[int], Optional[str]]  # (file_path, content, page, chunk_id)
```

y en `augment_with_semantic`, el candidato semántico incluye el id del chunk:

```python
        candidates.append((rid, (c.get("file_path", ""), c.get("content", ""), c.get("page"), c.get("id")), True))
```

En `backend/tools/agentTools.py`, justo antes de `def _create_coverage_tool(`:

```python
def _coverage_chunk(resource_id, snippet) -> dict:
    """Payload chunk for one coverage snippet ``(file_path, content, page[, chunk_id])``.

    ``id`` is LightRAG's own id for the chunk: the playground marks a cited chunk,
    and the entities whose ``source_id`` lists that id, by matching it, so a chunk
    without one can never be marked. It is omitted (never None or "") when the
    snippet does not carry one.
    """
    file_path, content, page, *rest = snippet
    chunk = {"file_path": file_path, "content": content, "resource_id": int(resource_id), "page": page}
    if rest and rest[0]:
        chunk["id"] = rest[0]
    return chunk
```

Y los tres sitios que construían el dict a mano pasan a usarlo. Primer bucle del modo con término (`for resource_id, snippets in grouped.items():` con `snippets[0]`):

```python
                    for resource_id, snippets in grouped.items():
                        file_path, content = snippets[0][:2]
                        n = _citation_offset[0] + len(chunks) + 1
                        flag = {"semantic": " (coincidencia semántica)", "possible": " (posible)"}.get(
                            semantic_tiers.get(resource_id), "")
                        label = f" — {labels[int(resource_id)]}" if int(resource_id) in labels else ""
                        enumerated_lines.append(f"- {file_path}{label} [{n}](cite://{n}){flag}")
                        chunks.append({**_coverage_chunk(resource_id, snippets[0]), "excerpt": _excerpt(file_path, content)})
                    for resource_id, snippets in grouped.items():
                        for snippet in snippets[1:]:
                            chunks.append({**_coverage_chunk(resource_id, snippet), "excerpt": _excerpt(snippet[0], snippet[1])})
```

Modo documento completo (el bucle sin `excerpt`):

```python
                    for resource_id, snippets in grouped.items():
                        for snippet in snippets:
                            chunks.append(_coverage_chunk(resource_id, snippet))
```

- [ ] **Step 4: Ejecutar y comprobar que pasan, y toda la suite del módulo**

Run: `$PY -m pytest tests/unit/tools -q -p no:cacheprovider -o addopts=""`
Expected: PASS (incluidos los tests de `test_list_documents_mentioning_tool.py` y `test_citation_offset_sharing.py` que parchean `find_chunks_mentioning` con snippets de 3 elementos, sin tocarlos).

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/coverage_union.py backend/tools/agentTools.py tests/unit/tools/test_coverage_union.py tests/unit/tools/test_list_documents_mentioning_tool.py tests/unit/tools/test_streaming_utils.py
git commit -m "fix(coverage): publish the LightRAG chunk id on coverage chunks so cited ones can be highlighted"
```

---

### Task 3: Claves de React únicas en los chips de chunks

**Files:**
- Modify: `frontend/src/components/playground/LightRAGGraphBubble.tsx:214`

**Interfaces:**
- Consumes: payloads con `id` repetido (chunk de cobertura y su gemelo) de la Task 2.
- Produces: nada que use otra tarea.

El frontend no tiene tests unitarios (`package.json` solo define `lint` y `build`); esta tarea se verifica con eslint y en la Task 4.

- [ ] **Step 1: Cambiar la clave**

En `LightRAGGraphBubble.tsx`, en el `chunks.map`, sustituye

```tsx
                      key={c.id ?? idx}
```

por

```tsx
                      key={`${c.id ?? 'chunk'}-${idx}`}
```

(Con el id de cobertura, dos chips de la misma página compartirían clave; el índice los distingue. La lista es estática por mensaje, así que no hay reordenación que lo penalice.)

- [ ] **Step 2: Lint**

Run: `cd frontend && npm run lint`
Expected: sin errores nuevos en `LightRAGGraphBubble.tsx`.

- [ ] **Step 3: `/ponytail-review` del diff y commit**

```bash
git add frontend/src/components/playground/LightRAGGraphBubble.tsx
git commit -m "fix(playground): index-qualified React keys for chunk chips (coverage twins share an id)"
```

---

### Task 4: Verificación en vivo y documentación

**Files:**
- Modify: `docs/dependencies/lightrag.md` (sección 10.3, tras el punto "Citas inline")

- [ ] **Step 1: Suite unitaria completa**

Run: `$PY -m pytest tests/unit -q -p no:cacheprovider -o addopts=""`
Expected: PASS (3578 de partida + los tests nuevos; 0 fallos).

- [ ] **Step 2: Reconstruir y levantar el backend**

```bash
cd docker && docker compose build backend && docker compose up -d --no-build backend
until [ "$(docker inspect docker-backend-1 --format '{{.State.Health.Status}}')" = healthy ]; do sleep 5; done
```

- [ ] **Step 3: Comprobar que toda cita apunta a un chunk con id**

Guarda una API key de evaluación válida del App 1 en `EVAL_API_KEY` (no la imprimas ni la pases por línea de comandos) y ejecuta:

```bash
python3 - <<'EOF'
import os, re, sys
sys.path.insert(0, "scripts")
import run_eval_multiling as m

question = "¿Qué documentos mencionan la función antilegionela, en todos los idiomas?"
text, graph = m.ask("http://localhost", 1, 7, os.environ["EVAL_API_KEY"], question, 600)
chunks = graph["data"]["chunks"]
cites = sorted({int(n) for n in re.findall(r"\(cite://(\d+)\)", text)})
missing = [n for n in cites if not chunks[n - 1].get("id")]
print(f"{len(cites)} citas, {len(missing)} sin id: {missing}")
assert not missing, "hay citas a chunks sin id"
EOF
```

Expected: `N citas, 0 sin id: []`. Antes del arreglo, una ejecución típica daba 37-50 citas sin id.

- [ ] **Step 4: Comprobar el playground**

En `http://localhost/`, agente `kr_superagente_citas`, la misma pregunta; expande el bubble de cada silo. Esperado: los chips de los chunks citados salen en verde (también los de las listas "- documento [N]"), hay entidades y relaciones en verde en el subgrafo, y la consola del navegador no muestra avisos de claves duplicadas.

- [ ] **Step 5: Documentar**

En `docs/dependencies/lightrag.md`, sección 10.3, añade tras el punto **Citas inline**:

```markdown
- **Chunks de cobertura con `id`**: `list_documents_mentioning` construye sus
  chunks con `_coverage_chunk` (`agentTools.py`), que publica el `id` de LightRAG
  del chunk (columna `lightrag_doc_chunks.id`, leída por `find_chunks_mentioning`
  y propagada en el cuarto elemento de cada snippet). El playground marca en verde
  un chunk citado y las entidades cuyo `source_id` lo contiene comparando ese `id`;
  sin él, las citas de las listas de cobertura no se pintan. Un chunk de cobertura
  y su gemelo de `retrieve_from_knowledge_base` pueden compartir `id` en un mismo
  payload: la numeración sigue siendo posicional y los chips usan el índice en su
  clave de React.
```

- [ ] **Step 6: Commit**

```bash
git add docs/dependencies/lightrag.md
git commit -m "docs(lightrag): coverage chunks carry the LightRAG chunk id"
```

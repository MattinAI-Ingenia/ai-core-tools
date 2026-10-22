# Dedup Citation Chunks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un chunk que vuelve a salir en otra llamada de tools del mismo turno conserve su número de cita y no se duplique en el payload, y que su línea para el LLM sea corta.

**Architecture:** Un `CitationCounter` (subclase de `list`, así que `counter[0] += 1` sigue valiendo) recuerda qué número recibió cada `id` de chunk. `number_chunks` numera una lista de chunks sobre él y devuelve los chunks nuevos (los que quedan en el payload) y, para cada chunk de entrada, su número y el tipo de línea (`full` o `ref`). `_append_lightrag_citation_sources` y la tool de cobertura lo usan; `IACTTool.run_counter` pasa a ser un `CitationCounter` que se reinicia en cada ejecución.

**Tech Stack:** Python 3.11, LangChain, Pydantic v2, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-14-dedup-citation-chunks-design.md`

## Global Constraints

- **Rama `feat/dedup-citation-chunks`**, creada desde `lightrag`. El árbol tiene ficheros sin trackear que no son de esta tarea (`capturas/`, `pass1_prompt.txt`, `presentacion_mattinai.html`, `resultado.json`) y `.claude/settings.json` modificado: `git add` **solo** de los ficheros listados en cada tarea.
- Entorno de tests: `PY=~/.cache/pypoetry/virtualenvs/ai-core-tools-OqwuKQTr-py3.12/bin/python`. Comando: `cd /home/miniguezdeonzono/LKS/ai-core-tools && $PY -m pytest <ruta> -q -p no:cacheprovider -o addopts=""`.
- `pytest tests/unit/tools` aislado tiene un fallo preexistente que depende del orden (`test_lightrag_variant_persistence.py::TestPersistVariantCounts::test_persists_drained_counts_for_silo`); la suite completa `tests/unit` pasa. En comandos por directorio se excluye con `--deselect`.
- `/ponytail-review` sobre el diff de cada tarea **antes** de su commit, aplicando o declinando (con motivo) lo que encuentre (CLAUDE.md).
- La numeración de citas es posicional: `cite://N` resuelve a `chunks[N-1]` del payload fusionado del turno. Con dedup los números de una ejecución siguen siendo **contiguos** (1..K de chunks únicos), porque `claim_citations` cuenta `len(chunks)` del payload de la ejecución y `rebase_citations(valid=count)` descarta lo que quede fuera.
- Un chunk **sin `id`** (ausente o `""`) nunca se deduplica.
- Una línea de chunk repetido **no se omite**: se escribe corta, `[N] (source: …) (already shown above)`, para que el LLM sepa qué número corresponde al texto que lee. Solo se vuelve a escribir completa si llega una ventana (`excerpt`) y antes no se había mostrado ninguna.
- Un contador que sea una lista plana `[0]` se comporta exactamente como hoy (sin dedup).
- `merge_lightrag_graph`, `rebase_citations`, `claim_citations` y el frontend **no se tocan**.
- `router_offset` (el contador del turno del router) se queda como lista plana: solo se usa como número en `claim_citations`.
- Pydantic v2 convierte un `List[int]` pasado explícitamente en una lista normal y pierde la subclase: el campo `run_counter` se declara `Any` con `default_factory`.

## Review Focus

1. **Chunk sin `id`.** Número nuevo cada vez, sin dedup, también con ids `""`. Test en la Task 1.
2. **Primero la búsqueda normal (inicio de página) y luego cobertura con ventana.** Mismo número, línea completa con la ventana; después otra ventana → línea corta. Al revés (primero ventana, luego inicio de página) → línea corta. Tests en la Task 1.
3. **La lista de cobertura sigue enumerando todos los documentos** con el número ya asignado al repetido, y el payload solo lleva los nuevos. Test en la Task 3.
4. **Llamada en la que todos los chunks son repetidos.** Sigue quitando la lista de referencias propia de LightRAG (si no, reaparece la doble numeración) y conserva la instrucción de citas. Test en la Task 2.
5. **Reinicio por ejecución de especialista.** La ejecución 2 no hereda ids ni números de la 1. Test en la Task 4.

---

### Task 1: `CitationCounter` y `number_chunks`

**Files:**
- Create: `backend/tools/citation_counter.py`
- Test: `tests/unit/tools/test_citation_counter.py`

**Interfaces:**
- Consumes: nada.
- Produces: `CitationCounter(list)` con `reset() -> None` y `claim(chunk: dict) -> Tuple[int, bool, str]` (número, es nuevo, tipo de línea `"full"|"ref"`); `number_chunks(counter, chunks: List[dict]) -> Tuple[List[dict], List[Tuple[int, dict, str]]]`, que devuelve `(kept, numbered)`: `kept` son los chunks nuevos en el orden de entrada y `numbered` tiene una tupla `(número, chunk, tipo)` por cada chunk de entrada, en el mismo orden. Con una lista plana `[0]` numera todos como `"full"` y `kept` es toda la entrada.

- [ ] **Step 1: Escribir los tests que fallan**

Crea `tests/unit/tools/test_citation_counter.py`:

```python
"""CitationCounter: a chunk that comes back in a later tool call of the same run keeps
its number and stays out of the payload; chunks without an id are never merged."""

from tools.citation_counter import CitationCounter, number_chunks


def _chunk(chunk_id, **extra):
    return {"id": chunk_id, "file_path": f"{chunk_id}.pdf", **extra}


def test_new_chunks_get_consecutive_numbers():
    counter = CitationCounter()
    kept, numbered = number_chunks(counter, [_chunk("a"), _chunk("b")])
    assert [(n, kind) for n, _, kind in numbered] == [(1, "full"), (2, "full")]
    assert counter[0] == 2 and len(kept) == 2


def test_a_chunk_that_comes_back_keeps_its_number_and_leaves_the_payload():
    counter = CitationCounter()
    number_chunks(counter, [_chunk("a"), _chunk("b")])
    kept, numbered = number_chunks(counter, [_chunk("b"), _chunk("c")])
    assert [(n, kind) for n, _, kind in numbered] == [(2, "ref"), (3, "full")]
    assert [c["id"] for c in kept] == ["c"]
    assert counter[0] == 3


def test_chunks_without_id_are_never_deduplicated():
    counter = CitationCounter()
    number_chunks(counter, [{"id": "", "file_path": "x"}, {"file_path": "y"}])
    kept, numbered = number_chunks(counter, [{"id": "", "file_path": "x"}])
    assert (numbered[0][0], numbered[0][2]) == (3, "full")
    assert len(kept) == 1


def test_a_window_upgrades_a_chunk_first_shown_without_one():
    counter = CitationCounter()
    number_chunks(counter, [_chunk("a")])  # page start, no window
    kept, numbered = number_chunks(counter, [_chunk("a", excerpt="…el término…")])
    assert (numbered[0][0], numbered[0][2]) == (1, "full")  # same number, the window is shown
    assert kept == []                                       # but the chunk is not added again
    _, again = number_chunks(counter, [_chunk("a", excerpt="otra ventana")])
    assert again[0][2] == "ref"                             # it already had a window


def test_a_chunk_first_shown_with_a_window_is_not_upgraded_by_the_page_start():
    counter = CitationCounter()
    number_chunks(counter, [_chunk("a", excerpt="ventana")])
    _, numbered = number_chunks(counter, [_chunk("a")])
    assert numbered[0][2] == "ref"


def test_reset_starts_numbering_over_and_forgets_ids():
    counter = CitationCounter()
    number_chunks(counter, [_chunk("a")])
    counter.reset()
    kept, numbered = number_chunks(counter, [_chunk("a")])
    assert (numbered[0][0], numbered[0][2]) == (1, "full")
    assert len(kept) == 1


def test_a_plain_list_counter_numbers_every_chunk_as_before():
    counter = [0]
    kept, numbered = number_chunks(counter, [_chunk("a"), _chunk("a")])
    assert [(n, kind) for n, _, kind in numbered] == [(1, "full"), (2, "full")]
    assert counter[0] == 2 and len(kept) == 2
```

- [ ] **Step 2: Ejecutar y comprobar que fallan**

Run: `$PY -m pytest tests/unit/tools/test_citation_counter.py -q -p no:cacheprovider -o addopts=""`
Expected: FAIL en la colección (`ModuleNotFoundError: No module named 'tools.citation_counter'`).

- [ ] **Step 3: Implementar**

Crea `backend/tools/citation_counter.py`:

```python
"""Turn-wide citation counter that remembers which chunk got which number.

Each LightRAG tool call of one run (``retrieve_from_knowledge_base``,
``list_documents_mentioning``) numbers the chunks it returns on a counter they share.
A page comes back in several calls, so without memory every appearance took a new
number and a new payload entry: 139 payload chunks for 74 distinct ones in a real
two-silo question. A chunk that already has a number now keeps it.

``cite://N`` stays positional (``chunks[N-1]`` of the payload): only new chunks enter
the payload, so a run's numbers remain contiguous.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

# "full": the SOURCES line carries the excerpt. "ref": the chunk was already shown, so the
# line only repeats its number and source (the LLM still needs them to cite it).
LINE_FULL = "full"
LINE_REF = "ref"


class CitationCounter(list):
    """``[last_number]``, plus the number each chunk id received.

    A ``list`` on purpose: the numbering code reads and bumps ``counter[0]``, and a plain
    ``[0]`` keeps working everywhere (without dedup).
    """

    def __init__(self) -> None:
        super().__init__([0])
        self._numbers: Dict[str, int] = {}
        self._windowed: Set[str] = set()  # ids already shown with a window around the term

    def reset(self) -> None:
        self[0] = 0
        self._numbers.clear()
        self._windowed.clear()

    def claim(self, chunk: dict) -> Tuple[int, bool, str]:
        """(number, is_new, line kind) for *chunk*. A chunk without an id is always new."""
        chunk_id = chunk.get("id") or None
        has_window = bool(chunk.get("excerpt"))
        if chunk_id is not None and chunk_id in self._numbers:
            # A window is worth a full line when the chunk was only shown from its page start.
            upgrade = has_window and chunk_id not in self._windowed
            if upgrade:
                self._windowed.add(chunk_id)
            return self._numbers[chunk_id], False, LINE_FULL if upgrade else LINE_REF
        self[0] += 1
        if chunk_id is not None:
            self._numbers[chunk_id] = self[0]
            if has_window:
                self._windowed.add(chunk_id)
        return self[0], True, LINE_FULL


def number_chunks(counter: List[int], chunks: List[dict]) -> Tuple[List[dict], List[Tuple[int, dict, str]]]:
    """Number *chunks* on *counter*: ``(kept, numbered)``.

    *kept* is what stays in the payload (the new chunks, input order); *numbered* has one
    ``(number, chunk, line kind)`` per input chunk, in order, for the SOURCES block and for
    the coverage list. A plain ``[0]`` counter numbers every chunk, as before.
    """
    if not isinstance(counter, CitationCounter):
        numbered = []
        for chunk in chunks:
            counter[0] += 1
            numbered.append((counter[0], chunk, LINE_FULL))
        return list(chunks), numbered
    kept: List[dict] = []
    numbered = []
    for chunk in chunks:
        number, is_new, kind = counter.claim(chunk)
        numbered.append((number, chunk, kind))
        if is_new:
            kept.append(chunk)
    return kept, numbered
```

- [ ] **Step 4: Ejecutar y comprobar que pasan**

Run: `$PY -m pytest tests/unit/tools/test_citation_counter.py -q -p no:cacheprovider -o addopts=""`
Expected: PASS (7 tests).

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/citation_counter.py tests/unit/tools/test_citation_counter.py
git commit -m "feat(citations): CitationCounter remembers which chunk got which number"
```

---

### Task 2: La línea de fuentes usa el contador (búsqueda normal)

**Files:**
- Modify: `backend/tools/agentTools.py` (`_append_lightrag_citation_sources`, líneas ~2590-2625; nuevas `_citation_line` y `_citation_block` justo encima; import de `citation_counter`; contador por defecto de `retrieve_from_knowledge_base`, línea ~105)
- Test: `tests/unit/tools/test_lightrag_citation_sources.py`

**Interfaces:**
- Consumes: `number_chunks`, `CitationCounter`, `LINE_REF` de la Task 1.
- Produces: `_citation_line(n: int, chunk: dict, labels: Optional[Dict[int, str]], kind: str) -> str` y `_citation_block(content: str, numbered: List[Tuple[int, dict, str]], labels: Optional[Dict[int, str]]) -> str` (devuelve `content` sin tocar si `numbered` está vacío). `_append_lightrag_citation_sources` conserva su firma y su resultado con contadores planos.

- [ ] **Step 1: Escribir los tests que fallan**

Añade al final de `tests/unit/tools/test_lightrag_citation_sources.py` (y `from tools.citation_counter import CitationCounter` a sus imports):

```python
def _doc_with(*chunks):
    return Document(page_content="ctx", metadata={"lightrag_raw_data": {"data": {"chunks": list(chunks)}}})


def _chunk(chunk_id, path, content):
    return {"id": chunk_id, "file_path": path, "content": content}


def test_a_repeated_chunk_keeps_its_number_gets_a_short_line_and_leaves_the_payload():
    counter = CitationCounter()
    first = _doc_with(_chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "primer texto largo " * 20))
    second = _doc_with(
        _chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "primer texto largo " * 20),
        _chunk("res2-p3-chunk-000", "B.pdf (p. 3)", "otro texto"),
    )
    out1 = _append_lightrag_citation_sources("c1", [first], counter)
    out2 = _append_lightrag_citation_sources("c2", [second], counter)
    assert "[1] (source: A.pdf (p. 1)) primer texto" in out1
    assert "[1] (source: A.pdf (p. 1)) (already shown above)" in out2
    assert "[2] (source: B.pdf (p. 3)) otro texto" in out2
    assert [c["id"] for c in second.metadata["lightrag_raw_data"]["data"]["chunks"]] == ["res2-p3-chunk-000"]
    assert counter[0] == 2


def test_a_call_with_only_repeated_chunks_still_strips_the_lightrag_reference_list():
    counter = CitationCounter()
    _append_lightrag_citation_sources("c1", [_doc_with(_chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "texto"))], counter)
    content = "ctx\n\nReference Document List (Use these as reference):\n\n```\n[1] A.pdf\n```\n"
    out = _append_lightrag_citation_sources(
        content, [_doc_with(_chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "texto"))], counter,
    )
    assert "Reference Document List" not in out          # LightRAG's own numbering must not reach the model
    assert _CITATION_INSTRUCTION in out
    assert "[1] (source: A.pdf (p. 1)) (already shown above)" in out


def test_a_plain_list_counter_never_deduplicates():
    counter = [0]
    doc = _doc_with(_chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "texto"))
    _append_lightrag_citation_sources("c", [doc], counter)
    second = _doc_with(_chunk("res1-p1-chunk-000", "A.pdf (p. 1)", "texto"))
    out = _append_lightrag_citation_sources("c", [second], counter)
    assert "[2] (source: A.pdf (p. 1)) texto" in out
    assert len(second.metadata["lightrag_raw_data"]["data"]["chunks"]) == 1
```

- [ ] **Step 2: Ejecutar y comprobar que fallan**

Run: `$PY -m pytest tests/unit/tools/test_lightrag_citation_sources.py -q -p no:cacheprovider -o addopts=""`
Expected: FAIL en los dos primeros tests nuevos (`[1] … primer texto` aparece como `[2]` y el segundo doc conserva 2 chunks); `test_a_plain_list_counter_never_deduplicates` pasa (fija el comportamiento actual) y los tests existentes también.

- [ ] **Step 3: Implementar**

En `backend/tools/agentTools.py`, importa junto al resto de imports del paquete `tools`:

```python
from tools.citation_counter import CitationCounter, LINE_REF, number_chunks
```

(`CitationCounter` se usa en la Task 3 y 4; añádelo ya.) Sustituye `_append_lightrag_citation_sources` por:

```python
def _citation_line(n: int, chunk: dict, labels: Optional[Dict[int, str]], kind: str) -> str:
    """One SOURCES line. A "ref" line is for a chunk already shown in this run: it keeps the
    number and the source, so the model can still match the text it reads to [N], and drops
    the excerpt it has already seen."""
    src = chunk.get("file_path") or "Unknown source"
    if labels and chunk.get("resource_id") in labels:
        src += f" — {labels[chunk['resource_id']]}"
    if kind == LINE_REF:
        return f"[{n}] (source: {src}) (already shown above)"
    excerpt = chunk.get("excerpt") or " ".join((chunk.get("content") or "").split())[:_CITATION_EXCERPT_CHARS]
    return f"[{n}] (source: {src}) {excerpt}"


def _citation_block(
    content: str, numbered: List[Tuple[int, dict, str]], labels: Optional[Dict[int, str]],
) -> str:
    """*content* + the citation instruction + one SOURCES line per numbered chunk; *content*
    untouched when there is none."""
    if not numbered:
        return content
    content = _strip_lightrag_references(content)
    lines = "\n".join(_citation_line(n, chunk, labels, kind) for n, chunk, kind in numbered)
    return f"{content}\n\n{_CITATION_INSTRUCTION}\n\n{lines}"


def _append_lightrag_citation_sources(
    content: str, docs: List[Document], offset: Optional[List[int]] = None,
    labels: Optional[Dict[int, str]] = None,
) -> str:
    """Append a numbered SOURCES block + citation instruction for LightRAG chunks.

    No-op when *docs* carry no LightRAG chunk provenance (e.g. PGVector/Qdrant
    silos), so it is safe to call on any retriever path. Numbering mirrors
    ``lightrag_raw_data.data.chunks`` order 1:1 to match the chunk order the frontend
    renders and resolves ``cite://N`` against.

    offset: same mutable-cell pattern as ``_call_count`` — pass the SAME counter across
    multiple retrieval calls in one turn so numbering continues (4, 5, 6...) instead of
    restarting at 1 each call. With a ``CitationCounter`` a chunk that already has a number
    keeps it: it gets a short line and is removed from the doc's payload, so ``chunks`` only
    holds new chunks. Omit for the old single-call behavior.

    labels: resource_id -> product label (see SiloService.get_resource_labels),
    appended to each source so the LLM does not guess the product.
    """
    counter = offset if offset is not None else [0]
    numbered: List[Tuple[int, dict, str]] = []
    for doc in docs:
        raw = (doc.metadata or {}).get("lightrag_raw_data") or {}
        data = raw.get("data") or {}
        chunks = data.get("chunks") or []
        kept, doc_numbered = number_chunks(counter, chunks)
        numbered.extend(doc_numbered)
        if len(kept) != len(chunks):
            doc.metadata = {**doc.metadata, "lightrag_raw_data": {**raw, "data": {**data, "chunks": kept}}}
    return _citation_block(content, numbered, labels)
```

Y en `_create_dynamic_lightrag_tool`, el contador por defecto pasa a ser uno con memoria:

```python
    _citation_offset: List[int] = offset if offset is not None else CitationCounter()
```

- [ ] **Step 4: Ejecutar y comprobar que pasan**

Run: `$PY -m pytest tests/unit/tools/test_lightrag_citation_sources.py tests/unit/tools/test_citation_counter.py -q -p no:cacheprovider -o addopts=""`
Expected: PASS (los existentes del fichero, que usan contadores planos o ninguno, sin tocarlos).

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/agentTools.py tests/unit/tools/test_lightrag_citation_sources.py
git commit -m "feat(citations): a chunk repeated across tool calls keeps its number and gets a short source line"
```

---

### Task 3: La tool de cobertura numera una vez y reutiliza números

**Files:**
- Modify: `backend/tools/agentTools.py` (`_create_coverage_tool`: contador por defecto, línea ~183; bloque del modo con término, líneas ~283-300; modo documento completo, líneas ~320-335; ensamblado final, líneas ~336-352)
- Test: `tests/unit/tools/test_list_documents_mentioning_tool.py`

**Interfaces:**
- Consumes: `number_chunks`, `CitationCounter` (Task 1), `_citation_block` (Task 2).
- Produces: el artefacto del tool lleva solo los chunks nuevos; la lista `- documento [N](cite://N)` usa el número asignado (el antiguo si el chunk se repite). Con contador plano, todo igual que hoy.

- [ ] **Step 1: Escribir los tests que fallan**

Añade al final de `tests/unit/tools/test_list_documents_mentioning_tool.py` (con `from tools.citation_counter import CitationCounter` en los imports):

```python
@pytest.mark.asyncio
async def test_a_document_already_numbered_in_this_run_keeps_its_number_in_the_list_and_stays_out_of_the_payload():
    counter = CitationCounter()
    tool = _create_coverage_tool(_fake_silo(), app_id=1, offset=counter)
    page_10 = ("CDOC004043.pdf (p. 10)", "aparece SG Ready aquí", 10, "res271-p10-chunk-000")
    first = {"271": [page_10]}
    second = {"271": [page_10], "275": [("CDOC004425.pdf (p. 5)", "también SG Ready", 5, "res275-p5-chunk-000")]}
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", side_effect=[(first, False), (second, False)]),
        patch("services.silo_service.SiloService.resolve_term_variants", return_value=[]),
        patch("tools.agentTools.augment_with_semantic", new=AsyncMock(side_effect=lambda g, *a, **k: (g, {}))),
    ):
        content_1, artifact_1 = await tool.coroutine(term="SG Ready")
        content_2, artifact_2 = await tool.coroutine(term="SG Ready")
    assert "- CDOC004043.pdf (p. 10) [1](cite://1)" in content_1
    # every document is still listed, the repeated one with its old number
    assert "- CDOC004043.pdf (p. 10) [1](cite://1)" in content_2
    assert "- CDOC004425.pdf (p. 5) [2](cite://2)" in content_2
    assert "[1] (source: CDOC004043.pdf (p. 10)) (already shown above)" in content_2
    chunks_2 = artifact_2[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert [c["id"] for c in chunks_2] == ["res275-p5-chunk-000"]
    assert counter[0] == 2


@pytest.mark.asyncio
async def test_whole_document_mode_also_leaves_repeated_chunks_out_of_the_payload():
    counter = CitationCounter()
    tool = _create_coverage_tool(_fake_silo(), app_id=1, offset=counter)
    grouped = {"271": [
        ("CDOC004043.pdf p.9", "P00...", 9, "res271-p9-chunk-000"),
        ("CDOC004043.pdf p.15", "P81...", 15, "res271-p15-chunk-000"),
    ]}
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)),
        patch("services.silo_service.SiloService.resolve_document_by_name", return_value=[271]),
    ):
        _, artifact_1 = await tool.coroutine(doc="Dual Clima R")
        _, artifact_2 = await tool.coroutine(doc="Dual Clima R")
    assert len(artifact_1[0].metadata["lightrag_raw_data"]["data"]["chunks"]) == 2
    assert artifact_2[0].metadata["lightrag_raw_data"]["data"]["chunks"] == []
    assert counter[0] == 2
```

- [ ] **Step 2: Ejecutar y comprobar que fallan**

Run: `$PY -m pytest tests/unit/tools/test_list_documents_mentioning_tool.py -q -p no:cacheprovider -o addopts=""`
Expected: FAIL en los dos tests nuevos (el segundo documento recibe `[2]`/`[3]` en vez de reutilizar y el payload no se filtra).

- [ ] **Step 3: Implementar**

En `_create_coverage_tool`, el contador por defecto:

```python
    _citation_offset: List[int] = offset if offset is not None else CitationCounter()
```

Modo con término: los chunks se construyen primero y se numeran **una vez**; la lista se arma con los números devueltos. Sustituye desde `enumerated_lines = []` hasta el final de los dos bucles de chunks por:

```python
                    for resource_id, snippets in grouped.items():
                        chunks.append({**_coverage_chunk(resource_id, snippets[0]), "excerpt": _excerpt(snippets[0])})
                    for resource_id, snippets in grouped.items():
                        for snippet in snippets[1:]:
                            chunks.append({**_coverage_chunk(resource_id, snippet), "excerpt": _excerpt(snippet)})
                    # Numbered once here: the list below needs the number each document's first
                    # chunk got (the old one when that chunk already appeared in this run).
                    kept, numbered = number_chunks(_citation_offset, chunks)
                    enumerated_lines = []
                    for (resource_id, snippets), (n, _, _) in zip(grouped.items(), numbered):
                        flag = {"semantic": " (coincidencia semántica)", "possible": " (posible)"}.get(
                            semantic_tiers.get(resource_id), "")
                        label = f" — {labels[int(resource_id)]}" if int(resource_id) in labels else ""
                        enumerated_lines.append(f"- {snippets[0][0]}{label} [{n}](cite://{n}){flag}")
```

(el resto del bloque, desde `summary = f"{len(grouped) - len(semantic_tiers)} documento(s) mencionan…` hasta `body = "\n".join(enumerated_lines)`, queda igual).

Modo documento completo: tras el bucle que llena `chunks`, y antes de `doc_word = …`:

```python
                    kept, numbered = number_chunks(_citation_offset, chunks)
```

(el `summary` sigue usando `len(chunks)`, que es el total de páginas pedidas.)

Ensamblado final, común a los dos modos:

```python
                page_content = f"{summary}\n\n{instruction}" + (f"\n\n{body}" if body else "")
                wrapper_doc = Document(
                    page_content=page_content,
                    metadata={"lightrag_raw_data": {"data": {"chunks": kept}}},
                )
                content = _citation_block(wrapper_doc.page_content, numbered, labels)
```

- [ ] **Step 4: Ejecutar y comprobar que pasan, y toda la suite del módulo**

Run: `$PY -m pytest tests/unit/tools -q -p no:cacheprovider -o addopts="" --deselect tests/unit/tools/vector_stores/test_lightrag_variant_persistence.py::TestPersistVariantCounts::test_persists_drained_counts_for_silo`
Expected: PASS (incluidos `test_citation_offset_sharing.py` y los tests de cobertura existentes, que usan contadores planos).

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/agentTools.py tests/unit/tools/test_list_documents_mentioning_tool.py
git commit -m "feat(coverage): number chunks once per call and reuse the number of a chunk already shown"
```

---

### Task 4: El contador de cada especialista es un `CitationCounter` y se reinicia

**Files:**
- Modify: `backend/tools/agentTools.py` (`IACTTool.run_counter`, línea ~1691; reinicio en `_run_subagent`, línea ~2097; contador por defecto en `_resolve_and_build_retriever_tool`, línea ~1448, y en `get_retriever_tool`, línea ~2872)
- Test: `tests/unit/tools/test_knowledge_router.py`

**Interfaces:**
- Consumes: `CitationCounter` (Task 1).
- Produces: `IACTTool.run_counter` es un `CitationCounter`; `_run_subagent` lo reinicia (números e ids) al empezar cada ejecución en modo router.

- [ ] **Step 1: Escribir el test que falla**

Añade al final de `tests/unit/tools/test_knowledge_router.py` (con `from tools.citation_counter import CitationCounter, number_chunks`):

```python
@pytest.mark.asyncio
async def test_a_specialist_run_forgets_the_chunk_ids_of_the_previous_run(no_writer):
    tool = _router_tool("ES", _CitingReactAgent(["x"]), [0], asyncio.Lock())
    assert isinstance(tool.run_counter, CitationCounter)   # Pydantic must not turn it into a plain list
    number_chunks(tool.run_counter, [{"id": "a"}])         # left over from an earlier run
    await tool._arun("q")
    _, numbered = number_chunks(tool.run_counter, [{"id": "a"}])
    assert numbered[0][2] == "full"                        # forgotten: new again, not a repeat
```

- [ ] **Step 2: Ejecutar y comprobar que falla**

Run: `$PY -m pytest tests/unit/tools/test_knowledge_router.py::test_a_specialist_run_forgets_the_chunk_ids_of_the_previous_run -q -p no:cacheprovider -o addopts=""`
Expected: FAIL en `assert isinstance(tool.run_counter, CitationCounter)` (hoy es `list`).

- [ ] **Step 3: Implementar**

`IACTTool.run_counter`:

```python
    # The sub-agent's own cite counter (shared with its retriever tools), reset on
    # every router-mode run so each run numbers its chunks from 1. Typed Any: Pydantic
    # would turn a List[int] into a plain list and lose the chunk memory.
    run_counter: Any = Field(default_factory=CitationCounter)
```

(comprueba que `Field` ya está importado de `pydantic` en `agentTools.py`; si no, impórtalo.) En `_run_subagent`, sustituye `self.run_counter[0] = 0` por:

```python
            self.run_counter.reset()
```

En `_resolve_and_build_retriever_tool` (`offset: List[int] = citation_counter if citation_counter is not None else [0]`) y en `get_retriever_tool` (`_citation_offset: List[int] = citation_counter if citation_counter is not None else [0]`), el valor por defecto pasa a `CitationCounter()`.

- [ ] **Step 4: Ejecutar y comprobar que pasa, y toda la suite del módulo**

Run: `$PY -m pytest tests/unit/tools -q -p no:cacheprovider -o addopts="" --deselect tests/unit/tools/vector_stores/test_lightrag_variant_persistence.py::TestPersistVariantCounts::test_persists_drained_counts_for_silo`
Expected: PASS.

- [ ] **Step 5: `/ponytail-review` del diff y commit**

```bash
git add backend/tools/agentTools.py tests/unit/tools/test_knowledge_router.py
git commit -m "feat(citations): each specialist run starts with an empty CitationCounter"
```

---

### Task 5: Verificación completa, en vivo y contra el benchmark

**Files:**
- Modify: `docs/dependencies/lightrag.md` (sección 10.3, tras el punto «Chunks de cobertura con `id`»)

- [ ] **Step 1: Suite unitaria completa**

Run: `$PY -m pytest tests/unit -q -p no:cacheprovider -o addopts=""`
Expected: PASS (3591 de partida + los tests nuevos; 0 fallos).

- [ ] **Step 2: Reconstruir y levantar el backend**

```bash
cd docker && docker compose build backend && docker compose up -d --no-build backend
until [ "$(docker inspect docker-backend-1 --format '{{.State.Health.Status}}')" = healthy ]; do sleep 5; done
```

- [ ] **Step 3: El payload no tiene ids repetidos y toda cita apunta a un chunk**

Guarda una API key de evaluación del App 1 en `EVAL_API_KEY` (no la imprimas) y ejecuta desde la raíz del repo:

```bash
python3 - <<'EOF'
import collections, os, re, sys
sys.path.insert(0, "scripts")
import run_eval_multiling as m

question = "¿Qué documentos mencionan la función antilegionela, en todos los idiomas?"
text, graph = m.ask("http://localhost", 1, 7, os.environ["EVAL_API_KEY"], question, 600)
chunks = graph["data"]["chunks"]
ids = [c.get("id") for c in chunks if c.get("id")]
repeated = {k: v for k, v in collections.Counter(ids).items() if v > 1}
cites = sorted({int(n) for n in re.findall(r"\(cite://(\d+)\)", text)})
out_of_range = [n for n in cites if not 1 <= n <= len(chunks)]
print(f"{len(chunks)} chunks en el payload, {len(repeated)} ids repetidos; {len(cites)} citas distintas, {len(out_of_range)} fuera de rango")
assert not repeated and not out_of_range
EOF
```

Expected: `ids repetidos: 0` y `0 fuera de rango`. Antes de este cambio, una ejecución típica daba 139 chunks con 46 páginas repetidas.

- [ ] **Step 4: La calidad no baja (ALL del benchmark)**

Compara las 8 preguntas ALL de la pasada oficial del 2026-10-14 (`benchmark/resultados/multiling/multiling_agent7_official_router_20261014.json`, con sus reintentos `_retry_` y `_retry2_`) contra una pasada nueva del agente 7:

```bash
python3 - <<'EOF'
import json, os, statistics as st, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
import run_eval_multiling as m

ids = [f"ML-ALL-{b}" for b in "UC1-1 UC1-10 UC2-3 UC2-10 UC3-4 UC4-4 UC5-1 UC6-2".split()]
qs = {q["id"]: q for q in json.load(open("benchmark/preguntas/multiling/eval_set_multiling_v1.json"))["preguntas"]}
base = {r["id"]: r for r in json.load(open("benchmark/resultados/multiling/multiling_agent7_official_router_20261014.json"))["results"]}

def run(i):
    text, graph = m.ask("http://localhost", 1, 7, os.environ["EVAL_API_KEY"], qs[i]["pregunta"], 900, {})
    return i, m.score(qs[i], text, graph, 37, 39)

with ThreadPoolExecutor(4) as ex:
    new = dict(ex.map(run, ids))
for i in ids:
    old = base[i]
    print(f"{i}: recall {old.get('doc_recall')} -> {new[i]['doc_recall']} | citas rotas {len(old.get('bad_cites', []))} -> {len(new[i]['bad_cites'])} | chunks {len(old.get('graph_chunks') or [])} -> {new[i]['n_chunks']}")
r_old = [base[i]["doc_recall"] for i in ids if base[i].get("doc_recall") is not None]
r_new = [new[i]["doc_recall"] for i in ids if new[i]["doc_recall"] is not None]
print(f"recall medio {st.mean(r_old):.3f} -> {st.mean(r_new):.3f}")
EOF
```

Expected: 0 citas rotas y recall medio dentro del ruido (±0,08) del anterior; menos chunks por pregunta. Si baja más, no se mergea: se revisa si el LLM dejó de citar los repetidos (línea corta) y se prueba a escribirla completa.

- [ ] **Step 5: Documentar**

En `docs/dependencies/lightrag.md`, sección 10.3, tras el punto **Chunks de cobertura con `id`**, añade:

```markdown
- **Un chunk, un número por ejecución**: `CitationCounter` (`tools/citation_counter.py`)
  recuerda qué número recibió cada `id` de chunk. Si una llamada posterior de las
  tools de LightRAG (búsqueda normal o cobertura) devuelve un chunk que ya salió, conserva
  su número, no entra otra vez en el payload y su línea de fuentes se escribe corta
  (`[N] (source: …) (already shown above)`), salvo que llegue ahora con una ventana
  alrededor del término y antes se hubiera mostrado solo el inicio de la página. Los
  números de cada ejecución siguen siendo contiguos. Un chunk sin `id` no se deduplica.
  El contador de cada especialista se reinicia al empezar su ejecución.
```

- [ ] **Step 6: Commit**

```bash
git add docs/dependencies/lightrag.md
git commit -m "docs(lightrag): one chunk, one citation number per run"
```

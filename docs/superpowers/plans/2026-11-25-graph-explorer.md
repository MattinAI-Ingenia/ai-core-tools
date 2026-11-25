# Graph Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the silo knowledge-graph explorer meet the client requirement: type-colored nodes in both graph views, type filtering, relationship selection with details, source documents (clickable to PDF page), neighbor expansion and node-rooted re-centering.

**Architecture:** Extend the read-only `SiloGraphService` (direct Cypher over Neo4j, workspace-scoped by label `silo_{id}`) with an entity-type filter/counts, a rooted-neighbors query and a chunk-id→document resolver. Rework `SiloGraphView` (global view: type colors + legend-filter, relationship panel, sources chips, expand/center controls) and `LightRAGGraphBubble` (type fill + ring overlays for cited/retrieved).

**Tech Stack:** FastAPI + SQLAlchemy (router/service), Neo4j Cypher (LightRAG 1.5.6 layout: `entity_type` is a **property**, not a label), PGKVStorage chunks, React 18 + TS + `@neo4j-nvl/react` (no native outline — rings via `overlayIcon` SVG data-URIs).

**Spec:** `docs/superpowers/specs/2026-11-25-graph-explorer-design.md` (read it first — D1–D9 carry the decisions)

## Global Constraints

- Workspace isolation is by Neo4j label `silo_{silo_id}`; every Cypher query must keep that label (cross-silo leakage impossible by construction).
- Entity type filters/counters go through the **property** `n.entity_type` (parameterized). Never interpolate a user string into a Cypher label.
- Only entity nodes (`entity_id IS NOT NULL`) and edges between them ever leave the service.
- Caps: canvas never exceeds `max_nodes` (default 200, slider) / `max_edges` (1000).
- Totals (`total_nodes`/`total_edges`, slider max) always reflect the **whole silo**, never the filtered set. Filters apply only to the base graph fetch; `neighbors` fetches unfiltered.
- UI copy in English (matches existing components); code comments only for the why.
- Frontend pure helpers go in `frontend/src/utils/` with vitest tests (`__tests__/*.test.ts`); backend tests follow the mocked-neo4j-session pattern in `tests/unit/services/test_silo_graph_service.py`.

## Review Focus

1. **Entity-type strings with unicode/quotes** (e.g. `Maquinaria@ES`): must survive the filter param and counts unharmed (parameterized Cypher; never a label). Test: Task 1 unit test with hostile type names.
2. **Display-name vs canonical id**: the API only exposes display variants; expansion must canonicalize `node_id` server-side, and two display variants folding to one canonical id must dedupe to one node. Test: Task 2 unit test.
3. **Root not found / canonicalizes to nothing** → HTTP 404 with a clear message, not an empty graph. Test: Task 2.
4. **Legacy chunk ids without the `res{id}-p{page}-chunk-N` shape** (domain pages, old indexes) → KV fallback, else `file_name=None` rendered as "Unknown source", never a crash. Test: Task 3.
5. **Bubble entities without `entity_type`** (partial nodes, some query modes) → gray fill, no ring, no crash. Test: Task 7 component test.

---

### Task 1: Entity-type filter + type counts (backend)

**Files:**
- Modify: `backend/services/silo_graph_service.py` (filter clause + counts; drop `_build_node_label` usage)
- Modify: `backend/routers/internal/silos.py:779-834` (params)
- Modify: `backend/schemas/silo_graph_schemas.py` (response shape)
- Test: `tests/unit/services/test_silo_graph_service.py`

**Interfaces:**
- Consumes: existing `SiloGraphService._cypher_graph` and display-rewrite block (unchanged).
- Produces: `get_silo_graph(silo_id, max_nodes, max_edges, search, entity_types: Optional[List[str]], db)` — `node_label` and `max_depth` params **removed** (dead code, verified never matching); response dict gains `"entity_types": [{"type": str, "count": int}]` sorted by count desc; each node dict gains `"entity_type": Optional[str]` (from `n.entity_type` property). `GraphNode` schema gains `entity_type: Optional[str] = None`; `SiloGraphResponse` gains `entity_types: List[GraphEntityType]`.

- [ ] **Step 1: Write failing tests** in `tests/unit/services/test_silo_graph_service.py` (extend existing mocked-session tests):
  - `test_entity_type_filter_uses_property_param`: called with `entity_types=["Machine", "Maquinaria@ES"]`, the node query binds `entity_types` and the WHERE clause contains `n.entity_type IN $entity_types` — and does **not** contain `` ` `` label interpolation of the type.
  - `test_type_counts_returned`: mocked session's count query returns `[(type, count)]` rows; result `entity_types` is `[{type,count}, ...]` sorted count desc.
  - `test_node_carries_entity_type`: a node with property `entity_type="Machine"` comes back with `entity_type="Machine"`.
- [ ] **Step 2: Run** `pytest tests/unit/services/test_silo_graph_service.py -v` → FAIL.
- [ ] **Step 3: Implement.** Replace the `node_label` label-interpolation branch with `WHERE n.entity_type IN $types` appended to `params`; add one cheap counts query `MATCH (n:`{workspace}`) WHERE n.entity_id IS NOT NULL RETURN n.entity_type AS type, count(*) AS count ORDER BY count DESC`; strip `workspace` from `labels` as today and pull `entity_type` from props. Remove `node_label`/`max_depth` params (they had no working effect) and `_NODE_LABEL_RE`.
- [ ] **Step 4: Run** full unit file → PASS. Also run `pytest tests/unit/services/test_silo_graph_display.py tests/unit/services/test_silo_graph_search_clause.py tests/integration/test_silo_graph_isolation.py tests/integration/test_silo_graph_guards.py -v` → PASS (update any fixture touching removed params).
- [ ] **Step 5: Commit** `git commit -m "feat: entity-type filter and counts for silo graph"` (no `-S`).

### Task 2: Rooted neighbors endpoint (backend)

**Files:**
- Modify: `backend/services/silo_graph_service.py` (new classmethod)
- Modify: `backend/routers/internal/silos.py` (new route next to the graph route)
- Modify: `backend/schemas/silo_graph_schemas.py` if the neighbors response reuses `SiloGraphResponse` (it does — plus a `root` marker is unnecessary)
- Test: `tests/unit/services/test_silo_graph_neighbors.py` (new)

**Interfaces:**
- Consumes: `canonicalize_entity_name` from `tools.vector_stores.lightrag.entity_name_normalization`; the display-rewrite block of Task 1's service (extract it into a shared `_rewrite_display(graph, silo_id, names, db)` helper — same file).
- Produces: `SiloGraphService.get_node_neighbors(silo_id: int, node_id: str, depth: int, max_nodes: int, max_edges: int, db=None) -> Dict[str, Any]` — same dict shape as `get_silo_graph` plus `entity_type`/`entity_types` fields. Router: `GET /{silo_id}/graph/neighbors?node_id=&depth=&max_nodes=` — 409 non-LightRAG, 503 Neo4j down, **404** `Node not found` when the canonical id has no entity node. Frontend (Task 4) calls it via `apiService.getSiloGraphNeighbors`.

- [ ] **Step 1: Write failing tests** (mocked neo4j session):
  - `test_incoming_display_name_canonicalized`: service called with `node_id="MCF-40"` (display variant) issues the root lookup with the canonical id (`canonicalize_entity_name("MCF-40")` — verify with a stub of the normalizer if needed, at minimum assert the bound param differs from the raw string).
  - `test_root_and_depth1_always_kept`: session data where the rooted match returns 500 depth-2 nodes but the limit is 10 → result contains root + all its depth-1 neighbors and 9 depth-2 nodes (limit applied only to deeper layers).
  - `test_entity_only_subgraph`: nodes without `entity_id` and edges touching them are excluded.
  - `test_root_not_found_raises`: lookup returns no row → `LookupError` (router maps to 404).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Algorithm (write the body from this spec): (1) canonicalize `node_id`; (2) root query `MATCH (n:\`{workspace}\`) WHERE n.entity_id = $root AND n.entity_id IS NOT NULL RETURN n` → none → `LookupError`; (3) depth-1 query `MATCH (n)-[r]-(m:\`{workspace}\`) WHERE n.entity_id = $root AND m.entity_id IS NOT NULL` → all neighbors, unlimited; (4) remaining budget = `max_nodes - 1 - len(depth1)`; if depth ≥ 2, fetch deeper nodes `MATCH (root)-[*2..{depth}]-(m:\`{workspace}\`) WHERE m.entity_id IS NOT NULL AND NOT m.entity_id IN $known WITH m, size([(m)-[]-() | 1]) AS degree ORDER BY degree DESC LIMIT $budget`; (5) edges induced among the fetched ids (same edge query as `get_silo_graph` with `$ids`); (6) `truncated = budget was hit`; (7) display rewrite via the shared helper — **and in that helper, after rewriting, merge any two nodes whose display id collides** (same id string from different canonical ids; keep the higher-degree one, union their edges): NVL breaks on duplicate ids.
- [ ] **Step 4: Run** → PASS; rerun Task 1 suites (the helper extraction touches shared code).
- [ ] **Step 5: Commit** `git commit -m "feat: rooted neighbor subgraph endpoint for silo graph"`.

### Task 3: Source-document resolver (backend)

**Files:**
- Create: `backend/services/silo_graph_sources_service.py`
- Modify: `backend/routers/internal/silos.py` (new route)
- Test: `tests/unit/services/test_silo_graph_sources.py` (new)

**Interfaces:**
- Consumes: `CHUNK_ID_RESOURCE_PAGE_RE` from `backend/tools/vector_stores/lightrag_store.py` (chunk id shape `res{id}-p{page}-chunk-N`); `Resource.name` (`models/resource`); LightRAG KV via `LightRAGVectorStore`'s `rag.text_chunks.get_by_ids(ids)` for ids that don't parse (file_path already carries the `(p. N)` label — see `_page_label`).
- Produces: `SiloGraphSourcesService.resolve_sources(silo_id: int, chunk_ids: List[str], db: Session) -> List[Dict[str, Any]]` returning `[{chunk_id, resource_id: Optional[int], page: Optional[int], file_name: Optional[str]}]`, deduped by `(file_name, page)`, input order preserved per group. Router: `GET /{silo_id}/graph/sources?ids=a,b,c` (max 100 ids, 422 beyond) — same 409/503 guards where LightRAG is required for the fallback path; the parse-only path needs no Neo4j. Frontend (Task 4) calls it via `apiService.getSiloGraphSources`.

- [ ] **Step 1: Write failing tests**:
  - `test_res_ids_parsed_without_db`: `["res7-p12-chunk-003", "res9-chunk-001"]` with a stubbed `db.query(Resource)` returning names → `[{chunk_id: "res7-p12-chunk-003", resource_id: 7, page: 12, file_name: "manual.pdf"}, {chunk_id: "res9-chunk-001", resource_id: 9, page: None, file_name: "ficha.pdf"}]` — one SQL query, no LightRAG touched.
  - `test_legacy_id_uses_kv_fallback`: id `doc-abc123` not matching the regex → `rag.text_chunks.get_by_ids(["doc-abc123"])` stubbed returning `[{file_path: "instrucciones.html (p. 3)"}]` → `file_name="instrucciones.html (p. 3)"`, `resource_id=None`.
  - `test_unknown_id_yields_none`: KV returns nothing → `file_name=None`.
  - `test_dedup_same_doc_page`: two chunk ids resolving to the same file+page → one entry.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Parse-first (regex match → resource_id/page), batch the parse misses through the KV, one `db.query(Resource.resource_id.in_(...))` for names. Build the rag instance via the silo's existing LightRAG store path (reuse whatever `SiloService`/`LightRAGVectorStore` construction the playground uses — do not instantiate a second store implementation).
- [ ] **Step 4: Run** → PASS; run `pytest tests/unit/services -v` → all PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: resolve graph node/edge source documents"`.

### Task 4: Frontend API client (thin)

**Files:**
- Modify: `frontend/src/services/api.ts:2581-2593` (`getSiloGraph`) and append two methods.

**Interfaces:**
- Produces:
  - `getSiloGraph(appId, siloId, { maxNodes?, search?, entityTypes?: string[] })` — `node_label`/`max_depth` keys removed; `entityTypes` serialized as repeated `entity_types=` params.
  - `getSiloGraphNeighbors(appId, siloId, { nodeId: string; depth: number; maxNodes?: number })` → `GET .../graph/neighbors?node_id=&depth=&max_nodes=`.
  - `getSiloGraphSources(appId, siloId, chunkIds: string[])` → `GET .../graph/sources?ids=` (comma-joined).
- No tests (HTTP thin client); verify with `cd frontend && npm run build`.

- [ ] **Step 1: Implement the three signatures.**
- [ ] **Step 2: Run** `npm run build` → PASS. Commit `git commit -m "feat: graph explorer API client methods"`.

### Task 5: Type palette + ring icons (frontend)

**Files:**
- Create: `frontend/src/utils/entityTypeColors.ts`
- Create: `frontend/src/utils/__tests__/entityTypeColors.test.ts`

**Interfaces:**
- Produces (consumed by Tasks 6 & 7):
  - `colorForType(type: string | null | undefined): string` — FNV-1a hash of the type → index into a fixed 12-color accessible palette; `null/undefined/""` → `"#94a3b8"` (slate, matches `COLOR_PARTIAL`).
  - `typeRingIcon(color: string, transparent = false): { url: string; position: number[]; size: number }` — `overlayIcon` payload for NVL: an SVG data-URI circle (stroke = the given color, width ~3, fill none) for cited nodes; keep one helper building both green (`#22c55e`) and indigo (`#6366f1`) rings.
  - `NEUTRAL_COLOR = "#6366f1"` — the "coloring off" fill both views use when the toggle is off (spec D1).

- [ ] **Step 1: Write failing vitest tests**: determinism (`colorForType("Machine") === colorForType("Machine")`), null → gray, output within the palette array, `typeRingIcon` returns a data-URI containing `stroke=` and the color, `position` `[0,0]`, `size` `1.15`.
- [ ] **Step 2: Run** `cd frontend && npx vitest run src/utils/__tests__/entityTypeColors.test.ts` → FAIL.
- [ ] **Step 3: Implement** (mirror the style of `systemOneColors.ts`).
- [ ] **Step 4: Run** → PASS. Commit `git commit -m "feat: entity-type palette and NVL ring icons"`.

### Task 6: SiloGraphView rework (frontend)

**Files:**
- Modify: `frontend/src/components/silo/SiloGraphView.tsx` (may split the detail panel into `frontend/src/components/silo/GraphDetailPanel.tsx` — builder's call)
- Create: `frontend/src/utils/siloGraphUtils.ts` (+ `__tests__/siloGraphUtils.test.ts`) — pure helpers, vitest-tested
- Uses: Task 4 API methods, Task 5 palette.

**Interfaces:**
- Produces (pure, in `siloGraphUtils.ts`):
  - `toNvlNodes(nodes: GraphNode[]): NvlNode[]` — fill `colorForType(properties.entity_type)`, caption = display id.
  - `toNvlRels(edges, nodes)` — unchanged behavior, extracted here.
  - `mergeGraphData(base: SiloGraphData, incoming: SiloGraphData): SiloGraphData` — dedupe nodes/edges by id, drop edges whose endpoints are missing, cap total nodes at `base`'s slider max, recompute counts.
  - `neighborsOf(nodeId, edges): { relatedId: string; type: string }[]` — for the connections list.
- State/props of the component: keep `appId`, `siloId`; add `activeTypes: Set<string>`, `selectedEdge`, `sourcesByChunkId: Map<string, SourceDoc[]>`.

- [ ] **Step 1: Write failing vitest tests for `siloGraphUtils`**: `mergeGraphData` dedupes ids, drops dangling edges, caps at max (canvas already at cap → returns base unchanged); `neighborsOf` returns both directions, deduped. Run `npx vitest run src/utils/__tests__/siloGraphUtils.test.ts` → FAIL.
- [ ] **Step 2: Implement the helpers** → vitest PASS.
- [ ] **Step 3: Wire the component.**
  - Colors + toggle: state `typeColoring: boolean` (default on); when off, every node fill is `NEUTRAL_COLOR` (D1) — a toolbar switch next to the legend, no refetch.
  - Legend: chips (from `entity_types` in the response) double as the multi-toggle filter → refetch with `entityTypes` (D2/D3). Chip click toggles; "All" resets.
  - Relationship panel: `onRelationshipClick` → panel with type/keywords/description/weight + source chips (same component as node's).
  - Node panel upgraded: full properties (scrollable), connections list from `neighborsOf` (click a neighbor opens its panel), and source chips.
  - Source chips: on panel open, call `getSiloGraphSources` with the SEP-split `source_id` (chunk ids; reuse `GRAPH_FIELD_SEP` semantics from the bubble — put the splitter in `siloGraphUtils.ts` as `splitSourceIds(sourceId?: string): string[]`); chips render "name (p. N)" or "Unknown source" and use `OpenPdfButton` (`appId`, `siloId`, `resourceId`, `page`) for the PDF-at-page jump.
  - Expand: button "Expand neighbors" (only when the panel node has edges beyond the canvas — check via degree) → `getSiloGraphNeighbors(nodeId, depth=1)` → `mergeGraphData`; if the canvas is already at cap, render a one-line warning in the panel instead (D6/D9).
  - Center: depth selector (1/2/3, friendly labels "Direct neighbors" / "+1 hop" / "+2 hops") + "Center graph here" → `getSiloGraphNeighbors(nodeId, depth)` → **replaces** the graph state (with a loading state on the canvas).
  - Keep: text search, slider (totals semantics unchanged), refresh, error/empty states.
- [ ] **Step 4: Verify** `npm run lint` and `npm run build` PASS; `npx vitest run src/utils/__tests__/siloGraphUtils.test.ts` PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: type-colored, filterable, explorable silo graph view"`.

### Task 7: LightRAGGraphBubble — type fill + rings (frontend)

**Files:**
- Modify: `frontend/src/components/playground/LightRAGGraphBubble.tsx`
- Test: `frontend/tests/LightRAGGraphBubble.test.cjs` (new — follow the pattern of an existing `.test.cjs` component test, e.g. `frontend/tests/BuildCitedGraph.test.cjs`)

**Interfaces:**
- Consumes: Task 5's `colorForType`, `typeRingIcon`; entity `entity_type` field (may be absent → Review Focus #5).

- [ ] **Step 1: Write failing component test** asserting the mapping function output (export the mappers or test via the util): entity with `entity_type: "Machine"` → node color = `colorForType("Machine")`; `partial: true` → gray `#94a3b8`, no `overlayIcon`; entity from a cited chunk → `overlayIcon` green ring; entity without `entity_type` → gray fill, no ring, no throw.
- [ ] **Step 2: Run** `npm test -- tests/LightRAGGraphBubble.test.cjs` → FAIL.
- [ ] **Step 3: Implement.** `toNvlNodes` fill becomes `e.partial ? COLOR_PARTIAL : (typeColoring ? colorForType(e.entity_type ?? null) : NEUTRAL_COLOR)` (state `typeColoring`, default on, toggle next to the legend — no refetch); cited entities add `overlayIcon: typeRingIcon(COLOR_USED)` when `typeColoring` is on (off = the old semantics: green fill for cited, indigo fill for retrieved, no rings — D1 fallback preserves the old look); non-cited get the indigo ring only when coloring is on. Cited edges stay green, non-cited edges stay slate. Legend line becomes: type dots (only types present, `colorForType`) + "ring = cited" explanation; keep the partial entry.
- [ ] **Step 4: Run** component test + `npm run lint` + `npm run build` → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: type-colored subgraph bubble with cited/retrieved rings"`.

### Task 8: End-to-end verification

- [ ] Run backend suites: `pytest tests/unit/services/test_silo_graph_service.py tests/unit/services/test_silo_graph_neighbors.py tests/unit/services/test_silo_graph_sources.py tests/unit/services/test_silo_graph_display.py tests/unit/services/test_silo_graph_search_clause.py tests/integration/test_silo_graph_isolation.py tests/integration/test_silo_graph_guards.py -v` → PASS.
- [ ] Run `cd frontend && npm run lint && npm run build && npm test` → PASS.
- [ ] Manual smoke (needs a LightRAG silo with Neo4j + at least one indexed document): load the silo Graph tab → type colors + legend; toggle a type chip → filtered refetch; click an edge → relationship panel with source chips; click a chip → PDF opens at page; select a node → expand neighbors (merge), center with depth 2 (replace); playground: ask a graph-mode question → bubble nodes type-colored with green rings on cited ones.
- [ ] Run `/ponytail-review` on the branch diff before declaring done (repo rule).

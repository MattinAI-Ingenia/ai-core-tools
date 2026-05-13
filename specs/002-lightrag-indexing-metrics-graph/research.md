# Phase 0 Research & Feasibility Analysis

**Feature**: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

This document answers the user's two explicit feasibility questions:

1. Can we record **exact LLM tokens, indexing time, and monetary cost per document** for LightRAG indexing?
2. Can we add **Neo4j knowledge-graph visualization** to the Silos frontend, and how hard is it?

**Short answer: both are feasible.** Feature 1 is low-to-moderate effort and builds directly on existing infrastructure. Feature 2 is moderate effort, the main cost being the frontend rendering library.

---

## Part 1 — Per-Document Indexing Metrics (tokens / time / cost)

### Where indexing actually happens

LightRAG is integrated as an **embedded Python library** through `LightRAGStore`
([backend/tools/vector_stores/lightrag_store.py](../../backend/tools/vector_stores/lightrag_store.py)).
The LLM that does entity/relationship extraction is **our own adapter**, built in
[backend/tools/vector_stores/lightrag/adapters.py](../../backend/tools/vector_stores/lightrag/adapters.py):

```python
# adapters.py — build_llm_model_func()
async def llm_model_func(prompt, system_prompt=None, history_messages=None, **_kwargs) -> str:
    ...
    response = await llm.ainvoke(messages)   # <-- LangChain chat model
    content = getattr(response, "content", response)
    return str(content)
```

This is the **key feasibility finding**: every LLM call LightRAG makes during indexing
passes through *our* coroutine. LangChain's `ainvoke()` returns an `AIMessage` whose
`response.usage_metadata` (and/or `response.response_metadata["token_usage"]`) carries
**exact** `input_tokens` / `output_tokens` / `total_tokens` reported by the provider. We
do not need LightRAG to expose anything — we already own the interception point.

### Decision: how to capture exact tokens

**Decision**: Wrap the existing `llm_model_func` (and `build_role_llm_configs`) so each call
accumulates `response.usage_metadata` into a per-run accumulator (a `contextvars`-scoped
collector or an object threaded through the store). Sum across all roles
(extract/keyword/query/vlm) for the indexing run.

**Rationale**:
- Provider-reported usage is the most accurate ("exactly how many were spent" — the user's words).
- It requires no LightRAG internals; it lives entirely in code we already maintain.
- It naturally covers all role LLMs because they all go through `build_llm_model_func`.

**Fallback**: When a provider omits usage metadata (some custom/Ollama endpoints do), count
tokens with `tiktoken` on the prompt + completion and **label the metric as "estimated"**.
The estimation machinery already exists in the cost-estimation service.

**Alternatives considered**:
- *LightRAG `TokenTracker` / `llm_response_cache`*: LightRAG 1.5.x ships a `TokenTracker`
  context manager and stores LLM responses in `kv_store_llm_response_cache`. Rejected as the
  primary mechanism because (a) it depends on LightRAG-internal behavior that can change
  between RC releases, and (b) the adapter approach is strictly simpler and equally exact. It
  remains a viable cross-check.
- *Parsing provider dashboards / LangSmith*: out of band, not per-document, rejected.

### Per-document attribution

Indexing is already driven per-resource (see `SiloService.index_*` and the per-resource
indexing flow in [backend/services/silo_service.py](../../backend/services/silo_service.py)).
LightRAG's `ainsert` is called with the texts of **one resource at a time** via
`LightRAGStore.index_documents()`. So scoping the token accumulator around a single
`index_documents()` call yields **per-document** totals. Timing is a simple
`time.perf_counter()` around the same call.

**Decision**: Attribute one `IndexingMetric` row per `(resource_id, run)`. For multi-file
batch indexing, wrap each file's `index_documents()` call independently.

### Cost computation — already have the pieces

[backend/models/pricing_catalog.py](../../backend/models/pricing_catalog.py) stores
`input_price_per_1m`, `output_price_per_1m`, `embedding_price_per_1m` per model + currency.
The cost-estimation feature (see [docs/INGESTION_PROGRESS.md](../../docs/INGESTION_PROGRESS.md)
and `POST /internal/apps/{id}/silos/{id}/estimate-indexing`) already converts tokens → cost
in the app currency.

**Decision**: Reuse the same pricing lookup to turn **actual** recorded tokens into an
**actual** cost. `cost = input_tokens/1e6 * input_price + output_tokens/1e6 * output_price
(+ embedding_tokens/1e6 * embedding_price)`. When the model is absent from the catalog,
store `cost = NULL` and surface "pricing unavailable".

**Rationale**: This turns the existing *estimate* path into an *actuals* path with no new
pricing infrastructure — only a new persistence target.

### Timing & progress

The system already has an `IngestionProgressManager` (SSE) and a documented progress schema
([docs/INGESTION_PROGRESS.md](../../docs/INGESTION_PROGRESS.md)). Per-document wall-clock time
is captured by timing the `index_documents()` call. We can additionally emit the final
metrics on the existing SSE completion event so the UI updates live.

### Persistence

`Resource` ([backend/models/resource.py](../../backend/models/resource.py)) has only a coarse
`status` column — no place for metrics. `Silo` carries LightRAG config but not run metrics.

**Decision**: Add a dedicated `IndexingMetric` table (1 row per indexing run, FK to
`resource_id` / `silo_id` / `app_id`) rather than widening `Resource`. Keeps the hot
`Resource` row small, supports re-indexing history, and works for non-Resource content
(media/domain) later. Requires one Alembic migration.

**Alternatives considered**: JSON column on `Resource` (rejected — no history, harder to
query/aggregate); reuse doc-status KV (rejected — LightRAG-internal, not app-queryable).

### Feasibility verdict — Feature 1

| Question | Answer |
|----------|--------|
| Exact tokens per document? | **Yes** — via provider usage metadata in the existing adapter. |
| Indexing time per document? | **Yes** — wall-clock around the per-resource insert. |
| Cost in money? | **Yes** — reuse `PricingCatalog`; NULL when pricing unknown. |
| Effort | **Low–moderate**: 1 model + migration, adapter token-capture wrapper, metric persistence, 1 read endpoint, 1 UI column/panel. |
| Risk | Low. Main edge: providers that omit usage → labeled estimate via tiktoken. |

---

## Part 2 — Neo4j Knowledge-Graph Visualization in Silos UI

### The graph already exists in Neo4j

LightRAG is configured with `graph_storage="Neo4JStorage"`
([storage_config.py](../../backend/tools/vector_stores/lightrag/storage_config.py)).
Every entity/relationship extracted during indexing is written to Neo4j, **scoped by a
`workspace` property** equal to the silo collection name (`silo_{silo_id}`). The cleanup code
already queries Neo4j by that property:

```python
session.run("MATCH (n) WHERE n.workspace = $ws DETACH DELETE n", ws=collection_name)
```

So the data is present and **already tenant-scoped by workspace** — no new extraction work.

### How to get graph data to the browser — the key design constraint

Neo4j is deployed **internal-network only, no published port** (original plan NFR-4 / FR-5).
Therefore the browser **must not** connect to Neo4j directly. This rules out the naive
`neovis.js` "connect from browser with bolt credentials" pattern.

**Decision**: Backend-proxied graph data. Add a read endpoint that returns
`{ nodes, edges }` JSON, and render it client-side with a Neo4j-ecosystem renderer that
accepts raw node/relationship arrays.

Two ways to source the JSON on the backend (pick at implementation):
1. **LightRAG API**: `LightRAG.get_knowledge_graph(node_label="*", max_depth=N, max_nodes=M)`
   returns a `KnowledgeGraph` with `nodes`/`edges` — this is exactly what LightRAG's own
   WebUI uses. Cleanest, stays within the LightRAG abstraction.
2. **Direct Cypher** against Neo4j scoped by `workspace` (same driver already used for
   cleanup). More control over limits; bypasses LightRAG.

**Recommendation**: Prefer option 1 (`get_knowledge_graph`) with option 2 as fallback if the
installed RC doesn't expose it. Both are low-risk; the driver and connection config already
exist.

### Frontend rendering — the "Neo4j ecosystem"

The user referenced Neo4j's visualization ecosystem. Options, in order of fit:

| Library | Fit | Notes |
|---------|-----|-------|
| **`@neo4j-nvl/react`** (Neo4j Visualization Library, NVL) | **Recommended** | Official Neo4j React component; consumes raw `nodes`/`relationships` arrays (no direct DB needed); pan/zoom/select built in. Aligns with "librería de Neo4J". |
| `neovis.js` | Not recommended | Designed to connect to Neo4j from the browser via bolt — violates the internal-network constraint. |
| `react-force-graph` / `cytoscape` / `sigma.js` | Viable fallback | Generic graph renderers if NVL licensing/bundle size is undesirable. Same backend JSON. |

**Decision**: Use `@neo4j-nvl/react` fed by the backend JSON endpoint. Falls back to a
generic force-graph renderer if NVL proves heavy. Either way the **backend contract is the
same** (`{nodes, edges}`), so the renderer choice is non-blocking.

### Where it slots into the frontend

The Silos UI already has multiple pages:
[SilosPage.tsx](../../frontend/src/pages/SilosPage.tsx),
[SiloPlaygroundPage.tsx](../../frontend/src/pages/SiloPlaygroundPage.tsx),
[SiloForm.tsx](../../frontend/src/components/forms/SiloForm.tsx), and the centralized
`api.ts` client. The Graph view is a new tab/section rendered only when
`silo.vector_db_type === 'LIGHTRAG'`, calling a new `api.ts` method.

### Complexity / effort

| Aspect | Effort | Why |
|--------|--------|-----|
| Backend graph endpoint | Low | Neo4j driver + workspace scoping already exist; or one LightRAG call. |
| Tenant isolation | Low | Reuse `@require_min_role` + existing app/silo ownership checks; scope by workspace. |
| Frontend renderer | **Moderate** | New dependency (NVL), a new tab, node-detail panel, bounded layout. Biggest single chunk of work. |
| Performance bounds | Low | Cap nodes/depth in the endpoint; lazy-load on tab open. |

### Feasibility verdict — Feature 2

| Question | Answer |
|----------|--------|
| Is the graph data available? | **Yes** — already in Neo4j, scoped by `workspace = silo_{id}`. |
| Can we use the Neo4j ecosystem? | **Yes** — `@neo4j-nvl/react`, fed by a backend JSON proxy. |
| How complicated? | **Moderate**, frontend-weighted. Backend is easy; the renderer integration is the main effort. |
| Hard constraint | Browser must **not** hit Neo4j directly → backend proxy endpoint (already designed in). |
| Risk | Low–moderate: large graphs need node/depth caps; NVL bundle size/licensing should be confirmed early. |

---

## Cross-cutting decisions

- **Feature flag**: Both features live behind the existing LightRAG enablement
  (`LIGHTRAG_ENABLED` / `is_lightrag_available()`); no impact when LightRAG is off.
- **No new infra**: Reuses Neo4j, Postgres, PricingCatalog, ingestion-progress SSE, and the
  existing adapter layer. Only additions: one DB table + migration, two read endpoints, one
  frontend dependency.
- **Backward compatibility**: Pre-existing documents show "metrics not recorded"; non-LightRAG
  silos hide the Graph tab.

## Open questions (low-risk, resolve during implementation)

1. Confirm the installed `lightrag-hku==1.5.0rc3` exposes `get_knowledge_graph`; if not, use
   direct Cypher (fallback already specified).
2. Confirm `@neo4j-nvl/react` license terms and bundle size are acceptable for the base
   library; otherwise use a generic force-graph renderer (backend contract unchanged).
3. Whether embedding-token cost is included in the displayed per-document cost or shown
   separately (default: include, with a breakdown tooltip).

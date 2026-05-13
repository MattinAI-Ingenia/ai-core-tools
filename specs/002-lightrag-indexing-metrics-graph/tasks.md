# Tasks: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

**Input**: Design documents from `specs/002-lightrag-indexing-metrics-graph/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/)

**Tests**: Test tasks are included **only** for the security-critical paths (tenant isolation, token recording) because Success Criteria SC-2 and SC-4 require verifiable guarantees. They are marked and can be skipped if you choose a no-test path.

**Organization**: Tasks are grouped by user story. US1 (metrics) is the MVP and is fully independent of US2 (graph).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1 = indexing metrics, US2 = graph visualization
- All paths are repository-relative

## Path Conventions

Web application: backend at `backend/`, frontend at `frontend/src/`, migrations at `alembic/versions/`, tests at `tests/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm LightRAG environment and add the frontend graph dependency.

- [X] T001 Verify LightRAG runtime prerequisites are documented for this feature (`LIGHTRAG_ENABLED=true`, `NEO4J_URI/USERNAME/PASSWORD`, Postgres URI) by cross-checking `backend/tools/vector_stores/lightrag/storage_config.py` and `docker/.env.example`; note any missing var in `specs/002-lightrag-indexing-metrics-graph/research.md` open questions.
- [X] T002 [P] Add the Neo4j visualization dependency `@neo4j-nvl/react` (with a generic force-graph fallback noted) to `frontend/package.json` and run install; confirm it builds in the base library (`frontend/` `npm run build:lib`).
- [X] T003 [P] Confirm `lightrag-hku==1.5.0rc3` exposes `get_knowledge_graph` in the deployed image (run a probe in the backend container); record result and chosen graph-source path (LightRAG API vs direct Cypher) in `specs/002-lightrag-indexing-metrics-graph/research.md`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the persistence and read layer that User Story 1 depends on. Nothing in US1 can land before T004–T006.

**⚠️ CRITICAL**: Complete this phase before starting User Story 1.

- [X] T004 Create the `IndexingMetric` SQLAlchemy model in `backend/models/indexing_metric.py` per [data-model.md](./data-model.md) (fields, FKs to App/Silo/Resource, indexes), and register it in `backend/models/__init__.py`.
- [X] T005 Create the Alembic migration in `alembic/versions/` adding the `indexing_metric` table with FKs and indexes on `(silo_id)`, `(resource_id)`, `(app_id)`; implement and test both `upgrade` and `downgrade` (`alembic upgrade head` then `alembic downgrade -1`).
- [X] T006 [P] Create the `IndexingMetricRepository` in `backend/repositories/indexing_metric_repository.py` with `create(...)`, `get_latest_by_resource(resource_id)`, and `list_latest_by_silo(silo_id, limit, offset)` returning latest run per resource.

**Checkpoint**: Metric table, model, and repository exist — US1 implementation can begin.

---

## Phase 3: User Story 1 - Per-document indexing metrics (Priority: P1) 🎯 MVP

**Goal**: After a document finishes indexing into a LightRAG silo, persist and display its exact LLM token usage, wall-clock time, and monetary cost.

**Independent Test**: Upload one document to a LightRAG silo, index it to completion, and verify the file row shows actual total tokens, duration, and cost (or "pricing unavailable"); verify the GET metrics endpoint returns the same values.

### Tests for User Story 1 (security/accuracy-critical)

- [X] T007 [P] [US1] Integration test in `tests/integration/test_indexing_metrics.py`: indexing a document records an `IndexingMetric` whose `total_tokens == prompt_tokens + completion_tokens` and whose `tokens_source` is `provider` when usage metadata is present (mock the LLM to return `usage_metadata`). Must FAIL before T008–T012.
- [X] T008 [P] [US1] Integration test in `tests/integration/test_indexing_metrics_isolation.py`: GET metrics for a resource under the wrong `app_id`/`silo_id` returns 403/404 and never leaks another app's metric.

### Implementation for User Story 1

- [X] T009 [US1] Add a token-usage capture wrapper in `backend/tools/vector_stores/lightrag/adapters.py`: wrap the coroutine returned by `build_llm_model_func` (and each role in `build_role_llm_configs`) to read `response.usage_metadata` / `response.response_metadata["token_usage"]` and accumulate prompt/completion/total tokens plus call count into a per-run collector (e.g. `contextvars`-scoped accumulator); fall back to a `tiktoken` count labeled `estimated` when usage is absent.
- [X] T010 [US1] Thread the per-run accumulator through `LightRAGStore.index_documents` in `backend/tools/vector_stores/lightrag_store.py`: start/stop the collector around the single-resource `_ainsert_with_progress` call and capture wall-clock duration with `time.perf_counter()`; return the collected totals to the caller.
- [X] T011 [US1] In `backend/services/silo_service.py` (per-resource indexing path, e.g. `index_resource`/`index_multiple_content`), after each document's `index_documents` call, compute cost from `PricingCatalog` (reuse the estimate-cost pricing lookup; NULL when model not found) and persist an `IndexingMetric` row via the repository, including `status` (`success`/`partial`/`failed`), `model_name`, `embedding_model_name`, and `currency`.
- [X] T012 [US1] Ensure failures still record partial usage: wrap the indexing call so an exception persists a `status='failed'` `IndexingMetric` with tokens/time consumed so far, then re-raises.
- [X] T013 [P] [US1] Create the response schema in `backend/schemas/indexing_metric_schemas.py` for the single-resource and silo roll-up payloads per [contracts/indexing-metrics.md](./contracts/indexing-metrics.md).
- [X] T014 [US1] Add `GET /internal/apps/{app_id}/silos/{silo_id}/resources/{resource_id}/indexing-metrics` and `GET /internal/apps/{app_id}/silos/{silo_id}/indexing-metrics` in `backend/routers/internal/silos.py` (or `resources.py`): `@require_min_role(AppRole.VIEWER)`, validate silo↔app ownership via `_validate_silo_app_ownership`, return 204 when no metric exists, and a `totals` roll-up for the silo list.
- [X] T015 [P] [US1] Add `getResourceIndexingMetrics(appId, siloId, resourceId)` and `getSiloIndexingMetrics(appId, siloId)` methods to `frontend/src/services/api.ts`.
- [X] T016 [US1] Create `frontend/src/components/repository/ResourceMetrics.tsx` to render total tokens, indexing duration, and cost (or "pricing unavailable") next to each indexed file; show "metrics not recorded" when the API returns 204.
- [X] T017 [US1] Wire `ResourceMetrics` into the repository/silo file list view (the page that lists resources for a silo) so it appears only for `LIGHTRAG` silos; optionally consume the metric from the existing ingestion-progress `complete` SSE event to update live.

**Checkpoint**: A user can upload, index, and see exact per-document tokens, time, and cost. US1 is independently shippable as the MVP.

---

## Phase 4: User Story 2 - Knowledge-graph visualization (Priority: P2)

**Goal**: Render a LightRAG silo's Neo4j knowledge graph (entities + relationships) in the Silos frontend, scoped strictly to that silo's workspace.

**Independent Test**: Open a LightRAG silo with indexed content, open the Graph tab, and confirm nodes/edges render interactively and that the endpoint returns only `workspace = silo_{id}` data.

### Tests for User Story 2 (security-critical isolation)

- [X] T018 [P] [US2] Integration test in `tests/integration/test_silo_graph_isolation.py`: the graph endpoint returns only nodes/edges where `workspace = silo_{silo_id}`; a second silo in the same app never appears in the first silo's response. Must FAIL before T020–T022.
- [ ] T019 [P] [US2] Integration test in `tests/integration/test_silo_graph_guards.py`: graph endpoint returns 409 for a non-LightRAG silo, 403/404 for wrong app ownership, and 503 when Neo4j is unreachable.

### Implementation for User Story 2

- [X] T020 [P] [US2] Create the graph response schema in `backend/schemas/silo_graph_schemas.py` (`GraphNode`, `GraphEdge`, `SiloGraphResponse` with `truncated`) per [contracts/silo-graph.md](./contracts/silo-graph.md).
- [X] T021 [US2] Create `backend/services/silo_graph_service.py` with `get_silo_graph(silo_id, max_nodes, max_depth, node_label, search)`: prefer `LightRAGStore._get_rag_instance("silo_{id}").get_knowledge_graph(...)`; fall back to direct Cypher via the existing `neo4j` driver filtered by `WHERE n.workspace = $ws LIMIT $max_nodes`. The `workspace = silo_{silo_id}` filter is mandatory on every path; set `truncated` when caps are hit.
- [X] T022 [US2] Add `GET /internal/apps/{app_id}/silos/{silo_id}/graph` in `backend/routers/internal/silos.py`: `@require_min_role(AppRole.VIEWER)`, validate silo↔app ownership, return 409 if `silo.vector_db_type != 'LIGHTRAG'`, 503 on Neo4j connection failure, and the bounded `SiloGraphResponse` otherwise (query params `max_nodes`, `max_depth`, `node_label`, `search`).
- [X] T023 [P] [US2] Add `getSiloGraph(appId, siloId, params)` to `frontend/src/services/api.ts`.
- [X] T024 [US2] Create `frontend/src/components/silo/SiloGraphView.tsx` using `@neo4j-nvl/react` (or the fallback renderer) to render `{nodes, edges}` from the backend, with pan/zoom/select and a node-detail panel showing entity properties and connected relationships; show a friendly error on 503 and a bounded-results notice when `truncated` is true.
- [X] T025 [US2] Add a **Graph** tab/section to the silo view (`frontend/src/pages/SilosPage.tsx` / `SiloPlaygroundPage.tsx`) that renders `SiloGraphView` and is shown only when `silo.vector_db_type === 'LIGHTRAG'`; hide/disable for other backends.

**Checkpoint**: Both user stories work independently; the graph view is isolated per silo.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T026 [P] Update `docs/ai/rag-vector-stores.md` (and/or `docs/INGESTION_PROGRESS.md`) to document per-document metrics and the graph endpoint.
- [X] T027 [P] Add unit test in `tests/unit/test_token_usage_capture.py` for the adapter accumulator: provider-usage path vs tiktoken-estimate fallback.
- [X] T028 Verify backward compatibility: PGVector/Qdrant silos show no Graph tab and no metrics columns, and pre-existing documents show "metrics not recorded" (manual + assert in an integration test).
- [X] T029 Run the `specs/002-lightrag-indexing-metrics-graph/quickstart.md` validation end-to-end against a `LIGHTRAG` silo.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — **blocks US1**.
- **User Story 1 (Phase 3)**: Depends on Foundational. Independently shippable MVP.
- **User Story 2 (Phase 4)**: Depends only on Setup (and shares no code with US1) — can run in parallel with US1 once Phase 1 is done. Its DB/repository needs are none (graph is read-only from Neo4j).
- **Polish (Phase 5)**: After the targeted stories are complete.

### Story Independence

- US1 and US2 touch disjoint files (metrics persistence/UI vs graph service/UI) and can be built by different people in parallel after Phase 1.
- US2 does **not** require Phase 2 (the `IndexingMetric` table); if you want graph-first, do Phase 1 → Phase 4.

### Within Each Story

- Tests (where included) before implementation.
- Model → migration → repository → service → router → `api.ts` → component → page wiring.

### Parallel Opportunities

- Phase 1: T002 and T003 in parallel.
- Phase 2: T006 after T004; T005 after T004.
- US1: T007/T008 in parallel; T013 and T015 in parallel; UI tasks after their endpoint/api.ts.
- US2: T018/T019 in parallel; T020 and T023 in parallel.
- Polish: T026 and T027 in parallel.

---

## Implementation Strategy

- **MVP** = Phase 1 + Phase 2 + Phase 3 (User Story 1). Delivers exact per-document token/time/cost accounting — the primary request — on its own.
- **Increment 2** = Phase 4 (User Story 2): graph visualization.
- **Finish** = Phase 5 polish, docs, and quickstart validation.

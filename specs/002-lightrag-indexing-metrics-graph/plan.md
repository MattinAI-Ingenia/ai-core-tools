# Implementation Plan: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

**Branch**: `lightrag` | **Date**: 2026-06-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-lightrag-indexing-metrics-graph/spec.md`

## Summary

Two additive features on top of the existing LightRAG integration:

1. **Per-document indexing metrics** — capture exact LLM token usage, wall-clock time, and
   monetary cost for each document indexed into a LightRAG silo, and display them next to the
   file once indexing completes. Implemented by wrapping the existing LightRAG LLM adapter to
   accumulate provider-reported token usage, timing the per-resource insert, persisting a new
   `IndexingMetric` row, and reusing `PricingCatalog` for cost.
2. **Knowledge-graph visualization** — a new backend read endpoint returns the silo's Neo4j
   knowledge graph (`{nodes, edges}`, scoped by `workspace = silo_{id}`), rendered in the
   Silos frontend with the Neo4j Visualization Library (`@neo4j-nvl/react`). The browser never
   connects to Neo4j directly; the backend proxies graph data.

Both are gated by the existing LightRAG enablement and have no effect on PGVector/Qdrant silos.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript / React 18 (frontend)

**Primary Dependencies**: FastAPI, SQLAlchemy, Alembic, LangChain, `lightrag-hku==1.5.0rc3`,
`neo4j` driver (already used); frontend adds `@neo4j-nvl/react` (or a generic force-graph
fallback). Reuses `PricingCatalog`, ingestion-progress SSE, and the LightRAG adapter layer.

**Storage**: PostgreSQL (new `IndexingMetric` table; existing pricing/doc-status tables);
Neo4j (existing, read-only for graph view).

**Testing**: pytest (unit + integration); frontend lint (no FE test harness yet).

**Target Platform**: Linux server (Docker single-host with Caddy), modern browsers.

**Project Type**: Web application (FastAPI backend + React frontend library).

**Performance Goals**: Graph endpoint bounded by node/depth caps (default ≤ a few hundred
nodes); metrics read is a single indexed query per resource/silo.

**Constraints**: Neo4j is internal-network only — graph data must be backend-proxied. Features
must be inert when `LIGHTRAG_ENABLED=false`. No regression for non-LightRAG silos.

**Scale/Scope**: Per-silo workspaces; typical knowledge bases < 10k documents.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is an **unfilled template** with
no ratified principles, so there are no binding gates. The plan nonetheless follows repository
conventions from `CLAUDE.md` / `.github/copilot-instructions.md`:

- Business logic in **services**, data access in **repositories**, routing in **routers**.
- DB sessions via dependency injection; role-based access via `@require_min_role`.
- All model changes ship with an Alembic migration (upgrade **and** downgrade tested).
- Frontend HTTP only via `api.ts`; no direct `fetch()`; no base-library changes for
  client-specific needs.
- Tenant isolation: every endpoint scoped by `app_id` / silo ownership.

**Result**: PASS (no violations; no complexity-tracking entries required).

## Project Structure

### Documentation (this feature)

```text
specs/002-lightrag-indexing-metrics-graph/
├── plan.md              # This file
├── research.md          # Phase 0 feasibility analysis
├── data-model.md        # Phase 1 — IndexingMetric + graph response shapes
├── quickstart.md        # Phase 1 — how to exercise both features
├── contracts/           # Phase 1 — endpoint contracts
│   ├── indexing-metrics.md
│   └── silo-graph.md
└── tasks.md             # Phase 2 (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
backend/
├── models/
│   ├── indexing_metric.py          # NEW — IndexingMetric ORM model
│   └── silo.py                     # (existing LightRAG config)
├── schemas/
│   ├── indexing_metric_schemas.py  # NEW — metric response schema
│   └── silo_graph_schemas.py       # NEW — {nodes, edges} response schema
├── repositories/
│   └── indexing_metric_repository.py  # NEW — persistence/read
├── services/
│   ├── silo_service.py             # MODIFY — record metrics around per-resource insert
│   └── silo_graph_service.py       # NEW — fetch graph from LightRAG/Neo4j (workspace-scoped)
├── tools/vector_stores/lightrag/
│   └── adapters.py                 # MODIFY — token-usage capture wrapper in llm_model_func
├── routers/internal/
│   ├── silos.py (or resources.py)  # MODIFY/NEW — GET metrics, GET graph endpoints
└── ...
alembic/versions/
└── xxxx_add_indexing_metric_table.py  # NEW migration

frontend/src/
├── services/api.ts                 # MODIFY — getIndexingMetrics(), getSiloGraph()
├── components/
│   ├── silo/SiloGraphView.tsx      # NEW — NVL graph renderer + node-detail panel
│   └── repository/ResourceMetrics.tsx  # NEW — per-file token/time/cost display
└── pages/
    └── SilosPage.tsx / SiloPlaygroundPage.tsx  # MODIFY — add Graph tab (LIGHTRAG only)
```

**Structure Decision**: Web-application layout (existing). New code follows the established
model → repository → service → router layering on the backend and the `api.ts` + component
pattern on the frontend. Names above are indicative; `/speckit.tasks` will finalize exact files.

## Phasing

- **Feature 1 (P1) first** — delivers standalone cost-accounting value and is lower risk.
- **Feature 2 (P2)** — graph visualization; backend endpoint can land before the frontend
  renderer since the contract is fixed.

## Complexity Tracking

No constitution violations; table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| — | — | — |

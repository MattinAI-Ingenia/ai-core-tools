# Feature Specification: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

**Feature Branch**: `lightrag` (feature dir `002-lightrag-indexing-metrics-graph`)

**Created**: 2026-06-04

**Status**: Draft (planning)

**Input**: User description (ES): "Quiero implementar dos nuevas funcionalidades dentro de LightRAG:
1. Al lado de cada archivo indexado, cuando se acaba su indexado, quiero que se incluya el número de tokens que se han necesitado de LLM exactamente, cuántos se han gastado. Primero haz un análisis de feasibility con LightRAG y su logging. ¿Es posible? Quiero saber: cuánto ha tardado el indexado de ese documento en total, cuántos tokens, y si es posible, cuánto ha valido esa indexación en dinero.
2. Quiero añadir en el frontend de SILOS de LightRAG la visualización del Grafo utilizando la librería de Neo4J. En teoría tienen ya un ecosistema montado para ello; mira a ver también feasibility y cuánto de complicado es añadirlo a la current pipeline."

> This feature builds on the existing LightRAG integration (`plans/lightrag-integration/`,
> already merged on branch `lightrag`). It extends FR-7 (post-indexing usage report) and
> realizes Non-Goal N3 (real-time knowledge-graph visualization) of that original plan.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — See exact indexing cost per document (Priority: P1)

As an app editor who uploaded files to a LightRAG silo, after a file finishes indexing I want to see, next to that specific file, the **exact LLM token usage**, the **total indexing time**, and (when pricing is known) the **monetary cost** that indexing consumed. This lets me understand the real cost of graph-enhanced ingestion per document.

**Why this priority**: This is the primary request and delivers immediate, standalone value — accurate per-document accounting for an expensive (LLM-heavy) operation.

**Independent Test**: Upload a single document to a LightRAG silo, run indexing to completion, and verify the file row shows actual prompt/completion/total tokens, elapsed time, and a cost figure (or "pricing unavailable" when no catalog entry exists).

**Acceptance Scenarios**:

1. **Given** a LightRAG silo with documents, **When** a document finishes indexing, **Then** its row shows actual total tokens (prompt + completion), wall-clock indexing duration, and computed cost in the app's currency.
2. **Given** a model with no pricing-catalog entry, **When** indexing completes, **Then** tokens and time are shown and cost is displayed as "unavailable" rather than a wrong number.
3. **Given** indexing fails partway, **When** the error surfaces, **Then** any tokens already consumed up to the failure are still recorded and shown.
4. **Given** a document indexed before this feature, **When** I view it, **Then** it shows "metrics not recorded" without breaking the UI.

---

### User Story 2 — Visualize the LightRAG knowledge graph in the Silos UI (Priority: P2)

As an app editor on a LightRAG silo, I want a **Graph** view in the Silos frontend that renders the silo's knowledge graph (entities and relationships) using the Neo4j visualization ecosystem, so I can explore the extracted entities and their connections.

**Why this priority**: High-value exploration/debugging capability, but secondary to cost accounting and not required for ingestion to work.

**Independent Test**: Open a LightRAG silo with indexed content, open the Graph view, and confirm nodes (entities) and edges (relationships) render, are pannable/zoomable, and are scoped to that silo only.

**Acceptance Scenarios**:

1. **Given** a LightRAG silo with an extracted graph, **When** I open the Graph view, **Then** entities and relationships render interactively (pan, zoom, node selection).
2. **Given** I select a node, **When** the detail panel opens, **Then** it shows the entity's properties (name, type, description) and connected relationships.
3. **Given** a large graph, **When** I open the view, **Then** results are bounded (node/depth limits) so the browser stays responsive.
4. **Given** a non-LightRAG silo (PGVector/Qdrant), **When** I view it, **Then** the Graph tab is hidden or disabled.
5. **Given** two silos in the same app, **When** I view each graph, **Then** each shows only its own workspace data (tenant isolation).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-1**: Capture exact LLM token usage (prompt, completion, total) consumed during indexing, attributed to the specific document being indexed.
- **FR-2**: Capture embedding token usage during indexing (best-effort; may be estimated when the provider does not report it).
- **FR-3**: Record total wall-clock indexing duration per document.
- **FR-4**: Compute monetary cost from recorded tokens using the existing `PricingCatalog`; degrade gracefully to "unavailable" when no pricing exists.
- **FR-5**: Persist per-document metrics so they survive page reloads and server restarts.
- **FR-6**: Surface the metrics next to each file in the repository/silo UI once indexing completes.
- **FR-7**: Expose a backend endpoint returning a silo's knowledge graph (nodes + edges) scoped to the silo's LightRAG workspace, with bounded size (node/depth limits).
- **FR-8**: Render the knowledge graph in the Silos frontend using the Neo4j visualization ecosystem, with pan/zoom/select interactions and a node-detail panel.
- **FR-9**: Show the Graph view only for `LIGHTRAG` silos; hide/disable for other backends.
- **FR-10**: Enforce tenant isolation — graph and metrics endpoints must validate app/silo ownership and never leak other workspaces' data.

### Key Entities

- **IndexingMetric**: Per-document (or per-indexing-run) record of prompt/completion/total LLM tokens, embedding tokens, duration, computed cost, currency, model name, and status.
- **GraphNode / GraphEdge** (transient response shapes): entity nodes and relationship edges returned by the graph endpoint, scoped by workspace.

## Success Criteria *(mandatory)*

- **SC-1**: After indexing a document, the user sees exact total token count, indexing time, and cost (or a clear "unavailable") within the silo UI.
- **SC-2**: Recorded token totals match the sum of LLM calls made during that document's indexing run (verified against provider usage where available).
- **SC-3**: A LightRAG silo's knowledge graph renders interactively in the Silos UI within a bounded node set, scoped to that silo only.
- **SC-4**: No metrics or graph data from one app/silo are ever visible from another (tenant isolation holds).
- **SC-5**: Existing PGVector/Qdrant silos and pre-existing documents continue to work unchanged.

## Assumptions

- Pricing uses the existing `PricingCatalog` (per-1M-token input/output/embedding prices) and app currency.
- Neo4j is the LightRAG graph backend (`Neo4JStorage`) and remains internal-network only; the browser must never connect to Neo4j directly.
- "Exact" tokens come from provider usage metadata when available; when a provider omits usage, a tokenizer-based count (tiktoken) is used and labeled as such.

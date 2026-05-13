# Data Model

**Feature**: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

## New entity: `IndexingMetric`

One row per indexing run of a document into a LightRAG silo. Supports re-indexing history.

**Table**: `indexing_metric`

| Field | Type | Null | Notes |
|-------|------|------|-------|
| `metric_id` | Integer PK | no | Surrogate key |
| `app_id` | Integer FK → `App.app_id` | no | Tenant scope; indexed |
| `silo_id` | Integer FK → `Silo.silo_id` | no | Indexed |
| `resource_id` | Integer FK → `Resource.resource_id` | yes | Null for non-Resource content (media/domain) |
| `content_ref` | String(1000) | yes | Free-form reference when no `resource_id` |
| `status` | String(20) | no | `success` \| `failed` \| `partial` |
| `prompt_tokens` | Integer | no | Sum of LLM input tokens across all roles; default 0 |
| `completion_tokens` | Integer | no | Sum of LLM output tokens; default 0 |
| `total_tokens` | Integer | no | `prompt + completion`; default 0 |
| `embedding_tokens` | Integer | yes | Best-effort; may be estimated |
| `tokens_source` | String(12) | no | `provider` \| `estimated` (tiktoken fallback) |
| `llm_calls` | Integer | yes | Number of LLM invocations during the run |
| `duration_seconds` | Float | no | Wall-clock indexing time for this document |
| `cost` | Float | yes | Computed cost in `currency`; NULL when pricing unknown |
| `currency` | String(3) | yes | ISO 4217, from app/pricing config |
| `model_name` | String(255) | yes | Extract/primary model used |
| `embedding_model_name` | String(255) | yes | Embedding model used |
| `created_at` | DateTime | no | Server default now |

**Relationships**:
- `IndexingMetric.app` → `App` (many-to-one)
- `IndexingMetric.silo` → `Silo` (many-to-one)
- `IndexingMetric.resource` → `Resource` (many-to-one, optional)

**Indexes**: `(silo_id)`, `(resource_id)`, `(app_id)`, optional `(resource_id, created_at)` to
fetch the latest run per resource quickly.

**Validation rules**:
- `total_tokens == prompt_tokens + completion_tokens` (computed, not user-supplied).
- `cost` is NULL iff the model has no `PricingCatalog` entry; otherwise `cost >= 0`.
- `status='failed'` rows may still carry partial token/time values consumed before failure.

**State / lifecycle**:
- Created when an indexing run for a document completes (success, partial, or failed).
- Re-indexing a document inserts a **new** row (history preserved). UI shows the latest by
  `created_at`.

**Migration**: One Alembic revision creating `indexing_metric` with the FKs and indexes above.
Downgrade drops the table. No changes to existing tables.

### Derived cost formula

```
cost = prompt_tokens     / 1_000_000 * input_price_per_1m
     + completion_tokens / 1_000_000 * output_price_per_1m
     + (embedding_tokens / 1_000_000 * embedding_price_per_1m)   # if embedding pricing known
```

Prices come from `PricingCatalog` (model_name + currency). Missing entry → `cost = NULL`.

## Transient response shapes (Feature 2 — not persisted)

These are Pydantic response schemas for the graph endpoint; the data lives in Neo4j.

### `GraphNode`

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Stable node identifier (entity id/name) |
| `label` | string | Display label (entity name) |
| `type` | string \| null | Entity type/category |
| `properties` | object | Arbitrary entity properties (description, source refs, …) |

### `GraphEdge`

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Edge identifier |
| `source` | string | Source node id |
| `target` | string | Target node id |
| `label` | string \| null | Relationship type/description |
| `properties` | object | Arbitrary relationship properties (weight, keywords, …) |

### `SiloGraphResponse`

| Field | Type | Notes |
|-------|------|-------|
| `silo_id` | integer | Echo of the requested silo |
| `workspace` | string | `silo_{silo_id}` |
| `nodes` | GraphNode[] | Bounded by `max_nodes` |
| `edges` | GraphEdge[] | Edges among returned nodes |
| `truncated` | boolean | True when results were capped |

**Scoping rule**: All nodes/edges are filtered by `workspace = silo_{silo_id}` (the LightRAG
isolation property) so one silo can never return another's graph.

# LightRAG Integration Guide

> Part of [Mattin AI Documentation](../index.md)

## Overview

LightRAG is a graph-enhanced RAG system that combines knowledge graphs (Neo4j) with vector search (Qdrant) to provide deeper document understanding. Unlike pure vector search backends (PGVector/Qdrant), LightRAG extracts entities and relationships from documents using LLM calls, building a knowledge graph that enables more contextual retrieval. It runs as an embedded Python library within the Mattin AI backend — not as a separate service.

LightRAG is the third vector store option alongside PGVector and Qdrant. When a silo uses the `LIGHTRAG` backend, documents are processed through an entity/relationship extraction pipeline and stored in both a Neo4j knowledge graph and a Qdrant vector index.

## Prerequisites

| Component | Purpose | Notes |
|-----------|---------|-------|
| **Neo4j 5 Community Edition** | Knowledge graph storage | Docker service, internal network only |
| **Qdrant** | Vector search | Already part of the standard deployment |
| **PostgreSQL with pgvector** | KV/doc-status storage | Already part of the standard deployment |
| **LLM API key** | Entity extraction during indexing | OpenAI, Anthropic, or compatible provider |
| **Embedding service** | Vector embeddings | Configured in the app |

## Setup

### 1. Configure Environment Variables

Add the following to your `.env` file:

```bash
LIGHTRAG_ENABLED=true
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
```

### 2. Start with the LightRAG Profile

```bash
docker compose --profile lightrag up -d
```

This adds a Neo4j container accessible only on the internal Docker network. No ports are exposed to the host by default.

## Creating a LightRAG Silo

1. Navigate to your **App → Silos → Create New Silo**
2. Select **Database type**: `LIGHTRAG`
3. Select **Embedding Service** — the model used for vector embeddings
4. Select **Indexing AI Service** — the LLM used for entity/relationship extraction during indexing (recommended: GPT-4o or equivalent)
5. Configure **Chunking strategy**:
   - `Token window` (default)
   - `Split by character`
   - `Split by character only`
6. For token window strategy, configure:
   - **Chunk token size** (default: 1200)
   - **Overlap token size** (default: 100)
7. Optionally enable **Graph context in retrieval metadata** (disabled by default)

> **Note:** LightRAG silos cannot be created if Neo4j is not deployed (`LIGHTRAG_ENABLED=true` required).

## Query Modes

LightRAG supports multiple query modes, selectable in the agent's retrieval configuration:

| Mode | Description | Best For |
|------|-------------|----------|
| `local` | Searches entity neighbors in the knowledge graph | Specific entity queries, factual lookups |
| `global` | Uses community summaries from the graph | Broad thematic questions, summaries |
| `hybrid` | Combines local + global (default) | General purpose, balanced retrieval |
| `mix` | All retrieval strategies combined | Maximum coverage, complex queries |
| `naive` | Pure vector search only (no graph) | Simple similarity search, baseline comparison |
| `bypass` | Skips retrieval entirely | Direct LLM responses without RAG |

- **`hybrid`** is the default and recommended mode for most use cases.
- **`naive`** mode is useful for comparing graph-enhanced retrieval against plain vector search.
- **`bypass`** mode skips retrieval entirely — the agent responds using only its system prompt and LLM knowledge.

## Cost Estimation

LightRAG indexing requires LLM calls for entity extraction, making it significantly more expensive than pure vector indexing (PGVector/Qdrant). Use the cost estimation endpoint before indexing large document sets.

### Estimate Endpoint

```bash
POST /internal/silos/{silo_id}/estimate-indexing
```

### Estimate Response

The estimate includes:

| Field | Description |
|-------|-------------|
| **Total chunks** | Number of text chunks after splitting |
| **Estimated LLM calls** | 2 per chunk (entity extraction + relationship extraction) |
| **Estimated embedding calls** | Number of embedding API calls |
| **Estimated input/output tokens** | Token usage for LLM calls |

> **Note:** Cost fields may be `null` if model pricing is not configured for the selected AI service.

## Agent Configuration

1. **Create or edit** an agent
2. **Link** it to a LightRAG silo
3. In **Retrieval Settings**, select the **LightRAG Query Mode** (default: `hybrid`)
4. Configure **Top K results** for the maximum number of results

Traditional vector search settings (Search Strategy, MMR, score threshold) are not shown for LightRAG silos — retrieval behavior is governed by the selected query mode instead.

## Workspace Isolation

Each LightRAG silo maintains strict data isolation:

| Component | Namespace |
|-----------|-----------|
| **LightRAG workspace** | `silo_{silo_id}` |
| **Qdrant collections** | `lightrag_silo_{id}_*` (separate from native Mattin collections `silo_{id}`) |
| **KV and doc-status** | PostgreSQL (not local filesystem) |
| **Neo4j data** | Isolated per workspace |

> LightRAG Qdrant collections use the `lightrag_` prefix to avoid conflicts with native Mattin AI Qdrant collections.

## Limitations

- **Slower indexing** — Entity extraction requires LLM calls per chunk (2 calls per chunk), making indexing significantly slower than pure vector backends.
- **Higher cost** — Each indexed document incurs LLM API costs for entity and relationship extraction.
- **Metadata filtering** — `get_distinct_metadata_values()` returns empty; graph-based retrieval replaces traditional metadata filtering.
- **Document metadata updates** — `update_documents_metadata()` is limited; re-indexing is recommended for metadata changes.
- **Model requirements** — GPT-4o-mini minimum for indexing; GPT-4o recommended for quality extraction.
- **Neo4j required** — LightRAG silos cannot be created if Neo4j is not deployed.

## See Also

- [RAG & Vector Stores](../ai/rag-vector-stores.md) — PGVector and Qdrant backends, silo system, and retrieval
- [LLM Integration](../ai/llm-integration.md) — AI service configuration for indexing LLMs
- [Agent System](../ai/agent-system.md) — How agents use RAG for retrieval

---

## Per-Document Indexing Metrics

After a document finishes indexing into a LightRAG silo, Mattin AI records exact provider-reported LLM token usage, wall-clock duration, and monetary cost in the `indexing_metric` table.

### What is captured

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Input tokens reported by the LLM provider |
| `completion_tokens` | Output tokens reported by the LLM provider |
| `total_tokens` | Sum of prompt + completion |
| `tokens_source` | `provider` (exact) or `estimated` (tiktoken fallback) |
| `llm_calls` | Number of LLM calls made during indexing |
| `duration_seconds` | Wall-clock time for the full indexing run |
| `cost` | Computed cost in USD (null when pricing unavailable) |
| `status` | `success` / `failed` / `partial` |

### How it works

Token counts are captured via a `contextvars`-scoped accumulator in `backend/tools/vector_stores/lightrag/adapters.py`. Each call to `llm_model_func` extracts usage from `response.usage_metadata` (standard LangChain contract); when the provider omits it, tiktoken is used as a fallback and the record is labelled `estimated`.

### API endpoints

```
GET /internal/apps/{app_id}/silos/{silo_id}/resources/{resource_id}/indexing-metrics
→ Latest IndexingMetric for that resource (204 when none exists yet)

GET /internal/apps/{app_id}/silos/{silo_id}/indexing-metrics
→ Latest metric per resource + aggregate totals for the silo
```

### Frontend

Metrics appear inline next to each file in the repository file list (tokens, duration, cost) for LightRAG silos only. No data = silent (no error shown).

---

## Knowledge Graph Visualization

LightRAG silos expose a read-only knowledge graph endpoint for exploring the Neo4j entity/relationship graph from the frontend.

### API endpoint

```
GET /internal/apps/{app_id}/silos/{silo_id}/graph
  ?max_nodes=200   (1–1000, default 200)
  &max_depth=2     (1–5, default 2)
  &node_label=...  (optional Cypher label filter)
  &search=...      (optional full-text entity filter)
→ SiloGraphResponse { nodes, edges, node_count, edge_count, truncated }
→ 409 when silo.vector_db_type != LIGHTRAG
→ 503 when Neo4j is unreachable
```

**Isolation guarantee**: every node and relationship in Neo4j carries a `workspace` property set to `silo_{silo_id}`. The endpoint enforces `WHERE n.workspace = $ws` on every query path — cross-silo data leakage is impossible.

### Frontend

A **Knowledge Graph** tab appears in the Silo Playground for LIGHTRAG silos. It shows a canvas-based force-directed graph with pan, zoom, node selection, and a detail panel. Results are truncated at `max_nodes` with a visible notice.


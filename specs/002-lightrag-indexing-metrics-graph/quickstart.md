# Quickstart

**Feature**: LightRAG Per-Document Indexing Metrics & Knowledge-Graph Visualization

This guide explains how to exercise both features once implemented. Prerequisites: a running
stack with `LIGHTRAG_ENABLED=true`, Neo4j reachable on the internal network, and an `AIService`
+ `EmbeddingService` configured.

## 0. Create a LightRAG silo

1. In the Silos UI, create a silo with **Database type = LIGHTRAG**.
2. Select the extract/query/keyword LLM service(s) and an embedding service.
3. Create a repository linked to the silo.

## 1. Per-document indexing metrics

1. Upload a document (PDF/DOCX/TXT) to the repository.
2. Trigger indexing and wait for completion (watch the existing progress indicator).
3. Verify in the UI, next to the file:
   - **Total tokens** (prompt + completion), exact when the provider reports usage;
   - **Indexing time** (wall-clock seconds);
   - **Cost** in the app currency, or "pricing unavailable" when the model is not in the
     pricing catalog.

Backend check:

```bash
# Latest metric for one resource
curl -s "$BASE/internal/apps/$APP/silos/$SILO/resources/$RES/indexing-metrics" | jq

# Silo roll-up
curl -s "$BASE/internal/apps/$APP/silos/$SILO/indexing-metrics" | jq '.totals'
```

Expected: `tokens_source: "provider"` for OpenAI/Anthropic/Mistral; `"estimated"` for
providers that omit usage metadata.

### Edge cases to verify

- Document indexed before this feature → UI shows "metrics not recorded" (204 from the API).
- Indexing failure → a `status: "failed"` metric still records tokens/time consumed so far.
- Model missing from `PricingCatalog` → tokens/time shown, `cost: null`.

## 2. Knowledge-graph visualization

1. With the silo containing indexed content, open the silo and switch to the **Graph** tab
   (visible only for LightRAG silos).
2. Confirm entities (nodes) and relationships (edges) render and that you can pan, zoom, and
   select nodes.
3. Click a node → a detail panel shows its properties and connected relationships.

Backend check:

```bash
curl -s "$BASE/internal/apps/$APP/silos/$SILO/graph?max_nodes=200&max_depth=2" | jq '{nodes: (.nodes|length), edges: (.edges|length), truncated}'
```

### Edge cases to verify

- Non-LightRAG silo → Graph tab hidden; API returns 409.
- Neo4j unreachable → API returns 503 with a clear message; UI shows a friendly error.
- Two silos in one app → each `/graph` returns only its own `workspace = silo_{id}` data.

## 3. Regression checks

- PGVector / Qdrant silos behave exactly as before (no Graph tab, no metrics columns).
- With `LIGHTRAG_ENABLED=false`, neither endpoint is active and the UI hides both features.

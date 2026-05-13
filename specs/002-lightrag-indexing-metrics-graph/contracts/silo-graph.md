# Contract: Silo Knowledge-Graph API

Internal API group (session/OIDC auth), scoped by app. Available only for `LIGHTRAG` silos.
The browser never connects to Neo4j directly — this endpoint proxies graph data from the
backend, scoped to the silo's LightRAG workspace (`silo_{silo_id}`).

## GET `/internal/apps/{app_id}/silos/{silo_id}/graph`

Return the silo's knowledge graph (entities + relationships), bounded in size.

**Auth**: `@require_min_role(AppRole.VIEWER)`; silo must belong to `app_id`.

**Preconditions**: `silo.vector_db_type == 'LIGHTRAG'`. Otherwise **409 Conflict** with a clear
message ("graph view is only available for LightRAG silos").

**Query params**:

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `max_nodes` | int | 200 | Hard cap to keep the browser responsive |
| `max_depth` | int | 2 | Traversal depth from seed nodes |
| `node_label` | string | `*` | Optional focus entity; `*` = whole graph |
| `search` | string | — | Optional substring match on entity name/label |

**200 Response** (`SiloGraphResponse`):

```json
{
  "silo_id": 3,
  "workspace": "silo_3",
  "nodes": [
    { "id": "ent:Boiler", "label": "Boiler", "type": "EQUIPMENT",
      "properties": { "description": "Industrial boiler unit", "source_id": "doc_5" } }
  ],
  "edges": [
    { "id": "rel:1", "source": "ent:Boiler", "target": "ent:Valve",
      "label": "CONNECTED_TO", "properties": { "weight": 0.82 } }
  ],
  "truncated": false
}
```

**409 Conflict**: silo is not a LightRAG silo.

**403 / 404**: caller lacks access, or silo not found under the app.

**503 Service Unavailable**: Neo4j is not reachable (graceful, with a clear message).

## Implementation notes (non-normative)

- **Primary source**: `LightRAG.get_knowledge_graph(node_label, max_depth, max_nodes)` for the
  silo's workspace instance (`LightRAGStore._get_rag_instance("silo_{id}")`).
- **Fallback source**: direct Cypher via the existing `neo4j` driver, filtered by
  `WHERE n.workspace = $ws`, with `LIMIT $max_nodes`.
- Both paths must enforce the `workspace = silo_{silo_id}` filter — this is the tenant-isolation
  boundary and must never be omitted.
- Response is read-only; no graph mutation endpoints are exposed in this feature.

# Contract: Indexing Metrics API

All endpoints are under the **internal** API group (session/OIDC auth), scoped by app and
require at least `VIEWER` role on the app. Metrics exist only for `LIGHTRAG` silos.

## GET `/internal/apps/{app_id}/silos/{silo_id}/resources/{resource_id}/indexing-metrics`

Return the latest indexing metric for a single resource.

**Auth**: `@require_min_role(AppRole.VIEWER)`; resource must belong to `silo_id` under `app_id`.

**200 Response**:

```json
{
  "metric_id": 12,
  "resource_id": 5,
  "silo_id": 3,
  "status": "success",
  "prompt_tokens": 18432,
  "completion_tokens": 2210,
  "total_tokens": 20642,
  "embedding_tokens": 4096,
  "tokens_source": "provider",
  "llm_calls": 14,
  "duration_seconds": 42.7,
  "cost": 0.0193,
  "currency": "USD",
  "model_name": "gpt-4o-mini",
  "embedding_model_name": "text-embedding-3-small",
  "created_at": "2026-06-04T10:21:03Z"
}
```

**204 No Content**: resource has no recorded metric (e.g. indexed before this feature).

**403 / 404**: caller lacks access, or resource/silo not found under the app.

## GET `/internal/apps/{app_id}/silos/{silo_id}/indexing-metrics`

Return the latest metric per resource for the whole silo (for list/table display), plus an
optional roll-up.

**Query params**: `limit` (default 100), `offset` (default 0).

**200 Response**:

```json
{
  "silo_id": 3,
  "totals": {
    "total_tokens": 184320,
    "cost": 0.21,
    "currency": "USD",
    "documents": 9
  },
  "items": [
    { "resource_id": 5, "total_tokens": 20642, "duration_seconds": 42.7, "cost": 0.0193, "status": "success" }
  ]
}
```

`cost` fields are `null` when pricing is unavailable for the model.

## Notes

- Metrics are produced server-side during indexing (no write endpoint is exposed to clients).
- The existing ingestion-progress SSE completion event MAY include the final metric payload so
  the UI can render it without a follow-up request.

"""
Cross-session user memory ("global memory") — mechanism package.

Per spec `user-memory-mem0` (AD-1), this package holds the small, pure,
independently testable primitives the feature is built on. Orchestration and
business policy live in `services/global_memory_service.py`, not here.

Modules (grown incrementally across the plan's steps):

- `settings.py` — the FR-10 kill switch, all tunables (read through thin
  accessors so call sites never import `config` directly), the shared
  turn-index helper (AD-13), the write/read trigger predicates (D-15/D-17),
  and the one structured logging helper for the feature.
- `identity.py` — the FR-4 eligibility predicate and the `MemoryIdentity` /
  scope-namespace builder (AD-2). This is the *only* place a memory
  namespace may be constructed.
- `resolution.py` — the FR-7 App-level service resolution (AD-5) and the
  D-24 dimensionality probe/cache. DB reads only, no LangGraph/store import.
- `client.py` — the single store seam (AD-9): `GlobalMemoryClient` (async
  `search`/`add`/`similarity`, all `MemoryIdentity`-only) plus the
  module-level, kill-switch-independent `purge`/`list_app_scopes`/
  `list_user_scopes` functions (sync, cascade-triggered erasure/enumeration)
  and `GlobalMemoryPoolService` (the shared `AsyncConnectionPool` + `setup()`
  lifecycle, wired from `backend/main.py`'s FastAPI lifespan). This is the
  **only** module in the codebase that may import `langgraph.store`. Every
  namespace-scoped read/list here re-checks the store's own `LIKE`-based
  namespace matching for exact equality before trusting it — see that
  module's docstring.
"""

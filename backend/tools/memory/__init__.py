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

Not present yet (later steps, see the plan): `resolution.py` (FR-7 service
resolution) and `client.py` (the single store seam). Nothing in this package
imports the LangGraph store.
"""

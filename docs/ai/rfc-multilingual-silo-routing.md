# RFC: Multilingual Routing via Multiple Silos per Agent (M:N)

> Part of [Mattin AI Documentation](../index.md)
> **Status**: Implemented as option C, generalised to any specialist (not only languages): the `is_knowledge_router` flag — see `docs/superpowers/specs/2026-10-07-knowledge-router-design.md` · Recorded September 30, 2026
> **Branch**: `lightrag`

## Use case

Users ask an agent questions in different languages. The corpus is split by language
(e.g. silo 39 = EN, the Domusa silo = ES), each silo a LightRAG graph. Multilingual
questions ("in which EN+ES manuals does X appear?") must be answered by querying several
silos and integrating the results, keeping **citations** and **subgraphs** correct.

## Options considered

### A. Central agent + one sub-agent per language (agent-as-tool) — broken as-is, fixable (see C)

A "Language_Silo_Router" skill would detect the query language and delegate to
sub-agents, each with a single-language silo. It breaks the two things that matter:

- **Citations.** `[N](cite://N)` markers survive as text, but the artifact that resolves
  them (`lightrag_raw_data`: chunks, `file_path`) is dropped at the agent-as-tool boundary:
  `IACTTool._extract_last_message_content` returns only `str(msg.content)`
  (`backend/tools/agentTools.py`). Numbering also collides: each sub-agent starts at `[1]`,
  so `[1]` ES and `[1]` EN are different chunks after merging. Result: citations that look
  right but open the wrong document (same class of bug fixed on 2026-09-08).
- **Subgraphs.** `_emit_subagent_stream_event` only forwards `tool_start`, `tool_end`,
  `thinking`, `code_output`; `_lightrag_graph` never leaves the sub-agent. The sub-agent
  does generate those events (`IACTTool._arun` runs `map_stream_event` on its own stream);
  they are just dropped. Since `IACTTool` returns plain text, the parent `ToolMessage` has
  no artifact either, so the graph is also missing when the conversation is reloaded
  from history (`agent_cache_service.py`).

### B. One agent with N silos (one retrieval tool per silo) — chosen direction

Keep the tools inside the same agent so nothing crosses the `IACTTool` boundary:

- Citations: the shared `offset` cell + `asyncio.Lock` pattern (`agentTools.py`, coverage
  router) is silo-agnostic; generalise it from 2 fixed tools to a list of tools.
- Graph: `merge_lightrag_graph` already accumulates several calls per turn (its docstring
  already mentions multi-silo). Entities dedup by id (= entity name), so "Quemador" and
  "Burner" never collapse: the payload is two disconnected clusters, no cross-language
  contamination.
- Documents: each chunk/entity carries `file_path` and a `resource_id`/page-encoded
  `chunk_id`, so `[N]` always resolves to the real PDF regardless of source silo.
- Missing: no `silo_id`/language field in the graph payload. Tag `graph_data` with its
  source silo before the merge if the frontend should show labelled per-language clusters.

### C. Option A made to work: sub-agents that return an artifact — viable, unprototyped

Keep sub-agents and fix the boundary instead of adding M:N:

1. **Subgraphs (cheap, low risk, ~20 lines).** `IACTTool` uses
   `response_format="content_and_artifact"`. While streaming the sub-agent it accumulates
   its `_lightrag_graph` events with `merge_lightrag_graph` and returns
   `(text, [Document(metadata={"lightrag_raw_data": merged})])`. The parent `ToolMessage`
   then looks like any direct LightRAG tool result, so the existing live merge
   (`agent_streaming_service.py`) and history reconstruction work untouched. Result: one
   graph payload with one disconnected cluster per silo (tag by language needs an extra field).
2. **Citation numbering.** Each sub-agent numbers from `[1]`, so numbers must be rebased
   at the boundary. Options:

   | Option | Pros | Cons |
   |---|---|---|
   | Shared offset + `asyncio.Lock` across sub-agents (existing pattern) | Proven pattern | Serialises whole sub-agents, not just LightRAG calls: languages no longer run in parallel, latency roughly doubles |
   | Claim offset when each sub-agent finishes, no Lock, rewrite `cite://N` -> `cite://N+base` | Keeps parallelism | Assumes finish order == chunk merge order — same class of assumption behind the 2026-09-08 bug. Plausible but **unverified**; needs a test |
   | Namespaced keys (`cite://<tool_call_id>:N`) resolved by key, not position | Order-independent, keeps parallelism, could also remove the current Lock | Frontend change (`MessageContent.tsx`, `ChatInterface.tsx`, `LightRAGGraphBubble.tsx` resolve by index today) + a key on each chunk |

   Namespaced keys are the preferred option if this goes ahead.
3. **Not fixable in code: the second LLM.** With one agent, the LLM writing the answer sees
   the SOURCES block and cites directly. With sub-agents, the super-agent only sees text
   already written by each sub-agent and must copy its markers without the sources:
   it can drop them, glue a citation onto the wrong claim when merging sentences, or
   invent a non-existent `cite://7` (the last one can be filtered by validating against
   the merged chunks; the others cannot). The sub-agent can also mis-cite, so there are
   two error hops. Also more tokens/latency (one LLM loop per sub-agent + synthesis).
   How much worse citations get can only be measured with the eval.

Trade-off vs B: C needs no migration and lets each language silo keep its own agent
(own prompt, own RAG config — which answers the shared-vs-per-silo config question), at
the price of less reliable citations and higher cost.

### Why separate silos rather than one mixed silo

The graph would not be contaminated either way. The real reason is that **language config
and entity types are per silo**: a mixed silo extracts EN PDFs with the ES extraction
prompt and single-language entity classes, degrading extraction.

## Why M:N is required

`Agent.silo_id` is a simple FK (N:1, optional) (`backend/models/agent.py`), and
`_resolve_and_build_retriever_tool` builds tools from `agent.silo` (singular). One tool per
silo inside the same agent needs an `agent_silos` association table.

## Scope of work (several layers, not a small patch)

| Layer | Change |
|---|---|
| DB | Migration `Agent.silo_id` -> `agent_silos` (M:N), downgrade tested |
| Model / `agent_service.py` | `agent.silo` -> `agent.silos`; `resolve_search_params`, RAG config precedence |
| `agentTools.py` | N silos -> N tools sharing offset/lock |
| `lightrag_store.py` | Tag `graph_data` with origin silo/language |
| Skill | New "Language Router" skill (pattern of `ensure_/cleanup_lightrag_router_skill`) so the LLM picks silo(s) |
| Schemas + API | Agent create/edit with several silos |
| Frontend | Silo selector single -> multi-select |
| Tests | Several unit tests assume singular `agent.silo` (`test_resolve_search_params.py`, `test_silo_service_search_params.py`, `test_repository_silo_field_forwarding.py`, ...) |

## Open questions / costs

- **RAG config** (`rag_k`, `rag_chunk_top_k`, ...): shared across silos or per silo? Affects the schema.
- **Latency**: the Lock serialises tool calls, so a multilingual turn queries silos in
  sequence instead of in parallel. Fine for 2 languages, painful at 4-5. Revisit with the
  inference-time improvements work.

## Decision

Deferred. First finish the EN silo (39), generate the EN eval set and measure the EN agent
alone. Only if real questions cross languages, choose between B (M:N) and C (sub-agents
with artifact). Cheapest way to decide: prototype C (points 1 + rebased numbering, behind a
test) and run the multilingual eval against a single-agent baseline; if citation quality
holds, C avoids the migration, otherwise go with B. Either path then runs as
`/spec` -> `/plan` -> `/implement`.

**Update 2026-10-07:** went with C, generalised as a "knowledge router" flag. Citation
numbering uses rebase onto a turn-wide counter (not namespaced keys): specialists are
claimed in list order inside `consultar_varios`, so finish order never matters and the
frontend keeps resolving `cite://N` by position.

## Known limitations / deferred decisions (2026-10-07)

Decided on purpose when the knowledge router shipped; each says when to reopen it.
Full detail: `docs/superpowers/specs/2026-10-07-knowledge-router-design.md` (untracked, `specs/` is gitignored).

1. **Citation numbering across a HITL pause.** A turn with human approval spans two HTTP
   requests (pause, then `stream_resume_agent_chat`), and the resumed request rebuilds the
   agent with the citation counter at 0. If the agent retrieved chunks both *before* and
   *after* the pause, `[N]` repeats between the two searches: live, the frontend resolves it
   against a graph holding only the post-resume chunks; after a reload, the history merges both
   searches in order, so the same `[N]` can open a different chunk. Separate user messages are
   unaffected (each builds its own agent and graph). Not fixed because it needs the
   `human_in_the_loop` middleware with a non-empty `interrupt_on`, retrieval on both sides of the
   pause, and a cited answer; it affects every HITL + LightRAG agent, not only routers.
   *Planned fix:* on resume, read the turn's already-retrieved chunks from the checkpoint, start
   the rebuilt agent's counter at that count and seed the accumulated graph with them.
   *Reopen when* a query finds an agent with `human_in_the_loop` (`config->'interrupt_on'` not
   empty) plus a LightRAG silo or specialists, or a user reports a citation opening the wrong
   document after approving.
2. **Chunks of nested agent-tools do not propagate.** Only the router's direct specialists hand
   their chunks up. In router → specialist (no silo) → sub-specialists, the sub-specialists'
   citations are dropped (range pruning in `claim_citations`), so those sentences appear without
   a link; they never point at a wrong document. *Planned fix:* nested agent-tools return an
   artifact and renumber too (`return_direct=False`), on top of `IACTTool.run_counter`; tag the
   silo only on untagged items. ~40-50 lines. *Reopen when* a hierarchy deeper than one level
   needs working citations.
3. **Citation opens a different document than the sentence names (to fix, separate task).**
   Measured 2026-10-07 on the multilingual eval (router run with memory): in ~14 % of the
   `[N]` markers of Spanish answers and ~27 % of English ones, the line names a document
   (CDOC…/DSAT…) and chunk `N` belongs to another. The standalone ES/EN agents show the same
   rate (15.4 % ES; 59/156 EN), and it is highest where the router renumbers nothing (single
   specialist, offset 0), so it is NOT caused by the router. Most likely cause: the order of the
   numbered list the model sees (e.g. the coverage tool `list_documents_mentioning`, which
   numbers its own enumerated lines) differs from the order of `lightrag_raw_data.chunks`
   that the frontend resolves `cite://N` against. Fix belongs in the coverage tool
   (`coverage_union.py` / `_create_coverage_tool`): make the enumerated numbers and the
   payload order the same list. This matters more to the user than the items above: a
   wrong-document citation is worse than a missing one. Check script used:
   `scratchpad/cite_check.py` logic (line names doc ∈ CDOC/DSAT, chunk N's `file_path` must be
   one of them) — worth turning into a regression test over saved eval results.
4. **Single-language answers with no citations at all.** 3 of 40 answers of the ES/EN blocks
   (`ML-ES-*`/`ML-EN-*`) came back without a single `[N](cite://N)`. Investigate whether the
   sub-agent skipped the SOURCES block (prompt) or the question type (e.g. enumeration /
   absence) never retrieves chunks.
5. Minor: agent export/import drops `is_knowledge_router`; cited-chunk highlighting compares
   chunk ids across silos (cosmetic).

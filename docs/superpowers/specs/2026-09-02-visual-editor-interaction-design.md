# Visual editor: interactive edge mutations — design

## Context

The visual editor (`frontend/src/components/visual-editor/`) renders an app's
resource graph with `@xyflow/react` as a **read-only** canvas: 7 node kinds
(agent, service, silo, embedding, skill, mcp, parser) and 7 edge kinds derived
from existing FK/M:N relationships. Nodes are draggable (position persisted to
`localStorage`) but `nodesConnectable={false}`, `edgesReconnectable={false}`,
`deleteKeyCode={null}` — no domain mutation happens from the canvas today.

This spec covers making 4 of the 7 relationship kinds interactive: dragging a
new connection or deleting an existing one on the canvas performs a real
domain mutation against the backend. Node creation/deletion from the canvas
is explicitly **out of scope** (future spec).

## Requirements

### In scope (editable relationships)

| Edge kind | Domain relationship | Cardinality |
|---|---|---|
| `silo` | `Agent.silo_id` (FK) | Agent → 0..1 Silo |
| `skill` | `agent_skills` (M:N) | Agent → 0..N Skills |
| `tool` | `agent_tools` (M:N, agent-as-tool) | Agent → 0..N tool-Agents |
| `mcp` | `agent_mcps` (M:N) | Agent → 0..N MCPConfigs |

`service`, `embedding`, `parser` edges and the `silo→embedding` edge stay
read-only (the latter is immutable backend-side after silo creation).

### Connection validation

- A connection may only be dropped between the node-kind pairs above
  (agent↔silo, agent↔skill, agent↔agent-tool, agent↔mcp). Enforced
  client-side via React Flow's `isValidConnection` so an invalid drop is
  rejected before any API call.
- Tool edges: target agent must have `is_tool=true`; no self-connection.
- Backend re-validates independently (already does, via
  `update_agent_tools/mcps/skills` in `backend/services/agent_service.py`) —
  the frontend check is a UX convenience, not the source of truth.
- Known gap, not blocking this spec: `update_agent_mcps`
  (`agent_service.py:395`) has no visible cross-app id check, unlike
  `update_agent_skills`. Flag during implementation; fix if confirmed missing.

### Mutation flow (optimistic + rollback)

- Connecting or deleting an edge updates the canvas immediately (optimistic),
  then calls the existing agent update endpoint
  (`POST /internal/apps/{appId}/agents/{agentId}`, `CreateUpdateAgentSchema`)
  with the full current agent payload (from the `AgentDetailGraphItem` already
  cached by `useAppGraph`) and just the changed relationship field
  (`silo_id`, `skill_ids`, `tool_ids`, or `mcp_config_ids`).
- On success: clear the optimistic overlay, refetch the graph so the
  canonical state (and any consequential changes) is reflected.
- On failure: revert the optimistic change and show an error toast (existing
  frontend notification pattern).
- `Agent.silo_id` is scalar: connecting a new silo edge implicitly replaces
  the previous one (no separate delete step needed) — the old edge just stops
  being derived once `silo_id` changes.
- No optimistic-locking / conflict detection for concurrent edits — single
  active editor is the assumed usage pattern for this phase.

### Deletion

- `deleteKeyCode` enabled (currently `null`); selecting an edge and pressing
  Delete/Backspace triggers `onEdgesDelete`, following the same
  optimistic-mutate-rollback flow, computing the new relationship value with
  that edge's target removed.
- `edgesReconnectable` stays `false` — dragging an edge endpoint elsewhere is
  not supported; delete + reconnect instead.

### Out of scope

- Persisting node layout/positions against the backend (stays
  `localStorage`-only, as today).
- Creating/deleting nodes (new agents, silos, etc.) from the canvas.
- Any backend endpoint or schema changes — this phase reuses existing
  endpoints as-is.

### UI language

All new UI strings (toasts, tooltips, inline errors) must be in English. Also
fix the two existing Spanish labels while touching this area:
`frontend/src/core/defaultNavigation.tsx:74` (`'Editor visual'` →
`'Visual editor'`) and `frontend/src/pages/VisualEditorPage.tsx:100`
(same text).

## Design

### New files

- `frontend/src/components/visual-editor/graphConnectionRules.ts` — the
  allowed-pair matrix + `isValidConnection(nodes, connection)` +
  `is_tool`/self-connection checks.
- `frontend/src/components/visual-editor/useGraphMutations.ts` — hook owning
  the optimistic overlay (`{added: Edge[], removedIds: Set<string>}`),
  `connect(sourceNode, targetNode)` and `disconnect(edge)`, both calling
  `apiService.updateAgent` and triggering the existing graph refetch from
  `useAppGraph` on success / reverting the overlay on failure.

### Changed files

- `GraphFlowView.tsx` — `nodesConnectable`, `isValidConnection`, `onConnect`,
  `onEdgesDelete`, `deleteKeyCode` wired through as props (component stays
  presentational; handlers come from the parent).
- `AppGraphCanvas.tsx` — instantiates `useGraphMutations`, merges its overlay
  over the edges produced by `graphAdapter`, passes handlers down to
  `GraphFlowView`.
- `frontend/src/core/defaultNavigation.tsx`, `frontend/src/pages/VisualEditorPage.tsx`
  — label fix noted above.

### Testing

- `graphConnectionRules.test.ts` (new) — table-driven unit test of the
  allowed/rejected node-kind pairs and the `is_tool`/self-connection rules.
  Run via `npx vitest <path> --run`, following the existing ad-hoc precedent
  in `frontend/src/core/__tests__/ConfigService.test.ts` (vitest is not an
  installed devDependency; no new dependency added).
- No backend changes in this phase, so no backend tests are added.
- End-to-end browser verification (`npm run dev` + manually dragging/deleting
  edges) is **not performed as part of this implementation session** — the
  local Docker stack (Postgres/backend) is currently occupied by a document
  indexing job on the ports this stack would use. This must be verified
  manually once the stack is free before considering the feature done; it is
  called out explicitly rather than claimed as tested.

## Open questions

- Confirm whether `update_agent_mcps` is missing the cross-app id check that
  `update_agent_skills` has, and fix if so (not blocking, discovered during
  exploration).

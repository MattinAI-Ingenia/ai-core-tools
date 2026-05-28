---
description: Compact ruleset for plan extensions (workflow used by @feature-planner and @plan-executor when adding related features to a completed plan)
applyTo: "plans/**"
---

# Plan Extension Rules

Extensions let you add **related** features to an already-completed plan without reopening it. They preserve continuous step numbering and a single execution status. The full tutorial (templates, examples, `status.yaml` schema) lives in [`docs/plan-extensions-workflow.md`](../../docs/plan-extensions-workflow.md).

## File Layout

```
plans/<slug>/
├── spec.md
└── execution/
    ├── status.yaml                 # Single status file for ALL steps
    ├── step_001.md … step_NNN.md   # Original + extension step files (flat)
    └── extensions/
        ├── extension-1.plan.md
        └── extension-N.plan.md
```

## Core Rules

- **One `status.yaml`** tracks original + every extension. Never create a second status file.
- **Step numbering is global and continuous**. If the original plan ended at `step_024`, extension-1 starts at `step_025`.
- **Specs live in `execution/extensions/`**, steps stay flat in `execution/`.
- **Extension specs must declare** `parent_plan: <slug>`, `extension_id: N`, `next_step_number: NNN` in their header.
- **Extension requirements** use the naming convention `FR-EN-{num}` and `AC-EN-{num}` (distinguishes them from the parent's `FR-{num}` / `AC-{num}`).
- **Every extension step** in `status.yaml` carries `extension_reference: extension-N` and a `[Extension-N]` title prefix.

## When to Create an Extension vs. a New Plan

- Use an **extension** when the new scope is closely related to the parent plan (builds on its FR/AC, shares architectural decisions).
- Use a **new plan** when the scope is independent — even if it touches the same area of the codebase.
- An extension can be **promoted** to a standalone plan if it grows large; mark the original as `archived` in the parent's index entry.

## For `@feature-planner`

1. Validate that the parent plan exists and is `implemented` (or nearly so).
2. Create `/plans/<slug>/execution/extensions/extension-N.plan.md` from the template in `docs/plan-extensions-workflow.md`.
3. Reference the parent: `parent_plan`, `extension_id`, citations of relevant `FR`/`AC`.
4. Read the parent's `status.yaml` to find the last step and write `next_step_number: NNN` in the extension header.
5. Hand off to `@plan-executor execute extension <slug> extension-N`.

## For `@plan-executor`

1. Read the extension spec and verify `next_step_number` matches the parent `status.yaml`.
2. Generate `step_NNN.md` files continuing from the parent's last step number.
3. Append entries to the existing `status.yaml`, each with `extension_reference: extension-N` and `[Extension-N]` prefix in the title.
4. Brief each implementer agent with one-line parent context when relevant.
5. Resumption: the executor must be able to pick up from any point across original + all extensions.

## Anti-Patterns

- ❌ Creating a second `status.yaml` per extension
- ❌ Resetting step numbering when starting an extension
- ❌ Putting unrelated features inside an extension (they belong in a new plan)
- ❌ Duplicating parent requirements in the extension spec
- ❌ Putting step files inside `extensions/` (specs go there, not steps)

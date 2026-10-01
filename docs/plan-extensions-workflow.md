# Plan Extensions Workflow

> **Audience**: developers and AI agents (`@feature-planner`, `@plan-executor`) working with feature plans in `/plans/`. The compact ruleset lives at [`.github/instructions/plan-extensions.instructions.md`](../.github/instructions/plan-extensions.instructions.md); this document contains the full tutorial, templates and examples.

## Overview

When a plan is completed and executed, new requirements, edge cases, or related features may be discovered. Rather than reopening the original plan, **extensions** let you add new features while maintaining context and continuous step numbering.

This workflow applies to both `@feature-planner` (creates extension specs) and `@plan-executor` (executes them).

## Extension Concepts

### File Layout

Extensions live as separate plan files alongside the original plan, inside a dedicated `extensions/` subfolder of the original's `execution/`:

```
plans/
└── agent-marketplace/             # Original plan folder
    ├── spec.md                    # Original specification (unchanged)
    ├── decisions.md
    ├── open-questions.md
    └── execution/
        ├── step_000_plan.md       # Original execution overview
        ├── step_001.md … step_024.md  # Original steps
        ├── status.yaml            # Single status file tracking ALL steps
        └── extensions/            # Extensions live here
            ├── extension-1.plan.md
            ├── extension-2.plan.md
            └── extension-N.plan.md
```

### Numbering Rules

- Extension plan files: `extension-1.plan.md`, `extension-2.plan.md`, …
- **Step numbering continues globally**: if the original plan ended at `step_024`, extension-1 starts at `step_025`.
- A **single `status.yaml`** tracks all steps (original + every extension).
- Extensions do NOT reset step numbering — the execution view stays unified.

### Relationships

Extensions must explicitly reference the parent plan:
- Header includes `parent_plan: <slug>` and `extension_id: <N>`
- May reference original spec sections (`see ../spec.md`)
- Requirements should be **related to** the original — not duplicate it, not unrelated

## Feature Planner Workflow (`@feature-planner`)

### When to Create an Extension

1. The parent plan is `implemented` (or nearly so)
2. A new conversation surfaces additional features
3. The new scope is **related** to the original (closely coupled, not independent)
4. There is a clear separation between original requirements and extension requirements

### Creating an Extension

User invocation:

```
@feature-planner extend the <slug> plan with extension-N: <description>
```

Planner workflow:

1. **Validate parent plan exists**: check `/plans/<slug>/spec.md` and `execution/status.yaml`, confirm parent is `implemented` or near completion.
2. **Create extension directory**: ensure `/plans/<slug>/execution/extensions/` exists.
3. **Write extension spec**: `/plans/<slug>/execution/extensions/extension-N.plan.md` (template below).
4. **Reference original plan**: include `parent_plan` and `extension_id` in the header; cite relevant FR/AC from the parent.
5. **Note next step number**: read `execution/status.yaml`, find the last step, and document `next_step_number: NNN` in the extension header.
6. **Hand off to executor**: after the spec is ready, the user invokes `@plan-executor execute extension <slug> extension-N`.

### Extension Spec Template

```markdown
# <Plan Name> — Extension N: <Extension Name>

> **Status**: ready
> **Parent Plan**: <slug>
> **Extension ID**: N
> **Created**: YYYY-MM-DD
> **Links To Original**: FR-1, FR-2 (see ../spec.md)
> **Next Step Number**: 025

## Context
This extension builds on … to add …

## Goals
- …

## Functional Requirements

### FR-EN-1: <name>
…

### FR-EN-2: <name>
…

## Acceptance Criteria
- [ ] AC-EN-1: …
- [ ] AC-EN-2: …

## Edge Cases & Constraints

## Dependencies
- Depends on original <slug> plan completion

## Open Questions
- …
```

**Naming convention for extension requirements**:
- `FR-EN-{num}` (e.g. `FR-E1-1`, `FR-E1-2` for extension 1) — distinguishes from original `FR-{num}`
- `AC-EN-{num}` mirrors it for acceptance criteria

## Plan Executor Workflow (`@plan-executor`)

### Executing an Extension

User invocation:

```
@plan-executor execute extension <slug> extension-N
```

Executor workflow:

1. **Read extension spec**: load `/plans/<slug>/execution/extensions/extension-N.plan.md`. Extract `parent_plan` and `next_step_number`.
2. **Read current status**: open `/plans/<slug>/execution/status.yaml`. Find the last completed step and verify the extension's `next_step_number` matches.
3. **Continue numbering**: if step 024 was the last original step, extension-1 starts at step 025. New files follow the same pattern: `step_025.md`, `step_026.md`, …
4. **Create steps**: place new `step_NNN.md` files alongside the original ones in the main `execution/` directory (specs live in `extensions/`, steps stay flat).
5. **Update `status.yaml`**: append new entries to the existing `steps:` list. Include `extension_reference: extension-N` on each extension step. Never split into a second status file.
6. **Reference parent context**: every extension step should briefly note it belongs to extension-N of plan `<slug>`, so implementer agents have the foundational context.

### `status.yaml` Example with Extensions

```yaml
plan_id: agent-marketplace
overall_status: in-progress
execution_mode: extended  # Indicates extensions are in flight

steps:
  - step: "001"
    title: "Create feature branch"
    status: done
    # ... original steps 001-024 ...

  # Extension-1 (continuation)
  - step: "025"
    title: "[Extension-1] Build analytics backend"
    target_agent: "@backend-expert"
    status: not-started
    extension_reference: extension-1
    fr: ["FR-E1-1"]
    ac: ["AC-E1-1", "AC-E1-2"]

  - step: "026"
    title: "[Extension-1] Analytics dashboard UI"
    target_agent: "@react-expert"
    status: not-started
    extension_reference: extension-1
    fr: ["FR-E1-2"]
    ac: ["AC-E1-3"]
    depends_on: ["025"]

  # Extension-2 follows extension-1
  - step: "027"
    title: "[Extension-2] Performance monitoring"
    target_agent: "@backend-expert"
    status: not-started
    extension_reference: extension-2
```

### Key Invariants

- **Single `status.yaml`** for the whole plan + every extension
- **Continuous numbering**: 001…024 (original), 025+ (extension-1), continuing through every extension
- **Extension markers**: `[Extension-N]` prefix in step titles, `extension_reference` field on every extension step
- **Specs vs. steps**: extension specs live in `execution/extensions/`, step files stay in `execution/`
- **Resumable**: the executor can resume from any point across original + extensions seamlessly

## Promoting an Extension to a Standalone Plan

If an extension grows large or becomes independent, promote it:

1. Copy the extension spec to a new plan folder: `/plans/<new-slug>/spec.md`
2. Mark the original extension as `archived` or `superseded` in the parent's `index.yaml` entry
3. Create a fresh `/plans/<new-slug>/execution/` and start numbering at `step_001`
4. The promoted plan no longer references the original via `parent_plan`

## Best Practices

### Feature Planner
- Always validate parent plan exists and is complete before creating an extension
- Reference original FR/AC by number to show the relationship
- Use consistent naming: `FR-EN-{num}`, `AC-EN-{num}`
- Include `next_step_number` in the extension header
- Only create an extension when scope is clear and ready to implement

### Plan Executor
- Always read the parent plan context before generating extension steps
- Maintain continuous step numbering across original + all extensions
- Use `[Extension-N]` prefixes in step titles for visual clarity
- Include `extension_reference` on every extension step
- Never create a second `status.yaml` — one file tracks everything
- Add a one-line parent context note to each agent prompt when relevant

### Developers
- Extensions are for **related features** discovered after plan completion
- **Unrelated features** must be new plans, not extensions
- Use extensions to refine/improve the original without losing context
- Multiple extensions (extension-1, extension-2, …) are allowed
- Extensions inherit all decisions/constraints from the parent plan

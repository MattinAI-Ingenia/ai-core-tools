---
description: "Execute a small ad-hoc task autonomously via @quick-executor — branch creation, implementer subagents and git operations end-to-end, with explicit confirmation only at commit / push / PR."
agent: quick-executor
argument-hint: "Brief task description (e.g. 'fix the login redirect loop on Safari')"
---

Please execute the following ad-hoc task autonomously: **${input:task}**.

Workflow:

1. **Read** the codebase using `read` / `search` to identify which files / modules the task touches.
2. **Decide the branch name**: if this conversation contains an `Issue Analysis` block from `@issue-reader`, use its `Suggested branch` verbatim; otherwise derive `<type>/<short-slug>` from the task description (`fix/`, `feat/`, `clean/`, `docs/`, `hotfix/`).
3. **State the planned subagent sequence** before invoking the first one (so I can intervene if you picked the wrong agents).
4. **Create the local feature branch** off `develop`.
5. **Invoke implementer subagents sequentially** (`@backend-expert`, `@react-expert`, `@alembic-expert`, `@test-expert`, `@docs-manager` as appropriate). After each subagent finishes file operations, pause for commit confirmation.
6. **Pause for push confirmation** once all commits are made — show me the commit list and the branch name.
7. **Pause for PR confirmation** once the push succeeds — show me the proposed title, body preview, base and head.
8. **Report** the final state: branch name, commits, PR URL (if opened), and any subagent that returned `blocked` or `needs-revision`.

If the task turns out to be substantial (3+ areas, new entities, multi-step migrations, non-trivial acceptance criteria), STOP and redirect me to `@feature-planner` for a tracked spec instead.

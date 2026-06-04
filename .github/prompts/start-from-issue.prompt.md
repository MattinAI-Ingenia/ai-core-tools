---
description: "Start a development workflow from a GitHub issue. Reads it via the @github MCP server and offers handoff to @feature-planner (formal spec) or @quick-executor (autonomous ad-hoc execution)."
agent: issue-reader
argument-hint: "Issue number or URL (e.g. 123, owner/repo#123, https://github.com/owner/repo/issues/123)"
---

Please read GitHub issue `${input:issueRef}` via the `@github` MCP server.

Steps:

1. **Resolve the reference**: accept a bare number (default repo `lksnext-ai-lab/ai-core-tools`), an `owner/repo#NN` reference, or a full GitHub URL.
2. **Fetch via MCP**: use the `@github` tool to retrieve title, body, labels, assignees, milestone and the most recent comments.
3. **Cross-check the codebase**: use `read`/`search` to verify entities and paths mentioned in the issue actually exist.
4. **Emit the Issue Analysis block** at the top of your response (see your agent definition for the exact format) — fully populated, no invented content. Include the **Suggested branch** field (`<type>/issue-<NN>-<short-slug>`).
5. **Recommend** one path based on issue size and scope:
   - **@feature-planner**: features, schema changes, multi-area work, anything >~3 files
   - **@quick-executor**: bugs, doc fixes, single-area refactors, <~3 files
6. **End** with a one-paragraph rationale. Mention that the chosen downstream agent will create the local feature branch (using the Suggested name) before any implementation begins, and that both `git push` and `gh pr create` will require explicit user confirmation later in the flow. The two handoff buttons (`Plan formally with @feature-planner` / `Execute autonomously with @quick-executor`) appear automatically — let the user click the one they want.

If the GitHub MCP server is unavailable, do NOT invent issue content — ask the user to paste the issue body and re-invoke instead.

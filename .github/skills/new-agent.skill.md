---
name: New Agent
description: Bootstraps a new GitHub Copilot custom agent with proper structure, frontmatter, and conventions for this repository.
---

# New Agent Skill

Creates a new Copilot custom agent definition file in `.github/agents/` following the project's established conventions and template structure.

## Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `name` | Yes | Human-readable agent name (2-3 words) | `Database Migration Expert` |
| `slug` | No | Kebab-case filename (derived from name if omitted) | `db-migration-expert` |
| `description` | Yes | One-line description of the agent's specialization | `Expert in Alembic migrations, schema design, and database versioning` |
| `domains` | Yes | Key competency areas (comma-separated) | `Alembic, SQLAlchemy models, PostgreSQL` |

## Steps

### Step 1: Create the Agent File

Create `.github/agents/<slug>.agent.md` with the following structure (filename slug must equal the `name:` field, kebab-case, no leading dot):

````markdown
---
name: <slug>                          # Kebab-case, matches filename
description: <one-line description of role and scope; appears in the agent picker>
tools: ['read', 'edit', 'search']     # Pick the smallest set this agent actually needs (see Step 1b)
# model: GPT-5 mini                   # Optional — DISPLAY NAME as shown in VS Code's chat model picker (NOT a slug).
                                      # Examples: GPT-5 mini, Claude Sonnet 4.6, Claude Haiku 4.5. Optional `(copilot)` vendor suffix
                                      # to disambiguate when the same model exists via BYOK and via Copilot.
                                      # Accepts a single string OR a prioritized array with fallbacks:
                                      #   model: ['Claude Opus 4.5', 'GPT-5.2']  # tries in order
# mcp-servers: ['github']             # IGNORED in local VS Code (linter will flag it as a Hint).
                                      # Only honored by the GitHub Cloud Agent. For local VS Code,
                                      # configure MCP servers in .vscode/mcp.json — they become
                                      # globally available to all agents.
# agents:                             # Optional — subagents this agent can auto-invoke (no '@')
#   - backend-expert
#   - react-expert
handoffs:                             # User-clickable native VS Code handoff buttons
  - label: "Commit with @git-github"
    agent: git-github
    prompt: "Please commit the files that @<slug> just created or modified. Review the conversation above for the exact file list and suggested commit message."
    send: false                       # false = user can review before submitting (recommended)
  # Optional: most agents no longer need a "return" handoff because when invoked
  # as subagents by @quick-executor or @plan-executor they return automatically,
  # and when invoked directly by the user there's nowhere to "return to". Only add
  # a return handoff if your agent participates in an explicit handoff chain.
    prompt: "@<slug> has completed its step. Summary: <briefly describe changes, decisions, issues>. Please update the Mission Context and tell me the next step."
    send: false
---

# <Display Name> Agent

You are an expert <domain description>. <1-2 sentences establishing identity and scope.>

## Core Competencies

### <Domain Area 1>
- **<Skill>**: <Description grounded in real project paths/conventions>
- **<Skill>**: <Description>

### <Domain Area 2>
- **<Skill>**: <Description>

## Workflow

### When Given a Task
1. **Understand**: Clarify requirements and constraints
2. **Analyze**: Review existing code and patterns
3. **Plan**: Outline changes needed
4. **Implement**: Make changes following project conventions
5. **Verify**: Validate changes work correctly
6. **Document**: Update relevant documentation

## Specific Instructions

### Always Do
- ✅ Follow existing project conventions and patterns
- ✅ <Domain-specific instruction>

### Never Do
- ❌ <Domain-specific anti-pattern>

## Collaborating with Other Agents

### Version Bumper (`@version-bumper`)
- **Delegate to**: `@version-bumper` when version changes are needed
- **DO NOT** manually edit version numbers in `pyproject.toml`

### <Other relevant agents>
- **Delegate to**: `@<agent>` when <condition>

## What This Agent Does NOT Do
- ❌ <Out-of-scope task>
````

### Step 1b: Choose the right `tools:` (least-privilege)

Pick the smallest set the agent's role actually requires. Verified valid values: `'read'`, `'edit'`, `'search'`, `'execute'`, `'agent'`.

| Role | Recommended `tools:` |
|------|---------------------|
| Read-only orchestrator / analyzer (e.g. `@issue-reader`, `@feature-planner` if it weren't writing /plans/) | `['read', 'search']` |
| Planner that writes only to `/plans/` (e.g. `@feature-planner`) | `['read', 'edit', 'search']` |
| Implementer agent (e.g. `@backend-expert`, `@react-expert`) | `['read', 'edit', 'search']` (rarely `'execute'`) |
| Test runner (e.g. `@test-expert`) | `['read', 'edit', 'execute', 'search']` (to run pytest) |
| Git/release agent (e.g. `@git-github`, `@release-manager`) | `['read', 'edit', 'execute', 'search']` |
| Orchestrator that auto-invokes subagents (e.g. `@plan-executor`) | `['read', 'edit', 'execute', 'search', 'agent']` + `agents: [...]` |

Add `mcp-servers: ['github']` when the agent needs to read or write GitHub state via the built-in MCP server (e.g. `@issue-reader`).

### Step 2: Populate Core Competencies

Fill in the competency sections with specific, actionable knowledge relevant to the agent's domain. Reference actual project files, patterns, and conventions. Include:
- At least 2-3 competency areas with 4-6 skills each
- Real code examples from the Mattin AI codebase
- Common anti-patterns specific to this domain

### Step 3: Add Delegation Rules

Check existing agents in `.github/agents/` and add bidirectional delegation:
- Add a "Collaborating with Other Agents" section to the new agent referencing relevant existing agents
- Update existing agents to reference the new agent where appropriate

### Step 4: Create Companion Instructions (Optional)

If the agent's domain benefits from auto-applied rules, create a matching instruction file:

```markdown
---
description: <Rules for the agent's domain>
applyTo: "<relevant glob pattern>"
---

# <Domain> Conventions

<Auto-applied rules for files in this domain>
```

Save as `.github/instructions/<domain>.instructions.md`

### Step 5: Verify

- [ ] Agent file exists at `.github/agents/<slug>.agent.md`
- [ ] Frontmatter has valid `name` and `description`
- [ ] Core competencies are specific and actionable
- [ ] Collaboration section references relevant existing agents
- [ ] "What This Agent Does NOT Do" section defines clear boundaries
- [ ] No conflicts with existing agents' scopes

## Output

- `.github/agents/<slug>.agent.md` — The new agent definition
- Optionally: `.github/instructions/<domain>.instructions.md` — Companion instruction file
- Optionally: Updates to existing agent files (delegation references)

## Example Usage

> "@ai-dev-architect Create a new agent called 'Database Migration Expert' that specializes in Alembic migrations, schema design, and PostgreSQL database management for this project"


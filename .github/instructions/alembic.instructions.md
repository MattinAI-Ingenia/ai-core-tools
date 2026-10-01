---
description: Project-specific conventions for Alembic migrations in the Mattin AI project — naming, structure, ignored tables, model registry, and the standard upgrade/downgrade round-trip test.
applyTo: "alembic/**"
---

# Mattin AI Alembic Conventions

These rules auto-apply whenever you edit any file under `alembic/`. The `@alembic-expert` agent provides generic Alembic / SQLAlchemy migration expertise; this file pins down the **Mattin-specific** patterns: table naming, the ignored-tables filter, the model registry, and the round-trip downgrade test.

## File Layout

```
alembic.ini                                # Alembic configuration
alembic/
├── env.py                                 # Migration environment (DB connection, model imports, include_name filter)
├── script.py.mako                         # Migration file template
├── README                                 # Alembic README
└── versions/                              # All migration scripts (~45+ files)
    ├── df947a43f4ba_db_base_1.py
    ├── 59e8d529b38a_initial_models.py
    └── skills001_add_skills_support.py    # Example recent migration
```

## Migration File Rules

- Every migration MUST have both `upgrade()` AND `downgrade()` functions.
- `downgrade()` MUST fully reverse `upgrade()`. Test the round trip before committing (see "Round-trip test" below).
- Use descriptive revision messages: `"add_memory_management_fields"`, not `"update"` or `"fix"`.
- Verify `down_revision` points to the correct parent revision before committing.
- **Never modify an existing migration that has already been applied** to any environment — create a new migration instead.
- Never delete files from `alembic/versions/` without understanding the full revision chain.

## Table Naming

- Primary entity tables use **PascalCase**: `Agent`, `Silo`, `App`, `Skill`, `Conversation`, `AIService`, `EmbeddingService`, `OutputParser`, `Repository`, `Resource`, `Folder`, `Domain`, `MCPServer`, `MCPConfig`, `APIKey`.
- Junction / association tables use **snake_case**: `agent_skills`, `agent_mcps`, `agent_tools`, `app_collaborator`, `mcp_server_agents`.

## Column Conventions

- Primary keys: `<table_name_lower>_id` (e.g. `agent_id`, `skill_id`, `silo_id`).
- Always be explicit about `nullable=True` or `nullable=False`.
- When adding a non-nullable column to an existing table, provide a `server_default` (or split into a multi-step migration that backfills first).
- Use `sa.DateTime()` for timestamps, with `default=datetime.now` defined on the model side.
- Foreign keys reference the full `TableName.column_name` format (e.g. `'App.app_id'`).
- Define `ondelete` / `onupdate` cascade rules explicitly.

## Model Registry — CRITICAL

All SQLAlchemy models are registered in `backend/models/__init__.py`. **A new model is invisible to Alembic autogenerate until it is imported there.**

```python
# backend/models/__init__.py — every model must appear in this file
from .user import User
from .app import App
from .agent import Agent
# … add your new model here ↑
```

If autogenerate produces an empty migration after you added a model, this is almost always the cause.

## Ignored Tables

The `include_name()` filter in `alembic/env.py` excludes the following tables from autogeneration. Do NOT create migrations for them:

- `langchain_pg_collection`, `langchain_pg_embedding` — managed by LangChain (PGVector store)
- `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` — managed by LangGraph's `AsyncPostgresSaver` (conversation memory)

If you ever introduce another externally-managed table (LangChain, LangGraph, third-party library), add it to the `include_name()` filter in `alembic/env.py` rather than letting Alembic generate spurious migrations for it.

## Round-trip Test (mandatory before committing)

```bash
poetry run alembic upgrade head        # apply migration
poetry run alembic downgrade -1        # revert it
poetry run alembic upgrade head        # apply again
```

If any step fails the migration is not ready. Never commit a migration whose `downgrade()` was not exercised locally.

## Project Migration Patterns (examples)

### Standard table creation

```python
def upgrade():
    op.create_table('Skill',
        sa.Column('skill_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(1000), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('create_date', sa.DateTime(), nullable=True),
        sa.Column('update_date', sa.DateTime(), nullable=True),
        sa.Column('app_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['app_id'], ['App.app_id'], ),
        sa.PrimaryKeyConstraint('skill_id'),
    )

def downgrade():
    op.drop_table('Skill')
```

### Junction table (many-to-many)

```python
op.create_table('agent_skills',
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('skill_id', sa.Integer(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['Agent.agent_id'], ),
    sa.ForeignKeyConstraint(['skill_id'], ['Skill.skill_id'], ),
    sa.PrimaryKeyConstraint('agent_id', 'skill_id'),
)
```

### Data seeding

```python
def upgrade():
    op.bulk_insert(
        sa.table('Model',
            sa.column('provider', sa.String),
            sa.column('name', sa.String),
            sa.column('description', sa.String),
        ),
        [
            {'provider': 'OpenAI', 'name': 'gpt-4o-mini', 'description': '...'},
        ],
    )
```

## Poetry Environment — ALL `alembic` commands

All Alembic commands MUST run through Poetry. Bare `alembic` may use a different Python interpreter or miss project dependencies.

```bash
poetry run alembic current                                       # current revision
poetry run alembic history --verbose                             # full history
poetry run alembic heads                                         # detect branches
poetry run alembic revision --autogenerate -m "<description>"    # autogenerate
poetry run alembic revision -m "<description>"                   # empty migration (manual)
poetry run alembic upgrade head                                  # apply all
poetry run alembic upgrade +1                                    # apply one
poetry run alembic downgrade -1                                  # revert one
poetry run alembic downgrade <revision_id>                       # revert to a target
poetry run alembic upgrade head --sql                            # dry-run SQL preview
poetry run alembic merge -m "merge heads"                        # resolve multiple heads
poetry run alembic stamp <revision_id>                           # mark as applied (use with extreme caution)
```

## Troubleshooting Checklist

1. Empty autogenerate? → check the model is imported in `backend/models/__init__.py`.
2. Multiple heads? → `poetry run alembic heads`, then `poetry run alembic merge -m "merge heads"`.
3. Migration partially applied? → inspect the `alembic_version` table; consider `stamp` only if you fully understand the state.
4. Type / nullable mismatch with the model? → autogenerate may have misread a custom type. Edit the migration by hand; verify with the round-trip test.
5. Failing in Docker but not local? → likely a model import path issue in `alembic/env.py` — paths differ between bare runs and the Docker entrypoint.

## Anti-Patterns (project-specific)

- ❌ Adding a model file without importing it in `backend/models/__init__.py`
- ❌ Adding a non-nullable column to an existing table with no `server_default` and no backfill step
- ❌ Skipping the `downgrade()` round-trip test
- ❌ Using bare `alembic` commands (must use `poetry run alembic`)
- ❌ Generating migrations for `langchain_pg_*` or `checkpoint*` tables — they are externally managed
- ❌ Editing a migration that has been applied to `develop` / `main` — create a new revision instead

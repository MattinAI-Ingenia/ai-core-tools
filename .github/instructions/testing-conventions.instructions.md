---
description: Project-specific testing conventions for the Mattin AI backend — pytest layout, savepoint-based transaction isolation, fixtures map, factory-boy, test DB on port 5433, and CI workflow.
applyTo: "tests/**"
---

# Mattin AI Testing Conventions

These rules auto-apply whenever you edit any file under `tests/`. Agents like `@test-expert` provide generic pytest expertise; this file pins down the **Mattin-specific** fixtures, isolation pattern and test infrastructure.

## Test Layout

```
tests/
├── conftest.py                            # Shared fixtures (test_engine, db, client, fake_*, *_headers)
├── factories.py                           # factory-boy model factories
├── unit/                                  # No DB, fast — services with mocked dependencies
│   └── services/
│       └── test_<service_name>.py
└── integration/                           # Real PostgreSQL, full HTTP stack via TestClient
    └── routers/
        ├── internal/
        │   └── test_<resource>.py
        └── public/
            └── test_<resource>.py
```

- **Unit tests** = pure Python, no DB, every dependency mocked (`pytest-mock`)
- **Integration tests** = real `pgvector/pgvector:pg17` running on port 5433, `TestClient`, full router → service → repo → DB stack
- E2E (frontend) is a future phase (Playwright + Vitest); not yet present

## Transaction Isolation Pattern (the Key Pattern)

Every test is fully isolated without touching real data because the `db` fixture wraps each test in a connection-level transaction that is **always rolled back at teardown**, even if the service code under test calls `session.commit()`.

```python
# tests/conftest.py
@pytest.fixture(scope="function")
def db(test_engine):
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",   # service .commit() → SAVEPOINT only
        autocommit=False,
        autoflush=True,
    )
    yield session
    session.close()
    transaction.rollback()                          # undoes everything
    connection.close()
```

The magic is `join_transaction_mode="create_savepoint"`: when service code calls `session.commit()`, SQLAlchemy emits a `SAVEPOINT` instead of a real commit. The outer `transaction.rollback()` at teardown wipes the lot.

Implication for test code:

- ✅ Use `db.flush()` to make data visible inside the current session without committing
- ❌ Never call `db.commit()` from a test or a test fixture — it breaks the isolation
- ❌ Never use `SessionLocal()` or `engine.connect()` directly — always use the `db` fixture

## Fixtures Map (`tests/conftest.py`)

```
test_engine (session-scoped)
   └── db (function-scoped — transactional, full rollback)
        ├── fake_user
        │    └── fake_app
        │         ├── fake_ai_service
        │         │    └── fake_agent
        │         └── fake_api_key
        └── client (TestClient with get_db overridden to the test session)
             ├── auth_headers   (fake_user logged in via /internal/auth/dev-login)
             └── owner_headers  (fake_user + OWNER AppCollaborator for fake_app)
```

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `test_engine` | session | Creates schema via `Base.metadata.create_all()` |
| `db` | function | Transactional session with full rollback |
| `client` | function | TestClient with `get_db` override |
| `fake_user` | function | User flushed to the test session |
| `fake_app` | function | App owned by fake_user |
| `fake_ai_service` | function | AIService in fake_app |
| `fake_agent` | function | Agent in fake_app |
| `fake_api_key` | function | Active APIKey for fake_app |
| `auth_headers` | function | `{"Authorization": "Bearer ..."}` for fake_user |
| `owner_headers` | function | Same + OWNER AppCollaborator for fake_app |

When adding test data inside a test, use `db.add(obj); db.flush()` (not `commit`).

## Factory-Boy (`tests/factories.py`)

For tests that need many objects:

```python
from tests.factories import configure_factories, UserFactory, AppFactory, AgentFactory

def test_many_agents(client, db, auth_headers):
    configure_factories(db)                    # ALWAYS call first to bind to the test session
    agents = [AgentFactory() for _ in range(5)]
    # All 5 exist for the duration of the test, rolled back at teardown
```

Available factories: `UserFactory`, `AppFactory`, `AIServiceFactory`, `AgentFactory`, `APIKeyFactory`, `AppCollaboratorFactory`.

## Project-Specific API Knowledge

- **Dev login endpoint**: `POST /internal/auth/dev-login` with body `{"email": "..."}` returns `{"access_token": "..."}`. There is no `/auth/fake-login` endpoint — do not use that path.
- **Internal API** has prefix `/internal/`, uses session/JWT auth, and is protected by `@require_min_role(AppRole.<LEVEL>)`. Use `auth_headers` for read endpoints and `owner_headers` for endpoints requiring OWNER (most mutations on App-scoped resources).
- **Public API** has prefix `/public/v1/` and authenticates via `X-API-KEY` (use `fake_api_key`).
- All resources are scoped by `app_id`. Missing app → expect `404`.

## Mocking Patterns

- `mocker.patch("target", return_value=...)` — replace a function or method
- `mocker.MagicMock()` / `mocker.AsyncMock()` — for sync / async fakes
- `mocker.spy(obj, "method")` — record calls but run the real code
- **Patch at the import location, not the definition location**: patch `"services.agent_service.AgentRepository.get"`, NOT `"repositories.agent_repository.AgentRepository.get"`
- Never call real external services (LLM providers, MCP servers, external APIs) — always mock them. Tests must be hermetic.

## Test Naming & Structure

- Files: `test_<thing>.py`
- Functions / methods: `test_<what>_<when_condition>`, e.g. `test_login_returns_401_for_unknown_email`
- Group with classes: `TestHappyPath`, `TestErrorCases`, `TestEdgeCases`
- Each test must be self-contained — never assume execution order

For every endpoint, cover at minimum:

1. Happy path (`200`/`201` + body assertion)
2. Missing resource (`404`)
3. Unauthorized (`401`/`403`)

## Coverage Targets

Aspirational, enforced gradually:

- After unit tests: ≥ 40 % overall
- After integration tests: ≥ 65 % overall

Generate reports with `pytest -v --cov=backend --cov-report=term-missing` or `--cov-report=html` for the browser view.

## Running Tests

```bash
# Fast unit tests — no DB needed, run constantly
poetry run pytest tests/unit/ -v

# Integration tests — auto-manage the ephemeral test DB on port 5433
./scripts/test.sh -m integration

# Or manually:
docker compose -f docker/docker-compose.yaml --profile test up -d db_test
poetry run pytest tests/integration/ -v

# Full suite with coverage
poetry run pytest -v --cov=backend --cov-report=term-missing

# Single test or class by name
poetry run pytest -k "test_blocks_at_limit" -v
poetry run pytest -k "TestRateLimit" -v
```

The test DB URL is hardcoded in `pyproject.toml` under `[tool.pytest.ini_options]` via `pytest-env`: `postgresql://test_user:test_pass@localhost:5433/test_db`. Do NOT hardcode this URL inside test files.

## CI/CD

Tests run in `.github/workflows/test.yaml`:

- **`unit-tests`** — `pytest tests/unit/ -v`, no DB service, fast
- **`integration-tests`** — service container `pgvector/pgvector:pg17` on port 5433, then `pytest tests/integration/ -v`
- **`frontend-lint`** — `npm run lint` on the frontend

Triggers: every push to `main`, `develop`, `feat/**`, `fix/**`; every PR to `main` or `develop`.

Full guide: `docs/testing/`.

## Common Failure Patterns (quick reference)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` | Wrong `pythonpath` | Verify `pyproject.toml` has `pythonpath = ["backend"]` |
| `connection refused` | Test DB not running | `./scripts/test.sh -m integration` or `docker compose --profile test up -d db_test` |
| `assert 404 == 200` | Wrong URL or missing fixture data | Verify path; ensure `db.flush()` was called |
| `assert 401 == 200` | Missing auth fixture | Add `auth_headers` or `owner_headers` |
| `assert 403 == 200` | Wrong role | Use `owner_headers` instead of `auth_headers` |
| `OperationalError` | Schema mismatch | `test_engine` fixture should `create_all()` — check it ran |
| Mock not taking effect | Patched the definition path, not the import path | Patch where the symbol is USED |

## Anti-Patterns (project-specific)

- ❌ `db.commit()` inside a test or fixture (breaks rollback isolation)
- ❌ `SessionLocal()` directly in tests (use the `db` fixture)
- ❌ Hardcoded DB connection strings in test files
- ❌ Calling a real LLM / external API from a test (always mock)
- ❌ Using `/auth/fake-login` (correct endpoint is `/internal/auth/dev-login`)
- ❌ Forgetting `configure_factories(db)` before factory calls
- ❌ Putting setup logic in `test_engine` (session-scoped — leaks between tests)
- ❌ Assuming a specific test execution order

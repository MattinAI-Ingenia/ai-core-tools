---
description: Project-specific conventions for the Mattin AI Python backend — paths, key utilities, layered architecture, AICT auth modes, tenant scoping, LangChain factories, and Poetry usage.
applyTo: "backend/**"
---

# Mattin AI Backend Conventions

These rules auto-apply whenever you edit any file under `backend/`. Agents like `@backend-expert` provide generic FastAPI / SQLAlchemy / LangChain expertise; this file pins down the **Mattin-specific** paths, tools and patterns.

## Layered Architecture

```
HTTP request
  → routers/  (HTTP concerns only — validation via Pydantic, status codes, deps)
    → services/  (business logic, transactional boundaries, cross-entity rules)
      → repositories/  (data access, SQLAlchemy queries; no HTTPException, no business rules)
        → models/  (SQLAlchemy ORM entities)
```

- **Routers** never contain business logic. They wire requests/responses and call a service.
- **Services** are where business rules live. Services consume repositories, never the DB directly.
- **Repositories** return ORM objects (or raise `NotFound`-style domain errors). They never raise `HTTPException`.
- **Models** are pure SQLAlchemy. No FastAPI / Pydantic imports.

## File Structure (real paths)

```
backend/
├── main.py                   # FastAPI app entry — lifespan (CheckpointerCacheService, OIDC startup)
├── models/                   # SQLAlchemy ORM entities — ALL must be imported in models/__init__.py
│   └── __init__.py           # Registry — adding a new model without registering it breaks Alembic autogenerate
├── schemas/                  # Pydantic v2 schemas (request/response)
├── repositories/             # Data access — one repository per aggregate root
├── services/                 # Business logic — one service per aggregate or workflow
├── routers/
│   ├── internal/             # Frontend ↔ backend API — session/OIDC auth
│   ├── public/v1/            # External API — X-API-KEY header, rate-limited, CORS-validated
│   ├── mcp/                  # JSON-RPC 2.0 MCP endpoints — X-API-KEY auth
│   └── controls/             # Cross-cutting middleware: rate limit, CORS, file size
├── tools/                    # AI / LLM utilities (see "Key Tools" below)
│   └── ai/                   # LLM provider implementations
├── auth/                     # FAKE / LOCAL / OIDC handlers
├── utils/                    # Cross-cutting helpers (logger, auth_config, etc.)
└── db/                       # Engine, session factory, Base
```

## Auth Modes (`AICT_LOGIN` env var)

| Mode | When | What you write against |
|------|------|------------------------|
| `FAKE` | Local dev | Simplified email-only login (`POST /internal/auth/dev-login` returns a JWT for any email) |
| `LOCAL` | SaaS deployments | Email + password authenticated against `UserCredential` (hashed); session JWTs |
| `OIDC` | Production (Azure Entra ID) | Token validated against issuer/audience/expiry; lifespan startup fetches JWKS |

Endpoints under `routers/internal/` accept any of these. Endpoints under `routers/public/v1/` and `routers/mcp/` accept only `X-API-KEY`.

## Tenant Scoping (mandatory)

Every business resource (Agent, Silo, Repository, AIService, EmbeddingService, OutputParser, MCPServer, MCPConfig, APIKey, Skill, etc.) is owned by an `App`. Service and repository code MUST filter by `app_id` in every query. The decorator `@require_min_role(AppRole.<LEVEL>)` enforces both presence and authorization in routers:

```python
from utils.auth_decorators import require_min_role
from models.app_collaborator import AppRole

@router.patch("/apps/{app_id}/agents/{agent_id}")
@require_min_role(AppRole.EDITOR)
async def update_agent(app_id: int, agent_id: int, ...):
    ...
```

Role hierarchy: `VIEWER < EDITOR < ADMINISTRATOR < OWNER < OMNIADMIN`. `OMNIADMIN` is set via the `AICT_OMNIADMINS` env var (comma-separated emails).

## Key Tools (use these, do not reinvent)

### `backend/tools/vector_store_factory.py`
Returns a PGVector or Qdrant instance based on `silo.vector_store_type`. **Never** instantiate `PGVector(...)` or `QdrantVectorStore(...)` directly from a service or router — always go through the factory. Each silo's collection is named `silo_{silo_id}`.

### `backend/tools/embeddingTools.py`
Returns the right `Embeddings` implementation for the silo's `EmbeddingService`. Supported providers: OpenAI, MistralAI, Ollama, Custom/HuggingFace, Azure OpenAI, **Google AI Studio**, **Google Cloud Vertex AI**. New providers go here, never inline.

### `backend/tools/langsmith_config.py`
Resolves LangSmith tracing config: per-app `App.langsmith_api_key` first (project = app name), then falls back to global env vars (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). Test endpoint: `POST /internal/apps/{id}/langsmith/test`. Do not read these env vars directly from services — go through this helper.

### `backend/tools/ai/`
Per-provider LLM implementations. Wired through `AIService` configs. When adding a provider, follow the existing module pattern (one file per provider, registered in the factory).

## Conversation Memory

Memory uses LangGraph's `AsyncPostgresSaver` as the canonical checkpointer.

- **Thread ID format**: `thread_{agent_id}_{session_id}` — always use this exact format
- **Per-Agent config** (model fields, with defaults):
  - `has_memory: bool` — enable/disable
  - `memory_max_messages: int = 20`
  - `memory_max_tokens: int = 4000`
  - `memory_summarize_threshold: int = 10`
- These defaults MUST stay in sync with `frontend/src/constants/agentConstants.ts` (e.g. `DEFAULT_MEMORY_SUMMARIZE_THRESHOLD = 20`). Change both sides in the same commit.

## Poetry Environment

This project uses **Poetry**. All Python commands must run through the Poetry virtual environment:

```bash
poetry run uvicorn backend.main:app --reload --port 8000
poetry run pytest tests/unit/ -v
poetry run alembic upgrade head
```

Never use bare `python`, `pytest`, `alembic`, or `uvicorn` unless you have already activated the venv with `poetry shell`.

## LangChain / LangGraph Project Patterns

- **Chains**: LCEL pipe syntax — `prompt | model | parser`. Use `model.with_structured_output(PydanticModel)` for typed responses (preferred over manual `JsonOutputParser`).
- **Agents**: Use LangGraph `StateGraph` with `TypedDict` or Pydantic state. Wire tools with `model.bind_tools(tools)` or a `ToolNode`. Do not use the deprecated `AgentExecutor`.
- **Tools**: `@tool` decorator for simple cases; `StructuredTool.from_function()` for complex inputs; `BaseTool` subclass when state or async is needed.
- **MCP tools**: load via `langchain-mcp-adapters` (`MultiServerMCPClient`). Both `stdio` and `http` transports are supported.
- **RAG**: `RecursiveCharacterTextSplitter` → embeddings (via `embeddingTools.py`) → vector store (via `vector_store_factory.py`) → retriever. Use `MultiQueryRetriever` for query expansion, `EnsembleRetriever` (BM25 + vector) for hybrid search.
- **Streaming**: `agent.astream_events(messages, version="v2")` for token-level events; wrap in an SSE response in the router.
- **Async-only LLM calls**: `ainvoke`, `astream`, `abatch`. Never mix sync and async in the same chain. Never block the event loop with sync I/O inside an `async def`.
- **Resilience**: `.with_fallbacks([...])` for provider failover, `.with_retry(...)` for transients.

## Pydantic v2

- Always use `model_config = ConfigDict(from_attributes=True)` instead of the old `class Config: orm_mode = True`.
- Use `model_dump()` / `model_validate()` (Pydantic v2 API), not `dict()` / `parse_obj()`.
- Validators use `@field_validator("field")` + `@classmethod` (not the old `@validator`).
- Define separate schemas for `list`, `detail`, `create`, `update` operations.

## SQLAlchemy 2.x

- Use the `select()` construct for new queries (`db.execute(select(Model).where(...))`) rather than legacy `db.query(Model).filter(...)`.
- Eager-load to avoid N+1: `joinedload(...)` for to-one, `selectinload(...)` for to-many.
- Use connection pooling (defaults in `backend/db/database.py`); for new engines, prefer `pool_pre_ping=True`.
- Use `db.flush()` to assign IDs without committing when needed inside a service operation.

## API Conventions

- Internal API: prefix `/internal/`. Public API: prefix `/public/v1/` (versioned). MCP: `/mcp/v1/`.
- Versioning is URL-based for the public API; never introduce header-based versioning.
- Responses use Pydantic response models — never return raw `dict`.
- Errors: raise `HTTPException` with proper status code; never expose stack traces or internal field names to clients.
- Pagination: cursor-based for large collections, `limit`/`offset` for small ones.

## Anti-Patterns (project-specific)

- ❌ Instantiating a vector store directly (use `vector_store_factory.get_vector_store(silo)`)
- ❌ Reading LangSmith env vars from a service (go through `langsmith_config.py`)
- ❌ Adding a model file without registering it in `backend/models/__init__.py`
- ❌ Putting business logic in a router (move it to a service)
- ❌ Raw SQL strings — use the SQLAlchemy ORM and `select()`
- ❌ Calling `print()` for application output — use the project logger from `utils.logger.get_logger(__name__)`
- ❌ Forgetting `@require_min_role` on a router that touches a tenant resource
- ❌ Reading `AICT_LOGIN` directly in business code — auth modes are abstracted in `backend/auth/`

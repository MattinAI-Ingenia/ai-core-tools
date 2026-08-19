# Mattin AI - Your AI Toolbox

[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Mattin AI is a comprehensive AI toolbox that provides a wide range of artificial intelligence capabilities and tools. This project offers various AI functionalities including:

- Large Language Models (LLMs) integration and management
- Retrieval-Augmented Generation (RAG) systems
- Semantic search capabilities
- Vector database management
- AI agents and automation
- And more...

The project aims to simplify the integration and use of AI technologies, providing a unified platform for various AI-powered solutions.

## Features

- **LLM Integration**: Easy access and management of various Large Language Models
- **RAG Systems**: Implementation of Retrieval-Augmented Generation for enhanced AI responses
- **Semantic Search**: Advanced search capabilities using semantic understanding
- **Vector Databases**: Efficient storage and retrieval of vector embeddings
- **AI Agents**: Framework for building and deploying AI agents
- **Modular Architecture**: Easy to extend and customize for specific needs

---

## Quick Start

### Prerequisites

- **For Docker**: Docker 20.10+ and Docker Compose v2+
- **For local development**: Python 3.11+, Node.js 18+, PostgreSQL with pgvector

### 1. Clone the Repository

```bash
git clone https://github.com/lksnext-ai-lab/ai-core-tools.git
cd ai-core-tools
```

---

## Option 1: Docker Compose (Recommended)

The fastest way to get started. Everything is served from the same origin via
Caddy — no CORS, no port juggling. Two ways to obtain the images:

- **Pull from GHCR** (fast, no source code needed) — for client servers and demos
- **Build locally** (includes your code changes) — for development

```bash
# 1. Copy environment template into the repo-root .env
#    (docker/.env is a symlink to it — never run this from inside docker/,
#     the copy would follow the link and overwrite the root file)
cp .env.example .env

# 2. Edit .env — fill in the required values (see "Configure your .env" below)

cd docker

# 3a. Pull prebuilt images (recommended)
docker compose pull backend frontend
docker compose up -d

#     — OR —

# 3b. Build from source (for dev with local changes)
docker compose up -d --build

# 4. Wait ~30 seconds and access:
#    - App:      http://localhost/
#    - API Docs: http://localhost/docs/internal
```

### Configure your `.env`

After `cp .env.example .env`, you only need to change the fields that do not have a working default or that block startup:

| Variable | Required | What to put | Notes |
|----------|----------|-------------|-------|
| `DATABASE_PASSWORD` | Yes | A strong random password | Compose will fail if it is empty |
| `SECRET_KEY` | Yes | Generate one with `python -c "import secrets; print(secrets.token_hex(32))"` | Used to sign sessions/JWTs |
| `FRONTEND_URL` | Only outside local Docker | `http://<server-ip-or-domain>` | In local Docker, `http://localhost` from `.env.example` already works |

If you set `AICT_LOGIN=OIDC`, you must also fill `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID`, `ENTRA_CLIENT_SECRET`, `VITE_OIDC_ENABLED=true`, `VITE_OIDC_AUTHORITY`, `VITE_OIDC_CLIENT_ID`, and `VITE_OIDC_REDIRECT_URI`.

### First login in `AICT_LOGIN=FAKE`

The default Docker setup uses `AICT_LOGIN=FAKE`. In that mode, the login email must already exist in the `User` table before you can sign in.

1. Start the stack with `docker compose up -d` or `docker compose up -d --build`.
2. Create the first user inside the running backend container:

```bash
docker compose exec backend python -m utils.seed_dev_users --yes \
  --users "you@company.com:Your Name"
```

3. Give that user write permissions. New users default to the `viewer` platform
   role, and **viewers can log in but cannot create or modify** apps, agents,
   silos or any other resource (every write returns `403 Viewer role cannot
   create or modify resources`). Pick one option:

   - **Make them an omniadmin** (simplest for your own admin account): add the
     email to `AICT_OMNIADMINS` in `.env` and re-run `docker compose up -d`.
     Omniadmins have full access and skip the step below.
   - **Promote from the UI**: log in as an omniadmin and go to
     **Admin → Users**, then set the role to **Editor**.
   - **Promote via SQL**:

     ```bash
     docker compose exec postgres psql -U mattin -d mattin_ai -c \
       "UPDATE \"User\" SET platform_role = 'editor' WHERE email = 'you@company.com';"
     ```

4. Log in with that email from the UI.
5. Create your first App, or import `scripts/demo-app.json`.

If you prefer to define the initial user in the `.env`, uncomment and fill
`AICT_DEV_SEED_USERS`, then run:

```bash
docker compose exec backend python -m utils.seed_dev_users --yes
```

### What `LIGHTRAG_ENABLED` does

`LIGHTRAG_ENABLED=true` enables the LightRAG integration in the backend. This means silos can use graph-augmented retrieval backed by Neo4j in addition to the configured vector store.

Set `LIGHTRAG_ENABLED=false` if you do not want to use LightRAG. Neo4j is still started by the Docker stack, but the backend will not expose or use the LightRAG functionality.

Published images: `ghcr.io/lksnext-ai-lab/mattinai-{backend,frontend}`. See
[docker/README.md](docker/README.md) for deployment details, primary login, and
the path to HTTPS.

### Try the Demo App

A pre-built demo workspace is available at `scripts/demo-app.json`. Import it to get a fully configured workspace with 6 agents, RAG, structured output, MCP integration, OCR, and more — no manual setup required.

From the UI: **Apps → Import App → select `demo-app.json` → provide your API keys → Import**.

See [App Export and Import — Demo App](docs/guides/app-export-import.md#demo-app) for details.

### Docker Commands

All commands run from the `docker/` directory.

```bash
# View logs in real-time
docker compose logs -f

# View logs for a specific service
docker compose logs -f backend

# Stop services
docker compose down

# Rebuild images (after code changes)
docker compose up -d --build

# Remove everything (including database data)
docker compose down -v
```

---

## Option 2: Local Development (Without Docker)

For active development on the source code.

### 1. Database Setup

You need PostgreSQL with the pgvector extension:

```bash
# Option A: Only PostgreSQL with Docker
docker run -d --name mattin-postgres \
  -e POSTGRES_DB=mattin_ai \
  -e POSTGRES_USER=mattin \
  -e POSTGRES_PASSWORD=mattin_secure_2024 \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Enable the pgvector extension in that database
docker exec -it mattin-postgres psql -U mattin -d mattin_ai -c \
  "CREATE EXTENSION IF NOT EXISTS vector;"

# Option B: PostgreSQL installed locally
# Make sure to install and enable the pgvector extension in your target database
```

### 2. Backend (FastAPI)

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install Poetry
pip install poetry

# Install dependencies with Poetry (from project root)
poetry install

# Configure environment variables — the repo-root .env, the same file Docker uses
# (python-dotenv resolves to it from any directory in the repo; a backend/.env
# would never be read when running from the project root)
cp .env.example .env
# Edit .env:
#   - set SQLALCHEMY_DATABASE_URI to your local database (the Docker stack ignores
#     this value: docker-compose.yaml hardcodes the container-side one)
#   - AICT_LOGIN=LOCAL for admin-provisioned email+password, or OIDC to test Entra
# Example:
# SQLALCHEMY_DATABASE_URI=postgresql://mattin:mattin_secure_2024@localhost:5434/mattin_ai

# Run migrations
alembic upgrade head

# Seed a local dev user (run from the backend/ directory)
cd backend
python -m utils.seed_dev_users --yes --users "you@company.com:Your Name"
cd ..

# Give that user write permissions
# If you are using the Docker Postgres from Option A:
docker exec -it mattin-postgres psql -U mattin -d mattin_ai -c \
  "UPDATE \"User\" SET platform_role = 'editor' WHERE email = 'you@company.com';"

# If you are using a locally installed Postgres from Option B:
psql postgresql://mattin:mattin_secure_2024@localhost:5432/mattin_ai -c \
  "UPDATE \"User\" SET platform_role = 'editor' WHERE email = 'you@company.com';"

# Start server
uvicorn backend.main:app --reload --port 8000
```

Without the `SQLALCHEMY_DATABASE_URI`, the backend will not boot. Without the seeded `User` row, the FAKE login flow cannot authenticate. Without
`platform_role='editor'`, the user can log in but cannot create or modify
resources.

### 3. Frontend (React)

```bash
# In another terminal
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
```

Edit `frontend/.env` with the following configuration for local development:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Authentication - disable OIDC for local development
VITE_OIDC_ENABLED=false

# IMPORTANT: If using OIDC in local development, change the redirect URI
# Docker uses port 3000, local development uses port 5173
VITE_OIDC_REDIRECT_URI=http://localhost:5173/auth/success
```

> **Note**: When running without Docker, the frontend runs on port 5173 (Vite default), so `VITE_OIDC_REDIRECT_URI` must be updated accordingly if you enable OIDC authentication.

```bash
# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

---

## Configuration

This repository contains multiple `.env.example` files. They are not for the same execution mode:

| Scenario | Template to copy | Resulting file | Used by |
|----------|------------------|----------------|---------|
| Docker Compose | `.env.example` | `.env` (repo root) | `cd docker && docker compose ...` |
| Local backend | `.env.example` | `.env` (repo root) | `uvicorn backend.main:app --reload` |
| Local frontend | `frontend/.env.example` | `frontend/.env` | `cd frontend && npm run dev` |

**Docker reads the repo-root `.env`.** `docker/.env` is a symlink to `../.env`, so
there is a single file to maintain. Two consequences:

- Copy the single template from the repo root: `cp .env.example .env`.
- A variable only reaches a container if `docker-compose.yaml` declares it under
  that service's `environment:`. Adding it to `.env` alone is not enough — this
  silently swallowed `UVICORN_WORKERS` until it was wired up.

There is a single template, `.env.example`, covering both.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_PASSWORD` | PostgreSQL password used by the Docker stack | `change-me-please` |
| `SECRET_KEY` | Session/JWT signing secret | `hex-string-generated-with-secrets.token_hex(32)` |

For local Docker, `FRONTEND_URL`, `AICT_LOGIN`, and `LIGHTRAG_ENABLED` already come with working defaults in `.env.example`. Change them only if your deployment needs something different.

For local development without Docker, the FastAPI app reads the **same repo-root
`.env`** (`load_dotenv()` walks up from the working directory), so there is one
backend configuration file for both modes. Only the React app has its own,
`frontend/.env`, because Vite loads it from `frontend/`. Note that the container-side topology
(`DATABASE_HOST`, `DATABASE_PORT`, `SQLALCHEMY_DATABASE_URI`) is hardcoded in
`docker-compose.yaml`, so a `localhost` value in the root `.env` cannot leak into
a container.

### Environment-Specific Variables

| Variable | Docker | Local |
|----------|--------|-------|
| `DATABASE_HOST` | (not needed) | `localhost` |
| `FRONTEND_URL` | `http://localhost` by default | (not used) |
| `VITE_API_BASE_URL` | relative same-origin via Caddy | `http://localhost:8000` |

For Docker, `AICT_OMNIADMINS` does not create users by itself. In `FAKE` mode you must seed or insert the login user into `User` first, and ensure the user has `platform_role='editor'` if they need to create or modify resources.

### AI Service Configuration

The platform supports multiple AI providers:

- OpenAI (GPT models)
- Anthropic (Claude models)
- Azure OpenAI
- Mistral AI
- Ollama (local models)

Configure these through the web interface or environment variables.

---

## Authentication Modes

### Development Mode (AICT_LOGIN=FAKE)

- Simple email login
- Ideal for testing and development
- No additional configuration required
- Email must exist in the database

### Production Mode (AICT_LOGIN=OIDC)

- Authentication with Microsoft Entra ID (Azure AD)
- Requires configuration:

```env
AICT_LOGIN=OIDC
ENTRA_TENANT_ID=your-tenant-id
ENTRA_CLIENT_ID=your-client-id
ENTRA_CLIENT_SECRET=your-client-secret
VITE_OIDC_ENABLED=true
VITE_OIDC_AUTHORITY=https://login.microsoftonline.com/{tenant-id}/v2.0
VITE_OIDC_CLIENT_ID=your-client-id
```

---

## Services and Ports

| Service | Docker (Caddy, single port) | Local |
|---------|------------------------------|-------|
| Frontend | http://localhost/ | http://localhost:5173 |
| Backend (API) | http://localhost/ (same origin via Caddy) | http://localhost:8000 |
| PostgreSQL | internal network only — not published to the host | localhost:5432 |
| API Docs | http://localhost/docs/internal | http://localhost:8000/docs/internal |

> In Docker, only Caddy's port (80 by default, configurable via `HTTP_PORT`) is exposed to the host. Backend, frontend, PostgreSQL, Qdrant and Neo4j stay on the internal Docker network.

---

## Architecture

The project consists of several main components:

- **Backend**: FastAPI-based REST API with Python
- **Frontend**: React-based web interface with TypeScript
- **Database**: PostgreSQL with pgvector for vector storage
- **AI Services**: Modular integration with various LLM providers

---

## Troubleshooting

### Frontend doesn't load

1. Verify the backend is running
2. Check browser console (F12)
3. Ensure `VITE_API_BASE_URL` points to the correct backend

### Database connection error

1. Verify PostgreSQL is running
2. Check credentials in `.env`
3. For Docker, wait ~30 seconds for PostgreSQL to fully start

### API keys don't work

1. Verify you edited the right file: the repo-root `.env` for both Docker (`docker/.env` is a symlink to it) and the local backend; `frontend/.env` for the React dev server (see the table in [Configuration](#configuration))
2. Reload after changing it: `docker compose up -d` for Docker (a plain `restart` does not pick up new env values), or restart `uvicorn` locally
3. Check that the key has no extra spaces or quotes

### Clean and start fresh (Docker)

```bash
cd docker
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/README.md#contributing) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Documentation

- [Full Documentation](docs/README.md)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## License

This project is available under a dual licensing model:

- **Open Source**: GNU Affero General Public License v3.0 (AGPL 3.0)
- **Commercial**: Proprietary license with enhanced rights and features

### Open Source (AGPL 3.0)
- Free to use for development and personal use
- Community contributions welcome
- Source code disclosure required for network use
- Copyleft obligations for modifications

### Commercial License
- Full AICT functionality without restrictions
- Commercial use rights without copyleft obligations
- Client modification rights for specific projects
- Enterprise features and support
- No source code disclosure requirements

For more information, see:
- [LICENSING.md](LICENSING.md) - Detailed licensing information
- [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) - Commercial license terms
- [CLIENT_LICENSE_AGREEMENT.md](CLIENT_LICENSE_AGREEMENT.md) - Client agreement template

**Contact LKS Next for commercial licensing inquiries.**

## Security

We take security seriously. If you discover a vulnerability, please **do not open a public GitHub issue**.

Report it privately via [GitHub Security Advisories](https://github.com/lksnext-ai-lab/ai-core-tools/security/advisories/new) or email **mattin-ai@lksnext.com**.

See [SECURITY.md](SECURITY.md) for the full vulnerability disclosure policy, supported versions, and response SLAs.

## Support

- Create an issue for bug reports or feature requests
- Check the documentation for common questions
- Join our community discussions

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Frontend powered by [React](https://reactjs.org/)
- Vector operations with [pgvector](https://github.com/pgvector/pgvector)
- AI integrations via [LangChain](https://langchain.com/)

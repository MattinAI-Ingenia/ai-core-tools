# Mattin AI - Your AI Toolbox

[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Quality gate status](https://sonarcloud.io/api/project_badges/measure?project=lks-ot-i-plus-d_mattin-ai&metric=alert_status&token=c319b34cc5a95f63cb61e299c00557054b459bb1)](https://sonarcloud.io/summary/new_code?id=lks-ot-i-plus-d_mattin-ai)

Mattin AI is an AI toolbox: LLM management, RAG (vector search and, with LightRAG, a
knowledge graph), semantic search and AI agents. FastAPI backend, React frontend,
PostgreSQL/pgvector, Qdrant and Neo4j, all behind Caddy on a single port.

---

## Quick start (Docker)

You need Docker 20.10+ with Compose v2, the host ports 80 and 5434 free, and an API key
of an LLM provider (the demo app uses OpenAI).

### 1. Clone

LightRAG lives in the `lightrag` branch of this repository (upstream `develop` does not
have it). The clone URL is behind the green **Code** button of the page you are reading:

```bash
git clone -b lightrag <clone-url-of-this-repository>
cd ai-core-tools
```

### 2. Configure

From the repo root:

```bash
cp .env.example .env
ln -s ../.env docker/.env
```

Compose reads `docker/.env`; the link keeps a single file to edit. Without symlink
support, skip the `ln` and add `--env-file ../.env` to every `docker compose` command.

Real values go only in this `.env` copy, never in `.env.example`. Fill the three secrets
with random values in one command (it only replaces those three lines):

```bash
python3 -c "import re,secrets;t=open('.env').read();[t:=re.sub(f'(?m)^{k}=.*',f'{k}={v}',t) for k,v in {'DATABASE_PASSWORD':secrets.token_urlsafe(24),'SECRET_KEY':secrets.token_hex(32),'NEO4J_PASSWORD':secrets.token_urlsafe(24)}.items()];open('.env','w').write(t)"
```

Then edit `.env` and set your email in `AICT_OMNIADMINS`. Required variables:

| Variable | Put | Why |
|----------|-----|-----|
| `DATABASE_PASSWORD` | a random password (the command above) | Compose fails if empty |
| `SECRET_KEY` | a random 64-character hex string (the command above) | signs sessions |
| `NEO4J_PASSWORD` | a random password, **not** `neo4j` (the command above) | Neo4j 5 refuses its default; Compose fails if empty |
| `AICT_OMNIADMINS` | your email | gets an admin account on first start (step 4) |

Everything else has a working default. To serve from another host, set
`AICT_PUBLIC_URL=http://<server-ip-or-domain>`.

### 3. Start

```bash
cd docker
docker compose up -d --build
```

The first build takes several minutes. Then open http://localhost/ (API docs:
http://localhost/docs/internal).

> On `lightrag` the images must be built locally, as above: the ones published in GHCR
> come from `develop`, which has no LightRAG, and there is no `lightrag` tag. Do not use
> `docker compose pull`.

### 4. First login, users and admins

Accounts are local (`AICT_LOGIN=LOCAL`: email + password stored by the app). Run the
commands from `docker/`.

The first admin: on first start the backend creates an admin account for each email in
`AICT_OMNIADMINS` and prints a one-time set-password link in its log:

```bash
docker compose logs backend | grep -A5 "omniadmin_bootstrap: NEW"
```

Open the link (`.../set-password?token=...`), choose a password and log in. The link
expires after 48 hours and is printed only once; if you miss it, set the password
directly:

```bash
docker compose exec backend python -m utils.seed_dev_users --yes \
  --users "<email>:<name>:<password>"
```

More users and admins, as an admin in **Admin → Users**:

- **Create user**: enter email and name; the app shows a one-time set-password link to
  send to that person. The same page issues a new link if it expires.
- **Roles**: `viewer` (the default: can log in, but every write returns `403`), `editor`
  (can create and modify) and `admin`. Change them on that page.
- **Another admin**: add the email to `AICT_OMNIADMINS` and run `docker compose up -d`
  (the account is created at startup), or set the `admin` role on that page.

### 5. Try LightRAG

LightRAG keeps a knowledge graph (in Neo4j) next to the vectors of a silo, which helps
with multi-hop questions. It is on by default (`LIGHTRAG_ENABLED=true`). Reference:
[docs/dependencies/lightrag.md](docs/dependencies/lightrag.md) (in Spanish).

Import the demo: **Apps → Import App → `scripts/lightrag-demo-app.json`**, and enter your
OpenAI API key for each service when asked. It creates:

- AI services `LightRAG Extract` and `LightRAG Keywords` (`gpt-6-luna`: build the graph
  while indexing, and read each question) and `LightRAG Query` (`gpt-6.1-terra`: the
  agent's LLM, writes the answer), plus an embedding service (`text-embedding-3-small`);
- the silo `LightRAG Knowledge Base` and an empty repository `Documentos`;
- the agent `LightRAG Agent` (no memory; picks the query mode for each question), and,
  optionally, a System One agent with its service (needs an API key with access to the
  Decisions API; delete both if you do not need them).

Upload your documents to `Documentos` (a dialog shows the estimated cost and time
first), wait for the indexing and chat with `LightRAG Agent` in its playground.

Before importing, know that:

- The language the graph is extracted in (`lightrag_language`, `Spanish` in the demo), the
  chunking and the extract and embedding services cannot be changed once the silo exists.
  Edit the JSON first if your documents are in another language.
- OpenAI services must have an empty **Endpoint** (the JSON does). With an explicit
  `https://api.openai.com/v1`, reasoning models such as `gpt-6-*` reject tools on that
  route and the agent fails.
- By hand: create the AI and embedding services, a silo of type *LightRAG* (it needs an
  extract service and an embedding service) and an agent with that silo and the query
  mode `skill-routed`.

More: `scripts/demo-app.json` is a demo with 6 agents (RAG, structured output, MCP, OCR)
without LightRAG; import it the same way. See
[App Export and Import](docs/guides/app-export-import.md#demo-app).

### Everyday commands

From `docker/`:

```bash
docker compose logs -f backend    # logs (omit the service for all of them)
docker compose down               # stop
docker compose up -d --build      # rebuild after changing code
docker compose down -v            # remove everything, including database data
```

---

## Local development (without Docker)

Needs Python 3.11+, Node.js 18+ and PostgreSQL with the pgvector extension.

```bash
# 1. PostgreSQL with pgvector (or your own, with the extension enabled)
docker run -d --name mattin-postgres -e POSTGRES_DB=mattin_ai -e POSTGRES_USER=mattin \
  -e POSTGRES_PASSWORD=<password> -p 5432:5432 pgvector/pgvector:pg17
docker exec -it mattin-postgres psql -U mattin -d mattin_ai -c \
  "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Backend, from the repo root
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install poetry && poetry install
cp .env.example .env              # the same root file Docker uses
```

In `.env`, set `DATABASE_PASSWORD` and uncomment the block "Desarrollo local SIN Docker"
(`DATABASE_HOST`, `DATABASE_PORT`, `SQLALCHEMY_DATABASE_URI`), with port 5432 and your
password. Alembic reads the `DATABASE_*` variables, the backend `SQLALCHEMY_DATABASE_URI`;
the backend does not boot without it.

```bash
alembic upgrade head
(cd backend && python -m utils.seed_dev_users --yes --users "<email>:<name>:<password>")
docker exec -it mattin-postgres psql -U mattin -d mattin_ai -c \
  "UPDATE \"User\" SET platform_role = 'editor' WHERE email = '<email>';"   # or psql <your URI>
uvicorn backend.main:app --reload --port 8000
```

Without the seeded password the user cannot log in with `AICT_LOGIN=LOCAL`, and without
the `editor` role they cannot create or modify anything. The frontend is separate:

```bash
cd frontend
npm install
cp .env.example .env      # VITE_API_BASE_URL=http://localhost:8000, VITE_OIDC_ENABLED=false
npm run dev               # http://localhost:5173
```

With OIDC in local development, set `VITE_OIDC_REDIRECT_URI=http://localhost:5173/auth/success`.

---

## Configuration

- Backend (Docker and local) reads one file: `.env` at the repo root, from
  `.env.example`. The React dev server reads `frontend/.env`, from
  `frontend/.env.example`.
- A variable reaches a container only if `docker/docker-compose.yaml` lists it under that
  service's `environment:`; adding it to `.env` alone is not enough.
- The compose file fixes the container topology (database host and port), so `localhost`
  values in `.env` for local development cannot leak into containers.
- AI providers (OpenAI, Anthropic, Azure OpenAI, Mistral AI, Ollama) are configured as AI
  services in the web UI.

To use Microsoft Entra ID instead of local accounts, set `AICT_LOGIN=OIDC` and
`VITE_OIDC_REDIRECT_URI` (see `.env.example`), plus:

```env
AICT_LOGIN=OIDC
ENTRA_TENANT_ID=your-tenant-id
ENTRA_CLIENT_ID=your-client-id
ENTRA_CLIENT_SECRET=your-client-secret
VITE_OIDC_ENABLED=true
VITE_OIDC_AUTHORITY=https://login.microsoftonline.com/{tenant-id}/v2.0
VITE_OIDC_CLIENT_ID=your-client-id
```

## Services and ports

| Service | Docker (through Caddy) | Local |
|---------|------------------------|-------|
| Frontend | http://localhost/ | http://localhost:5173 |
| Backend (API) | http://localhost/ (same origin) | http://localhost:8000 |
| API docs | http://localhost/docs/internal | http://localhost:8000/docs/internal |
| PostgreSQL | localhost:5434 (published for development) | localhost:5432 |

Caddy's port (80, set with `HTTP_PORT`) is the only one meant for outside access. Backend,
frontend, Qdrant and Neo4j stay on the internal Docker network.

## Troubleshooting

- `required variable ... is missing a value` from `docker compose`: fill that variable in
  `.env` and check that `docker/.env` is the link to `../.env` (step 2).
- The `neo4j` container keeps restarting: `NEO4J_PASSWORD` must be set and not `neo4j`.
- "port is already allocated": free 80 and 5434, or change `HTTP_PORT` (and set
  `AICT_PUBLIC_URL` to match, e.g. `http://localhost:8080`).
- The page does not load or the database refuses connections: wait about 30 seconds after
  starting, then check `docker compose ps` and `docker compose logs -f backend`.
- You changed `.env` and nothing happens: run `docker compose up -d` (a plain `restart`
  does not reload it) and check that keys have no extra spaces or quotes.

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
- [Internal API](docs/api/internal-api.md) and [public API](docs/api/public-api.md)
- [Deployment guide](docker/README.md) (in Spanish)

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

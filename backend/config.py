import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

load_dotenv()

from utils.secret_key import get_secret_key  # noqa: E402 – must follow load_dotenv()

class ClientConfig(BaseModel):
    client_id: str
    client_name: str
    oidc_enabled: bool = True
    oidc_authority: Optional[str] = None
    oidc_client_id: Optional[str] = None
    custom_domain: Optional[str] = None
    
def load_client_config() -> ClientConfig:
    """Load client configuration from environment variables."""
    login_mode = os.getenv('AICT_LOGIN', 'OIDC').upper()
    oidc_enabled = (login_mode == 'OIDC')
    
    return ClientConfig(
        client_id=os.getenv('CLIENT_ID', 'default'),
        client_name=os.getenv('CLIENT_NAME', 'Mattin AI'),
        oidc_enabled=oidc_enabled,
        oidc_authority=os.getenv('OIDC_AUTHORITY'),
        oidc_client_id=os.getenv('OIDC_CLIENT_ID'),
        custom_domain=os.getenv('CUSTOM_DOMAIN')
    )

CLIENT_CONFIG = load_client_config()

DATABASE_URL = os.getenv('SQLALCHEMY_DATABASE_URI', 'postgresql://iacoretoolsdev:iacoretoolsdev@localhost:5432/iacoretoolsdev')

# Fails fast at import if SECRET_KEY is absent, too short, or a known weak value.
SECRET_KEY: str = get_secret_key()
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')

VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'PGVECTOR').upper()

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_PREFER_GRPC = os.getenv('QDRANT_PREFER_GRPC', 'false').lower() == 'true'

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')

# MCP Server Configuration
# Base URL for generating MCP endpoint URLs (e.g., https://your-domain.com)
MCP_BASE_URL = os.getenv('MCP_BASE_URL', 'http://localhost:8000')

# ---------------------------------------------------------------------------
# Sandbox provider configuration (IT-2)
# ---------------------------------------------------------------------------

# System-wide default provider when an app has no explicit sandbox_provider.
# Allowed values: 'opensandbox' (self-hosted), 'daytona' (managed SaaS),
# 'e2b' (managed cloud).
SANDBOX_DEFAULT_PROVIDER = os.getenv('SANDBOX_DEFAULT_PROVIDER', 'opensandbox').lower()

# Comma-separated list of provider names apps may select in this deployment.
SANDBOX_ALLOWED_PROVIDERS = [
    p.strip()
    for p in os.getenv('SANDBOX_ALLOWED_PROVIDERS', 'opensandbox,daytona,e2b').split(',')
    if p.strip()
]

# OpenSandbox server connection.
OPENSANDBOX_DOMAIN = os.getenv('OPENSANDBOX_DOMAIN', 'localhost:8080')
OPENSANDBOX_API_KEY = os.getenv('OPENSANDBOX_API_KEY', '')
OPENSANDBOX_CODE_INTERPRETER_IMAGE = os.getenv(
    'OPENSANDBOX_CODE_INTERPRETER_IMAGE',
    'opensandbox/code-interpreter:v1.0.2',
)

# Timeout for creating/reconnecting a sandbox (seconds).
SANDBOX_CREATE_TIMEOUT_S = int(os.getenv('SANDBOX_CREATE_TIMEOUT_S', '60'))

# Per-execution timeout (seconds).
SANDBOX_DEFAULT_TIMEOUT_S = int(os.getenv('SANDBOX_DEFAULT_TIMEOUT_S', '30'))

# Maximum sandbox session lifetime (hours).
SANDBOX_SESSION_TTL_H = float(os.getenv('SANDBOX_SESSION_TTL_H', '2'))

# Maximum idle time before cached sandboxes are stopped/destroyed (seconds).
SANDBOX_IDLE_TIMEOUT_S = int(os.getenv('SANDBOX_IDLE_TIMEOUT_S', '120'))

# How often the in-process sandbox reaper checks for idle sandboxes (seconds).
SANDBOX_REAPER_INTERVAL_S = int(os.getenv('SANDBOX_REAPER_INTERVAL_S', '30'))

# Maximum number of characters returned by run_code (truncates at this limit).
SANDBOX_MAX_OUTPUT_CHARS = int(os.getenv('SANDBOX_MAX_OUTPUT_CHARS', '20000'))

# Maximum number of REPL executions per agent turn (budget guard).
SANDBOX_MAX_EXECUTIONS_PER_TURN = int(os.getenv('SANDBOX_MAX_EXECUTIONS_PER_TURN', '5'))

# Maximum number of OpenSandbox code contexts created per language in one
# sandbox. Values above 1 allow parallel tool executions without sharing a busy
# interpreter session.
OPENSANDBOX_MAX_CONTEXTS_PER_LANGUAGE = int(os.getenv('OPENSANDBOX_MAX_CONTEXTS_PER_LANGUAGE', '4'))

# Daytona SaaS sandbox connection and creation options. The Daytona SDK also
# reads DAYTONA_API_KEY, DAYTONA_API_URL, and DAYTONA_TARGET directly from the
# environment; these constants document the integration surface.
DAYTONA_API_KEY = os.getenv('DAYTONA_API_KEY', '')
DAYTONA_API_URL = os.getenv('DAYTONA_API_URL', '')
DAYTONA_TARGET = os.getenv('DAYTONA_TARGET', '')
DAYTONA_IMAGE = os.getenv('DAYTONA_IMAGE', '')
DAYTONA_SNAPSHOT = os.getenv('DAYTONA_SNAPSHOT', '')
DAYTONA_WORKSPACE = os.getenv('DAYTONA_WORKSPACE', 'workspace')

# E2B cloud sandbox connection and creation options. The E2B SDK reads
# E2B_API_KEY directly from the environment.
E2B_API_KEY = os.getenv('E2B_API_KEY', '')
E2B_TEMPLATE = os.getenv('E2B_TEMPLATE', '')
E2B_WORKSPACE = os.getenv('E2B_WORKSPACE', '/home/user/workspace')

# Minutes before a sandbox TTL is proactively renewed.
SANDBOX_RENEW_MINUTES = int(os.getenv('SANDBOX_RENEW_MINUTES', '30'))

# ---------------------------------------------------------------------------
# Cross-session user memory configuration (global memory, spec user-memory-mem0)
# ---------------------------------------------------------------------------

# Deployment-wide kill switch (FR-10). Defaults to enabled (D-9): the real
# opt-in gate is the per-agent Agent.has_global_memory flag, so a default-on
# switch is still zero behavior change on upgrade. Tolerant boolean parsing:
# 'false'/'0'/'no' (case-insensitive) -> disabled, anything else including
# unset -> enabled.
GLOBAL_MEMORY_ENABLED = os.getenv('AICT_GLOBAL_MEMORY_ENABLED', 'true').strip().lower() not in (
    'false', '0', 'no',
)

# Max facts injected into the read-path context block per turn.
GLOBAL_MEMORY_TOP_K = int(os.getenv('AICT_GLOBAL_MEMORY_TOP_K', '5'))

# Timeout budgets (seconds) so a memory operation can never block a turn.
GLOBAL_MEMORY_READ_TIMEOUT_S = float(os.getenv('AICT_GLOBAL_MEMORY_READ_TIMEOUT_S', '1.5'))
GLOBAL_MEMORY_WRITE_TIMEOUT_S = float(os.getenv('AICT_GLOBAL_MEMORY_WRITE_TIMEOUT_S', '30'))

# Hard character cap on the rendered "remembered context" block (AD-7).
GLOBAL_MEMORY_CONTEXT_MAX_CHARS = int(os.getenv('AICT_GLOBAL_MEMORY_CONTEXT_MAX_CHARS', '1500'))

# Hard character cap on the extraction input window (FR-5, R-6).
GLOBAL_MEMORY_EXTRACTION_INPUT_MAX_CHARS = int(
    os.getenv('AICT_GLOBAL_MEMORY_EXTRACTION_INPUT_MAX_CHARS', '4000')
)

# Back-pressure bound on concurrent fire-and-forget write tasks (D-6).
GLOBAL_MEMORY_MAX_INFLIGHT_WRITES = int(os.getenv('AICT_GLOBAL_MEMORY_MAX_INFLIGHT_WRITES', '50'))

# CODE CONSTANT — never an env var (R-25). It participates in every memory
# namespace (D-2); making it configurable would strand existing facts beyond
# both enumeration (AD-10) and erasure (FR-8) -- undeletable personal data.
GLOBAL_MEMORY_NAMESPACE_ROOT = 'user_memory'

# Bounds for the extraction output (FR-5, D-23): caps the number of facts an
# extraction call may produce and the length of each fact.
GLOBAL_MEMORY_MAX_FACTS_PER_EXTRACTION = int(os.getenv('AICT_GLOBAL_MEMORY_MAX_FACTS_PER_EXTRACTION', '5'))

# Deployment-wide memory vector width (D-13). Every memory embedding service
# used on this deployment must match this dimensionality.
GLOBAL_MEMORY_EMBEDDING_DIMS = int(os.getenv('AICT_GLOBAL_MEMORY_EMBEDDING_DIMS', '1536'))

# Similarity ceiling above which a candidate fact is considered already known
# and dropped instead of written (D-14 dedup).
GLOBAL_MEMORY_DEDUP_THRESHOLD = float(os.getenv('AICT_GLOBAL_MEMORY_DEDUP_THRESHOLD', '0.95'))

# Batching trigger (D-15): extraction runs on turn 1 and every Nth turn after.
GLOBAL_MEMORY_WRITE_EVERY_N_TURNS = int(os.getenv('AICT_GLOBAL_MEMORY_WRITE_EVERY_N_TURNS', '5'))

# Read condition (D-17): for has_memory=True agents, retrieval only runs while
# turn_index <= this value (later turns already carry the block in-thread).
GLOBAL_MEMORY_READ_FIRST_N_TURNS = int(os.getenv('AICT_GLOBAL_MEMORY_READ_FIRST_N_TURNS', '2'))

# FR-11 `remember` tool (D-19). Off by default (R-18, a High security finding
# gated on deliberate per-deployment enablement) -- unlike the feature's own
# kill switch, which defaults to enabled.
GLOBAL_MEMORY_TOOL_ENABLED = os.getenv('AICT_GLOBAL_MEMORY_TOOL_ENABLED', 'false').strip().lower() in (
    'true', '1', 'yes',
)
GLOBAL_MEMORY_FACT_MAX_CHARS = int(os.getenv('AICT_GLOBAL_MEMORY_FACT_MAX_CHARS', '300'))
GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_TURN = int(os.getenv('AICT_GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_TURN', '3'))
GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_DAY = int(os.getenv('AICT_GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_DAY', '50'))

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

# LightRAG / Neo4j Configuration (optional, opt-in)
# These settings are only consumed when LIGHTRAG_ENABLED is true and a silo
# is configured to use the LightRAG backend. They default to safe no-op
# values so the backend boots cleanly without Neo4j running.
LIGHTRAG_ENABLED: bool = os.getenv('LIGHTRAG_ENABLED', 'false').lower() == 'true'
NEO4J_URI: Optional[str] = os.getenv('NEO4J_URI') or None
NEO4J_USERNAME: str = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD: Optional[str] = os.getenv('NEO4J_PASSWORD') or None
# Entity extraction configuration
# Maximum number of "gleaning" iterations the entity-extraction pipeline
# should perform when processing documents during indexing. Set to 0 to
# disable additional gleaning. This value is configurable via the
# ENTITY_EXTRACT_MAX_GLEANING environment variable (or entity_extract_max_gleaning).
ENTITY_EXTRACT_MAX_GLEANING: int = int(
    os.getenv('ENTITY_EXTRACT_MAX_GLEANING', os.getenv('entity_extract_max_gleaning', '0'))
)
# Output cap for the entity-extraction LLM (the `extract` role).
# Without it, an OpenAI-compatible server such as vLLM defaults max_tokens to
# "the rest of the context window" (~30k), and a model that starts inventing
# relationships generates until it hits that ceiling — minutes for one page.
# The extraction prompt asks for at most 100 records, which fits in ~4-6k
# tokens, so a cap truncates only runaway generations. A truncated response is
# not lost work: LightRAG parses it with json_repair, which closes the open
# structures and keeps every complete record extracted before the cut.
# Unset (default) keeps the provider's own behaviour.
LIGHTRAG_EXTRACT_MAX_TOKENS: Optional[int] = (
    int(os.environ['LIGHTRAG_EXTRACT_MAX_TOKENS'])
    if os.getenv('LIGHTRAG_EXTRACT_MAX_TOKENS')
    else None
)

# Send the extraction JSON schema to the LLM server as `response_format`, so
# decoding is constrained instead of the prompt merely asking for JSON. Only
# applied to providers with an OpenAI-compatible json_schema parameter (vLLM,
# OpenAI, Azure, OpenRouter) and only to entity-extraction calls. Set to false
# if your server rejects the parameter.
LIGHTRAG_EXTRACT_GUIDED_JSON: bool = (
    os.getenv('LIGHTRAG_EXTRACT_GUIDED_JSON', 'true').lower() == 'true'
)

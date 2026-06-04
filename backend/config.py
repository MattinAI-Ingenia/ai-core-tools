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
    """Load client configuration from environment variables"""
    # Determine if OIDC is enabled based on AICT_LOGIN mode
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

# SECRET_KEY has no insecure default. get_secret_key() validates and raises
# RuntimeError if the value is absent, too short, or a known weak default.
# Module-level evaluation means any process that imports config without a
# valid SECRET_KEY in the env will fail immediately — which is the intended
# fail-fast behaviour. Tests set this via pytest-env in pyproject.toml.
SECRET_KEY: str = get_secret_key()
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')

GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')

# Vector Database Configuration
# Supported types: PGVECTOR, QDRANT, PINECONE, WEAVIATE, CHROMA
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'PGVECTOR').upper()

# PGVector Configuration (uses existing PostgreSQL connection)
# PGVECTOR uses SQLALCHEMY_DATABASE_URI defined above

# Qdrant Configuration
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')  # Optional, for cloud instances
QDRANT_PREFER_GRPC = os.getenv('QDRANT_PREFER_GRPC', 'false').lower() == 'true'

# Pinecone Configuration (future support)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# Weaviate Configuration (future support)
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

# Chroma Configuration (future support)
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')

# MCP Server Configuration
# Base URL for generating MCP endpoint URLs (e.g., https://your-domain.com)
MCP_BASE_URL = os.getenv('MCP_BASE_URL', 'http://localhost:8000') 
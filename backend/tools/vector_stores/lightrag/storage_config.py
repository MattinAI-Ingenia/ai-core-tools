"""LightRAG storage backend configuration.

Builds the storage-backend dict that future steps will pass to ``LightRAG(...)``
and exports the matching environment variables that LightRAG's storage
implementations read directly via ``os.environ``.

Per D2 in ``plans/lightrag-integration/decisions.md`` the chosen backends are:

* ``graph_storage="Neo4JStorage"`` (Neo4j)
* ``vector_storage="QdrantVectorDBStorage"`` (Qdrant)
* ``kv_storage="PGKVStorage"`` (PostgreSQL)
* ``doc_status_storage="PGDocStatusStorage"`` (PostgreSQL)

This module only reads settings from :mod:`config` and the environment; it
does not import ``lightrag`` and does not open any connection.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


GRAPH_STORAGE = "Neo4JStorage"
VECTOR_STORAGE = "QdrantVectorDBStorage"
KV_STORAGE = "PGKVStorage"
DOC_STATUS_STORAGE = "PGDocStatusStorage"


def _parse_postgres_uri(uri: str) -> Dict[str, str]:
    """Decompose a SQLAlchemy Postgres URI into LightRAG's POSTGRES_* parts.

    LightRAG's ``postgres_impl`` reads ``POSTGRES_HOST`` / ``POSTGRES_PORT`` /
    ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` / ``POSTGRES_DATABASE`` from the
    environment rather than a single URI.
    """
    parsed = urlparse(uri)
    if not parsed.hostname or not parsed.path:
        raise RuntimeError(
            "SQLALCHEMY_DATABASE_URI is not a valid Postgres URI; cannot "
            "configure LightRAG PGKVStorage/PGDocStatusStorage."
        )
    return {
        "POSTGRES_HOST": parsed.hostname,
        "POSTGRES_PORT": str(parsed.port or 5432),
        "POSTGRES_USER": unquote(parsed.username) if parsed.username else "",
        "POSTGRES_PASSWORD": unquote(parsed.password) if parsed.password else "",
        "POSTGRES_DATABASE": parsed.path.lstrip("/"),
    }


def _require(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(
            f"LIGHTRAG_ENABLED=true but {name} is not configured. "
            "Set it in the environment (see docker/.env.example)."
        )
    return value


def build_storage_config() -> Dict[str, str]:
    """Validate config and return the storage-backend dict for ``LightRAG``.

    Side effect: also calls ``os.environ.setdefault(...)`` for each variable
    that LightRAG's storage implementations read directly from the environment
    (NEO4J_*, QDRANT_*, POSTGRES_*). Existing env values are preserved so an
    operator can always override.

    Raises:
        RuntimeError: If ``LIGHTRAG_ENABLED`` is true but a required setting
            (Neo4j URI/password or the Postgres URI) is missing.
    """
    import config  # local import to avoid coupling at module import time

    if not getattr(config, "LIGHTRAG_ENABLED", False):
        raise RuntimeError(
            "build_storage_config() called but LIGHTRAG_ENABLED is false. "
            "Set LIGHTRAG_ENABLED=true to enable the LightRAG integration."
        )

    neo4j_uri = _require("NEO4J_URI", getattr(config, "NEO4J_URI", None))
    neo4j_username = getattr(config, "NEO4J_USERNAME", None) or "neo4j"
    neo4j_password = _require("NEO4J_PASSWORD", getattr(config, "NEO4J_PASSWORD", None))

    postgres_uri = _require(
        "SQLALCHEMY_DATABASE_URI", getattr(config, "DATABASE_URL", None)
    )
    postgres_env = _parse_postgres_uri(postgres_uri)

    qdrant_url = getattr(config, "QDRANT_URL", None) or "http://localhost:6333"
    qdrant_api_key = getattr(config, "QDRANT_API_KEY", None)

    env_vars: Dict[str, str] = {
        "NEO4J_URI": neo4j_uri,
        "NEO4J_USERNAME": neo4j_username,
        "NEO4J_PASSWORD": neo4j_password,
        "QDRANT_URL": qdrant_url,
        **postgres_env,
    }
    if qdrant_api_key:
        env_vars["QDRANT_API_KEY"] = qdrant_api_key

    for key, value in env_vars.items():
        os.environ.setdefault(key, value)

    return {
        "graph_storage": GRAPH_STORAGE,
        "vector_storage": VECTOR_STORAGE,
        "kv_storage": KV_STORAGE,
        "doc_status_storage": DOC_STATUS_STORAGE,
    }


__all__ = [
    "DOC_STATUS_STORAGE",
    "GRAPH_STORAGE",
    "KV_STORAGE",
    "VECTOR_STORAGE",
    "build_storage_config",
]

"""Unit tests for ``tools.vector_stores.lightrag.storage_config``."""

from __future__ import annotations

import os

import pytest

from tools.vector_stores.lightrag import storage_config


@pytest.fixture
def patched_config(monkeypatch):
    """Set the config attributes ``build_storage_config`` reads.

    Tests can mutate the returned namespace via further ``monkeypatch.setattr``
    calls before invoking ``build_storage_config``.
    """
    import config

    monkeypatch.setattr(config, "LIGHTRAG_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "NEO4J_URI", "bolt://neo4j:7687", raising=False)
    monkeypatch.setattr(config, "NEO4J_USERNAME", "neo4j", raising=False)
    monkeypatch.setattr(config, "NEO4J_PASSWORD", "secret", raising=False)
    monkeypatch.setattr(
        config,
        "DATABASE_URL",
        "postgresql://app_user:p%40ss@db:5432/appdb",
        raising=False,
    )
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant:6333", raising=False)
    monkeypatch.setattr(config, "QDRANT_API_KEY", None, raising=False)
    return config


@pytest.fixture(autouse=True)
def _clear_lightrag_env(monkeypatch):
    """Strip any pre-existing LightRAG env vars so ``setdefault`` is exercised."""
    for key in (
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)


class TestBuildStorageConfig:
    def test_raises_when_lightrag_disabled(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "LIGHTRAG_ENABLED", False, raising=False)
        with pytest.raises(RuntimeError, match="LIGHTRAG_ENABLED is false"):
            storage_config.build_storage_config()

    def test_raises_when_neo4j_uri_missing(self, patched_config, monkeypatch):
        monkeypatch.setattr(patched_config, "NEO4J_URI", None, raising=False)
        with pytest.raises(RuntimeError, match="NEO4J_URI"):
            storage_config.build_storage_config()

    def test_raises_when_neo4j_password_missing(self, patched_config, monkeypatch):
        monkeypatch.setattr(patched_config, "NEO4J_PASSWORD", None, raising=False)
        with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
            storage_config.build_storage_config()

    def test_raises_when_postgres_uri_missing(self, patched_config, monkeypatch):
        monkeypatch.setattr(patched_config, "DATABASE_URL", None, raising=False)
        with pytest.raises(RuntimeError, match="SQLALCHEMY_DATABASE_URI"):
            storage_config.build_storage_config()

    def test_returns_expected_backend_names(self, patched_config):
        result = storage_config.build_storage_config()

        assert result == {
            "graph_storage": "Neo4JStorage",
            "vector_storage": "QdrantVectorDBStorage",
            "kv_storage": "PGKVStorage",
            "doc_status_storage": "PGDocStatusStorage",
        }

    def test_exports_neo4j_and_qdrant_env_vars(self, patched_config):
        storage_config.build_storage_config()

        assert os.environ["NEO4J_URI"] == "bolt://neo4j:7687"
        assert os.environ["NEO4J_USERNAME"] == "neo4j"
        assert os.environ["NEO4J_PASSWORD"] == "secret"
        assert os.environ["QDRANT_URL"] == "http://qdrant:6333"
        # No API key configured => env var must not be set.
        assert "QDRANT_API_KEY" not in os.environ

    def test_decomposes_postgres_uri_into_env_vars(self, patched_config):
        storage_config.build_storage_config()

        assert os.environ["POSTGRES_HOST"] == "db"
        assert os.environ["POSTGRES_PORT"] == "5432"
        assert os.environ["POSTGRES_USER"] == "app_user"
        # URL-encoded "@" must be decoded back.
        assert os.environ["POSTGRES_PASSWORD"] == "p@ss"
        assert os.environ["POSTGRES_DATABASE"] == "appdb"

    def test_setdefault_preserves_existing_env(self, patched_config, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://override:7687")
        storage_config.build_storage_config()
        assert os.environ["NEO4J_URI"] == "bolt://override:7687"

    def test_exports_qdrant_api_key_when_configured(self, patched_config, monkeypatch):
        monkeypatch.setattr(patched_config, "QDRANT_API_KEY", "qd-key", raising=False)
        storage_config.build_storage_config()
        assert os.environ["QDRANT_API_KEY"] == "qd-key"

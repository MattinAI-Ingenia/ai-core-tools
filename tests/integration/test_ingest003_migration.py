"""Integration tests for the ingest003 migration schema state.

Asserts the schema produced by ingest003 is present after create_all:
- the ``azure_blob_source`` JSON column on Repository (nullable, so existing
  rows survive the upgrade untouched)

NOTE on migration cycle testing (same constraint as
``test_localauth001_migration.py`` / ``test_migration_user_deletion.py``):
  Full alembic upgrade → downgrade → upgrade cycle testing is not automated
  here — the session-scoped test_engine uses Base.metadata.create_all without
  an alembic_version row, so running alembic against the same DB would attempt
  to re-create what create_all already made.

  The up/down/up cycle was manually validated during ingest003 development:
    - alembic upgrade head: azure_blob_source added to Repository
    - alembic downgrade -1: column dropped
    - Re-upgrade: column restored correctly

Requires: test DB (docker compose --profile test, port 5433).
"""

import pytest
from sqlalchemy import inspect


class TestIngest003SchemaState:
    """Assert the schema produced by ingest003 is present after create_all."""

    def test_repository_has_azure_blob_source_column(self, test_engine):
        inspector = inspect(test_engine)
        cols = {c["name"] for c in inspector.get_columns("Repository")}
        assert "azure_blob_source" in cols, (
            "Repository.azure_blob_source missing — ingest003 not applied"
        )

    def test_azure_blob_source_is_nullable_json(self, test_engine):
        """Existing rows must survive the upgrade — the column is nullable."""
        inspector = inspect(test_engine)
        col = next(
            c for c in inspector.get_columns("Repository")
            if c["name"] == "azure_blob_source"
        )
        assert col["nullable"] is True, (
            "Repository.azure_blob_source must be nullable so pre-ingest003 "
            "rows upgrade without a default"
        )

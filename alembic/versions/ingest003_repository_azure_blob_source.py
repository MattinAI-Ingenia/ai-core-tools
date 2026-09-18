"""add azure_blob_source to Repository

Remembers the last successfully-validated Azure Blob source (account_url,
container, prefix, auth_mode) so the repository page's "Actualizar desde
Azure Blob" button can re-run the ingestion without re-typing the config.
Written only after a real listing against the container succeeded, so a
stored source is one that was proven reachable.

Deliberately non-secret: the SAS token of an SAS_TOKEN run is never stored —
those runs always re-ask for the token in the UI, mirroring the ingestion
endpoint's request-scoped-only handling of sas_token.

Revision ID: ingest003
Revises: da0c9d2daf57
Create Date: 2026-09-18
"""
from alembic import op
import sqlalchemy as sa


revision = 'ingest003'
down_revision = 'da0c9d2daf57'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Repository', sa.Column('azure_blob_source', sa.JSON(), nullable=True))


def downgrade():
    op.drop_column('Repository', 'azure_blob_source')

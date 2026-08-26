"""add ingestion_stop_mode to Repository

Lets a pause/cancel request reach the indexing thread. The signal has to be a
row, not process memory: the thread lives in one uvicorn worker while the HTTP
request is balanced across all of them (UVICORN_WORKERS is 4 in this
deployment), so an in-memory flag would be invisible to most requests.

NULL = no stop requested; 'pause' / 'cancel' = requested.

Revision ID: ingest001
Revises: ragcfg003
Create Date: 2026-08-25
"""
from alembic import op
import sqlalchemy as sa


revision = 'ingest001'
down_revision = 'ragcfg003'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Repository', sa.Column('ingestion_stop_mode', sa.String(length=10), nullable=True))


def downgrade():
    op.drop_column('Repository', 'ingestion_stop_mode')

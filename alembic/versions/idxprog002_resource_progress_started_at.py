"""Add Resource.progress_started_at — batch stamp for DB-backed ingestion progress.

Revision ID: idxprog002
Revises: idxprog001
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa

revision = 'idxprog002'
down_revision = 'idxprog001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Resource', sa.Column('progress_started_at', sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column('Resource', 'progress_started_at')

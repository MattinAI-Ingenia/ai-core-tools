"""Add Resource.progress_done / progress_total for persisted indexing progress.

Revision ID: idxprog001
Revises: csvimp001
Create Date: 2026-08-18
"""
from alembic import op
import sqlalchemy as sa

revision = 'idxprog001'
down_revision = 'csvimp001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Resource', sa.Column('progress_done', sa.Integer(), nullable=True))
    op.add_column('Resource', sa.Column('progress_total', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('Resource', 'progress_total')
    op.drop_column('Resource', 'progress_done')

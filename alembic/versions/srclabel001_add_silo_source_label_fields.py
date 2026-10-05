"""add Silo.source_label_fields

Comma-separated Resource.extra_metadata keys (the import CSV's columns) shown
next to every source the agent reads, so it knows which product each manual
is about. NULL = no labels, the behaviour of every existing silo.

Revision ID: srclabel001
Revises: entvar001
Create Date: 2026-10-05
"""
from alembic import op
import sqlalchemy as sa

revision = 'srclabel001'
down_revision = 'entvar001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Silo', sa.Column('source_label_fields', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('Silo', 'source_label_fields')

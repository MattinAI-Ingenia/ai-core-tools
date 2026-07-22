"""add lightrag_entity_types column to silo model

Revision ID: lightrag010
Revises: lightrag009
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa

revision = 'lightrag010'
down_revision = 'lightrag009'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Silo', sa.Column('lightrag_entity_types', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('Silo', 'lightrag_entity_types')

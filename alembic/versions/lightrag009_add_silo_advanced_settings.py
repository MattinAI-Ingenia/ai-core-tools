"""add lightrag advanced-settings columns to silo model

Revision ID: lightrag009
Revises: lightrag008
Create Date: 2026-07-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'lightrag009'
down_revision = 'lightrag008'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Silo', sa.Column('lightrag_entity_extract_max_gleaning', sa.Integer(), nullable=True))
    op.add_column('Silo', sa.Column('lightrag_max_source_ids_per_entity', sa.Integer(), nullable=True))
    op.add_column('Silo', sa.Column('lightrag_max_source_ids_per_relation', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('Silo', 'lightrag_max_source_ids_per_relation')
    op.drop_column('Silo', 'lightrag_max_source_ids_per_entity')
    op.drop_column('Silo', 'lightrag_entity_extract_max_gleaning')

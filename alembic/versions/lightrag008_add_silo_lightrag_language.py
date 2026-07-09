"""add lightrag_language column to silo model

Revision ID: lightrag008
Revises: lightrag007
Create Date: 2026-07-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'lightrag008'
down_revision = 'lightrag007'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Silo', sa.Column('lightrag_language', sa.String(length=45), nullable=True))


def downgrade():
    op.drop_column('Silo', 'lightrag_language')

"""add use_agent_as_query to Silo

Revision ID: lagq001
Revises: lightrag006
Create Date: 2026-06-12
"""
from alembic import op
import sqlalchemy as sa

revision = 'lagq001'
down_revision = 'lightrag006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'Silo',
        sa.Column('use_agent_as_query', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade() -> None:
    op.drop_column('Silo', 'use_agent_as_query')

"""add rag_k_mode to agent

Revision ID: ragcfg002
Revises: lightrag010
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa


revision = 'ragcfg002'
down_revision = 'lightrag010'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Agent', sa.Column('rag_k_mode', sa.String(20), nullable=False, server_default='fixed'))


def downgrade():
    op.drop_column('Agent', 'rag_k_mode')

"""add rag_chunk_top_k to agent

Revision ID: ragcfg003
Revises: lightrag012
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa


revision = 'ragcfg003'
down_revision = 'lightrag012'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Agent', sa.Column('rag_chunk_top_k', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('Agent', 'rag_chunk_top_k')

"""add agent rag search method and rerank strategy columns

Revision ID: ragcfg002
Revises: e5e1e987bcc4
Create Date: 2026-07-21

Adds rag_search_method / rag_strategy / rag_rerank_top_n / rag_rerank_similarity_threshold
to Agent. Non-disruptive: rag_search_method defaults to 'dense' (today's implicit behaviour)
and rag_strategy defaults to NULL (no post-retrieval strategy), so existing agents keep their
current retrieval behaviour unchanged after this migration.
"""
from alembic import op
import sqlalchemy as sa


revision = 'ragcfg002'
down_revision = 'e5e1e987bcc4'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Agent', sa.Column('rag_search_method', sa.String(45), nullable=False, server_default='dense'))
    op.add_column('Agent', sa.Column('rag_strategy', sa.String(45), nullable=True))
    op.add_column('Agent', sa.Column('rag_rerank_top_n', sa.Integer(), nullable=True))
    op.add_column('Agent', sa.Column('rag_rerank_similarity_threshold', sa.Float(), nullable=True))


def downgrade():
    op.drop_column('Agent', 'rag_rerank_similarity_threshold')
    op.drop_column('Agent', 'rag_rerank_top_n')
    op.drop_column('Agent', 'rag_strategy')
    op.drop_column('Agent', 'rag_search_method')

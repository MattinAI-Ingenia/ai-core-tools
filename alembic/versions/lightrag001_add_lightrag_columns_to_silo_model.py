"""add lightrag columns to silo model

Revision ID: lightrag001
Revises: mw002
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'lightrag001'
down_revision = 'mw002'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Silo', sa.Column('indexing_service_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_silo_indexing_service_id',
        'Silo', 'AIService',
        ['indexing_service_id'], ['service_id']
    )
    op.add_column('Silo', sa.Column('lightrag_chunk_strategy', sa.String(length=45), nullable=True))
    op.add_column('Silo', sa.Column('lightrag_chunk_token_size', sa.Integer(), nullable=True))
    op.add_column('Silo', sa.Column('lightrag_chunk_overlap_token_size', sa.Integer(), nullable=True))
    op.add_column('Silo', sa.Column('lightrag_graph_context_enabled', sa.Boolean(), nullable=True, server_default=sa.text('false')))


def downgrade():
    op.drop_column('Silo', 'lightrag_graph_context_enabled')
    op.drop_column('Silo', 'lightrag_chunk_overlap_token_size')
    op.drop_column('Silo', 'lightrag_chunk_token_size')
    op.drop_column('Silo', 'lightrag_chunk_strategy')
    op.drop_constraint('fk_silo_indexing_service_id', 'Silo', type_='foreignkey')
    op.drop_column('Silo', 'indexing_service_id')

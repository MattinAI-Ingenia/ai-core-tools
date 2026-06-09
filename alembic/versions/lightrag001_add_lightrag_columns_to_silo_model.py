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


def _column_exists(conn, table, column):
    result = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name=:t AND column_name=:c"
        ),
        {"t": table, "c": column},
    ).fetchone()
    return result is not None


def _fk_exists(conn, constraint_name):
    result = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.table_constraints "
            "WHERE constraint_name=:n AND constraint_type='FOREIGN KEY'"
        ),
        {"n": constraint_name},
    ).fetchone()
    return result is not None


def upgrade():
    conn = op.get_bind()
    if not _column_exists(conn, 'Silo', 'indexing_service_id'):
        op.add_column('Silo', sa.Column('indexing_service_id', sa.Integer(), nullable=True))
    if not _fk_exists(conn, 'fk_silo_indexing_service_id'):
        op.create_foreign_key(
            'fk_silo_indexing_service_id',
            'Silo', 'AIService',
            ['indexing_service_id'], ['service_id']
        )
    if not _column_exists(conn, 'Silo', 'lightrag_chunk_strategy'):
        op.add_column('Silo', sa.Column('lightrag_chunk_strategy', sa.String(length=45), nullable=True))
    if not _column_exists(conn, 'Silo', 'lightrag_chunk_token_size'):
        op.add_column('Silo', sa.Column('lightrag_chunk_token_size', sa.Integer(), nullable=True))
    if not _column_exists(conn, 'Silo', 'lightrag_chunk_overlap_token_size'):
        op.add_column('Silo', sa.Column('lightrag_chunk_overlap_token_size', sa.Integer(), nullable=True))
    if not _column_exists(conn, 'Silo', 'lightrag_graph_context_enabled'):
        op.add_column('Silo', sa.Column('lightrag_graph_context_enabled', sa.Boolean(), nullable=True, server_default=sa.text('false')))


def downgrade():
    op.drop_column('Silo', 'lightrag_graph_context_enabled')
    op.drop_column('Silo', 'lightrag_chunk_overlap_token_size')
    op.drop_column('Silo', 'lightrag_chunk_token_size')
    op.drop_column('Silo', 'lightrag_chunk_strategy')
    op.drop_constraint('fk_silo_indexing_service_id', 'Silo', type_='foreignkey')
    op.drop_column('Silo', 'indexing_service_id')

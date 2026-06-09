"""add lightrag_vector_db_type to Silo

Revision ID: lightrag003
Revises: pricing003
Create Date: 2026-06-03
"""
from alembic import op
import sqlalchemy as sa


revision = 'lightrag003'
down_revision = 'pricing003'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='Silo' AND column_name='lightrag_vector_db_type'"
        )
    ).fetchone()
    if not exists:
        op.add_column('Silo', sa.Column('lightrag_vector_db_type', sa.String(length=45), nullable=True))

    # Backfill legacy LightRAG silos to preserve prior behaviour.
    op.execute(
        'UPDATE "Silo" '
        "SET lightrag_vector_db_type = 'QDRANT' "
        "WHERE UPPER(COALESCE(vector_db_type, '')) = 'LIGHTRAG' "
        'AND lightrag_vector_db_type IS NULL'
    )


def downgrade():
    op.drop_column('Silo', 'lightrag_vector_db_type')

"""add Silo.lightrag_entity_types_mode

Splits "how do we get the entity types" from "what are they":
  'infer'  — propose them from the documents before the first index (new default)
  'manual' — typed in by hand, or NULL to use the language defaults (historical)

server_default='manual' so every pre-existing silo keeps its current behaviour;
only silos created through the ORM after this migration default to 'infer'.

Revision ID: lightrag012
Revises: lightrag011
Create Date: 2026-08-21
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'lightrag012'
down_revision = 'lightrag011'
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


def upgrade():
    conn = op.get_bind()
    if not _column_exists(conn, 'Silo', 'lightrag_entity_types_mode'):
        op.add_column(
            'Silo',
            sa.Column('lightrag_entity_types_mode', sa.String(length=20),
                      nullable=False, server_default='manual'),
        )


def downgrade():
    conn = op.get_bind()
    if _column_exists(conn, 'Silo', 'lightrag_entity_types_mode'):
        op.drop_column('Silo', 'lightrag_entity_types_mode')

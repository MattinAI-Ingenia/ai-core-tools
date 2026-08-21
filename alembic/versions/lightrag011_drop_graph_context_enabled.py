"""drop unused lightrag_graph_context_enabled from Silo

The column was added by lightrag001 but never read anywhere in the backend
(no schema, service, router or frontend reference). Dropping it.

Revision ID: lightrag011
Revises: idxprog002
Create Date: 2026-08-21
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'lightrag011'
down_revision = 'idxprog002'
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
    if _column_exists(conn, 'Silo', 'lightrag_graph_context_enabled'):
        op.drop_column('Silo', 'lightrag_graph_context_enabled')


def downgrade():
    conn = op.get_bind()
    if not _column_exists(conn, 'Silo', 'lightrag_graph_context_enabled'):
        op.add_column(
            'Silo',
            sa.Column('lightrag_graph_context_enabled', sa.Boolean(),
                      nullable=True, server_default=sa.text('false')),
        )

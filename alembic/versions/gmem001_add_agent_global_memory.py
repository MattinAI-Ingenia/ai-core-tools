"""Add Agent.has_global_memory and App memory service FKs

Schema for cross-session, store-backed user memory. Alembic owns only these
three columns; the vector store (LangGraph's ``AsyncPostgresStore``) creates
and migrates its own four tables, which is why this revision also adds them
to the ``include_name`` ignore-list in ``alembic/env.py``.

Both ``App`` FKs default to ``NULL``, meaning memory is inactive for that App
— there is no system-level fallback. ``ondelete='SET NULL'`` therefore
degrades the App to "memory inactive" rather than cascading.

Table-name casing is intentionally non-uniform here; see the comment in
``backend/models/app.py``.

Downgrade is data-lossy: dropping these three columns irreversibly discards
every per-agent memory toggle and every App memory-service selection;
re-upgrading yields ``false``/``NULL``, not the prior values. The store's own
tables (``store``, ``store_vectors``, ...) deliberately survive a downgrade
and keep holding memory content regardless.

Revision ID: gmem001
Revises: ec5b82391242
Create Date: 2026-09-14 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'gmem001'
down_revision = 'ec5b82391242'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Agent',
        sa.Column('has_global_memory', sa.Boolean(), server_default='false', nullable=False),
    )

    op.add_column('App', sa.Column('memory_extraction_service_id', sa.Integer(), nullable=True))
    op.add_column('App', sa.Column('memory_embedding_service_id', sa.Integer(), nullable=True))

    op.create_foreign_key(
        'App_memory_extraction_service_id_fkey',
        'App', 'AIService',
        ['memory_extraction_service_id'], ['service_id'],
        ondelete='SET NULL',
    )
    op.create_foreign_key(
        'App_memory_embedding_service_id_fkey',
        'App', 'embedding_service',
        ['memory_embedding_service_id'], ['service_id'],
        ondelete='SET NULL',
    )


def downgrade():
    op.drop_constraint('App_memory_embedding_service_id_fkey', 'App', type_='foreignkey')
    op.drop_constraint('App_memory_extraction_service_id_fkey', 'App', type_='foreignkey')

    op.drop_column('App', 'memory_embedding_service_id')
    op.drop_column('App', 'memory_extraction_service_id')

    op.drop_column('Agent', 'has_global_memory')

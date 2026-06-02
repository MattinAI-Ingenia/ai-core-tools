"""add role-specific LLM columns to Silo (query/extract/keywords/vlm)

LightRAG 2026.05 introduced role-specific LLM configuration with four roles:
EXTRACT (entity/relationship extraction), QUERY (final answer generation),
KEYWORDS (query keyword extraction), and VLM (vision-language for images).

This migration adds four nullable FK columns to Silo so each LightRAG silo
can configure an independent AIService per role. The legacy
``indexing_service_id`` column is kept for backward compatibility — when set
on legacy silos it acts as a fallback for ``extract_service_id``.

Revision ID: lightrag002
Revises: lightrag001
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa


revision = 'lightrag002'
down_revision = 'lightrag001'
branch_labels = None
depends_on = None


_NEW_COLUMNS = (
    'query_service_id',
    'extract_service_id',
    'keywords_service_id',
    'vlm_service_id',
)


def upgrade():
    for col in _NEW_COLUMNS:
        op.add_column('Silo', sa.Column(col, sa.Integer(), nullable=True))
        op.create_foreign_key(
            f'fk_silo_{col}',
            'Silo', 'AIService',
            [col], ['service_id'],
        )

    # Backfill: legacy silos used indexing_service_id for extraction; copy it
    # into extract_service_id so existing LightRAG silos keep working without
    # operator intervention.
    op.execute(
        'UPDATE "Silo" SET extract_service_id = indexing_service_id '
        'WHERE indexing_service_id IS NOT NULL AND extract_service_id IS NULL'
    )


def downgrade():
    for col in _NEW_COLUMNS:
        op.drop_constraint(f'fk_silo_{col}', 'Silo', type_='foreignkey')
        op.drop_column('Silo', col)

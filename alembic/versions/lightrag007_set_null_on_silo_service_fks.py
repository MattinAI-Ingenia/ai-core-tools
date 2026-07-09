"""set ON DELETE SET NULL on Silo service FKs (indexing/query/extract/keywords/vlm/embedding)

Deleting an AIService (or EmbeddingService) currently raises a ForeignKeyViolation
because these 6 Silo FK columns were created with ON DELETE NO ACTION (the default),
inconsistent with the rest of the app (e.g. Agent.service_id already uses SET NULL).
All 6 columns are nullable, so SET NULL is safe and preserves the Silo row.

Revision ID: lightrag007
Revises: lightragmode001
Create Date: 2026-07-08
"""
from alembic import op


revision = 'lightrag007'
down_revision = 'lightragmode001'
branch_labels = None
depends_on = None

_FK_CHANGES = [
    # (constraint_name, child_table, child_col, parent_table, parent_col)
    ('fk_silo_indexing_service_id', 'Silo', 'indexing_service_id', 'AIService', 'service_id'),
    ('fk_silo_extract_service_id', 'Silo', 'extract_service_id', 'AIService', 'service_id'),
    ('fk_silo_keywords_service_id', 'Silo', 'keywords_service_id', 'AIService', 'service_id'),
    ('fk_silo_query_service_id', 'Silo', 'query_service_id', 'AIService', 'service_id'),
    ('fk_silo_vlm_service_id', 'Silo', 'vlm_service_id', 'AIService', 'service_id'),
    ('Silo_embedding_service_id_fkey', 'Silo', 'embedding_service_id', 'embedding_service', 'service_id'),
]


def upgrade():
    for name, child, child_col, parent, parent_col in _FK_CHANGES:
        op.drop_constraint(name, child, type_='foreignkey')
        op.create_foreign_key(
            name, child, parent,
            [child_col], [parent_col],
            ondelete='SET NULL',
        )


def downgrade():
    for name, child, child_col, parent, parent_col in _FK_CHANGES:
        op.drop_constraint(name, child, type_='foreignkey')
        op.create_foreign_key(
            name, child, parent,
            [child_col], [parent_col],
        )

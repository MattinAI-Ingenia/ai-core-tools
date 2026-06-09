"""set ON DELETE SET NULL on indexing_metric.resource_id FK

Allows deleting a Resource without violating the FK constraint in
indexing_metric. Historical metric rows are preserved with resource_id = NULL.

Revision ID: lightrag005
Revises: lightrag_merge001
Create Date: 2026-06-08
"""
from alembic import op


revision = 'lightrag005'
down_revision = 'lightrag_merge001'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint('indexing_metric_resource_id_fkey', 'indexing_metric', type_='foreignkey')
    op.create_foreign_key(
        'indexing_metric_resource_id_fkey',
        'indexing_metric', 'Resource',
        ['resource_id'], ['resource_id'],
        ondelete='SET NULL',
    )


def downgrade():
    op.drop_constraint('indexing_metric_resource_id_fkey', 'indexing_metric', type_='foreignkey')
    op.create_foreign_key(
        'indexing_metric_resource_id_fkey',
        'indexing_metric', 'Resource',
        ['resource_id'], ['resource_id'],
    )

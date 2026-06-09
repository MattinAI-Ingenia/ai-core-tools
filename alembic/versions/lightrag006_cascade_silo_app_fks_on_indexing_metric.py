"""set ON DELETE CASCADE on indexing_metric silo_id and app_id FKs

When a Silo or App is deleted, its indexing_metric rows are deleted too
(metrics without a silo/app have no meaning).

Revision ID: lightrag006
Revises: lightrag005
Create Date: 2026-06-08
"""
from alembic import op


revision = 'lightrag006'
down_revision = 'lightrag005'
branch_labels = None
depends_on = None

_FK_CHANGES = [
    # (constraint_name, child_table, child_col, parent_table, parent_col)
    ('indexing_metric_silo_id_fkey', 'indexing_metric', 'silo_id', 'Silo', 'silo_id'),
    ('indexing_metric_app_id_fkey', 'indexing_metric', 'app_id', 'App', 'app_id'),
]


def upgrade():
    for name, child, child_col, parent, parent_col in _FK_CHANGES:
        op.drop_constraint(name, child, type_='foreignkey')
        op.create_foreign_key(
            name, child, parent,
            [child_col], [parent_col],
            ondelete='CASCADE',
        )


def downgrade():
    for name, child, child_col, parent, parent_col in _FK_CHANGES:
        op.drop_constraint(name, child, type_='foreignkey')
        op.create_foreign_key(
            name, child, parent,
            [child_col], [parent_col],
        )

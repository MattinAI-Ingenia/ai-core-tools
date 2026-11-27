"""App.default_rerank_service_id FK: ON DELETE SET NULL -> ON DELETE RESTRICT.

The rerank service delete guard (FR-16) relies on the database refusing to delete a service still referenced
as an app default; with SET NULL a missed pre-check would silently clear an app's default.
AppService.delete_app clears the default (and flushes) before deleting rerank services, so app deletion is unaffected.

Downgrade restores ON DELETE SET NULL exactly. No data change.

Revision ID: rerank002
Revises: rerank001
"""

from alembic import op

revision = "rerank002"
down_revision = "rerank001"
branch_labels = None
depends_on = None

FK = "App_default_rerank_service_id_fkey"


def upgrade():
    op.drop_constraint(FK, "App", type_="foreignkey")
    op.create_foreign_key(
        FK, "App", "rerank_service", ["default_rerank_service_id"], ["service_id"], ondelete="RESTRICT"
    )


def downgrade():
    op.drop_constraint(FK, "App", type_="foreignkey")
    op.create_foreign_key(
        FK, "App", "rerank_service", ["default_rerank_service_id"], ["service_id"], ondelete="SET NULL"
    )

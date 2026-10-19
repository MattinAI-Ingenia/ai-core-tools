"""merge lightrag branch heads with upstream develop (a2a, media silos, indexing failures)

Revision ID: merge003
Revises: indexerr001, kr001
Create Date: 2026-10-19

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge003'
down_revision = ('indexerr001', 'kr001')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

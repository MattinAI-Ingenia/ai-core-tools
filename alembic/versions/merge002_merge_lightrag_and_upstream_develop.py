"""merge lightrag branch heads with upstream develop (skills, metrics, scheduled tasks)

Revision ID: merge002
Revises: apikeyhash001, srclabel001
Create Date: 2026-10-07

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge002'
down_revision = ('apikeyhash001', 'srclabel001')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

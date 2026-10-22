"""merge the system-one branch heads with the lightrag chain heads

Revision ID: merge004
Revises: merge003, sysone003
Create Date: 2026-10-19

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge004'
down_revision = ('merge003', 'sysone003')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

"""merge the lightrag chain head with the upstream service-lifecycle heads

Revision ID: merge005
Revises: merge004, siloparser001
Create Date: 2026-10-22

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge005'
down_revision = ('merge004', 'siloparser001')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

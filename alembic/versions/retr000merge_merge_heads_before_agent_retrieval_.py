"""merge heads before agent retrieval config

Revision ID: retr000merge
Revises: 5ccb27fe08c1, spoint001
Create Date: 2026-06-15 16:08:07.794816

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'retr000merge'
down_revision = ('5ccb27fe08c1', 'spoint001')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

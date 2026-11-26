"""merge rag config and silo indexing heads

Revision ID: ragsilomerge001
Revises: ragcfg003, siloidx001
Create Date: 2026-11-26 12:32:39.155261

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ragsilomerge001'
down_revision = ('ragcfg003', 'siloidx001')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

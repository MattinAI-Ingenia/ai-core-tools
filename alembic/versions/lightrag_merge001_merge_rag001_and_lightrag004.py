"""merge rag001 and lightrag004 heads

Revision ID: lightrag_merge001
Revises: lightrag004, rag001
Create Date: 2026-06-08

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'lightrag_merge001'
down_revision = ('lightrag004', 'rag001')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

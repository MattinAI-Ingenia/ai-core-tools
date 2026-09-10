"""merge lightrag branch heads with develop

Revision ID: 595bfe0662bf
Revises: unaccnt1, 1eeb4ba697f8
Create Date: 2026-09-10 15:26:41.424103

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '595bfe0662bf'
down_revision = ('unaccnt1', '1eeb4ba697f8')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

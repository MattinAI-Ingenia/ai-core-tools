"""add error_message to Resource

Revision ID: da0c9d2daf57
Revises: 595bfe0662bf
Create Date: 2026-09-10 15:31:04.626951

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'da0c9d2daf57'
down_revision = '595bfe0662bf'
branch_labels = None
depends_on = None


def upgrade():
    # Hand-written: autogenerate also picked up unrelated drift from other
    # pending model/DB differences (Middleware, Media, lightrag_* tables,
    # index renames) that don't belong in this change. This migration adds
    # only the one column.
    op.add_column('Resource', sa.Column('error_message', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('Resource', 'error_message')

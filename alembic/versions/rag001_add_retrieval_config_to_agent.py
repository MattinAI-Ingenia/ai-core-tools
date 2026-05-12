"""add_retrieval_config_to_agent

Revision ID: rag001
Revises: 14b4c9c42164
Create Date: 2026-04-21

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'rag001'
down_revision = '14b4c9c42164'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Agent',
        sa.Column('retrieval_config', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )


def downgrade():
    op.drop_column('Agent', 'retrieval_config')

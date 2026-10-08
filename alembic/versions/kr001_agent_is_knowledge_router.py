"""add Agent.is_knowledge_router

Opt-in mode where the agent routes each question to its agent-tools
(specialists) and passes a single specialist's answer through unchanged.
See docs/superpowers/specs/2026-10-07-knowledge-router-design.md.

Revision ID: kr001
Revises: srclabel001
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa

revision = 'kr001'
down_revision = 'merge002'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Agent',
        sa.Column('is_knowledge_router', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade():
    op.drop_column('Agent', 'is_knowledge_router')

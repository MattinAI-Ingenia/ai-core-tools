"""add_retrieval_config_to_agent

Adds per-agent retrieval configuration to the Agent table.

Two stable *selectors* are stored as typed columns:
    - retrieval_search_type: how the vector store is queried (Phase 1)
    - retrieval_strategy:    post-retrieval strategy applied to candidates (Phase 2)

Every component-specific knob (k, top_n, similarity_threshold, and any future
search-method / strategy parameter such as a hybrid ``alpha``) lives in a single
``retrieval_params`` JSON column, so adding a new retrieval parameter never
requires a schema migration.

Defaults reproduce the previous hard-coded behaviour (similarity / passthrough,
with k falling back to 30 in code when absent), so existing agents are
unaffected.

Revision ID: retr001
Revises: retr000merge
Create Date: 2026-06-15
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'retr001'
down_revision = 'retr000merge'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Agent',
        sa.Column(
            'retrieval_search_type',
            sa.String(length=45),
            nullable=False,
            server_default='similarity',
        ),
    )
    op.add_column(
        'Agent',
        sa.Column(
            'retrieval_strategy',
            sa.String(length=45),
            nullable=False,
            server_default='passthrough',
        ),
    )
    op.add_column(
        'Agent',
        sa.Column(
            'retrieval_params',
            sa.JSON(),
            nullable=False,
            server_default='{}',
        ),
    )


def downgrade():
    op.drop_column('Agent', 'retrieval_params')
    op.drop_column('Agent', 'retrieval_strategy')
    op.drop_column('Agent', 'retrieval_search_type')

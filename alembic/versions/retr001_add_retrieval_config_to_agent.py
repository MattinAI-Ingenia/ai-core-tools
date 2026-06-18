"""add_retrieval_config_to_agent

Adds per-agent retrieval configuration columns to the Agent table:
    - retrieval_search_type: how the vector store is queried (Phase 1)
    - retrieval_k: number of candidate documents retrieved (Phase 1)
    - retrieval_strategy: post-retrieval strategy applied to candidates (Phase 2)
    - retrieval_top_n: documents kept after reranking (Phase 2, nullable)

Defaults reproduce the previous hard-coded behaviour (similarity / k=30 /
passthrough), so existing agents are unaffected.

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
            'retrieval_k',
            sa.Integer(),
            nullable=False,
            server_default='30',
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
            'retrieval_top_n',
            sa.Integer(),
            nullable=True,
        ),
    )


def downgrade():
    op.drop_column('Agent', 'retrieval_top_n')
    op.drop_column('Agent', 'retrieval_strategy')
    op.drop_column('Agent', 'retrieval_k')
    op.drop_column('Agent', 'retrieval_search_type')

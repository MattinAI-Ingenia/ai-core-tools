"""collapse Agent.retrieval_config JSON into flat lightrag_query_mode column

Drops the redundant ``retrieval_config`` JSON column. Its only runtime-relevant
field was ``lightrag_query_mode`` (the generic search_type/k/fetch_k/lambda_mult/
score_threshold were already duplicated by the ``rag_*`` columns and unread after
the upstream merge). That single field is promoted to a flat ``lightrag_query_mode``
column and backfilled from the existing JSON.

This revision also unifies the three migration heads that existed on the branch
(``d3adbeef1234``, ``lagq001``, ``merge001_userdel_platform_role``) so there is a
single linear head again.

Revision ID: lightragmode001
Revises: d3adbeef1234, lagq001, merge001_userdel_platform_role
Create Date: 2026-06-25
"""
from alembic import op
import sqlalchemy as sa


revision = 'lightragmode001'
down_revision = ('d3adbeef1234', 'lagq001', 'merge001_userdel_platform_role')
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('Agent', sa.Column('lightrag_query_mode', sa.String(20), nullable=True))
    # Backfill the single field we still read from the old JSON blob.
    op.execute(
        """
        UPDATE "Agent"
        SET lightrag_query_mode = retrieval_config->>'lightrag_query_mode'
        WHERE retrieval_config IS NOT NULL
          AND retrieval_config->>'lightrag_query_mode' IS NOT NULL
        """
    )
    op.drop_column('Agent', 'retrieval_config')


def downgrade() -> None:
    op.add_column('Agent', sa.Column('retrieval_config', sa.JSON(), nullable=True))
    # Restore the JSON blob with just the query mode (the only field that was live).
    op.execute(
        """
        UPDATE "Agent"
        SET retrieval_config = json_build_object('lightrag_query_mode', lightrag_query_mode)
        WHERE lightrag_query_mode IS NOT NULL
        """
    )
    op.drop_column('Agent', 'lightrag_query_mode')

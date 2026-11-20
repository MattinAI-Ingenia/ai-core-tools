"""Cached input tokens per agent execution and the cached-input price per model.

agent_execution_event.cached_input_tokens: part of input_tokens the provider served from its prompt cache
(NULL for events recorded before this revision or when the run reported no usage).
pricing_catalog.cached_input_price_per_1m: its USD price per 1M tokens (NULL = not known).
"""

from alembic import op
import sqlalchemy as sa

revision = "cachetok001"
down_revision = "merge006"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("agent_execution_event", sa.Column("cached_input_tokens", sa.Integer(), nullable=True))
    op.add_column("pricing_catalog", sa.Column("cached_input_price_per_1m", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("pricing_catalog", "cached_input_price_per_1m")
    op.drop_column("agent_execution_event", "cached_input_tokens")

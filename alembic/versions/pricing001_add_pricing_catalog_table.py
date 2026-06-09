"""Add pricing catalog table for dynamic price management.

Revision ID: pricing001
Revises: lightrag002
Create Date: 2026-05-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'pricing001'
down_revision = 'lightrag002'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    exists = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name='pricing_catalog'"
        )
    ).fetchone()
    if exists:
        return
    op.create_table(
        'pricing_catalog',
        sa.Column('model_name', sa.String(255), nullable=False, primary_key=True),
        sa.Column('provider', sa.String(50), nullable=False),  # openai, anthropic, mistral, google
        sa.Column('currency', sa.String(3), nullable=False, server_default='USD'),  # USD, EUR
        sa.Column('input_price_usd_per_1m', sa.Float, nullable=True),  # renamed by pricing003
        sa.Column('output_price_usd_per_1m', sa.Float, nullable=True),  # renamed by pricing003
        sa.Column('embedding_price_usd_per_1m', sa.Float, nullable=True),  # renamed by pricing003
        sa.Column('last_updated', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('source', sa.String(100), nullable=False),  # "openai_api", "anthropic_docs", etc.
        sa.Index('idx_provider', 'provider'),
        sa.Index('idx_currency', 'currency'),
    )


def downgrade():
    op.drop_table('pricing_catalog')

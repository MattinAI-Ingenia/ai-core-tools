"""Rename price columns: drop _usd_ infix to make them currency-agnostic.

Revision ID: pricing003
Revises: pricing002
Create Date: 2026-05-28 14:00:00.000000

Renames:
  input_price_usd_per_1m  → input_price_per_1m
  output_price_usd_per_1m → output_price_per_1m
  embedding_price_usd_per_1m → embedding_price_per_1m
"""
from alembic import op


import sqlalchemy as sa


revision = 'pricing003'
down_revision = 'pricing002'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    # Check if the rename already happened (column already has the new name)
    already_renamed = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='pricing_catalog' AND column_name='input_price_per_1m'"
        )
    ).fetchone()
    if already_renamed:
        return
    op.alter_column('pricing_catalog', 'input_price_usd_per_1m',
                    new_column_name='input_price_per_1m')
    op.alter_column('pricing_catalog', 'output_price_usd_per_1m',
                    new_column_name='output_price_per_1m')
    op.alter_column('pricing_catalog', 'embedding_price_usd_per_1m',
                    new_column_name='embedding_price_per_1m')


def downgrade():
    op.alter_column('pricing_catalog', 'input_price_per_1m',
                    new_column_name='input_price_usd_per_1m')
    op.alter_column('pricing_catalog', 'output_price_per_1m',
                    new_column_name='output_price_usd_per_1m')
    op.alter_column('pricing_catalog', 'embedding_price_per_1m',
                    new_column_name='embedding_price_usd_per_1m')

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


revision = 'pricing003'
down_revision = 'pricing002'
branch_labels = None
depends_on = None


def upgrade():
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

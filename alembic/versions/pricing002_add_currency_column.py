"""Add currency column to pricing_catalog.

Revision ID: pricing002
Revises: pricing001
Create Date: 2026-05-28 13:00:00.000000

Adds the currency column that was omitted from the initial pricing_catalog
migration when the feature was extended to support multi-currency.
"""
from alembic import op
import sqlalchemy as sa


revision = 'pricing002'
down_revision = 'pricing001'
branch_labels = None
depends_on = None


def upgrade():
    # pricing001 already creates the pricing_catalog table with the currency
    # column and idx_currency index, so this migration is a no-op for any DB
    # that ran pricing001 in its current form.
    pass


def downgrade():
    op.drop_index('idx_currency', table_name='pricing_catalog')
    op.drop_column('pricing_catalog', 'currency')

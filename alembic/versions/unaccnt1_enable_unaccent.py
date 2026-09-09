"""Enable the unaccent extension for accent-insensitive coverage search.

list_documents_mentioning's literal ILIKE search missed real matches purely
over spelling ("anodo" vs "ánodo") — unaccent() lets Postgres compare content
and search term with diacritics stripped from both sides, without turning the
search into fuzzy/semantic matching.

Revision ID: unaccnt1
Revises: etjob001
Create Date: 2026-09-04
"""
from alembic import op

revision = 'unaccnt1'
down_revision = 'etjob001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS unaccent")

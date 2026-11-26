"""add full-text GIN index for lexical (BM25) search on langchain_pg_embedding

Revision ID: lexfts001
Revises: ragsilomerge001
Create Date: 2026-11-26

BM25SearchMethod now delegates keyword search to Postgres full-text search on
PGVector silos (PGVectorStore.lexical_search) instead of rebuilding an in-memory
BM25 index per query. This index matches that query's expression exactly.

langchain_pg_embedding is created by langchain_postgres on the first collection,
not by Alembic, so on a fresh database it may not exist yet: the upgrade is a
no-op then and PGVectorStore creates the index lazily on first use.
"""
from alembic import op


revision = 'lexfts001'
down_revision = 'ragsilomerge001'
branch_labels = None
depends_on = None

INDEX_NAME = 'ix_langchain_pg_embedding_document_fts'


def upgrade():
    op.execute(f"""
        DO $$
        BEGIN
            IF to_regclass('public.langchain_pg_embedding') IS NOT NULL THEN
                CREATE INDEX IF NOT EXISTS {INDEX_NAME}
                ON langchain_pg_embedding
                USING gin (to_tsvector('simple', coalesce(document, '')));
            END IF;
        END $$;
    """)


def downgrade():
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")

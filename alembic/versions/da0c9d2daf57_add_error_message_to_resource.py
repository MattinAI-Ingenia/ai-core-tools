"""add error_message to Resource

Revision ID: da0c9d2daf57
Revises: 595bfe0662bf
Create Date: 2026-09-10 15:31:04.626951

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = 'da0c9d2daf57'
down_revision = '595bfe0662bf'
branch_labels = None
depends_on = None


def upgrade():
    # Merged-tree no-op: indexerr001 (lightrag develop, #277) adds the same
    # Resource.error_message column earlier in the combined chain. Kept as a
    # pass-through so the sysone chain (ingest003) stays connected.
    pass


def downgrade():
    pass

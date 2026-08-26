"""add ingestion_elapsed_seconds to Repository

Makes the progress bar's timer survive a pause. The clock is
``now() - progress_started_at`` of the active batch, and resuming stamps a new
batch, so without an accumulator the timer restarts at 00:00:00. Keeping the old
stamp instead is not an option: it would count the time spent paused, which can
be days.

Only finished stretches are stored; the live batch's own elapsed time is added
on read. Reset to 0 when an upload starts a new job, kept across pause -> resume.

Revision ID: ingest002
Revises: ingest001
Create Date: 2026-08-26
"""
from alembic import op
import sqlalchemy as sa


revision = 'ingest002'
down_revision = 'ingest001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'Repository',
        sa.Column('ingestion_elapsed_seconds', sa.Integer(), nullable=False,
                  server_default='0'),
    )


def downgrade():
    op.drop_column('Repository', 'ingestion_elapsed_seconds')

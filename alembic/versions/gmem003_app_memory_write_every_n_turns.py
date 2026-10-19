"""Add App.memory_write_every_n_turns

Per-App cadence for cross-session memory fact extraction: extraction runs on
turn 1 and every Nth turn after. NULL means "use the deployment default"
(``AICT_GLOBAL_MEMORY_WRITE_EVERY_N_TURNS``, 5 unless set). A CHECK keeps it
at 1 or more, because the trigger computes ``turn % N``.

Downgrade drops the column: every App falls back to the deployment default.

Revision ID: gmem003
Revises: gmem002
Create Date: 2026-10-09 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'gmem003'
down_revision = 'gmem002'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('App', sa.Column('memory_write_every_n_turns', sa.Integer(), nullable=True))
    op.create_check_constraint(
        'ck_App_memory_write_every_n_turns_min', 'App', 'memory_write_every_n_turns >= 1'
    )


def downgrade():
    op.drop_constraint('ck_App_memory_write_every_n_turns_min', 'App', type_='check')
    op.drop_column('App', 'memory_write_every_n_turns')

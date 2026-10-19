"""Add App.memory_language

The language cross-session memory facts are written in, per App, so users
read their remembered preferences in their own language. Stored as a short
code from a closed set (see ``tools/memory/settings.py::MEMORY_LANGUAGES``);
NULL means English, the behaviour before this column existed. A CHECK keeps
it to that set because the value is turned into an instruction for the
extraction model.

Changing it does not translate facts already stored: they stay in the
language they were written in. Downgrade drops the column (back to English).

Revision ID: gmem004
Revises: gmem003
Create Date: 2026-10-09 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'gmem004'
down_revision = 'gmem003'
branch_labels = None
depends_on = None

_CODES = ('en', 'es', 'eu', 'ca', 'gl', 'fr', 'de', 'it', 'pt')


def upgrade():
    op.add_column('App', sa.Column('memory_language', sa.String(length=8), nullable=True))
    op.create_check_constraint(
        'ck_App_memory_language_known',
        'App',
        "memory_language IN (" + ", ".join(f"'{code}'" for code in _CODES) + ")",
    )


def downgrade():
    op.drop_constraint('ck_App_memory_language_known', 'App', type_='check')
    op.drop_column('App', 'memory_language')

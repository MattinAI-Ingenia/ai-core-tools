"""Add Conversation.memory_pending_user_messages

Cross-session memory extracts facts only on write-due turns (turn 1 and every
Nth after). This column buffers the user's own messages from the turns in
between, so a write-due turn extracts from every turn since the previous
extraction instead of only its own. It holds a JSON list of the user's
original messages (never attachment text, never assistant replies), is
emptied on every write-due turn and on memory erasure, and is deleted with
its conversation row.

Downgrade drops the column and any messages still buffered in it — at most
the last few user messages per conversation, which also remain in the
conversation transcript.

Revision ID: gmem002
Revises: gmem001
Create Date: 2026-10-02 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'gmem002'
down_revision = 'gmem001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('Conversation', sa.Column('memory_pending_user_messages', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('Conversation', 'memory_pending_user_messages')

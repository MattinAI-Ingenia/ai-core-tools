"""merge media and bedrock migration heads

Revision ID: e5e1e987bcc4
Revises: bedrock001, mediaemb001
Create Date: 2026-07-20 13:01:53.162475

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e5e1e987bcc4'
down_revision = ('bedrock001', 'mediaemb001')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

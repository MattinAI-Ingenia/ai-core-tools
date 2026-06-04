"""indexing_metric table

Revision ID: lightrag004
Revises: lightrag003
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa


revision = 'lightrag004'
down_revision = 'lightrag003'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'indexing_metric',
        sa.Column('metric_id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('app_id', sa.Integer(), nullable=False),
        sa.Column('silo_id', sa.Integer(), nullable=False),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('content_ref', sa.String(length=1000), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('completion_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('embedding_tokens', sa.Integer(), nullable=True),
        sa.Column('tokens_source', sa.String(length=12), nullable=False, server_default='provider'),
        sa.Column('llm_calls', sa.Integer(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=True),
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('embedding_model_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['app_id'], ['App.app_id']),
        sa.ForeignKeyConstraint(['silo_id'], ['Silo.silo_id']),
        sa.ForeignKeyConstraint(['resource_id'], ['Resource.resource_id']),
        sa.PrimaryKeyConstraint('metric_id'),
    )

    op.create_index('idx_indexing_metric_silo_id', 'indexing_metric', ['silo_id'])
    op.create_index('idx_indexing_metric_resource_id', 'indexing_metric', ['resource_id'])
    op.create_index('idx_indexing_metric_app_id', 'indexing_metric', ['app_id'])
    op.create_index('idx_indexing_metric_resource_created', 'indexing_metric', ['resource_id', 'created_at'])


def downgrade():
    op.drop_index('idx_indexing_metric_resource_created', table_name='indexing_metric')
    op.drop_index('idx_indexing_metric_app_id', table_name='indexing_metric')
    op.drop_index('idx_indexing_metric_resource_id', table_name='indexing_metric')
    op.drop_index('idx_indexing_metric_silo_id', table_name='indexing_metric')
    op.drop_table('indexing_metric')

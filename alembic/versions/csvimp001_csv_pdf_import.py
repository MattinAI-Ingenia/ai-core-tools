"""CSV → PDF import: add import_job, import_job_row tables; add Resource.extra_metadata.

Revision ID: csvimp001
Revises: ragcfg002
Create Date: 2026-08-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'csvimp001'
down_revision = 'ragcfg002'
branch_labels = None
depends_on = None


def upgrade():
    import_job_status = postgresql.ENUM('DOWNLOADING', 'REVIEW', name='import_job_status', create_type=True)
    import_job_status.create(op.get_bind(), checkfirst=True)

    import_row_status = postgresql.ENUM(
        'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'FAILED', 'CONFIRMED', 'DISCARDED',
        name='import_row_status', create_type=True,
    )
    import_row_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'import_job',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('repository_id', sa.Integer(), nullable=False),
        sa.Column('status', postgresql.ENUM('DOWNLOADING', 'REVIEW', name='import_job_status', create_type=False),
                   nullable=False, server_default='DOWNLOADING'),
        sa.Column('source_filename', sa.String(length=255), nullable=True),
        sa.Column('link_column', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('NOW()')),
        sa.Column('last_activity_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['repository_id'], ['Repository.repository_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_import_job_repository_id', 'import_job', ['repository_id'])
    op.create_index('ix_import_job_created_at', 'import_job', ['created_at'])
    op.create_index('ix_import_job_last_activity_at', 'import_job', ['last_activity_at'])

    op.create_table(
        'import_job_row',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('import_job_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('row_metadata', sa.JSON(), nullable=True),
        sa.Column('status', postgresql.ENUM(
            'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'FAILED', 'CONFIRMED', 'DISCARDED',
            name='import_row_status', create_type=False), nullable=False, server_default='PENDING'),
        sa.Column('failure_reason', sa.String(length=32), nullable=True),
        sa.Column('staged_path', sa.String(length=1000), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['import_job_id'], ['import_job.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resource_id'], ['Resource.resource_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_import_job_row_import_job_id', 'import_job_row', ['import_job_id'])

    op.add_column('Resource', sa.Column('extra_metadata', sa.JSON(), nullable=True))


def downgrade():
    op.drop_column('Resource', 'extra_metadata')

    op.drop_index('ix_import_job_row_import_job_id', table_name='import_job_row')
    op.drop_table('import_job_row')

    op.drop_index('ix_import_job_last_activity_at', table_name='import_job')
    op.drop_index('ix_import_job_created_at', table_name='import_job')
    op.drop_index('ix_import_job_repository_id', table_name='import_job')
    op.drop_table('import_job')

    postgresql.ENUM(name='import_row_status', create_type=False).drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name='import_job_status', create_type=False).drop(op.get_bind(), checkfirst=True)

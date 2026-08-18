import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

from models.enums.import_job_status import ImportJobStatus


class ImportJob(Base):
    """A single CSV → PDF import run for a Repository."""
    __tablename__ = 'import_job'

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, sa.ForeignKey('Repository.repository_id', ondelete='CASCADE'), nullable=False, index=True)

    status = Column(
        sa.Enum(ImportJobStatus, name='import_job_status', create_type=False),
        nullable=False,
        default=ImportJobStatus.DOWNLOADING,
        server_default='DOWNLOADING',
    )
    source_filename = Column(String(255), nullable=True)
    link_column = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    repository = relationship('Repository')
    rows = relationship('ImportJobRow', back_populates='import_job', cascade='all, delete-orphan')

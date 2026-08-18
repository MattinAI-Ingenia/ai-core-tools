import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

from models.enums.import_row_status import ImportRowStatus


class ImportJobRow(Base):
    """One (url, metadata) row surviving dedup within an ImportJob."""
    __tablename__ = 'import_job_row'

    id = Column(Integer, primary_key=True, autoincrement=True)
    import_job_id = Column(Integer, sa.ForeignKey('import_job.id', ondelete='CASCADE'), nullable=False, index=True)

    url = Column(String(2048), nullable=False)
    row_metadata = Column(JSON, nullable=True)

    status = Column(
        sa.Enum(ImportRowStatus, name='import_row_status', create_type=False),
        nullable=False,
        default=ImportRowStatus.PENDING,
        server_default='PENDING',
    )
    failure_reason = Column(String(32), nullable=True)
    staged_path = Column(String(1000), nullable=True)
    resource_id = Column(Integer, sa.ForeignKey('Resource.resource_id', ondelete='SET NULL'), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    import_job = relationship('ImportJob', back_populates='rows')

from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

class Resource(Base):
    __tablename__ = 'Resource'
    resource_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    create_date = Column(DateTime, default=datetime.now)
    uri = Column(String(1000))
    type = Column(String(45))
    status = Column(String(45))
    repository_id = Column(Integer,
                        ForeignKey('Repository.repository_id'),
                        nullable=True)
    folder_id = Column(Integer,
                       ForeignKey('Folder.folder_id'),
                       nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # CSV-import metadata, merged into indexed chunk metadata
    # Indexing progress, persisted so the UI survives F5 and backend restarts
    # (the in-memory ingestion session does not).  Unit: LightRAG documents
    # (one per PDF page).  NULL while the backend cannot report sub-file progress.
    progress_done = Column(Integer, nullable=True)
    progress_total = Column(Integer, nullable=True)
    # Same value for every resource of one upload batch: identifies the batch and
    # provides the start time the ETA is extrapolated from.
    progress_started_at = Column(DateTime, nullable=True)

    repository = relationship('Repository',
                           back_populates='resources',
                           foreign_keys=[repository_id])
    folder = relationship('Folder',
                         back_populates='resources',
                         foreign_keys=[folder_id]) 
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from enum import Enum
from datetime import datetime

class SiloType(Enum):
    CUSTOM = "CUSTOM"
    REPO = "REPO"
    DOMAIN = "DOMAIN"
    SHAREPOINT = "SHAREPOINT"
    
class Silo(Base):
    __tablename__ = 'Silo'
    silo_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    create_date = Column(DateTime, default=datetime.now)
    status = Column(String(45))
    silo_type = Column(String(45))  # Store as String in DB
    app_id = Column(Integer, ForeignKey('App.app_id'))
    app = relationship('App', back_populates='silos')
    fixed_metadata = Column(Boolean, default=False)
    metadata_definition_id = Column(Integer, ForeignKey('OutputParser.parser_id'), nullable=True)
    metadata_definition = relationship('OutputParser', uselist=False)
    embedding_service_id = Column(Integer, ForeignKey('embedding_service.service_id'), nullable=True)
    embedding_service = relationship('EmbeddingService', uselist=False)
    indexing_service_id = Column(Integer, ForeignKey('AIService.service_id'), nullable=True)
    indexing_service = relationship('AIService', uselist=False, foreign_keys=[indexing_service_id])

    # LightRAG 2026.05 role-specific LLM configuration. Each role is an
    # independent AIService so operators can size models per task
    # (small/fast for KEYWORDS, mid-tier for EXTRACT, large for QUERY,
    # multimodal for VLM). ``indexing_service_id`` above is kept as a
    # legacy alias that maps to ``extract_service_id``.
    query_service_id = Column(Integer, ForeignKey('AIService.service_id'), nullable=True)
    query_service = relationship('AIService', uselist=False, foreign_keys=[query_service_id])
    extract_service_id = Column(Integer, ForeignKey('AIService.service_id'), nullable=True)
    extract_service = relationship('AIService', uselist=False, foreign_keys=[extract_service_id])
    keywords_service_id = Column(Integer, ForeignKey('AIService.service_id'), nullable=True)
    keywords_service = relationship('AIService', uselist=False, foreign_keys=[keywords_service_id])
    vlm_service_id = Column(Integer, ForeignKey('AIService.service_id'), nullable=True)
    vlm_service = relationship('AIService', uselist=False, foreign_keys=[vlm_service_id])

    vector_db_type = Column(String(45), default='PGVECTOR')

    # Secondary selector used only when vector_db_type == LIGHTRAG.
    # Determines which vector storage backend LightRAG should use.
    lightrag_vector_db_type = Column(String(45), nullable=True, default='QDRANT')

    lightrag_chunk_strategy = Column(String(45), nullable=True)
    lightrag_chunk_token_size = Column(Integer, nullable=True)
    lightrag_chunk_overlap_token_size = Column(Integer, nullable=True)
    lightrag_graph_context_enabled = Column(Boolean, default=False, nullable=True)

    is_frozen = Column(Boolean, default=False, nullable=False)
    agents = relationship('Agent', lazy=True)
    repository = relationship('Repository', back_populates='silo')
    domain = relationship('Domain', back_populates='silo') 
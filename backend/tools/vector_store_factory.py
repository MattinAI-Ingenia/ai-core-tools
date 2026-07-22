"""
Unified VectorStore facade that replaces PGVectorTools.

This module provides a unified interface for vector database operations,
abstracting away the underlying implementation (PGVector, Qdrant, etc.).
It serves as a drop-in replacement for the deprecated PGVectorTools class.

Includes factory logic for creating vector store instances based on configuration.
"""

import config
from typing import List, Optional, Dict

from tools.vector_stores.vector_store_interface import VectorStoreInterface
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:

    # Supported vector database types (including future planned support)
    SUPPORTED_TYPES = {
        'PGVECTOR': 'PGVector (PostgreSQL with pgvector extension)',
        'QDRANT': 'Qdrant vector database',
        'LIGHTRAG': 'LightRAG graph-enhanced RAG (Neo4j + Qdrant + PostgreSQL)',
        'PINECONE': 'Pinecone vector database (future support)',
        'WEAVIATE': 'Weaviate vector database (future support)',
        'CHROMA': 'Chroma vector database (future support)',
    }

    # Types that are currently implemented and can be selected by users
    IMPLEMENTED_TYPES = ('PGVECTOR', 'QDRANT', 'LIGHTRAG')

    _instances: Dict[str, VectorStoreInterface] = {}

    @staticmethod
    def get_vector_store(db, vector_db_type: Optional[str] = None, **kwargs) -> VectorStoreInterface:
        """Return a cached vector store instance for the requested backend."""

        resolved_type = (vector_db_type or 'PGVECTOR').upper()

        if resolved_type not in VectorStoreFactory.SUPPORTED_TYPES:
            supported = ', '.join(VectorStoreFactory.SUPPORTED_TYPES.keys())
            raise ValueError(
                f"Unsupported VECTOR_DB_TYPE: {resolved_type}. Supported types: {supported}"
            )

        if resolved_type not in VectorStoreFactory.IMPLEMENTED_TYPES:
            raise NotImplementedError(
                f"{resolved_type} support is planned but not yet implemented. Currently available: "
                f"{', '.join(VectorStoreFactory.IMPLEMENTED_TYPES)}"
            )

        if resolved_type in VectorStoreFactory._instances:
            return VectorStoreFactory._instances[resolved_type]

        logger.info("Initializing vector store backend: %s", resolved_type)

        if resolved_type == 'PGVECTOR':
            instance = VectorStoreFactory._create_pgvector_backend(db)
        elif resolved_type == 'QDRANT':
            instance = VectorStoreFactory._create_qdrant_backend(db)
        elif resolved_type == 'LIGHTRAG':
            instance = VectorStoreFactory._create_lightrag_backend(db, **kwargs)
        else:
            # Guard clause for future implementations
            raise NotImplementedError(f"Vector DB type {resolved_type} is not implemented yet")

        if resolved_type != 'LIGHTRAG':
            VectorStoreFactory._instances[resolved_type] = instance
        return instance

    @staticmethod
    def get_available_type_options() -> List[Dict[str, str]]:
        """Expose implemented vector DB choices with human-friendly labels."""

        options: List[Dict[str, str]] = []
        for key in VectorStoreFactory.IMPLEMENTED_TYPES:
            label = VectorStoreFactory.SUPPORTED_TYPES.get(key, key)
            options.append({
                'code': key,
                'label': label
            })
        return options
        
    
    @staticmethod
    def _create_pgvector_backend(db) -> VectorStoreInterface:
        """
        Create PGVector backend instance.
        
        Args:
            db: Database object
            
        Returns:
            PGVectorStore instance
        """
        from tools.vector_stores.pgvector_store import PGVectorStore
        
        logger.debug("Creating PGVector store with existing database connection")
        instance = PGVectorStore(db)
        try:
            # Ensure the backend tables exist so first-index workflows don't fail
            instance.ensure_backend_ready()
        except Exception as exc:
            logger.warning("PGVector backend readiness check failed: %s", exc)
        return instance
    
    @staticmethod
    def _create_qdrant_backend(db) -> VectorStoreInterface:
        """
        Create Qdrant backend instance.
        
        Args:
            db: Database object (passed for API consistency)
            
        Returns:
            QdrantStore instance
            
        Raises:
            ValueError: If required Qdrant configuration is missing
        """
        from tools.vector_stores.qdrant_store import QdrantStore
        
        if not config.QDRANT_URL:
            raise ValueError(
                "QDRANT_URL environment variable is required when VECTOR_DB_TYPE=QDRANT"
            )
        
        logger.debug(f"Creating Qdrant store with URL: {config.QDRANT_URL}")
        return QdrantStore(
            db=db,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            prefer_grpc=config.QDRANT_PREFER_GRPC
        )

    @staticmethod
    def _create_lightrag_backend(db, **kwargs) -> VectorStoreInterface:
        from tools.vector_stores.lightrag.adapters import is_lightrag_available
        from tools.vector_stores.lightrag_store import LightRAGStore

        if not is_lightrag_available():
            raise RuntimeError(
                "LightRAG is not available. Install lightrag-hku[offline-storage] "
                "and set LIGHTRAG_ENABLED=true."
            )

        ai_service = kwargs.get('ai_service')
        embedding_service = kwargs.get('embedding_service')

        if ai_service is None or embedding_service is None:
            raise ValueError(
                "LightRAG requires both ai_service and embedding_service. "
                "Pass them via VectorStoreFactory.get_vector_store(db, 'LIGHTRAG', "
                "ai_service=..., embedding_service=...)."
            )

        logger.debug("Creating LightRAG store")
        return LightRAGStore(
            db=db,
            ai_service=ai_service,
            embedding_service=embedding_service,
            keywords_service=kwargs.get('keywords_service'),
            vlm_service=kwargs.get('vlm_service'),
            lightrag_vector_db_type=kwargs.get('lightrag_vector_db_type'),
            lightrag_chunk_token_size=kwargs.get('lightrag_chunk_token_size'),
            lightrag_chunk_overlap_token_size=kwargs.get('lightrag_chunk_overlap_token_size'),
            lightrag_chunk_strategy=kwargs.get('lightrag_chunk_strategy'),
            lightrag_language=kwargs.get('lightrag_language'),
            lightrag_entity_extract_max_gleaning=kwargs.get('lightrag_entity_extract_max_gleaning'),
            lightrag_max_source_ids_per_entity=kwargs.get('lightrag_max_source_ids_per_entity'),
            lightrag_max_source_ids_per_relation=kwargs.get('lightrag_max_source_ids_per_relation'),
            lightrag_entity_types=kwargs.get('lightrag_entity_types'),
        )
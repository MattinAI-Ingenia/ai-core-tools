"""
Playground File Service

Manages a temporary repository + silo per (agent, conversation session) so that
PDF and text files attached in the agent playground are vectorized and retrieved
via RAG, instead of being injected verbatim into the prompt.

Only PDF and text files are handled here. The temp repo is created lazily on the
first vectorizable upload and cleaned up on conversation reset.
"""

import os
import logging
from typing import Optional, List

from sqlalchemy.orm import Session

from models.repository import Repository
from repositories.repository_repository import RepositoryRepository
from repositories.embedding_service_repository import EmbeddingServiceRepository
from services.repository_service import RepositoryService
from services.silo_service import SiloService
from utils.config import get_app_config

logger = logging.getLogger(__name__)

TEMP_REPO_PREFIX = "_playground_"

# File types vectorized into the temp silo instead of injected into the prompt.
VECTORIZABLE_FILE_TYPES = {"pdf", "text"}


def _temp_repo_name(agent_id: int, session_id: str) -> str:
    return f"{TEMP_REPO_PREFIX}{agent_id}_{session_id}"


class PlaygroundFileService:
    """Temporary PDF/text repositories for the agent playground."""

    @staticmethod
    def get_temp_repository(
        app_id: int,
        agent_id: int,
        session_id: str,
        db: Session,
    ) -> Optional[Repository]:
        """Find the temp playground repository for a given agent + session."""
        name = _temp_repo_name(agent_id, session_id)
        repos = RepositoryRepository.get_by_app_id(db, app_id)
        return next((r for r in repos if r.name == name), None)

    @staticmethod
    def get_or_create_temp_repository(
        app_id: int,
        agent_id: int,
        session_id: str,
        db: Session,
        embedding_service_id: Optional[int] = None,
    ) -> Optional[Repository]:
        """Get or create a temporary repository for playground files.

        The temp repo name follows the convention
        ``_playground_{agent_id}_{session_id}``. A new silo is automatically
        created with the repo (standard flow). Returns ``None`` when no
        embedding service is available in the app.
        """
        existing = PlaygroundFileService.get_temp_repository(
            app_id, agent_id, session_id, db
        )
        if existing:
            return existing

        # Resolve embedding service: explicit > first available in app
        if not embedding_service_id:
            app_emb_services = EmbeddingServiceRepository.get_by_app_id(db, app_id)
            if app_emb_services:
                embedding_service_id = app_emb_services[0].service_id

        if not embedding_service_id:
            logger.warning(
                "No embedding service available for app %s; cannot create temp "
                "playground repository",
                app_id,
            )
            return None

        repo = Repository(
            name=_temp_repo_name(agent_id, session_id),
            type="playground_file",
            status="active",
            app_id=app_id,
        )

        created = RepositoryService.create_repository(
            repository=repo,
            embedding_service_id=embedding_service_id,
            db=db,
        )
        logger.info(
            "Created temp playground repo %s (silo %s) for agent %s session %s",
            created.repository_id,
            created.silo_id,
            agent_id,
            session_id,
        )
        return created

    @staticmethod
    def get_temp_silo_ids_for_agent(
        app_id: int,
        agent_id: int,
        session_id: Optional[str],
        db: Session,
    ) -> List[int]:
        """Return silo IDs from the temp playground repo for this agent/session.

        Used by agent execution to include temp file silos in retrieval.
        """
        if not session_id:
            return []
        repo = PlaygroundFileService.get_temp_repository(
            app_id, agent_id, session_id, db
        )
        if repo and repo.silo_id:
            return [repo.silo_id]
        return []

    @staticmethod
    def vectorize_uploaded_file(
        app_id: int,
        agent_id: int,
        session_id: str,
        file_id: str,
        filename: str,
        file_path: Optional[str],
        content: Optional[str],
        db: Session,
        embedding_service_id: Optional[int] = None,
    ) -> bool:
        """Vectorize a single PDF/text file into the shared temp silo.

        Args:
            app_id: App ID.
            agent_id: Agent ID.
            session_id: Conversation session ID (e.g. ``conv_5_abc123``).
            file_id: Unique file identifier.
            filename: Original filename.
            file_path: Relative path to ``TMP_BASE_FOLDER`` (may be ``None``).
            content: Pre-extracted text content (fallback if file unavailable).
            db: Database session.
            embedding_service_id: Optional explicit embedding service ID.

        Returns:
            ``True`` if the file was successfully vectorized.
        """
        repo = PlaygroundFileService.get_or_create_temp_repository(
            app_id, agent_id, session_id, db,
            embedding_service_id=embedding_service_id,
        )
        if not repo or not repo.silo_id:
            logger.warning(
                "Could not create temp repository/silo for file vectorization"
            )
            return False

        app_config = get_app_config()
        tmp_base = app_config['TMP_BASE_FOLDER']

        base_metadata = {
            "file_id": file_id,
            "filename": filename,
            "source": "playground_upload",
        }

        docs_indexed = False

        # Preferred path: file-based extraction with proper loaders/chunking.
        if file_path:
            abs_path = os.path.join(tmp_base, file_path)
            if os.path.exists(abs_path):
                ext = os.path.splitext(filename)[1].lower()
                try:
                    docs = SiloService.extract_documents_from_file(
                        abs_path, ext, base_metadata
                    )
                    if docs:
                        SiloService.index_multiple_content(
                            repo.silo_id,
                            [
                                {"content": d.page_content, "metadata": d.metadata}
                                for d in docs
                            ],
                            db,
                        )
                        docs_indexed = True
                        logger.info(
                            "Vectorized %s via file extraction into silo %s: %d chunk(s)",
                            filename, repo.silo_id, len(docs),
                        )
                except Exception as exc:
                    logger.warning(
                        "File-based extraction failed for %s, falling back to "
                        "content: %s",
                        filename, exc,
                    )
            else:
                logger.warning(
                    "File path '%s' not found on disk for %s; using content fallback",
                    abs_path, filename,
                )

        # Fallback: index the pre-extracted text content.
        if not docs_indexed and content:
            SiloService.index_multiple_content(
                repo.silo_id,
                [{"content": content, "metadata": base_metadata}],
                db,
            )
            docs_indexed = True
            logger.info(
                "Vectorized %s via content fallback into silo %s as a single chunk "
                "(%d chars) — no file-based chunking applied",
                filename, repo.silo_id, len(content),
            )

        if docs_indexed:
            logger.info(
                "Vectorized file %s (%s) into silo %s",
                file_id, filename, repo.silo_id,
            )
        return docs_indexed

    @staticmethod
    def cleanup(
        app_id: int,
        agent_id: int,
        session_id: str,
        db: Session,
    ) -> bool:
        """Delete the temp playground repository, its silo, and vector data."""
        repo = PlaygroundFileService.get_temp_repository(
            app_id, agent_id, session_id, db
        )
        if not repo:
            return False

        logger.info(
            "Cleaning up playground files: repo %s, silo %s (agent %s, session %s)",
            repo.repository_id,
            repo.silo_id,
            agent_id,
            session_id,
        )
        RepositoryService.delete_repository(repo, db)
        return True

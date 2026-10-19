from typing import Optional, List
from sqlalchemy.orm import Session
from models.app import App
from repositories.app_repository import AppRepository
from repositories.app_collaboration_repository import AppCollaborationRepository
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryServiceConfigError(ValueError):
    """Raised only by `_validate_memory_service_ids` (AC-38).

    A distinct subclass of `ValueError` — not a bare one — so the
    `/internal` router can translate *this* failure into a 400 with the
    (safe, user-facing) message it carries, while any other, unrelated
    `ValueError` raised elsewhere in `create_or_update_app` still falls
    through to the router's generic handling instead of being misreported
    as a memory-configuration problem (security-audit LOW, step_009
    follow-up).
    """


class AppService:

    def __init__(self, db: Session):
        self.db = db
        self.app_repo = AppRepository(db)
        self.collaboration_repo = AppCollaborationRepository(db)

    def get_apps_by_user(self, user_id: int) -> List[App]:
        """Get apps owned by a specific user ordered by creation date"""
        return self.app_repo.get_by_owner(user_id)

    def get_collaborated_apps(self, user_id: int):
        """Get apps where user is an accepted collaborator"""
        return self.app_repo.get_collaborated_apps(user_id)

    def get_all_apps(self) -> List[App]:
        """Get all apps."""
        return self.app_repo.get_all()

    def get_apps(self, user_id: int) -> List[App]:
        """Get all apps for a specific user (owned + collaborated) ordered by creation date"""
        return self.collaboration_repo.get_user_accessible_apps(user_id)

    def get_app(self, app_id: int) -> Optional[App]:
        """Get a specific app by ID"""
        return self.app_repo.get_by_id(app_id)
    
    def create_or_update_app(self, app_data: dict, user_id: int = None) -> App:
        """Create a new app or update an existing one.

        In SaaS mode, enforces the per-user app limit before creation.

        Raises:
            MemoryServiceConfigError: if `memory_extraction_service_id` or
                `memory_embedding_service_id` is present in `app_data` and
                fails validation (AC-38). A `ValueError` subclass —
                callers at the `/internal` boundary translate *this specific
                type* into a 400.
        """
        app_id = app_data.get('app_id')
        existing_app = self.app_repo.get_by_id(app_id) if app_id else None

        if 'memory_extraction_service_id' in app_data or 'memory_embedding_service_id' in app_data:
            self._validate_memory_service_ids(
                app_id=app_id,
                extraction_service_id=app_data.get('memory_extraction_service_id'),
                embedding_service_id=app_data.get('memory_embedding_service_id'),
            )

        if existing_app:
            return self.app_repo.update(existing_app, app_data)

        # Enforce app limit (SaaS mode only, no-op in self-managed)
        if user_id is not None:
            from services.tier_enforcement_service import TierEnforcementService
            TierEnforcementService.check_app_limit(self.db, user_id)

        # Create new app
        return self.app_repo.create(app_data)

    def _validate_memory_service_ids(
        self,
        app_id: Optional[int],
        extraction_service_id: Optional[int],
        embedding_service_id: Optional[int],
    ) -> None:
        """Validate the two App-level memory service FKs (D-16, FR-2b, FR-3(b)).

        Mirrors the `default_sandbox_service_id` App-or-system visibility rule:
        a non-`None` id must load and either belong to this App or be
        system-level (`app_id IS NULL`); anything else — a nonexistent id or
        another App's service — is rejected (AC-38). Any embedding width is
        accepted: the memory store is dimension-agnostic
        (`tools/memory/client.py`).

        Args:
            app_id: The App these ids are being configured for.
            extraction_service_id: Candidate `App.memory_extraction_service_id`.
            embedding_service_id: Candidate `App.memory_embedding_service_id`.

        Raises:
            MemoryServiceConfigError: naming the offending field, on any of
                the failures above.
        """
        from repositories.ai_service_repository import AIServiceRepository
        from repositories.embedding_service_repository import EmbeddingServiceRepository

        if extraction_service_id is not None:
            ai_service = AIServiceRepository.get_by_id(self.db, extraction_service_id)
            if ai_service is None or not (ai_service.app_id == app_id or ai_service.app_id is None):
                raise MemoryServiceConfigError(
                    f"memory_extraction_service_id '{extraction_service_id}' does not reference "
                    "a valid AI service for this app."
                )

        if embedding_service_id is not None:
            embedding_service = EmbeddingServiceRepository.get_by_id(self.db, embedding_service_id)
            if embedding_service is None or not (
                embedding_service.app_id == app_id or embedding_service.app_id is None
            ):
                raise MemoryServiceConfigError(
                    f"memory_embedding_service_id '{embedding_service_id}' does not reference "
                    "a valid embedding service for this app."
                )


    def get_memory_service_options(self, app_id: int) -> dict:
        """Build the App-visible + system-level option lists for the App-settings
        memory service selectors (FR-3(b)).

        Reuses the same combination and labelling convention as the agent
        form's `ai_services` list (`AgentRepository.get_form_data_for_agent`):
        App-scoped services first, then system-level (`app_id IS NULL`)
        services prefixed with `"[System] "`.

        Args:
            app_id: The App to build the option lists for.

        Returns:
            A dict with `'ai_services'` and `'embedding_services'` keys, each a
            list of `{"service_id": int, "name": str}` dicts.
        """
        from repositories.ai_service_repository import AIServiceRepository
        from repositories.embedding_service_repository import EmbeddingServiceRepository

        ai_services = AIServiceRepository.get_by_app_id(self.db, app_id) + AIServiceRepository.get_system_services(
            self.db
        )
        embedding_services = EmbeddingServiceRepository.get_by_app_id(
            self.db, app_id
        ) + EmbeddingServiceRepository.get_system_services(self.db)

        return {
            'ai_services': [
                {"service_id": s.service_id, "name": f"[System] {s.name}" if s.app_id is None else s.name}
                for s in ai_services
            ],
            'embedding_services': [
                {"service_id": s.service_id, "name": f"[System] {s.name}" if s.app_id is None else s.name}
                for s in embedding_services
            ],
        }

    def delete_app(self, app_id: int) -> bool:
        """Delete an app and all its related data with proper cascade deletion
        
        Note: Access control should be handled at the API level
        """
        from .repository_service import RepositoryService
        from .domain_service import DomainService
        from .agent_service import AgentService
        from .output_parser_service import OutputParserService
        from .ai_service_service import AIServiceService
        from .silo_service import SiloService
        from .embedding_service_service import EmbeddingServiceService
        from .sandbox_service_service import SandboxServiceService
        from .api_key_service import APIKeyService
        from .mcp_config_service import MCPConfigService
        from .resource_service import ResourceService
        from .skill_service import SkillService
        from .mcp_server_service import MCPServerService
        
        # Get the app
        app = self.app_repo.get_by_id(app_id)
        if not app:
            logger.warning(f"App {app_id} not found for deletion")
            return False
        
        logger.info(f"Starting cascade deletion for app {app_id}: {app.name}")
        
        try:
            # 0. Purge global user memory for all scopes this App holds (FR-8/D-8).
            # Enumerated directly from the memory store (AD-10), never from
            # AppCollaborator/Conversation rows, so it has no ordering dependency
            # on agents/collaborators still existing — it goes first only so a
            # later step's failure cannot skip it. Never touches the checkpointer
            # (D-8/AC-42): this purges durable memory facts only, not conversation
            # history.
            #
            # Security-audit decision (MEDIUM 2, step_006 follow-up): unlike the
            # "vectors are orphaned (acceptable for app deletion)" convention this
            # cascade otherwise follows for non-DB side effects, an incomplete or
            # failed memory purge here IS re-raised, aborting the deletion. Once
            # the App row below is actually deleted, the operator erasure endpoint
            # (`DELETE /internal/apps/{app_id}/global-memory/users/{user_id}`) 404s
            # — there is no App left to check the ADMINISTRATOR role against —
            # so any surviving memory becomes reachable ONLY if the affected user
            # happens to call their own cross-App self-service endpoint. Reporting
            # a deletion as "successful" while quietly leaving that behind would
            # violate NFR-5/AC-61. This step runs first, before any row belonging
            # to this App has been touched, so re-raising is cheap and safe: the
            # caller simply gets a failed, retryable App deletion — never one that
            # silently orphaned personal data. `GlobalMemoryPurgeIncompleteError`
            # is a distinct type (not a bare `RuntimeError`) precisely so the
            # router can tell "retry, the memory store hiccuped" (503) apart from
            # an ordinary cascade-deletion bug (500) — see the router for that
            # mapping.
            from .global_memory_service import GlobalMemoryPurgeIncompleteError, delete_app_global_memory
            try:
                memory_result = delete_app_global_memory(self.db, app_id)
            except GlobalMemoryPurgeIncompleteError:
                raise
            except Exception as memory_exc:
                raise GlobalMemoryPurgeIncompleteError(
                    f"Global memory purge for app {app_id} raised an unexpected error: {memory_exc}"
                ) from memory_exc

            if not memory_result.complete:
                raise GlobalMemoryPurgeIncompleteError(
                    f"Global memory purge for app {app_id} did not complete "
                    f"(partial count={memory_result.deleted})"
                )

            logger.info(f"Purged {memory_result.deleted} global memory record(s) for app {app_id}")

            # Initialize all required services
            repository_service = RepositoryService()
            domain_service = DomainService()
            agent_service = AgentService()
            output_parser_service = OutputParserService()
            ai_service_service = AIServiceService()
            silo_service = SiloService()
            embedding_service_service = EmbeddingServiceService()
            sandbox_service_service = SandboxServiceService()
            api_key_service = APIKeyService()
            mcp_config_service = MCPConfigService()
            resource_service = ResourceService()

            # Delete in the correct order to respect foreign key constraints
            
            # 1. Delete agents (they depend on AI services, silos, output parsers)
            agents = self.app_repo.get_agents_by_app_id(app_id)
            for agent in agents:
                logger.info(f"Deleting agent {agent.agent_id}: {agent.name}")
                agent_service.delete_agent(self.db, agent.agent_id)

            # 2. Delete skills (after agents, since agents may reference skills)
            skills = self.app_repo.get_skills_by_app_id(app_id)
            for skill in skills:
                logger.info(f"Deleting skill {skill.skill_id}: {skill.name}")
                SkillService.delete_skill(self.db, app_id, skill.skill_id)

            # 3. Delete MCP servers (after agents, since they may expose agents)
            mcp_servers = self.app_repo.get_mcp_servers_by_app_id(app_id)
            for server in mcp_servers:
                logger.info(f"Deleting MCP server {server.server_id}: {server.name}")
                MCPServerService.delete_mcp_server(self.db, app_id, server.server_id)

            # 4. Delete resources (they depend on repositories)
            repositories = self.app_repo.get_repositories_by_app_id(app_id)
            for repository in repositories:
                resources = resource_service.get_resources_by_repo_id(repository.repository_id, self.db)
                for resource in resources:
                    logger.info(f"Deleting resource {resource.resource_id}: {resource.name}")
                    resource_service.delete_resource(resource.resource_id, self.db)
            
            # 5. Get domains (for later deletion; DomainUrl/CrawlPolicy/CrawlJob cascade from domain FK)
            domains = self.app_repo.get_domains_by_app_id(app_id)

            # 5b. Delete SharePoint sources and their silos (before main silo deletion)
            from repositories.sharepoint_source_repository import SharePointSourceRepository
            sp_sources = SharePointSourceRepository.list_by_app(app_id, self.db)
            for sp_source in sp_sources:
                logger.info(f"Deleting SharePoint source {sp_source.id}: {sp_source.name}")
                silo_id = sp_source.silo_id
                # SharePointFile rows cascade-delete via FK; vectors are orphaned (acceptable for app deletion)
                self.db.delete(sp_source)
                self.db.flush()
                if silo_id:
                    silo_service.delete_silo(silo_id, self.db)

            # 6. Delete repositories (they depend on silos)
            for repository in repositories:
                logger.info(f"Deleting repository {repository.repository_id}: {repository.name}")
                repository_service.delete_repository(repository, self.db)
            
            # 7. Delete domains (they depend on silos)
            for domain in domains:
                logger.info(f"Deleting domain {domain.domain_id}: {domain.name}")
                domain_service.delete_domain(domain.domain_id, self.db)
            
            # 8. Delete silos (they depend on embedding services and output parsers)
            silos = self.app_repo.get_silos_by_app_id(app_id)
            for silo in silos:
                logger.info(f"Deleting silo {silo.silo_id}: {silo.name}")
                silo_service.delete_silo(silo.silo_id, self.db)
            
            # 9. Delete output parsers
            output_parsers = self.app_repo.get_output_parsers_by_app_id(app_id)
            for parser in output_parsers:
                logger.info(f"Deleting output parser {parser.parser_id}: {parser.name}")
                output_parser_service.delete_output_parser(self.db, app_id, parser.parser_id)
            
            # 10. Delete MCP configs
            mcp_configs = self.app_repo.get_mcp_configs_by_app_id(app_id)
            for config in mcp_configs:
                logger.info(f"Deleting MCP config {config.config_id}: {config.name}")
                mcp_config_service.delete_mcp_config(self.db, app_id, config.config_id)
            
            # 11. Delete API keys
            api_keys = self.app_repo.get_api_keys_by_app_id(app_id)
            for api_key in api_keys:
                logger.info(f"Deleting API key {api_key.key_id}: {api_key.name}")
                api_key_service.delete_api_key(self.db, app_id, api_key.key_id)
            
            # 12. Delete AI services
            ai_services = self.app_repo.get_ai_services_by_app_id(app_id)
            for service in ai_services:
                logger.info(f"Deleting AI service {service.service_id}: {service.name}")
                ai_service_service.delete_ai_service(self.db, app_id, service.service_id)
            
            # 13. Delete embedding services
            embedding_services = self.app_repo.get_embedding_services_by_app_id(app_id)
            for service in embedding_services:
                logger.info(f"Deleting embedding service {service.service_id}: {service.name}")
                embedding_service_service.delete_embedding_service(self.db, app_id, service.service_id)
            
            # 13b. Delete sandbox services
            sandbox_services = self.app_repo.get_sandbox_services_by_app_id(app_id)
            for service in sandbox_services:
                logger.info(f"Deleting sandbox service {service.service_id}: {service.name}")
                sandbox_service_service.delete_sandbox_service(self.db, app_id, service.service_id)

            # 14. Delete collaborations
            collaborations = self.collaboration_repo.get_collaborations_by_app(app_id)
            for collab in collaborations:
                logger.info(f"Deleting collaboration {collab.id}")
                self.collaboration_repo.delete_collaboration(collab)
            
            # 15. Finally, delete the app
            success = self.app_repo.delete(app)
            
            if success:
                logger.info(f"Successfully deleted app {app_id} and all related data")
            
            return success

        except GlobalMemoryPurgeIncompleteError as e:
            # Nothing has been mutated for this App yet (this is step 0) — the
            # rollback below is a no-op for this App's own rows, but re-running
            # it here is harmless and keeps this branch symmetric with the
            # generic one. Re-raised (not converted to `return False`) so the
            # router can answer 503/retryable instead of the generic 500 the
            # branch below reports for an ordinary cascade bug.
            logger.error(f"Aborting deletion of app {app_id}: {e}; safe to retry")
            self.db.rollback()
            raise

        except Exception as e:
            logger.error(f"Error during cascade deletion of app {app_id}: {str(e)}")
            # Rollback the transaction
            self.db.rollback()
            return False
"""A repository's silo-owned fields must survive both write paths.

These live on the Silo, so the repository endpoints forward them explicitly —
and twice now one was added to the schema and the update path but not to
creation. The symptom is silent: accepted, no error, silo keeps its default.
lightrag_entity_types_mode did it, leaving "Write them yourself" unclearable.
"""

import inspect

from schemas.repository_schemas import CreateRepositorySchema, UpdateRepositorySchema
from services.repository_service import _SILO_OWNED_FIELDS, RepositoryService


def test_creation_forwards_every_silo_owned_field():
    """create_repository must be able to receive each one."""
    accepted = set(inspect.signature(RepositoryService.create_repository).parameters)
    assert not _SILO_OWNED_FIELDS - accepted


def test_both_schemas_declare_them():
    """Pydantic drops undeclared fields silently, so a missing one here means
    the value never even reaches the service."""
    for schema in (CreateRepositorySchema, UpdateRepositorySchema):
        assert not _SILO_OWNED_FIELDS - set(schema.model_fields), schema.__name__

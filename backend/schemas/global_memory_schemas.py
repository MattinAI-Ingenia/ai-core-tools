"""Response schemas for cross-session global memory erasure (spec
`user-memory-mem0`, FR-8, step_006).
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class GlobalMemoryPurgeResponseSchema(BaseModel):
    """Response for a **self-service** global-memory erasure request.

    `purged` is safe to return here because the caller is only ever asking
    about their own data (D-11). **Never reuse this schema for the operator
    endpoint** (`DELETE /internal/apps/{app_id}/global-memory/users/{user_id}`)
    — that endpoint must not return a varying purge count at all (AC-62): an
    ADMINISTRATOR could otherwise use the count as a destructive oracle for
    which user PKs have memory in this App, including people who are not (or
    are no longer) members. The operator endpoint returns `204 No Content`
    instead.
    """

    purged: int

    model_config = ConfigDict(from_attributes=True)


class GlobalMemoryFactSchema(BaseModel):
    """One remembered fact, as shown to the user it is about (self-service only)."""

    key: str
    text: str
    agent_id: Optional[int] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class GlobalMemoryFactListSchema(BaseModel):
    """The caller's own remembered facts in one App, newest first."""

    facts: List[GlobalMemoryFactSchema]

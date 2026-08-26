"""_claim_pending_resource_for_indexing: the compare-and-swap the feeder uses.

Reproduces a real incident: cancel clicked right after upload, while phase 1
(chunk counting) was still running and every resource was still 'pending'.
request_ingestion_stop parked all of them to 'cancelled' and told the user so
— but the feeder's queue had been snapshotted before that, independent of
resource status, and it fed three of the four anyway: a plain read-then-write
in the feeder overwrote 'cancelled' straight back to 'indexing', so cancel's
own promise was silently reneged on for the resources the window still had
room for.

Uses committed_repo (see test_resume_indexing_status_reset.py) because this is
inherently a cross-session race — two independent connections contending for
the same row — not something a single in-memory mock can represent.
"""

from sqlalchemy.orm import Session

from models.resource import Resource
from services.resource_service import ResourceService

# Reuse the real-commit scaffolding from the sibling stamping-behavior tests
# instead of duplicating it.
from tests.integration.services.test_resume_indexing_status_reset import (  # noqa: F401
    committed_repo, _add,
)


def test_claims_a_pending_resource(committed_repo):
    session, repo = committed_repo
    resource = _add(session, repo, name='f.pdf', status='pending')

    claimed = ResourceService._claim_pending_resource_for_indexing(session, resource.resource_id)

    assert claimed is not None
    assert claimed.status == 'indexing'
    session.expire_all()
    assert session.get(Resource, resource.resource_id).status == 'indexing'


def test_loses_the_race_to_a_stop_that_already_parked_it(committed_repo, test_engine):
    """The scenario that actually happened: a stop request, from an
    independent session, parks the row to 'cancelled' before the feeder's
    queue (built earlier, oblivious to it) gets around to claiming it."""
    session, repo = committed_repo
    resource = _add(session, repo, name='f.pdf', status='pending')

    stopper = Session(bind=test_engine)
    try:
        stopper.query(Resource).filter_by(resource_id=resource.resource_id).update(
            {'status': 'cancelled'}
        )
        stopper.commit()
    finally:
        stopper.close()

    claimed = ResourceService._claim_pending_resource_for_indexing(session, resource.resource_id)

    assert claimed is None
    session.expire_all()
    assert session.get(Resource, resource.resource_id).status == 'cancelled'


def test_a_resource_that_no_longer_exists_is_not_claimed(committed_repo):
    session, repo = committed_repo

    claimed = ResourceService._claim_pending_resource_for_indexing(session, 9_999_999)

    assert claimed is None

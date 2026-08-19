"""``get_by_repository_id`` must return resources in upload/indexing order.

Without ORDER BY, Postgres returns rows in an unspecified order — the repo
page rendered files in whatever order the planner felt like, which looked
shuffled to users. resource_id is assigned in upload order, and that is also
the order the indexing queue processes files in, so ordering by it doubles
as indexing order without a dedicated "indexed_at" column.

Display priority (surfaces what needs attention first): indexing, then
error, then ready (completed), then pending (not started) last — each group
ordered by resource_id ascending. The pipeline processes one resource at a
time strictly in upload order (resource_service._run's
`for resource_id, ... in resource_snapshots` loop), so the currently-indexing
resource always holds the highest resource_id among anything already
'ready' — meaning the moment it finishes and flips to 'ready', it lands at
the *end* of that group, not the start.
"""

from models.repository import Repository
from models.resource import Resource
from repositories.resource_repository import ResourceRepository


def _make_repo(db, fake_app, fake_silo, name="order-repo"):
    repository = Repository(name=name, app_id=fake_app.app_id, silo_id=fake_silo.silo_id)
    db.add(repository)
    db.flush()
    return repository


def _add(db, repo, *, name, status):
    resource = Resource(name=name, uri=name, type=".pdf", status=status, repository_id=repo.repository_id)
    db.add(resource)
    db.flush()
    return resource


def test_returns_resources_in_upload_order(db, fake_app, fake_silo):
    repository = _make_repo(db, fake_app, fake_silo)

    # Insert out of name order so a name-based sort would not accidentally pass.
    third = _add(db, repository, name="c.pdf", status="ready")
    first = _add(db, repository, name="a.pdf", status="ready")
    second = _add(db, repository, name="b.pdf", status="ready")

    resources = ResourceRepository.get_by_repository_id(db, repository.repository_id)

    assert [r.resource_id for r in resources] == [third.resource_id, first.resource_id, second.resource_id]


def test_orders_by_status_priority_then_upload_order(db, fake_app, fake_silo):
    repository = _make_repo(db, fake_app, fake_silo)

    ready_1 = _add(db, repository, name="1.pdf", status="ready")
    ready_2 = _add(db, repository, name="2.pdf", status="ready")
    indexing = _add(db, repository, name="3.pdf", status="indexing")
    pending = _add(db, repository, name="4.pdf", status="pending")
    error = _add(db, repository, name="5.pdf", status="error")

    names = [r.name for r in ResourceRepository.get_by_repository_id(db, repository.repository_id)]
    assert names == ["3.pdf", "5.pdf", "1.pdf", "2.pdf", "4.pdf"]

    # The indexing resource finishes: it must move to the *end* of the ready
    # group, past the resources that finished before it — not jump to the front.
    # pending stays last regardless.
    indexing.status = "ready"
    db.flush()

    names_after = [r.name for r in ResourceRepository.get_by_repository_id(db, repository.repository_id)]
    assert names_after == ["5.pdf", "1.pdf", "2.pdf", "3.pdf", "4.pdf"]

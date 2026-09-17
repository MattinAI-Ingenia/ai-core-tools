"""AC-7: RBAC — a ``viewer`` gets 403; an ``editor`` can trigger the ingestion.

App-level roles come from ``AppCollaborator`` (``resolve_user_app_role``),
not from the platform role, so both users here get their own collaborator row
against the fixture app.
"""
import time

from models.app_collaborator import AppCollaborator, CollaborationRole, CollaborationStatus
from tests.integration.blob_ingest_helpers import BLOB_INGEST_URL, ingest_payload
from utils.local_auth_tokens import mint_access_token


def _headers_for(db, app_id, role):
    from models.user import User

    user = User(
        name=f"Blob RBAC {role.name}",
        email=f"blob-rbac-{role.name.lower()}-{time.time_ns()}@test.com",
        is_active=True,
        platform_role="editor",
    )
    db.add(user)
    db.flush()
    db.add(AppCollaborator(
        app_id=app_id, user_id=user.user_id, role=role,
        invited_by=user.user_id, status=CollaborationStatus.ACCEPTED,
    ))
    db.flush()
    token, _ = mint_access_token(user.user_id, user.email, user.name)
    return {"Authorization": f"Bearer {token}"}


def _post(client, repository, headers):
    return client.post(
        BLOB_INGEST_URL.format(app_id=repository.app_id, repository_id=repository.repository_id),
        json=ingest_payload(),
        headers=headers,
    )


def test_viewer_gets_403_and_never_reaches_the_ingestion(client, repository, db, fake_blob_client):
    viewer_headers = _headers_for(db, repository.app_id, CollaborationRole.VIEWER)

    response = _post(client, repository, viewer_headers)

    assert response.status_code == 403
    # The RoleChecker dependency runs before the handler: no listing, no
    # downloads, nothing touched.
    assert fake_blob_client.list_calls == 0


def test_editor_can_trigger_the_ingestion(
    client, repository, db, fake_blob_client, fake_dns, instant_indexing, tmp_repo_base,
):
    fake_blob_client.add_blob("docs/a.pdf")
    editor_headers = _headers_for(db, repository.app_id, CollaborationRole.EDITOR)

    response = _post(client, repository, editor_headers)

    assert response.status_code == 202
    body = response.json()
    assert body["queued"] == 1
    assert body["session_id"] == "sid"

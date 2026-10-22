"""Fixtures shared by the Azure Blob ingestion integration tests.

Kept in a conftest (not in ``blob_ingest_helpers.py``) so pytest discovers
them without each test module importing fixture functions — which both trips
pyflakes' unused-import analysis and re-registers the fixtures per module.

All of these are opt-in: a test only activates them by requesting them.
"""
import socket
from types import SimpleNamespace

import pytest

from services import silo_indexing_lock
from services.blob import azure_blob_client
from services.resource_service import ResourceService
from services.silo_service import SiloService

from tests.integration.blob_ingest_helpers import FakeContainerClient, fake_dns_global


@pytest.fixture
def fake_dns(monkeypatch):
    """Stub ``socket.getaddrinfo`` so no test resolves real DNS."""
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns_global)


@pytest.fixture
def fake_blob_client(monkeypatch):
    """Install a fresh :class:`FakeContainerClient` into ``azure_blob_client``.

    Tests add blobs to the returned object before triggering; every SDK call
    the ingestion makes (list + one download per blob) goes through it.
    """
    fake = FakeContainerClient()
    monkeypatch.setattr(azure_blob_client, "build_container_client", lambda cfg: fake)
    return fake


@pytest.fixture
def instant_indexing(monkeypatch):
    """Make background indexing complete immediately, like the real thread
    would on success: the silo lock is released and a session_id returned.

    Records the resource ids handed to each run so tests can assert what got
    indexed without depending on real LLM/embedding work.
    """
    runs = []

    def _fake(resources, silo_id=0, lock_conn=None, **_kwargs):
        silo_indexing_lock.release(lock_conn, silo_id)
        runs.append([getattr(r, "resource_id", None) for r in resources])
        return "sid"

    monkeypatch.setattr(ResourceService, "_index_resources_background", _fake)
    return runs


@pytest.fixture
def forbid_index_single_content(monkeypatch):
    """The FR-1 architectural guard: the Azure blob path must go through
    ``create_multiple_resources`` and never call ``index_single_content``."""
    sentinel = SimpleNamespace(
        side_effect=AssertionError(
            "SiloService.index_single_content must never be called by the "
            "Azure blob ingestion path (AD-1/AC-1)"
        )
    )
    monkeypatch.setattr(SiloService, "index_single_content", sentinel)
    return sentinel


@pytest.fixture
def tmp_repo_base(tmp_path, monkeypatch):
    """Redirect resource files to a throwaway directory (no repo pollution)."""
    monkeypatch.setattr("services.resource_service.REPO_BASE_FOLDER", str(tmp_path))
    return tmp_path


@pytest.fixture
def tmp_staging_root(tmp_path, monkeypatch):
    """Redirect the ingest staging root so AC-10 can assert it empties out."""
    staging_root = tmp_path / "blob_ingest_staging"
    monkeypatch.setattr(
        "services.azure_blob_ingest_service.get_app_config",
        lambda: {"TMP_BASE_FOLDER": str(tmp_path)},
    )
    return staging_root

# --- Service end-of-service fixtures (steps 011/012): builders live in
# ``routers/internal/_eos_helpers.py`` and ``_eos_enforcement_helpers.py``.
from models.app_collaborator import CollaborationRole
from tests.factories import UserFactory, configure_factories
from tests.integration.routers.internal._eos_helpers import WARNING_DAYS_ENV, headers_for, member_headers


@pytest.fixture(autouse=True)
def _clean_warning_days_env(monkeypatch):
    """A developer's local .env must not change the 30-day default."""
    monkeypatch.delenv(WARNING_DAYS_ENV, raising=False)


@pytest.fixture
def factories(db):
    configure_factories(db)
    return db


@pytest.fixture
def admin_headers(db, monkeypatch):
    """Headers of a platform admin (OMNIADMIN) who is NOT a collaborator of any app."""
    configure_factories(db)
    user = UserFactory(email="eos-platform-admin@mattin-test.com")
    monkeypatch.setenv("AICT_OMNIADMINS", user.email)
    db.flush()
    return headers_for(user)


@pytest.fixture
def app_admin_headers(db, fake_app):
    return member_headers(db, fake_app, CollaborationRole.ADMINISTRATOR)


@pytest.fixture
def app_editor_headers(db, fake_app):
    return member_headers(db, fake_app, CollaborationRole.EDITOR)


@pytest.fixture
def app_viewer_headers(db, fake_app):
    return member_headers(db, fake_app, CollaborationRole.VIEWER)

"""Validation-cap sampling + last-source persistence (frontend "Actualizar").

AC for the UI feature: each run ingests at most ``sample_size`` randomly
picked pending blobs (the frontend always sends 20), the response reports the
whole-run ``total_blobs``/``pending_blobs`` the informative message needs, and
the last successfully-listed source (account_url/container/prefixes/auth_mode —
never the SAS token) is persisted on the Repository row so the "Actualizar"
button survives page reloads.
"""
from contextlib import contextmanager
import random

import pytest

from tests.integration.blob_ingest_helpers import (
    ACCOUNT_URL,
    CONTAINER,
    SAS_TOKEN,
    FakeBlob,
    azure_blob_resources,
    trigger_ingestion,
)


@contextmanager
def seeded_random(seed):
    """Deterministic sampling for one block, without perturbing the suite."""
    state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(state)


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_sample_size_caps_each_run_and_reports_whole_run_counts(db, repository, fake_blob_client, instant_indexing):
    for i in range(5):
        fake_blob_client.add_blob(FakeBlob(f"docs/etiqueta-{i}.pdf", etag=f'"etag-{i}"'))

    with seeded_random(42):
        result = trigger_ingestion(repository, db, sample_size=2)

    assert result["total_blobs"] == 5
    assert result["pending_blobs"] == 5
    assert result["queued"] == 2
    assert result["skipped_unchanged"] == 0
    assert result["skipped_unsupported"] == 0
    assert len(azure_blob_resources(db, repository)) == 2

    with seeded_random(42):
        second = trigger_ingestion(repository, db, sample_size=2)

    assert second["pending_blobs"] == 3
    assert second["queued"] == 2
    assert second["skipped_unchanged"] == 2
    assert len(azure_blob_resources(db, repository)) == 4


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_sample_size_only_caps_pending_not_already_ingested(db, repository, fake_blob_client, instant_indexing):
    """The cap applies to the pending set only: unchanged blobs are counted,
    reported, and never sampled away."""
    for i in range(3):
        fake_blob_client.add_blob(FakeBlob(f"docs/etiqueta-{i}.pdf", etag=f'"etag-{i}"'))
    trigger_ingestion(repository, db)  # ingest all 3

    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-new.pdf", etag='"etag-new"'))

    result = trigger_ingestion(repository, db, sample_size=2)

    assert result["total_blobs"] == 4
    assert result["pending_blobs"] == 1
    assert result["queued"] == 1
    assert result["skipped_unchanged"] == 3


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_no_sample_size_ingests_everything_pending(db, repository, fake_blob_client, instant_indexing):
    for i in range(5):
        fake_blob_client.add_blob(FakeBlob(f"docs/etiqueta-{i}.pdf", etag=f'"etag-{i}"'))

    result = trigger_ingestion(repository, db)

    assert result["pending_blobs"] == 5
    assert result["queued"] == 5
    assert len(azure_blob_resources(db, repository)) == 5


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_successful_run_persists_the_source_without_secrets(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))

    trigger_ingestion(repository, db, prefixes=["docs/"])

    assert repository.azure_blob_source == {
        "account_url": ACCOUNT_URL,
        "container": CONTAINER,
        "prefixes": ["docs/"],
        "name_excludes": [],
        "auth_mode": "ANONYMOUS",
    }
    assert "sas_token" not in repository.azure_blob_source


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_sas_run_persists_the_mode_but_never_the_token(db, repository, fake_blob_client, instant_indexing):
    from services.azure_blob_ingest_service import AzureBlobIngestService
    from tests.integration.blob_ingest_helpers import ingest_payload

    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))

    AzureBlobIngestService.trigger_ingestion(
        repository.app_id, repository.repository_id,
        ingest_payload(auth_mode="SAS_TOKEN", sas_token=SAS_TOKEN), db,
    )

    assert repository.azure_blob_source["auth_mode"] == "SAS_TOKEN"
    assert "sas_token" not in repository.azure_blob_source
    assert SAS_TOKEN not in str(repository.azure_blob_source)


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_failed_listing_does_not_overwrite_the_persisted_source(db, repository, fake_blob_client, instant_indexing):
    from azure.core.exceptions import HttpResponseError

    from services.blob import azure_blob_client

    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))
    trigger_ingestion(repository, db)
    good_source = dict(repository.azure_blob_source)

    fake_blob_client.list_error = HttpResponseError(message="listing failed")
    with pytest.raises(azure_blob_client.BlobConnectionError):
        trigger_ingestion(repository, db)

    assert repository.azure_blob_source == good_source


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_single_blob_ingests_only_that_exact_file(db, repository, fake_blob_client, instant_indexing):
    for i in range(3):
        fake_blob_client.add_blob(FakeBlob(f"docs/etiqueta-{i}.pdf", etag=f'"etag-{i}"'))

    result = trigger_ingestion(repository, db, blob_name="docs/etiqueta-1.pdf")

    assert result["total_blobs"] == 1
    assert result["pending_blobs"] == 1
    assert result["queued"] == 1
    assert [r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)] == ["docs/etiqueta-1.pdf"]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_single_blob_already_ingested_is_up_to_date(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))
    trigger_ingestion(repository, db, blob_name="docs/etiqueta-a.pdf")

    result = trigger_ingestion(repository, db, blob_name="docs/etiqueta-a.pdf")

    assert result["pending_blobs"] == 0
    assert result["queued"] == 0
    assert result["skipped_unchanged"] == 1


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_single_blob_not_found_is_a_validation_error(db, repository, fake_blob_client, instant_indexing):
    from utils.error_handlers import ValidationError

    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))

    with pytest.raises(ValidationError, match="not found"):
        trigger_ingestion(repository, db, blob_name="docs/missing.pdf")


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_single_blob_and_prefix_are_mutually_exclusive(db, repository, fake_blob_client, instant_indexing):
    from utils.error_handlers import ValidationError

    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))

    with pytest.raises(ValidationError, match="mutually exclusive"):
        trigger_ingestion(repository, db, prefixes=["docs/"], blob_name="docs/etiqueta-a.pdf")


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_single_file_run_persists_the_container_not_the_blob_name(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))

    trigger_ingestion(repository, db, blob_name="docs/etiqueta-a.pdf")

    assert repository.azure_blob_source == {
        "account_url": ACCOUNT_URL,
        "container": CONTAINER,
        "prefixes": [],
        "name_excludes": [],
        "auth_mode": "ANONYMOUS",
    }

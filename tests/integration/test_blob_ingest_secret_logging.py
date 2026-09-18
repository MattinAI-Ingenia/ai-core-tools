"""AC-9: secret hygiene — a recognizable ``sas_token`` must appear in neither
the API responses nor the logs of a run, on both a success path and an
``AUTH_FAILED`` error path.

The backend loggers set ``propagate=False`` (``utils/logger.py``), so pytest's
``caplog`` cannot see them; :class:`LogCapture` attaches a handler to the
ingestion path's loggers directly instead.
"""
import pytest

from tests.integration.blob_ingest_helpers import (
    AUTH_FAILED_ERROR,
    BLOB_INGEST_URL,
    SAS_TOKEN,
    FakeBlob,
    LogCapture,
    ingest_payload,
    trigger_ingestion,
)

SECRET_MARKERS = (SAS_TOKEN, "SUPERSECRETSIG", "SharedAccessSignature=")


def assert_no_secret_in(text: str, context: str):
    for marker in SECRET_MARKERS:
        assert marker not in text, f"secret marker {marker!r} leaked into {context}"


@pytest.mark.usefixtures("tmp_repo_base", "fake_dns")
def test_success_path_never_logs_or_returns_the_token(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/a.pdf", etag='"etag-a"'))

    with LogCapture() as capture:
        result = trigger_ingestion(repository, db, auth_mode="SAS_TOKEN", sas_token=SAS_TOKEN)

    assert result["queued"] == 1
    assert_no_secret_in(str(result), "the response dict")
    assert_no_secret_in(capture.all_text(), "the service logs")


def test_auth_failure_path_never_logs_or_returns_the_token(
    client, repository, owner_headers, fake_blob_client, fake_dns,
):
    fake_blob_client.list_error = AUTH_FAILED_ERROR

    with LogCapture() as capture:
        response = client.post(
            BLOB_INGEST_URL.format(app_id=repository.app_id, repository_id=repository.repository_id),
            json=ingest_payload(auth_mode="SAS_TOKEN", sas_token=SAS_TOKEN),
            headers=owner_headers,
        )

    assert response.status_code == 422
    assert_no_secret_in(response.text, "the HTTP error response")
    assert_no_secret_in(capture.all_text(), "the router/service logs")

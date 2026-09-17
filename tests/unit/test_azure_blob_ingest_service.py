"""Unit tests for the idempotency diff and path-safety logic in
services.azure_blob_ingest_service.

classify_blobs is pure and DB-free (see its docstring): no database, no Azure
SDK, and no threads are touched by anything in this module. Existing
``Resource`` rows are represented as lightweight fixtures exposing only
``.resource_id``/``.extra_metadata`` — everything classify_blobs actually reads.

``_sanitize_blob_filename``/``_resolve_dest_path`` are likewise pure string/path
logic — no filesystem access needed to exercise the path-containment guard.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.azure_blob_ingest_service as azure_blob_ingest_service
from services.azure_blob_ingest_service import (
    PendingBlob,
    _process_batch,
    _resolve_dest_path,
    _sanitize_blob_filename,
    classify_blobs,
)
from services.blob.azure_blob_client import BlobSourceConfig, RemoteBlob
from services.resource_service import ResourceService

_SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}
_CONTAINER = "etiquetas"
_ACCOUNT_URL = "https://acct.blob.core.windows.net"
_OTHER_ACCOUNT_URL = "https://other-acct.blob.core.windows.net"


def _resource(resource_id: int, **extra_metadata) -> SimpleNamespace:
    return SimpleNamespace(resource_id=resource_id, extra_metadata=extra_metadata or None)


def _azure_resource(
    resource_id: int, blob_name: str, etag: str, container: str = _CONTAINER, account_url: str = _ACCOUNT_URL,
) -> SimpleNamespace:
    return _resource(
        resource_id,
        source_type='azure_blob', account_url=account_url, container=container, blob_name=blob_name, etag=etag,
    )


def _blob(name: str, etag: str = '"abc"') -> RemoteBlob:
    return RemoteBlob(
        name=name,
        url=f"{_ACCOUNT_URL}/{_CONTAINER}/{name}",
        size_bytes=100,
        content_type="application/pdf",
        etag=etag,
        last_modified=datetime(2026, 1, 1),
    )


def _classify(resources, remote, supported=_SUPPORTED, account_url=_ACCOUNT_URL, container=_CONTAINER):
    return classify_blobs(resources, account_url, container, remote, supported)


class TestClassifyBlobsUnchanged:
    def test_same_etag_is_unchanged_and_not_queued(self):
        resources = [_azure_resource(1, "docs/a.pdf", '"etag-1"')]
        remote = [_blob("docs/a.pdf", etag='"etag-1"')]

        result = _classify(resources, remote)

        assert result.to_ingest == []
        assert result.unchanged_count == 1
        assert result.unsupported_count == 0

    def test_unchanged_is_scoped_to_the_same_container(self):
        """A Resource with a matching blob_name/etag under a different container
        must not suppress re-ingesting the same-named blob from this run's container."""
        resources = [_azure_resource(1, "docs/a.pdf", '"etag-1"', container="other-container")]
        remote = [_blob("docs/a.pdf", etag='"etag-1"')]

        result = _classify(resources, remote)

        assert [p.kind for p in result.to_ingest] == ['new']
        assert result.unchanged_count == 0


class TestClassifyBlobsChanged:
    def test_different_etag_is_changed_and_carries_resource_id(self):
        resources = [_azure_resource(42, "docs/a.pdf", '"old-etag"')]
        remote = [_blob("docs/a.pdf", etag='"new-etag"')]

        result = _classify(resources, remote)

        assert len(result.to_ingest) == 1
        pending = result.to_ingest[0]
        assert pending.kind == 'changed'
        assert pending.resource_id == 42
        assert result.unchanged_count == 0

    def test_missing_remote_etag_is_always_changed_never_a_match(self):
        """Edge case (spec.md): a blob with no ETag is treated as changed in the
        worst case (re-ingest), never silently skipped as unchanged."""
        resources = [_azure_resource(7, "docs/a.pdf", '"some-etag"')]
        remote = [_blob("docs/a.pdf", etag=None)]

        result = _classify(resources, remote)

        assert [p.kind for p in result.to_ingest] == ['changed']
        assert result.unchanged_count == 0


class TestClassifyBlobsNew:
    def test_no_matching_resource_is_new(self):
        result = _classify([], [_blob("docs/a.pdf")])

        assert len(result.to_ingest) == 1
        assert result.to_ingest[0].kind == 'new'
        assert result.to_ingest[0].resource_id is None

    def test_non_azure_blob_resources_are_ignored_by_the_index(self):
        """A Resource from manual upload / CSV import (no source_type=azure_blob)
        must never be mistaken for a prior run of this same blob."""
        resources = [_resource(1, name="unrelated"), _resource(2)]
        remote = [_blob("docs/a.pdf")]

        result = _classify(resources, remote)

        assert [p.kind for p in result.to_ingest] == ['new']

    def test_two_blobs_same_basename_different_prefixes_are_distinguishable(self):
        """Edge case (spec.md): classified by full blob path, not basename."""
        resources = [_azure_resource(1, "a/report.pdf", '"etag-a"')]
        remote = [_blob("a/report.pdf", etag='"etag-a"'), _blob("b/report.pdf", etag='"etag-b"')]

        result = _classify(resources, remote)

        assert result.unchanged_count == 1
        assert [p.blob.name for p in result.to_ingest] == ["b/report.pdf"]


class TestClassifyBlobsAccountUrl:
    """The idempotency key is (account_url, container, blob_name), not just
    (container, blob_name) — a second storage account with a same-named
    container must never be mistaken for a prior run of the first account's
    blobs (would otherwise supersede-and-delete the wrong Resources)."""

    def test_same_container_different_account_is_new_not_unchanged(self):
        resources = [_azure_resource(1, "docs/a.pdf", '"etag-1"', account_url=_OTHER_ACCOUNT_URL)]
        remote = [_blob("docs/a.pdf", etag='"etag-1"')]

        result = _classify(resources, remote, account_url=_ACCOUNT_URL)

        assert [p.kind for p in result.to_ingest] == ['new']
        assert result.unchanged_count == 0

    def test_same_container_different_account_is_new_not_changed(self):
        """Also must not be mis-detected as 'changed' — that would carry the
        other account's resource_id and delete a Resource that belongs to a
        completely different storage account (item G / corpus poisoning)."""
        resources = [_azure_resource(1, "docs/a.pdf", '"stale-etag"', account_url=_OTHER_ACCOUNT_URL)]
        remote = [_blob("docs/a.pdf", etag='"etag-1"')]

        result = _classify(resources, remote, account_url=_ACCOUNT_URL)

        assert len(result.to_ingest) == 1
        assert result.to_ingest[0].kind == 'new'
        assert result.to_ingest[0].resource_id is None

    def test_same_account_and_container_is_still_matched(self):
        resources = [_azure_resource(1, "docs/a.pdf", '"etag-1"', account_url=_ACCOUNT_URL)]
        remote = [_blob("docs/a.pdf", etag='"etag-1"')]

        result = _classify(resources, remote, account_url=_ACCOUNT_URL)

        assert result.to_ingest == []
        assert result.unchanged_count == 1


class TestClassifyBlobsUnsupported:
    def test_extension_outside_supported_set_is_unsupported_not_queued(self):
        result = _classify([], [_blob("docs/a.exe")])

        assert result.to_ingest == []
        assert result.unsupported_count == 1
        assert result.unchanged_count == 0

    def test_empty_supported_set_disables_the_filter(self):
        """supported_extensions is only ever empty in a defensive/edge-case call —
        classify_blobs must not reject everything when that happens."""
        result = _classify([], [_blob("docs/a.exe")], supported=set())

        assert len(result.to_ingest) == 1
        assert result.unsupported_count == 0


class TestClassifyBlobsOrderingAndMix:
    def test_preserves_listing_order_across_new_and_changed(self):
        resources = [_azure_resource(1, "b.pdf", '"old"')]
        remote = [_blob("a.pdf"), _blob("b.pdf", etag='"new"'), _blob("c.pdf")]

        result = _classify(resources, remote)

        assert [p.blob.name for p in result.to_ingest] == ["a.pdf", "b.pdf", "c.pdf"]
        assert [p.kind for p in result.to_ingest] == ["new", "changed", "new"]

    def test_pending_blob_is_a_frozen_value_object(self):
        pending = PendingBlob(kind='new', blob=_blob("a.pdf"), resource_id=None)
        with pytest.raises(Exception):
            pending.kind = 'changed'


class TestSanitizeBlobFilename:
    """Table-driven coverage of the path-containment guard (security-critical,
    previously untested): every blob name here is attacker-influenced, coming
    from whatever storage account the caller points the endpoint at."""

    @pytest.mark.parametrize("blob_name", [
        "../x.pdf",
        "../../etc/passwd.pdf",
        "docs/../../x.pdf",
        "a/./../b.pdf",
        "/etc/x.pdf",
        "/x.pdf",
        "",
        "a/\x00b.pdf",
    ])
    def test_rejects_unsafe_blob_names(self, blob_name):
        with pytest.raises(ValueError):
            _sanitize_blob_filename(blob_name)

    def test_flattens_nested_path_with_double_underscore(self):
        result = _sanitize_blob_filename("a/b.pdf")
        assert result.startswith("a__b_")
        assert result.endswith(".pdf")

    def test_flattens_deeply_nested_path(self):
        result = _sanitize_blob_filename("a/b/c/d.pdf")
        assert result.startswith("a__b__c__d_")
        assert result.endswith(".pdf")

    def test_strips_leading_and_trailing_slashes_before_flattening(self):
        result = _sanitize_blob_filename("a/b.pdf/")
        assert result.startswith("a__b_")
        assert result.endswith(".pdf")

    def test_plain_filename_keeps_its_extension_with_a_hash_suffix(self):
        result = _sanitize_blob_filename("report.pdf")
        assert result.startswith("report_")
        assert result.endswith(".pdf")

    def test_overlong_basename_is_truncated_with_a_disambiguating_hash(self):
        long_name = ("x" * 300) + ".pdf"
        result = _sanitize_blob_filename(long_name)

        assert result.endswith(".pdf")
        assert len(result) < len(long_name)
        # Truncating two different overlong names must not collide.
        other = _sanitize_blob_filename(("y" * 300) + ".pdf")
        assert result != other

    def test_same_blob_name_always_maps_to_the_same_filename(self):
        """Stability is required: a re-ingested (unchanged or 'changed') blob
        must keep landing on the same local slot across runs."""
        assert _sanitize_blob_filename("docs/report.pdf") == _sanitize_blob_filename("docs/report.pdf")

    def test_different_blob_names_that_flatten_identically_do_not_collide(self):
        """Regression for the round-2/round-3 HIGH: 'a/b.pdf' flattens to the
        same string as the literal blob name 'a__b.pdf' — without an
        unconditional hash suffix these would alias onto the same local file,
        corrupting concurrent downloads and cross-blob Resources."""
        assert _sanitize_blob_filename("a/b.pdf") != _sanitize_blob_filename("a__b.pdf")


class TestResolveDestPath:
    """Defense in depth on top of _sanitize_blob_filename: the resolved path
    must never land outside staging_dir even if a sanitized filename somehow
    still carried something dangerous."""

    def test_positive_containment_for_a_flattened_filename(self, tmp_path):
        staging_dir = str(tmp_path)
        dest = _resolve_dest_path(staging_dir, "a__b.pdf")

        assert dest == str(tmp_path / "a__b.pdf")

    def test_rejects_a_path_that_escapes_staging_dir(self, tmp_path):
        staging_dir = str(tmp_path)
        with pytest.raises(ValueError):
            _resolve_dest_path(staging_dir, "../escaped.pdf")

    def test_rejects_an_absolute_path_outside_staging_dir(self, tmp_path):
        staging_dir = str(tmp_path)
        with pytest.raises(ValueError):
            _resolve_dest_path(staging_dir, "/etc/passwd")


def _cfg() -> BlobSourceConfig:
    return BlobSourceConfig(
        account_url=_ACCOUNT_URL, container=_CONTAINER, prefix=None, auth_mode='ANONYMOUS', sas_token=None,
    )


def _fake_resource(resource_id: int, uri: str) -> SimpleNamespace:
    return SimpleNamespace(resource_id=resource_id, uri=uri)


class TestProcessBatchSupersede:
    """``_process_batch``'s supersede-and-delete step for 'changed' blobs.

    ``_download_batch`` and ``ResourceService.create_multiple_resources`` are
    stubbed out (they need Azure/DB/filesystem) so only the supersede
    decision itself — which of round 2's HIGH bugs it must avoid — is under
    test. ``get_app_config`` is stubbed to stage under ``tmp_path`` so
    ``_process_batch``'s own ``os.makedirs``/``shutil.rmtree`` still run for
    real against a throwaway directory.
    """

    def _run(self, monkeypatch, tmp_path, pending: PendingBlob, created_resource: SimpleNamespace):
        filename = "staged.pdf"

        def fake_download_batch(cfg, pendings, staging_dir, max_bytes, concurrency):
            return [(pendings[0], f"{staging_dir}/{filename}", filename, None)]

        monkeypatch.setattr(azure_blob_ingest_service, 'get_app_config', lambda: {'TMP_BASE_FOLDER': str(tmp_path)})
        monkeypatch.setattr(azure_blob_ingest_service, '_download_batch', fake_download_batch)
        monkeypatch.setattr(
            ResourceService, 'create_multiple_resources',
            MagicMock(return_value=([created_resource], [], 'session-1')),
        )
        delete_mock = MagicMock(return_value=True)
        monkeypatch.setattr(ResourceService, 'delete_resource', delete_mock)

        outcome = _process_batch([pending], MagicMock(), repository_id=1, cfg=_cfg(), max_bytes=None, concurrency=1)
        return outcome, delete_mock

    def test_supersedes_the_old_resource_without_deleting_the_new_files_backing_file(self, monkeypatch, tmp_path):
        """Item 1: the new Resource (id 99) landed on the same ``uri`` the old
        Resource (id 42) used. Only the old DB row/vectors must be removed —
        never the file, since it now belongs to the new Resource."""
        pending = PendingBlob(kind='changed', blob=_blob("docs/a.pdf"), resource_id=42)
        created = _fake_resource(resource_id=99, uri="staged.pdf")

        outcome, delete_mock = self._run(monkeypatch, tmp_path, pending, created)

        delete_mock.assert_called_once()
        assert delete_mock.call_args.args[0] == 42
        assert delete_mock.call_args.kwargs == {'delete_file': False}
        assert outcome.failed_entries == []

    def test_skips_supersede_when_the_created_resource_is_the_same_row(self, monkeypatch, tmp_path):
        """Item 2: ``_process_single_file`` resurrected the pending Resource's
        own (uri, repository_id, folder_id) row from status='error' instead of
        inserting a new one — same resource_id on both sides. Deleting it
        would delete the row/file that is being indexed right now."""
        pending = PendingBlob(kind='changed', blob=_blob("docs/a.pdf"), resource_id=42)
        created = _fake_resource(resource_id=42, uri="staged.pdf")

        outcome, delete_mock = self._run(monkeypatch, tmp_path, pending, created)

        delete_mock.assert_not_called()
        assert outcome.failed_entries == []

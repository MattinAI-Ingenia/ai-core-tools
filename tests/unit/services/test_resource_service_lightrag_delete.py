"""Unit tests for the LightRAG single-file delete flow in ``ResourceService``."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from services.resource_service import (
    LIGHTRAG_DELETE_FAILED_DETAIL,
    LIGHTRAG_DELETE_PARTIAL_DETAIL,
    LIGHTRAG_SILO_BUSY_DETAIL,
    ResourceService,
)
from utils.service_lifecycle import ServiceRetiredError


def _resource(db_type="LIGHTRAG"):
    silo = SimpleNamespace(silo_id=7, vector_db_type=db_type)
    repo = SimpleNamespace(silo=silo, silo_id=7)
    return SimpleNamespace(resource_id=3, repository=repo)


def _run(resource, graph, lock="conn", delete_ok=True, row_exists=False):
    """Run the public delete with mocks; return (result_or_exc, mocks)."""
    with patch.object(ResourceService, "get_repository_resource", return_value=resource), \
            patch("services.silo_indexing_lock.acquire", return_value=lock) as acquire, \
            patch("services.silo_indexing_lock.release") as release, \
            patch.object(ResourceService, "delete_resource", return_value=delete_ok) as delete, \
            patch("services.resource_service.ResourceRepository.get_by_id",
                  return_value=object() if row_exists else None), \
            patch("services.resource_service.SiloService.delete_lightrag_resource_document", **graph) as g:
        try:
            out = ResourceService.delete_resource_from_repository(1, 2, 3, MagicMock())
        except Exception as exc:  # noqa: BLE001
            out = exc
        return out, acquire, release, delete, g


@pytest.mark.parametrize("outcome", ["success", "not_found"])
def test_success_and_not_found_delete_file_and_row(outcome):
    res = _resource()
    out, acquire, release, delete, g = _run(res, {"return_value": outcome})
    assert out == {"message": "Resource deleted successfully"}
    acquire.assert_called_once_with(7)
    g.assert_called_once_with(res)
    assert delete.call_args.kwargs == {"delete_from_silo": False}
    release.assert_called_once_with("conn", 7)


def test_not_allowed_is_409_and_nothing_deleted():
    out, _, release, delete, _ = _run(_resource(), {"return_value": "not_allowed"})
    assert isinstance(out, HTTPException) and out.status_code == 409
    assert out.detail == LIGHTRAG_SILO_BUSY_DETAIL
    delete.assert_not_called()
    release.assert_called_once()


@pytest.mark.parametrize("graph", [{"return_value": "fail"}, {"side_effect": TimeoutError("secret")}])
def test_fail_or_exception_is_500_and_nothing_deleted(graph):
    out, _, release, delete, _ = _run(_resource(), graph)
    assert isinstance(out, HTTPException) and out.status_code == 500
    assert out.detail == LIGHTRAG_DELETE_FAILED_DETAIL
    delete.assert_not_called()
    release.assert_called_once()


def test_lock_held_is_409_without_graph_step():
    out, _, release, delete, g = _run(_resource(), {"return_value": "success"}, lock=None)
    assert isinstance(out, HTTPException) and out.status_code == 409
    assert out.detail == LIGHTRAG_SILO_BUSY_DETAIL
    g.assert_not_called()
    delete.assert_not_called()
    release.assert_not_called()


def test_retired_service_is_reraised_and_lock_released():
    exc = ServiceRetiredError(SimpleNamespace(service_id=1, name="svc", end_of_service_at=datetime(2020, 1, 1)), "embedding")
    out, _, release, delete, _ = _run(_resource(), {"side_effect": exc})
    assert out is exc and out.status_code == 409 and out.error_code == "SERVICE_RETIRED"
    delete.assert_not_called()
    release.assert_called_once()


def test_file_row_failure_maps_to_404_and_releases_lock():
    out, _, release, _, _ = _run(_resource(), {"return_value": "success"}, delete_ok=False)
    assert isinstance(out, HTTPException) and out.status_code == 404
    release.assert_called_once()


def test_file_row_failure_with_row_still_present_is_500_partial():
    out, _, release, _, _ = _run(_resource(), {"return_value": "success"}, delete_ok=False, row_exists=True)
    assert isinstance(out, HTTPException) and out.status_code == 500
    assert out.detail == LIGHTRAG_DELETE_PARTIAL_DETAIL
    release.assert_called_once_with("conn", 7)


def test_non_lightrag_path_unchanged():
    out, acquire, _, delete, g = _run(_resource("PGVECTOR"), {"return_value": "success"})
    assert out == {"message": "Resource deleted successfully"}
    acquire.assert_not_called()
    g.assert_not_called()
    assert delete.call_args.kwargs == {}  # default delete_from_silo=True


def test_delete_resource_from_silo_flag():
    res = SimpleNamespace(resource_id=3, repository_id=2, uri="a.txt")
    for flag, calls in ((True, 1), (False, 0)):
        with patch("services.resource_service.ResourceRepository") as repo, \
                patch("services.resource_service.SiloService.delete_resource") as silo_del:
            repo.get_by_id.return_value = res
            assert ResourceService.delete_resource(3, MagicMock(), delete_file=False, delete_from_silo=flag)
            assert silo_del.call_count == calls

"""Unit tests for ``SiloService.delete_lightrag_resource_document``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.silo_service import SiloService

CACHE = "services.metadata_values_cache_service.MetadataValuesCacheService.invalidate"


def _resource(embedding=object()):
    silo = SimpleNamespace(silo_id=7, embedding_service=embedding)
    return SimpleNamespace(resource_id=3, repository=SimpleNamespace(silo=silo))


def _call(outcome=None, store_exc=None, cache_exc=None, resource=None):
    store = MagicMock()
    store.delete_resource_document.return_value = outcome
    store.delete_resource_document.side_effect = store_exc
    with patch("services.silo_service._get_vector_store", return_value=store), \
            patch(CACHE, side_effect=cache_exc) as inv:
        res = SiloService.delete_lightrag_resource_document(resource or _resource())
    return res, store, inv


def test_no_embedding_service_raises():
    with pytest.raises(ValueError):
        _call("success", resource=_resource(embedding=None))


@pytest.mark.parametrize("outcome", ["success", "not_found"])
def test_success_and_not_found_invalidate_cache(outcome):
    res, store, inv = _call(outcome)
    assert res == outcome
    store.delete_resource_document.assert_called_once_with("silo_7", 3)
    inv.assert_called_once_with(7)


@pytest.mark.parametrize("outcome", ["fail", "not_allowed"])
def test_fail_and_not_allowed_do_not_invalidate(outcome):
    res, _, inv = _call(outcome)
    assert res == outcome
    inv.assert_not_called()


def test_invalidate_error_is_swallowed():
    res, _, inv = _call("success", cache_exc=RuntimeError("boom"))
    assert res == "success"
    inv.assert_called_once_with(7)


def test_store_error_propagates():
    with pytest.raises(TimeoutError):
        _call(store_exc=TimeoutError("x"))

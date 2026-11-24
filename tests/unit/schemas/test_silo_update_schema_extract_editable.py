"""UpdateSiloSchema must carry the LightRAG extract/indexing service ids so a
change made in the silo form actually reaches the service layer (regression:
the schema dropped them silently, so a successful PUT reverted the change)."""

from unittest.mock import MagicMock, patch

from schemas.silo_schemas import UpdateSiloSchema
from services.silo_service import SiloService


def _forwarded(**overrides) -> dict:
    """Run an UpdateSiloSchema through create_or_update_silo_router and return the
    silo_data dict it forwards to create_or_update_silo."""
    data = UpdateSiloSchema(name="Existing Silo", **overrides)
    forwarded = {}

    def _capture(silo_data, db=None):
        forwarded.update(silo_data)

    with patch("services.silo_service.SiloService.create_or_update_silo", side_effect=_capture):
        SiloService.create_or_update_silo_router(10, 1, data, MagicMock())
    return forwarded


def test_router_forwards_extract_service_id():
    assert _forwarded(extract_service_id=42)["extract_service_id"] == 42


def test_router_forwards_indexing_service_id_alias():
    assert _forwarded(indexing_service_id=42)["indexing_service_id"] == 42


def test_omitted_fields_are_none():
    payload = _forwarded()
    assert payload["extract_service_id"] is None
    assert payload["indexing_service_id"] is None

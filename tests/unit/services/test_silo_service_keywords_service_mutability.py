"""
Unit tests: keywords_service_id must stay editable after silo creation,
while extract_service_id/vlm_service_id/indexing_service_id/lightrag_language
stay immutable.

keywords_service_id only feeds LightRAG's query-time keyword-extraction LLM —
it never touches already-indexed data, unlike extract/vlm which would mix
entities extracted with different models in the same graph. lightrag_language
is immutable too because it drives that same extraction prompt.
"""

from unittest.mock import MagicMock, patch

from services.silo_service import SiloService


def _make_silo() -> MagicMock:
    silo = MagicMock()
    silo.silo_id = 1
    silo.app_id = 10
    silo.vector_db_type = "LIGHTRAG"
    silo.silo_type = "CUSTOM"
    silo.metadata_definition_id = None
    silo.embedding_service_id = 5
    silo.name = "Existing Silo"
    silo.description = ""
    silo.status = None
    silo.fixed_metadata = False
    silo.extract_service_id = 100
    silo.keywords_service_id = 200
    silo.vlm_service_id = 300
    silo.indexing_service_id = 100
    silo.lightrag_language = "English"
    return silo


def _call(existing_silo: MagicMock, extra: dict):
    data = {
        "silo_id": 1,
        "app_id": 10,
        "name": "Existing Silo",
        "description": "",
        **extra,
    }
    mock_db = MagicMock()
    with patch("services.silo_service.SiloService.get_silo", return_value=existing_silo):
        return SiloService.create_or_update_silo(data, db=mock_db)


def test_keywords_service_id_is_updatable_after_creation():
    silo = _make_silo()
    result = _call(silo, {"keywords_service_id": 999})
    assert result.keywords_service_id == 999


def test_extract_service_id_stays_immutable_after_creation():
    silo = _make_silo()
    result = _call(silo, {"extract_service_id": 999})
    assert result.extract_service_id == 100


def test_vlm_service_id_stays_immutable_after_creation():
    silo = _make_silo()
    result = _call(silo, {"vlm_service_id": 999})
    assert result.vlm_service_id == 300


def test_lightrag_language_stays_immutable_after_creation():
    silo = _make_silo()
    result = _call(silo, {"lightrag_language": "Spanish"})
    assert result.lightrag_language == "English"


def test_lightrag_entity_extract_max_gleaning_stays_immutable_after_creation():
    silo = _make_silo()
    silo.lightrag_entity_extract_max_gleaning = 0
    result = _call(silo, {"lightrag_entity_extract_max_gleaning": 3})
    assert result.lightrag_entity_extract_max_gleaning == 0

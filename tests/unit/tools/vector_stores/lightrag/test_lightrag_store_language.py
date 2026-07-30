"""Unit tests for LightRAGStore's per-silo "Advanced settings" wiring
(language, entity-extraction gleaning, max source-ids per entity/relation).

Verifies ``_build_rag`` forwards each setting to the LightRAG instance when
configured, and falls back to LightRAG's/config's own default when not.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.vector_stores.lightrag_store import LightRAGStore


def _make_ai_service():
    return SimpleNamespace(provider="OpenAI", name="test-llm", description="gpt-4o", api_key="sk-test", endpoint=None)


def _make_embedding_service():
    return SimpleNamespace(
        provider="OpenAI", name="test-embed", description="text-embedding-3-small",
        api_key="sk-test", endpoint=None, api_version=None,
    )


def _build_store(lightrag_language=None, **advanced):
    return LightRAGStore(
        db=MagicMock(),
        ai_service=_make_ai_service(),
        embedding_service=_make_embedding_service(),
        lightrag_language=lightrag_language,
        **advanced,
    )


def _patched_build_rag(store, collection_name="silo_1"):
    """Call store._build_rag with every LightRAG dependency mocked out."""
    with patch("lightrag.LightRAG") as mock_lightrag_cls, \
         patch("tools.vector_stores.lightrag.adapters.build_embedding_func"), \
         patch("tools.vector_stores.lightrag.adapters.build_llm_model_func"), \
         patch("tools.vector_stores.lightrag.adapters.build_role_llm_configs"), \
         patch(
             "tools.vector_stores.lightrag.storage_config.build_storage_config",
             return_value={
                 "graph_storage": "Neo4JStorage",
                 "vector_storage": "QdrantVectorDBStorage",
                 "kv_storage": "PGKVStorage",
                 "doc_status_storage": "PGDocStatusStorage",
             },
         ):
        rag = store._build_rag(collection_name)
        return rag, mock_lightrag_cls


def test_language_is_applied_to_addon_params_when_configured():
    # English (not Spanish) so this test stays isolated from the
    # Spanish-default-entity_types side effect covered separately below.
    store = _build_store(lightrag_language="English")
    rag, _ = _patched_build_rag(store)
    rag.addon_params.__setitem__.assert_called_once_with("language", "English")


def test_addon_params_untouched_when_language_not_configured():
    store = _build_store(lightrag_language=None)
    rag, _ = _patched_build_rag(store)
    rag.addon_params.__setitem__.assert_not_called()


def test_advanced_settings_forwarded_to_lightrag_constructor_when_configured():
    store = _build_store(
        lightrag_entity_extract_max_gleaning=3,
        lightrag_max_source_ids_per_entity=42,
        lightrag_max_source_ids_per_relation=7,
    )
    _, mock_lightrag_cls = _patched_build_rag(store)
    call_kwargs = mock_lightrag_cls.call_args.kwargs
    assert call_kwargs["entity_extract_max_gleaning"] == 3
    assert call_kwargs["max_source_ids_per_entity"] == 42
    assert call_kwargs["max_source_ids_per_relation"] == 7


def test_entity_extraction_use_json_is_always_forced_on(monkeypatch):
    """Delimited-text mode causes silent data loss (see
    docs/testing/lightrag_extraction_benchmark_corpus.md) — this must stay on
    regardless of the ENTITY_EXTRACTION_USE_JSON env var LightRAG itself reads.
    """
    monkeypatch.delenv("ENTITY_EXTRACTION_USE_JSON", raising=False)
    store = _build_store()
    _, mock_lightrag_cls = _patched_build_rag(store)
    assert mock_lightrag_cls.call_args.kwargs["entity_extraction_use_json"] is True


def test_advanced_settings_fall_back_to_global_default_when_not_configured():
    import config
    store = _build_store()
    _, mock_lightrag_cls = _patched_build_rag(store)
    call_kwargs = mock_lightrag_cls.call_args.kwargs
    assert call_kwargs["entity_extract_max_gleaning"] == config.ENTITY_EXTRACT_MAX_GLEANING
    assert "max_source_ids_per_entity" not in call_kwargs
    assert "max_source_ids_per_relation" not in call_kwargs


def test_entity_types_blank_and_english_does_not_override_addon_params():
    store = _build_store(lightrag_language="English")
    rag, _ = _patched_build_rag(store)
    for call in rag.addon_params.__setitem__.call_args_list:
        assert call.args[0] != "entity_types_guidance"


def test_entity_types_blank_and_spanish_uses_spanish_default_guidance():
    store = _build_store(lightrag_language="Spanish")
    rag, _ = _patched_build_rag(store)
    rag.addon_params.__setitem__.assert_any_call(
        "entity_types_guidance",
        "Classify each entity using one of the following types. "
        "If no type fits, use `Other`.\n\n"
        "- Persona\n- Criatura\n- Organización\n- Lugar\n- Evento\n"
        "- Concepto\n- Método\n- Contenido\n- Datos\n- Artefacto\n- ObjetoNatural",
    )


def test_entity_types_explicit_text_is_parsed_and_used_regardless_of_language():
    store = _build_store(
        lightrag_language="English",
        lightrag_entity_types="Person, Organization,, Person , Location ",
    )
    rag, _ = _patched_build_rag(store)
    rag.addon_params.__setitem__.assert_any_call(
        "entity_types_guidance",
        "Classify each entity using one of the following types. "
        "If no type fits, use `Other`.\n\n"
        "- Person\n- Organization\n- Location",
    )

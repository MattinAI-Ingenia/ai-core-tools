"""Unit tests for SiloService.get_silo_retriever search-kwargs merge.

After collapsing the agent ``retrieval_config`` JSON column into flat ``rag_*``
columns, ``get_silo_retriever`` merges only two layers: system defaults and the
per-call ``search_params`` (which is where ``lightrag_query_mode`` now travels).
"""


def _merged(search_params=None):
    """Replicate the 2-layer merge from SiloService.get_silo_retriever()."""
    known_params = {"k", "filter", "score_threshold", "fetch_k", "lambda_mult", "search_type", "lightrag_query_mode"}

    merged: dict = {"k": 30}  # Layer 1: defaults

    if search_params:  # Layer 2: per-call overrides
        filter_fields = {}
        direct_params = {}
        for key, value in search_params.items():
            if key in known_params:
                direct_params[key] = value
            else:
                filter_fields[key] = value
        merged.update(direct_params)
        if filter_fields:
            merged.setdefault("filter", {}).update(filter_fields)

    return merged


def test_no_params_uses_defaults():
    assert _merged() == {"k": 30}


def test_search_params_override_defaults():
    result = _merged(search_params={"k": 5, "search_type": "mmr"})
    assert result["k"] == 5
    assert result["search_type"] == "mmr"


def test_search_params_filter_wrapped():
    result = _merged(search_params={"custom_field": "value"})
    assert result["filter"] == {"custom_field": "value"}


def test_search_params_known_and_unknown_mixed():
    result = _merged(search_params={"k": 7, "category": "docs"})
    assert result["k"] == 7
    assert result["filter"] == {"category": "docs"}


def test_lightrag_query_mode_passes_through():
    """The dynamic LightRAG tool injects the mode via search_params."""
    result = _merged(search_params={"lightrag_query_mode": "local"})
    assert result["lightrag_query_mode"] == "local"

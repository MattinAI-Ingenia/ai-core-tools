"""Pins the exhaustive-coverage query for cobertura/G07/G09-style questions.

No chunk_top_k here on purpose: these questions are membership ("does X
appear in doc D, yes or no"), not ranking — a document with one incidental
mention must count exactly the same as one with five relevant ones, which a
similarity top-k cannot guarantee.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.vector_stores.lightrag_store import LightRAGStore, _group_chunk_rows
# find_chunks_mentioning is a thin wrapper around this (it just runs the SQL
# and calls _group_chunk_rows) — exercised indirectly in Task 3's tool tests,
# which mock it at the LightRAGStore instance level.


def test_groups_snippets_by_resource_id():
    rows = [
        ("res271-p10", "CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí."),
        ("res271-p42", "CDOC004043.pdf p.42", "Revisar la válvula de seguridad cada año."),
        ("res275-p3", "CDOC001961.pdf p.3", "La válvula de seguridad debe purgarse."),
    ]
    grouped = _group_chunk_rows(rows)
    assert set(grouped.keys()) == {"271", "275"}
    assert len(grouped["271"]) == 2
    assert len(grouped["275"]) == 1
    # file_path and page travel with the snippet — Task 3's citations need the
    # real "file.pdf p.N" label, and the frontend's "Open PDF" button needs
    # the page number, not just the resource id.
    assert grouped["271"][0] == ("CDOC004043.pdf p.10", "La válvula de seguridad se instala aquí.", 10)


def test_caps_snippets_per_document():
    rows = [
        (f"res271-p{i}", f"CDOC004043.pdf p.{i}", f"mención {i}")
        for i in range(10)
    ]
    grouped = _group_chunk_rows(rows, per_doc_cap=3)
    assert len(grouped["271"]) == 3


def test_ignores_rows_with_unparseable_doc_id():
    """Non-PDF sources (crawled Domain pages) get a content-hash id, not
    res{N}-p{N} — see _lightrag_doc_id's fallback. Skip them rather than crash."""
    rows = [("doc-abc123hash", "https://example.com/page", "algo de contenido")]
    grouped = _group_chunk_rows(rows)
    assert grouped == {}


def test_empty_rows_returns_empty_dict():
    assert _group_chunk_rows([]) == {}


# ---------------------------------------------------------------------------
# find_chunks_mentioning's term=None "whole document" mode — added for
# G07-style questions ("list every parameter in manual Z"), where there is no
# single literal string that means "a parameter" to search for.
# ---------------------------------------------------------------------------


def _make_store():
    db = MagicMock()
    return LightRAGStore(
        db=db,
        ai_service=SimpleNamespace(provider="OpenAI", name="t", description="gpt-4o", api_key="k", endpoint=None),
        embedding_service=SimpleNamespace(
            provider="OpenAI", name="t", description="text-embedding-3-small",
            api_key="k", endpoint=None, api_version=None,
        ),
    )


def test_requires_term_or_doc_filter():
    store = _make_store()
    with pytest.raises(ValueError):
        store.find_chunks_mentioning("silo_1", term=None, doc_filter=None)


def test_term_none_omits_the_ilike_clause():
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = []
    store.find_chunks_mentioning("silo_1", term=None, doc_filter="271")
    sql_arg = conn.execute.call_args[0][0]
    assert "ILIKE" not in str(sql_arg)
    assert "full_doc_id LIKE" in str(sql_arg)


def test_term_none_does_not_cap_snippets_to_three():
    """The whole point of the doc-only mode is the WHOLE document — the
    default per_doc_cap=3 would silently defeat it."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = [
        (f"res271-p{i}", f"CDOC004043.pdf p.{i}", f"contenido {i}") for i in range(10)
    ]
    grouped, _ = store.find_chunks_mentioning("silo_1", term=None, doc_filter="271")
    assert len(grouped["271"]) == 10


def test_doc_filter_accepts_a_list_and_ors_the_conditions():
    """An ambiguous name (resolve_document_by_name) can resolve to several
    real documents — every one of them must be searched, not just the first."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = []
    store.find_chunks_mentioning("silo_1", term="P20", doc_filter=[280, 282, 296])
    sql_arg = str(conn.execute.call_args[0][0])
    params = conn.execute.call_args[0][1]
    assert "OR" in sql_arg
    assert params["doc_pattern_0"] == "res280-p%"
    assert params["doc_pattern_1"] == "res282-p%"
    assert params["doc_pattern_2"] == "res296-p%"


def test_term_matching_is_accent_insensitive():
    """"anodo" (no accent, what a user/LLM is likely to type) must still find
    "ánodo" in the indexed text — accent is a spelling difference, not a
    different word, and the graph already stores the correctly-accented form."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = []
    store.find_chunks_mentioning("silo_1", term="anodo", doc_filter=None)
    sql_arg = str(conn.execute.call_args[0][0])
    assert "unaccent(content)" in sql_arg
    assert "unaccent(:term_pattern_0)" in sql_arg


def test_terms_present_literally_keeps_only_real_matches():
    """A graph-entity name that isn't actually verbatim page text (e.g. an
    LLM-paraphrased label) must be dropped, not treated as a real spelling."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    # "Dual Clima R" is found (row returned), "Dual Climatización" is not (None).
    conn.execute.return_value.fetchone.side_effect = [(1,), None]
    result = store.terms_present_literally("silo_1", ["Dual Clima R", "Dual Climatización"])
    assert result == ["Dual Clima R"]


def test_terms_present_literally_empty_list_makes_no_query():
    store = _make_store()
    assert store.terms_present_literally("silo_1", []) == []
    store.db.engine.connect.assert_not_called()


def test_term_accepts_a_list_and_ors_the_conditions():
    """Several literal spellings of the same thing (e.g. the caller's term
    plus graph-entity variants like "Cenicero Compresor Automatico") must all
    be tried — a document matching any one of them counts as a match."""
    store = _make_store()
    conn = store.db.engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchall.return_value = []
    store.find_chunks_mentioning("silo_1", term=["cenicero", "Cenicero Compresor Automatico"])
    sql_arg = str(conn.execute.call_args[0][0])
    params = conn.execute.call_args[0][1]
    assert "OR" in sql_arg
    assert params["term_pattern_0"] == "%cenicero%"
    assert params["term_pattern_1"] == "%Cenicero Compresor Automatico%"

"""Unit tests for ``tools.vector_stores.lightrag_store.LightRAGStore``.

All LightRAG / Neo4j / Qdrant / PostgreSQL dependencies are mocked so these
tests run without network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from tools.vector_stores.lightrag_store import (
    LightRAGStore,
    LightRAGRetriever,
    CHUNK_ID_RESOURCE_PAGE_RE,
    _FULL_DOC_ID_RESOURCE_RE,
    _lightrag_doc_id,
    _normalize_lightrag_graph,
    _page_label,
)

# pytest.ini does not enable pytest-asyncio's auto mode, so mark every
# async test in this module explicitly.
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_service():
    return SimpleNamespace(
        provider="OpenAI",
        name="test-llm",
        description="gpt-4o",
        api_key="sk-test",
        endpoint=None,
    )


def _make_embedding_service():
    return SimpleNamespace(
        provider="OpenAI",
        name="test-embed",
        description="text-embedding-3-small",
        api_key="sk-test",
        endpoint=None,
        api_version=None,
    )


@pytest.fixture
def store():
    db = MagicMock()
    return LightRAGStore(
        db=db,
        ai_service=_make_ai_service(),
        embedding_service=_make_embedding_service(),
    )


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.ainsert = AsyncMock()
    rag.aquery = AsyncMock(return_value="LightRAG response")
    rag.adelete_by_doc_id = AsyncMock()
    rag.initialize_storages = AsyncMock()
    return rag


# ---------------------------------------------------------------------------
# index_documents
# ---------------------------------------------------------------------------


def test_index_documents_calls_ainsert(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [
        Document(page_content="Hello world"),
        Document(page_content="Second doc"),
    ]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    mock_insert.assert_called_once()
    texts = mock_insert.call_args[0][1]
    assert texts == ["Hello world", "Second doc"]


def test_index_documents_skips_empty(store):
    store._get_rag_instance = MagicMock()

    store.index_documents("silo_1", [])

    store._get_rag_instance.assert_not_called()


def test_index_documents_skips_blank_content(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    # page_content="" is falsy → filtered out by `if doc.page_content`
    docs = [
        Document(page_content=""),
    ]
    store.index_documents("silo_1", docs)

    mock_rag.ainsert.assert_not_called()


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


def test_delete_documents_is_noop(store, mock_rag):
    # delete_documents is a deliberate no-op (per-document graph deletion unsupported)
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    store.delete_documents("silo_1", ["doc1", "doc2"])

    mock_rag.adelete_by_doc_id.assert_not_called()


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


def test_delete_collection_removes_cached_instance(store, mock_rag):
    store._rag_instances["silo_1"] = mock_rag

    with patch.object(store, "_cleanup_neo4j"), \
         patch.object(store, "_cleanup_qdrant"), \
         patch.object(store, "_cleanup_postgres"):
        store.delete_collection("silo_1")

    assert "silo_1" not in store._rag_instances


# ---------------------------------------------------------------------------
# search_similar_documents
# ---------------------------------------------------------------------------


def test_search_similar_documents_calls_aquery(store, mock_rag):
    mock_rag.aquery.return_value = "Some response"
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    with patch("tools.vector_stores.lightrag_store.QueryParam", create=True) as MockQP:
        # Patch the lazy import of QueryParam inside the method
        with patch.dict("sys.modules", {"lightrag.base": MagicMock(QueryParam=MockQP)}):
            results = store.search_similar_documents("silo_1", "test query", k=5)

    assert len(results) == 1
    assert results[0].page_content == "Some response"
    assert results[0].metadata == {"source": "lightrag", "query_mode": "hybrid", "lightrag_keywords": {}}


def test_search_similar_documents_bypass_returns_empty(store):
    store._get_rag_instance = MagicMock()

    with patch.dict("sys.modules", {"lightrag.base": MagicMock()}):
        results = store.search_similar_documents(
            "silo_1", "test query", search_type="bypass"
        )

    assert results == []
    store._get_rag_instance.assert_not_called()


# ---------------------------------------------------------------------------
# get_retriever
# ---------------------------------------------------------------------------


def test_get_retriever_returns_lightrag_retriever(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    retriever = store.get_retriever("silo_1")

    assert isinstance(retriever, LightRAGRetriever)
    assert retriever.query_mode == "hybrid"


def test_get_retriever_with_custom_query_mode(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    retriever = store.get_retriever(
        "silo_1", search_params={"lightrag_query_mode": "local"}
    )

    assert isinstance(retriever, LightRAGRetriever)
    assert retriever.query_mode == "local"


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------


def test_count_documents_queries_postgres(store):
    mock_result = MagicMock()
    mock_result.scalar.return_value = 42
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    store.db.engine.begin.return_value.__enter__.return_value = mock_conn

    count = store.count_documents("silo_1")

    assert count == 42
    mock_conn.execute.assert_called_once()


# ---------------------------------------------------------------------------
# update_documents_metadata
# ---------------------------------------------------------------------------


def test_update_documents_metadata_returns_zero(store):
    result = store.update_documents_metadata(
        "silo_1",
        filter_metadata={"source": "test"},
        metadata_updates={"tag": "new"},
    )

    assert result == 0


# ---------------------------------------------------------------------------
# get_distinct_metadata_values
# ---------------------------------------------------------------------------


def test_get_distinct_metadata_values_returns_empty(store):
    result = store.get_distinct_metadata_values("silo_1", "source")

    assert result == []


# ---------------------------------------------------------------------------
# _lightrag_doc_id / chunk-id → (resource_id, page) round trip
#
# The "open source PDF at this page" feature needs every chunk's own id to be
# parseable back to which resource and page it came from — see
# CHUNK_ID_RESOURCE_PAGE_RE. LightRAG derives a chunk's id from the document
# id we supply (f"{doc_id}-chunk-{n}"), so this is set at index time.
# ---------------------------------------------------------------------------


def test_lightrag_doc_id_encodes_resource_and_page():
    assert _lightrag_doc_id({"resource_id": 61, "page": 5}, "some text") == "res61-p5"


def test_lightrag_doc_id_falls_back_to_content_hash_without_resource_or_page():
    # Domain/web-crawled content has neither — must still get a valid,
    # LightRAG-shaped id instead of erroring.
    doc_id = _lightrag_doc_id({}, "crawled page text")
    assert doc_id.startswith("doc-")
    assert not doc_id.startswith("res")


def test_index_documents_passes_resource_and_page_ids_to_ainsert(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [
        Document(page_content="page one", metadata={"resource_id": 61, "page": 1}),
        Document(page_content="page two", metadata={"resource_id": 61, "page": 2}),
    ]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    assert mock_insert.call_args.kwargs["ids"] == ["res61-p1", "res61-p2"]


# ---------------------------------------------------------------------------
# _normalize_lightrag_graph: chunk resource_id/page extraction
# ---------------------------------------------------------------------------


def test_normalize_extracts_resource_and_page_from_chunk_id():
    raw = {"data": {"chunks": [{"chunk_id": "res61-p5-chunk-000", "content": "text", "file_path": "f.pdf (p. 5)"}]}}

    result = _normalize_lightrag_graph(raw)

    chunk = result["data"]["chunks"][0]
    assert chunk["resource_id"] == 61
    assert chunk["page"] == 5


def test_normalize_omits_resource_and_page_for_content_hash_ids():
    # Content indexed before this feature, or from a source with no page
    # metadata — must not error, and must not fabricate a resource/page.
    raw = {"data": {"chunks": [{"chunk_id": "doc-abc123-chunk-000", "content": "text", "file_path": "f.pdf"}]}}

    result = _normalize_lightrag_graph(raw)

    chunk = result["data"]["chunks"][0]
    assert "resource_id" not in chunk
    assert "page" not in chunk


# ---------------------------------------------------------------------------
# CHUNK_ID_RESOURCE_PAGE_RE / _FULL_DOC_ID_RESOURCE_RE: accept both the old
# per-page PDF id shape and the new per-file shape (step_002 of
# lightrag-pdf-document-schema — no writer changes yet, readers only).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_id,expected", [
    ("res7-chunk-000", ("7", None)),            # new DOCX/TXT/MD shape
    ("res7-p12-chunk-037", ("7", "12")),         # new PDF-chunk shape
    ("res7-p12-chunk-037-s01", ("7", "12")),     # new PDF shape, hard-split child
    ("res7-p12-chunk-000", ("7", "12")),         # old per-page PDF shape
])
def test_chunk_id_resource_page_re_matches_both_shapes(chunk_id, expected):
    match = CHUNK_ID_RESOURCE_PAGE_RE.match(chunk_id)
    assert match is not None
    assert (match.group(1), match.group(2)) == expected


def test_chunk_id_resource_page_re_does_not_match_non_resource_ids():
    assert CHUNK_ID_RESOURCE_PAGE_RE.match("doc-abc-chunk-000") is None


def test_chunk_id_resource_page_re_requires_a_digit_chunk_index():
    # "-chunk-" must be followed by digits — a malformed/non-digit index
    # (never produced by _lightrag_doc_id, but not something to silently
    # accept either) must not match.
    assert CHUNK_ID_RESOURCE_PAGE_RE.match("res7-chunk-abc") is None


@pytest.mark.parametrize("doc_id", ["res7", "res7-p12"])
def test_full_doc_id_resource_re_matches_both_shapes(doc_id):
    assert _FULL_DOC_ID_RESOURCE_RE.match(doc_id) is not None


@pytest.mark.parametrize("doc_id", ["doc-abc", "res7-p12-chunk-000"])
def test_full_doc_id_resource_re_rejects_non_doc_ids(doc_id):
    assert _FULL_DOC_ID_RESOURCE_RE.match(doc_id) is None


@pytest.mark.parametrize("file_path", ["", "unknown_source"])
def test_page_label_does_not_attach_a_page_to_a_sourceless_file_path(file_path):
    # No real file name to attach a page to — must stay exactly "" or
    # "unknown_source" so the frontend's "Unknown source" fallback (an
    # exact/empty match, see CitationBadge.tsx / LightRAGGraphBubble.tsx)
    # still fires, instead of showing a bare "(p. 12)" or "unknown_source (p. 12)".
    assert _page_label(file_path, 12) == file_path


# ---------------------------------------------------------------------------
# _normalize_lightrag_graph: new-shape chunk ids
# ---------------------------------------------------------------------------


def test_normalize_appends_page_to_a_bare_file_name_for_new_shape_pdf_chunk():
    raw = {"data": {"chunks": [{"chunk_id": "res7-p12-chunk-037", "content": "text", "file_path": "m.pdf"}]}}

    chunk = _normalize_lightrag_graph(raw)["data"]["chunks"][0]

    assert chunk["resource_id"] == 7
    assert chunk["page"] == 12
    assert chunk["file_path"] == "m.pdf (p. 12)"


def test_normalize_new_docx_shape_has_no_page_and_leaves_file_path_bare():
    raw = {"data": {"chunks": [{"chunk_id": "res8-chunk-000", "content": "text", "file_path": "d.docx"}]}}

    chunk = _normalize_lightrag_graph(raw)["data"]["chunks"][0]

    assert chunk["resource_id"] == 8
    assert "page" not in chunk
    assert chunk["file_path"] == "d.docx"


def test_normalize_old_shape_label_already_has_page_suffix_is_not_doubled():
    raw = {"data": {"chunks": [{"chunk_id": "res61-p5-chunk-000", "content": "text", "file_path": "f.pdf (p. 5)"}]}}

    chunk = _normalize_lightrag_graph(raw)["data"]["chunks"][0]

    assert chunk["resource_id"] == 61
    assert chunk["page"] == 5
    assert chunk["file_path"] == "f.pdf (p. 5)"

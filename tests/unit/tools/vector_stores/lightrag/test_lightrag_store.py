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
    _source_label_from_metadata,
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


def test_index_documents_domain_crawl_pages_still_get_content_hash_ids(store, mock_rag):
    # AC-35 (NFR-1): Domain/URL crawl Documents (no resource_id, no .id set)
    # are completely untouched by this spec — they must keep getting
    # doc-<hash> ids, exactly like before the whole-file PDF refactor.
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [
        Document(page_content="crawled page one", metadata={"url": "https://example.com/a", "domain_id": 1}),
        Document(page_content="crawled page two", metadata={"url": "https://example.com/b", "domain_id": 1}),
    ]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    ids = mock_insert.call_args.kwargs["ids"]
    assert len(ids) == 2
    assert all(doc_id.startswith("doc-") for doc_id in ids)
    assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


def test_delete_documents_is_noop(store, mock_rag, caplog):
    # AC-9: delete_documents never touches the graph, whatever the filter
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    ids_list = [
        {"resource_id": {"$eq": 42}},
        ["doc1", "doc2"],
        {"url": {"$eq": "https://x"}},
        {"media_id": {"$eq": 3}},
    ]
    for ids in ids_list:
        caplog.clear()
        with caplog.at_level("WARNING"):
            store.delete_documents("silo_1", ids)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    mock_rag.adelete_by_doc_id.assert_not_called()


@pytest.mark.parametrize("status", ["success", "not_found", "not_allowed", "fail"])
@pytest.mark.parametrize("llm_cache", [True, False])
def test_delete_resource_document_passes_through(
    store, mock_rag, monkeypatch, caplog, status, llm_cache
):
    import asyncio

    import config

    monkeypatch.setattr(config, "LIGHTRAG_DELETE_LLM_CACHE_ON_DOC_DELETE", llm_cache)
    store._get_rag_instance = MagicMock(return_value=mock_rag)
    mock_rag.adelete_by_doc_id = AsyncMock(
        return_value=SimpleNamespace(status=status, status_code=418, message="SECRET-MSG")
    )

    with patch.object(
        store, "_run_on_collection_loop",
        side_effect=lambda name, coro: asyncio.run(coro),
    ) as run, caplog.at_level("WARNING"):
        assert store.delete_resource_document("silo_7", 42) == status

    mock_rag.adelete_by_doc_id.assert_awaited_once_with("res42", delete_llm_cache=llm_cache)
    assert run.call_args.args[0] == "silo_7"
    warned = any(r.levelname == "WARNING" for r in caplog.records)
    assert warned == (status in ("not_allowed", "fail"))
    assert "SECRET-MSG" not in caplog.text
    if warned:
        warning = next(r for r in caplog.records if r.levelname == "WARNING")
        assert "418" in warning.getMessage()


def test_delete_resource_document_times_out(store, mock_rag, monkeypatch):
    import asyncio

    import tools.vector_stores.lightrag_store as mod

    cancelled = []

    async def _hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(mod, "LIGHTRAG_DOC_DELETE_TIMEOUT_S", 0.05)
    store._get_rag_instance = MagicMock(return_value=mock_rag)
    mock_rag.adelete_by_doc_id = _hang

    with patch.object(
        store, "_run_on_collection_loop",
        side_effect=lambda name, coro: asyncio.run(coro),
    ), pytest.raises(asyncio.TimeoutError):
        store.delete_resource_document("silo_7", 42)

    # The cancel ran (and finished) before the method returned.
    assert cancelled


def test_delete_resource_document_propagates_rag_errors(store):
    store._get_rag_instance = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        store.delete_resource_document("silo_7", 42)


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


def test_delete_collection_removes_cached_instance(store, mock_rag):
    store._rag_instances["silo_1"] = mock_rag

    with patch.object(store, "_cleanup_neo4j"), \
         patch.object(store, "_cleanup_vector_storage"), \
         patch.object(store, "_cleanup_postgres"):
        store.delete_collection("silo_1")

    assert "silo_1" not in store._rag_instances


def test_delete_collection_pgvector_cleans_vector_storage(store):
    store.lightrag_vector_db_type = "PGVECTOR"

    with patch.object(store, "_cleanup_neo4j"), \
         patch.object(store, "_cleanup_vector_storage") as vec, \
         patch.object(store, "_cleanup_postgres"):
        store.delete_collection("silo_1")

    vec.assert_called_once_with("silo_1")


def _rag_with_vdbs(results):
    rag = MagicMock()
    rag.entities_vdb.drop = AsyncMock(return_value=results[0])
    rag.relationships_vdb.drop = AsyncMock(return_value=results[1])
    rag.chunks_vdb.drop = AsyncMock(return_value=results[2])
    return rag


def test_cleanup_vector_storage_drops_all_three_vdbs(store):
    import asyncio

    ok = {"status": "success"}
    rag = _rag_with_vdbs([ok, ok, ok])
    store._get_rag_instance = MagicMock(return_value=rag)
    store._run_on_collection_loop = lambda name, coro: asyncio.run(coro)

    store._cleanup_vector_storage("silo_1")

    rag.entities_vdb.drop.assert_awaited_once()
    rag.relationships_vdb.drop.assert_awaited_once()
    rag.chunks_vdb.drop.assert_awaited_once()


def test_cleanup_vector_storage_warns_on_error_status(store, caplog):
    import asyncio

    ok = {"status": "success"}
    rag = _rag_with_vdbs([ok, {"status": "error", "message": "x"}, ok])
    store._get_rag_instance = MagicMock(return_value=rag)
    store._run_on_collection_loop = lambda name, coro: asyncio.run(coro)

    with caplog.at_level("WARNING"):
        store._cleanup_vector_storage("silo_77")

    assert any("silo_77" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_cleanup_vector_storage_logs_error_when_instance_unavailable(store, caplog):
    store._get_rag_instance = MagicMock(side_effect=RuntimeError("no rag"))

    with caplog.at_level("ERROR"):
        store._cleanup_vector_storage("silo_55")

    assert any(r.levelname == "ERROR" and "silo_55" in r.getMessage() for r in caplog.records)


def _pg_error(pgcode):
    err = Exception("boom")
    err.orig = SimpleNamespace(pgcode=pgcode)
    return err


def test_cleanup_postgres_warns_on_unexpected_error(store, caplog):
    store.db = MagicMock()
    store.db.engine.begin.return_value.__enter__.return_value.execute.side_effect = _pg_error("57014")

    with caplog.at_level("WARNING"):
        store._cleanup_postgres("silo_88")

    assert any(
        r.levelname == "WARNING" and "silo_88" in r.getMessage() and "lightrag_" in r.getMessage()
        for r in caplog.records
    )


def test_cleanup_postgres_undefined_table_is_not_a_warning(store, caplog):
    store.db = MagicMock()
    store.db.engine.begin.return_value.__enter__.return_value.execute.side_effect = _pg_error("42P01")

    with caplog.at_level("WARNING"):
        store._cleanup_postgres("silo_88")

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


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
    doc = Document(page_content="some text", metadata={"resource_id": 61, "page": 5})
    assert _lightrag_doc_id(doc) == "res61-p5"


def test_lightrag_doc_id_falls_back_to_content_hash_without_resource_or_page():
    # Domain/web-crawled content has neither — must still get a valid,
    # LightRAG-shaped id instead of erroring.
    doc_id = _lightrag_doc_id(Document(page_content="crawled page text"))
    assert doc_id.startswith("doc-")
    assert not doc_id.startswith("res")


def test_lightrag_doc_id_trusts_an_explicitly_set_id():
    # step_004: only the Resource whole-file extraction sets Document.id —
    # when it's set, it is trusted directly, regardless of metadata.
    doc = Document(id="res7", page_content="whole file text", metadata={"resource_id": 999, "page": 3})
    assert _lightrag_doc_id(doc) == "res7"


def test_lightrag_doc_id_resource_id_alone_in_metadata_cannot_fake_a_whole_file_id():
    # AD-2 guarantee: caller-supplied metadata alone (no real .id set, no
    # page) must never produce a bare "res{N}" id — only extract_documents_
    # from_file's whole-file path can do that, by setting .id directly.
    doc = Document(page_content="some text", metadata={"resource_id": 7})
    doc_id = _lightrag_doc_id(doc)
    assert doc_id != "res7"
    assert doc_id.startswith("doc-")


@pytest.mark.parametrize("malformed_id", ["../../etc", "res7; DROP TABLE", "res7-p12", "res"])
def test_lightrag_doc_id_rejects_a_malformed_explicit_id(malformed_id):
    # Defence in depth: only a bare "res{N}"-shaped .id is trusted. Anything
    # else set on .id (not reachable today, but not something to trust
    # blindly either) falls through to the metadata/hash logic instead of
    # being used verbatim as the LightRAG doc id.
    doc = Document(id=malformed_id, page_content="some text")
    doc_id = _lightrag_doc_id(doc)
    assert doc_id != malformed_id
    assert doc_id.startswith("doc-")


def test_index_documents_passes_resource_and_page_ids_to_ainsert(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [
        Document(page_content="page one", metadata={"resource_id": 61, "page": 1}),
        Document(page_content="page two", metadata={"resource_id": 61, "page": 2}),
    ]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    assert mock_insert.call_args.kwargs["ids"] == ["res61-p1", "res61-p2"]


def test_index_documents_whole_file_document_uses_its_explicit_id(store, mock_rag):
    # step_004: a single whole-file Document (resource_id folded into .id by
    # extract_documents_from_file) must be indexed under that exact id.
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [Document(id="res61", page_content="whole file text", metadata={"resource_id": 61, "name": "f.pdf"})]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    assert mock_insert.call_args.kwargs["ids"] == ["res61"]


def test_enqueue_documents_whole_file_document(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [Document(id="res7", page_content="<<<page:1>>>\nx", metadata={"resource_id": 7, "name": "m.pdf"})]
    with patch("tools.vector_stores.lightrag_store._aenqueue", new_callable=AsyncMock) as mock_enqueue:
        result = store.enqueue_documents("silo_1", docs)

    assert result == {7: ["res7"]}
    assert mock_enqueue.call_args.kwargs["file_paths"] == ["m.pdf"]
    assert mock_enqueue.call_args.kwargs["ids"] == ["res7"]


# ---------------------------------------------------------------------------
# _source_label_from_metadata
# ---------------------------------------------------------------------------


def test_source_label_strips_path_and_keeps_resource_id_out_of_the_label():
    assert _source_label_from_metadata({"name": "a/b/m.pdf", "resource_id": 7}) == "m.pdf"


def test_source_label_appends_page_when_present():
    assert _source_label_from_metadata({"name": "a/b/m.pdf", "page": 3}) == "m.pdf (p. 3)"


# ---------------------------------------------------------------------------
# NFR-1: caller-supplied metadata alone can never trigger the whole-file
# res{N} id path — only extract_documents_from_file's Resource extraction can.
# ---------------------------------------------------------------------------


def test_create_documents_for_indexing_never_collapses_chunk_ids():
    from services.silo_service import SiloService

    # Realistic multi-chunk caller-metadata scenario (e.g. the public
    # index-file endpoint's payload shape): varied content so chunk
    # boundaries differ, same shape a public caller would send.
    paragraph = "The quick brown fox jumps over the lazy dog near the riverbank. "
    content = "\n\n".join(f"{paragraph}Paragraph number {i}." for i in range(40))
    docs = SiloService._create_documents_for_indexing(1, [{"content": content, "metadata": {"resource_id": 5}}])

    assert len(docs) > 1
    ids = [_lightrag_doc_id(doc) for doc in docs]
    assert all(doc_id.startswith("doc-") for doc_id in ids)
    assert "res5" not in ids
    assert len(set(ids)) == len(ids)


def test_create_documents_for_indexing_single_chunk_resource_id_without_page_gets_a_hash_id():
    # AC-35: the public index-file shape — a single caller-supplied dict with
    # resource_id and no "page" key — must never produce the whole-file
    # "res{N}" id; only extract_documents_from_file's own Resource extraction
    # (which sets Document.id directly) can do that.
    from services.silo_service import SiloService

    docs = SiloService._create_documents_for_indexing(1, [{"content": "short content", "metadata": {"resource_id": 5}}])

    assert len(docs) == 1
    doc_id = _lightrag_doc_id(docs[0])
    assert doc_id.startswith("doc-")
    assert doc_id != "res5"


def test_create_documents_for_indexing_single_chunk_resource_id_and_page_keeps_legacy_shape():
    # AC-35: today's legacy per-page shape (res{N}-p{P}) must still work
    # through the public-API-metadata path when the caller supplies both
    # resource_id and page (step_002's backward-compat guarantee).
    from services.silo_service import SiloService

    docs = SiloService._create_documents_for_indexing(
        1, [{"content": "short content", "metadata": {"resource_id": 5, "page": 2}}],
    )

    assert len(docs) == 1
    assert _lightrag_doc_id(docs[0]) == "res5-p2"


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

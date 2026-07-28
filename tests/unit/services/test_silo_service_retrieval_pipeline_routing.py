"""Unit tests for SiloService.find_docs_in_collection / search_silo_documents_router
routing between the legacy dense-similarity path and the new RetrievalPipeline
(dense/bm25 + optional rerank) path.

Default behaviour (no search_method / no strategy, or search_method="dense") MUST
keep using vector_store.search_similar_documents unchanged (same `_score` metadata
contract). Only an explicit non-dense search_method or a strategy routes through
`_build_pipeline_retriever` / `RetrievalPipeline.build` instead.
"""
from unittest.mock import MagicMock, patch

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from services.silo_service import SiloService


def _make_silo() -> MagicMock:
    silo = MagicMock()
    silo.embedding_service_id = 42
    return silo


class _FakeEmbeddings(Embeddings):
    """EmbeddingsFilter validates its ``embeddings`` field as an Embeddings
    instance — a plain MagicMock fails pydantic validation."""

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _FakeDenseRetriever(BaseRetriever):
    """ContextualCompressionRetriever validates ``base_retriever`` as a Runnable
    instance — a plain MagicMock fails pydantic validation."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return []


# ---------------------------------------------------------------------------
# Default path (unchanged behaviour) — no search_method/strategy requested
# ---------------------------------------------------------------------------

def test_default_search_uses_search_similar_documents_not_pipeline():
    silo = _make_silo()
    vector_store = MagicMock()
    vector_store.search_similar_documents.return_value = [
        Document(page_content="hit", metadata={"_score": 0.9})
    ]
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever") as mock_build_pipeline:
        docs = SiloService.find_docs_in_collection(silo_id=7, query="invoice", db=db)

    vector_store.search_similar_documents.assert_called_once()
    mock_build_pipeline.assert_not_called()
    assert docs[0].metadata["_score"] == 0.9


def test_explicit_search_method_dense_uses_search_similar_documents_not_pipeline():
    """search_method='dense' (explicitly) must behave identically to the default."""
    silo = _make_silo()
    vector_store = MagicMock()
    vector_store.search_similar_documents.return_value = []
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever") as mock_build_pipeline:
        SiloService.find_docs_in_collection(silo_id=7, query="invoice", search_method="dense", db=db)

    vector_store.search_similar_documents.assert_called_once()
    mock_build_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline path — explicit non-dense search_method or a strategy
# ---------------------------------------------------------------------------

def test_bm25_search_method_routes_through_pipeline():
    silo = _make_silo()
    vector_store = MagicMock()
    db = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="bm25 hit", metadata={})]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever", return_value=mock_retriever) as mock_build_pipeline:
        docs = SiloService.find_docs_in_collection(
            silo_id=7, query="invoice", search_method="bm25", limit=10, db=db
        )

    vector_store.search_similar_documents.assert_not_called()
    mock_build_pipeline.assert_called_once()
    args, _ = mock_build_pipeline.call_args
    called_vector_store, called_collection, called_embedding_service, called_kwargs = args
    assert called_vector_store is vector_store
    assert called_collection == "silo_7"
    assert called_embedding_service == "emb"
    assert called_kwargs["search_method"] == "bm25"
    assert called_kwargs["k"] == 10
    mock_retriever.invoke.assert_called_once_with("invoice")
    assert docs[0].page_content == "bm25 hit"


def test_rerank_strategy_routes_through_pipeline():
    silo = _make_silo()
    vector_store = MagicMock()
    db = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever", return_value=mock_retriever) as mock_build_pipeline:
        SiloService.find_docs_in_collection(
            silo_id=7,
            query="invoice",
            strategy="rerank",
            top_n=5,
            similarity_threshold=0.4,
            db=db,
        )

    vector_store.search_similar_documents.assert_not_called()
    mock_build_pipeline.assert_called_once()
    _, _, _, called_kwargs = mock_build_pipeline.call_args[0]
    assert called_kwargs["strategy"] == "rerank"
    assert called_kwargs["top_n"] == 5
    assert called_kwargs["similarity_threshold"] == 0.4
    # Regression: this call site invokes the retriever synchronously (.invoke()),
    # so it must build it in sync mode — an async-mode retriever raises "This
    # method must be called without async_mode" when invoked this way.
    assert mock_build_pipeline.call_args.kwargs["use_async"] is False


def test_dense_plus_rerank_builds_a_sync_mode_retriever():
    """End-to-end regression (no mocking of _build_pipeline_retriever/DenseSearchMethod):
    dense search_method + rerank strategy must produce a retriever built with
    use_async=False, since find_docs_in_collection calls it via .invoke()."""
    silo = _make_silo()
    embedding_service = MagicMock()
    vector_store = MagicMock()
    vector_store.get_retriever.return_value = _FakeDenseRetriever()
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value=embedding_service), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("tools.retrieval.strategies.rerank_strategy.get_embeddings_model", return_value=_FakeEmbeddings()):
        SiloService.find_docs_in_collection(silo_id=7, query="invoice", strategy="rerank", db=db)

    vector_store.get_retriever.assert_called_once()
    assert vector_store.get_retriever.call_args.kwargs["use_async"] is False


def test_pipeline_path_forwards_filter_and_search_type():
    silo = _make_silo()
    vector_store = MagicMock()
    db = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever", return_value=mock_retriever) as mock_build_pipeline:
        SiloService.find_docs_in_collection(
            silo_id=7,
            query="invoice",
            filter_metadata={"doc_type": {"$eq": "pdf"}},
            search_type="mmr",
            search_method="bm25",
            db=db,
        )

    _, _, _, called_kwargs = mock_build_pipeline.call_args[0]
    assert called_kwargs["filter"] == {"doc_type": {"$eq": "pdf"}}
    assert called_kwargs["search_type"] == "mmr"


def test_pipeline_path_still_applies_content_length_post_filter():
    silo = _make_silo()
    vector_store = MagicMock()
    db = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="x" * 10, metadata={}),
        Document(page_content="x" * 200, metadata={}),
    ]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_pipeline_retriever", return_value=mock_retriever):
        docs = SiloService.find_docs_in_collection(
            silo_id=7, query="invoice", search_method="bm25", min_content_length=100, db=db
        )

    assert len(docs) == 1
    assert len(docs[0].page_content) == 200


# ---------------------------------------------------------------------------
# search_silo_documents_router — same routing, response shape unaffected
# ---------------------------------------------------------------------------

def test_router_default_path_returns_score_from_metadata():
    silo = _make_silo()
    db = MagicMock()

    with patch("services.silo_service.SiloService.get_silo", return_value=silo), \
         patch(
             "services.silo_service.SiloService.find_docs_in_collection",
             return_value=[Document(page_content="hit", metadata={"_score": 0.42})],
         ) as mock_find:
        result = SiloService.search_silo_documents_router(1, "query", db=db)

    assert result["results"][0]["score"] == 0.42
    # search_method/strategy/top_n/similarity_threshold default to None and are forwarded
    _, kwargs = mock_find.call_args
    assert kwargs["search_method"] is None
    assert kwargs["strategy"] is None


def test_router_pipeline_path_returns_none_score_when_absent():
    silo = _make_silo()
    db = MagicMock()

    with patch("services.silo_service.SiloService.get_silo", return_value=silo), \
         patch(
             "services.silo_service.SiloService.find_docs_in_collection",
             return_value=[Document(page_content="hit", metadata={})],
         ) as mock_find:
        result = SiloService.search_silo_documents_router(
            1, "query", search_method="bm25", strategy="rerank", top_n=3,
            similarity_threshold=0.6, db=db,
        )

    assert result["results"][0]["score"] is None
    _, kwargs = mock_find.call_args
    assert kwargs["search_method"] == "bm25"
    assert kwargs["strategy"] == "rerank"
    assert kwargs["top_n"] == 3
    assert kwargs["similarity_threshold"] == 0.6

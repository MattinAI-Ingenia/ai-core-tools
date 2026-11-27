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

from services.silo_service import RerankRequest, SiloService


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


def _patched(silo, vector_store, retriever, plan=None):
    """Patch the collaborators of the search path; returns the patch list (entered by the caller)."""
    return [
        patch("services.silo_service.SiloRepository.get_by_id", return_value=silo),
        patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True),
        patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"),
        patch("services.silo_service._get_vector_store", return_value=vector_store),
        patch("services.silo_service._build_pipeline_retriever", return_value=retriever),
    ]


def test_rerank_with_dense_routes_through_sync_pipeline_with_plan():
    """Regression (sync mode): dense + rerank must go through the pipeline, built with use_async=False."""
    from tools.retrieval.rerank_transformer import RerankPlan

    silo = _make_silo()
    vector_store = MagicMock()
    db = MagicMock()
    plan = RerankPlan(top_n=5, candidate_limit=20)
    retriever = MagicMock()
    retriever.invoke.return_value = []

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=vector_store), \
         patch("services.silo_service._build_rerank_plan", return_value=plan) as mock_plan, \
         patch("services.silo_service._build_pipeline_retriever", return_value=retriever) as mock_build:
        SiloService.find_docs_in_collection(
            silo_id=7, query="invoice", rerank=RerankRequest(True, None, 5), limit=10, db=db
        )

    vector_store.search_similar_documents.assert_not_called()
    mock_plan.assert_called_once()
    assert mock_plan.call_args.args[2:] == (None, 5, 10)
    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["use_async"] is False
    assert mock_build.call_args.kwargs["rerank_plan"] is plan
    # first stage supplies the candidate limit
    assert mock_build.call_args.args[3]["k"] == 20


def test_explicit_rerank_service_not_visible_raises_validation_error():
    from utils.error_handlers import ValidationError

    silo = _make_silo()
    silo.app_id = 3
    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service.RerankServiceRepository.get_visible_by_id", return_value=None) as mock_vis, \
         patch("services.silo_service._build_pipeline_retriever") as mock_build:
        try:
            SiloService.find_docs_in_collection(
                silo_id=7, query="q", rerank=RerankRequest(True, 99), db=MagicMock()
            )
        except ValidationError:
            pass
        else:
            raise AssertionError("expected ValidationError")

    mock_vis.assert_called_once()
    assert mock_vis.call_args.args[1:] == (99, 3)
    mock_build.assert_not_called()


def test_rerank_scores_stay_aligned_after_content_length_filter():
    from tools.retrieval.rerank_transformer import RerankOutcome, RerankPlan

    silo = _make_silo()
    plan = RerankPlan(top_n=3, candidate_limit=10)
    retriever = MagicMock()

    def _invoke(_q):
        plan.outcome = RerankOutcome(applied=True, scores=[0.9, 0.8, 0.7])
        return [
            Document(page_content="x" * 200, metadata={}),
            Document(page_content="short", metadata={}),
            Document(page_content="y" * 300, metadata={}),
        ]

    retriever.invoke.side_effect = _invoke
    with patch("services.silo_service.SiloService.get_silo", return_value=silo), \
         patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value="emb"), \
         patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
         patch("services.silo_service._build_rerank_plan", return_value=plan), \
         patch("services.silo_service._build_pipeline_retriever", return_value=retriever):
        result = SiloService.search_silo_documents_router(
            1, "q", rerank=True, min_content_length=100, db=MagicMock()
        )

    assert [r["rerank_score"] for r in result["results"]] == [0.9, 0.7]
    assert result["rerank"]["requested"] is True
    assert result["rerank"]["applied"] is True


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

def _router_search(docs_scores, plan=None, **kwargs):
    silo = _make_silo()
    with patch("services.silo_service.SiloService.get_silo", return_value=silo), \
         patch("services.silo_service.SiloService._search_docs", return_value=(*docs_scores, plan)) as mock_search:
        result = SiloService.search_silo_documents_router(1, "query", db=MagicMock(), **kwargs)
    return result, mock_search


def test_router_default_path_returns_score_from_metadata():
    result, mock_search = _router_search(([Document(page_content="hit", metadata={"_score": 0.42})], [None]))

    assert result["results"][0]["score"] == 0.42
    assert result["results"][0]["rerank_score"] is None
    assert "rerank" not in result
    # search_method / rerank flags default and are forwarded positionally
    args = mock_search.call_args.args
    assert args[11] is None and args[12] is None


def test_router_pipeline_path_returns_none_score_when_absent():
    result, mock_search = _router_search(([Document(page_content="hit", metadata={})], [None]), search_method="bm25")

    assert result["results"][0]["score"] is None
    assert mock_search.call_args.args[11] == "bm25"


def test_router_reports_rerank_outcome_and_scores():
    from tools.retrieval.rerank_transformer import RerankOutcome, RerankPlan

    plan = RerankPlan(top_n=2)
    plan.outcome = RerankOutcome(
        applied=True, service_id=4, service_name="svc", source="app_default", latency_ms=12, scores=[0.9, 0.1]
    )
    result, _ = _router_search(
        ([Document(page_content="a", metadata={}), Document(page_content="b", metadata={})], [0.9, 0.1]),
        plan=plan, rerank=True,
    )

    assert [r["rerank_score"] for r in result["results"]] == [0.9, 0.1]
    assert result["rerank"] == {
        "requested": True, "applied": True, "service_id": 4, "service_name": "svc",
        "source": "app_default", "skipped_reason": None, "latency_ms": 12,
    }


def test_router_reports_skipped_rerank():
    from tools.retrieval.rerank_transformer import RerankOutcome, RerankPlan

    plan = RerankPlan(top_n=2)
    plan.outcome = RerankOutcome(applied=False, skipped_reason="timeout", service_id=4)
    result, _ = _router_search(([Document(page_content="a", metadata={})], [None]), plan=plan, rerank=True)

    assert result["rerank"]["applied"] is False
    assert result["rerank"]["skipped_reason"] == "timeout"
    assert result["results"][0]["rerank_score"] is None

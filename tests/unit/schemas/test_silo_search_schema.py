"""Unit tests for SiloSearchSchema and SiloCountRequestSchema validation."""
import pytest
from pydantic import ValidationError

from schemas.silo_schemas import SiloSearchSchema, SiloCountRequestSchema


# ---------------------------------------------------------------------------
# search_type validation
# ---------------------------------------------------------------------------

def test_valid_search_type_similarity():
    s = SiloSearchSchema(query="test", search_type="similarity")
    assert s.search_type == "similarity"


def test_valid_search_type_mmr():
    s = SiloSearchSchema(query="test", search_type="mmr", fetch_k=20, lambda_mult=0.5)
    assert s.search_type == "mmr"


def test_valid_search_type_score_threshold():
    s = SiloSearchSchema(
        query="test",
        search_type="similarity_score_threshold",
        score_threshold=0.8,
    )
    assert s.score_threshold == 0.8


def test_invalid_search_type_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", search_type="bm25")


# ---------------------------------------------------------------------------
# score_threshold consistency
# ---------------------------------------------------------------------------

def test_score_threshold_without_correct_search_type_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", search_type="similarity", score_threshold=0.5)


# ---------------------------------------------------------------------------
# MMR param consistency
# ---------------------------------------------------------------------------

def test_fetch_k_without_mmr_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", search_type="similarity", fetch_k=10)


def test_lambda_mult_without_mmr_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", search_type="similarity", lambda_mult=0.5)


# ---------------------------------------------------------------------------
# content-length validation
# ---------------------------------------------------------------------------

def test_negative_min_content_length_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", min_content_length=-1)


def test_negative_max_content_length_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", max_content_length=-5)


def test_min_greater_than_max_raises():
    with pytest.raises(ValidationError):
        SiloSearchSchema(query="test", min_content_length=500, max_content_length=100)


def test_valid_content_length_range():
    s = SiloSearchSchema(query="test", min_content_length=10, max_content_length=500)
    assert s.min_content_length == 10
    assert s.max_content_length == 500


def test_zero_min_content_length_is_valid():
    s = SiloSearchSchema(query="test", min_content_length=0)
    assert s.min_content_length == 0


# ---------------------------------------------------------------------------
# search_method (RetrievalPipeline) and rerank (playground schema only)
# ---------------------------------------------------------------------------

def test_search_method_defaults_to_none():
    s = SiloSearchSchema(query="test")
    assert s.search_method is None


def test_valid_search_method_dense():
    s = SiloSearchSchema(query="test", search_method="dense")
    assert s.search_method == "dense"


def test_valid_search_method_bm25():
    s = SiloSearchSchema(query="test", search_method="bm25")
    assert s.search_method == "bm25"


def test_invalid_search_method_raises():
    with pytest.raises(ValidationError, match="search_method"):
        SiloSearchSchema(query="test", search_method="sparse")


def test_public_schema_has_no_rerank_or_legacy_fields():
    for name in ("strategy", "top_n", "similarity_threshold", "rerank", "rerank_service_id", "rerank_top_n"):
        assert name not in SiloSearchSchema.model_fields


def test_public_schema_ignores_rerank_payload():
    s = SiloSearchSchema(query="test", rerank=True, rerank_service_id=3, strategy="rerank", top_n=5)
    dumped = s.model_dump()
    for name in ("strategy", "top_n", "rerank", "rerank_service_id"):
        assert name not in dumped


def test_playground_schema_rerank_defaults():
    from schemas.silo_schemas import PlaygroundSiloSearchSchema
    s = PlaygroundSiloSearchSchema(query="test")
    assert s.rerank is False
    assert s.rerank_service_id is None
    assert s.rerank_top_n is None


def test_playground_schema_accepts_rerank_fields():
    from schemas.silo_schemas import PlaygroundSiloSearchSchema
    s = PlaygroundSiloSearchSchema(query="test", rerank=True, rerank_service_id=4, rerank_top_n=20)
    assert (s.rerank, s.rerank_service_id, s.rerank_top_n) == (True, 4, 20)


@pytest.mark.parametrize("value", [0, 51])
def test_playground_schema_rerank_top_n_bounds(value):
    from schemas.silo_schemas import PlaygroundSiloSearchSchema
    with pytest.raises(ValidationError, match="rerank_top_n"):
        PlaygroundSiloSearchSchema(query="test", rerank_top_n=value)


def test_playground_schema_old_fields_absent():
    from schemas.silo_schemas import PlaygroundSiloSearchSchema
    for name in ("strategy", "top_n", "similarity_threshold"):
        assert name not in PlaygroundSiloSearchSchema.model_fields

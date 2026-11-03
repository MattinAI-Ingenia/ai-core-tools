"""Unit tests for ``SiloService.count_resource_chunks``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.silo_service import SiloService, _chunks_from_tokens, _count_tokens


def _resource(silo):
    repository = SimpleNamespace(silo_id=silo.silo_id, silo=silo)
    return SimpleNamespace(
        resource_id=1, repository_id=80, uri="doc.pdf", folder_id=None,
        repository=repository,
    )


def _session(resource):
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = resource
    return session


def _run(resource, docs):
    with patch("services.silo_service.SessionLocal", return_value=_session(resource)), \
            patch.object(SiloService, "extract_documents_from_file", return_value=docs):
        return SiloService.count_resource_chunks(resource)


def test_lightrag_silo_falls_back_to_default_overlap():
    """overlap=None must fall back to exactly 100 (LightRAG's own default), not 0.

    chunk_token_size=1200 matters here: at chunk_token_size=100,
    ``_chunks_from_tokens`` takes the same no-credit ceiling-division branch
    for ANY overlap >= 100 (0 included would differ, but 100/150/1000 would
    not), so that size can't distinguish "correctly defaulted to 100" from
    "wrongly defaulted to something else >= chunk size". 1200 (with enough
    tokens) forces the overlap-credited branch, which only matches at
    overlap=100 exactly.
    """
    silo = SimpleNamespace(
        silo_id=1, vector_db_type="LIGHTRAG",
        lightrag_chunk_token_size=1200, lightrag_chunk_overlap_token_size=None,
    )
    resource = _resource(silo)
    clean_text = "word " * 7000
    docs = [SimpleNamespace(page_content=f"<<<page:1>>>\n{clean_text}")]

    result = _run(resource, docs)

    tokens = _count_tokens(clean_text)
    expected = _chunks_from_tokens(tokens, 1200, 100)
    assert result == expected
    # Proves the fallback is 100 and not 0 (a legitimate, distinct DB value).
    assert expected != _chunks_from_tokens(tokens, 1200, 0)


def test_lightrag_silo_explicit_zero_overlap_differs_from_none_fallback():
    """overlap=0 is a legitimate DB value (nullable Integer, no check constraint)
    and must be honored as-is, not conflated with overlap=None's 100 fallback."""
    silo = SimpleNamespace(
        silo_id=1, vector_db_type="LIGHTRAG",
        lightrag_chunk_token_size=1200, lightrag_chunk_overlap_token_size=0,
    )
    resource = _resource(silo)
    clean_text = "word " * 7000
    docs = [SimpleNamespace(page_content=f"<<<page:1>>>\n{clean_text}")]

    result = _run(resource, docs)

    tokens = _count_tokens(clean_text)
    expected = _chunks_from_tokens(tokens, 1200, 0)
    assert result == expected
    assert result != _chunks_from_tokens(tokens, 1200, 100)


def test_lightrag_silo_honors_explicit_overlap():
    silo = SimpleNamespace(
        silo_id=1, vector_db_type="LIGHTRAG",
        lightrag_chunk_token_size=100, lightrag_chunk_overlap_token_size=10,
    )
    resource = _resource(silo)
    # Enough tokens that the overlap=10 stride-credited count diverges from
    # what a wrongly-defaulted overlap=100 would give (asserted below).
    clean_text = "word " * 550
    docs = [SimpleNamespace(page_content=f"<<<page:1>>>\n{clean_text}")]

    result = _run(resource, docs)

    expected = _chunks_from_tokens(_count_tokens(clean_text), 100, 10)
    assert result == expected
    # Confirm the explicit overlap actually diverges from the 100-fallback for
    # this input, otherwise this test would pass even if the explicit value
    # were silently ignored and replaced by the fallback.
    assert expected != _chunks_from_tokens(_count_tokens(clean_text), 100, 100)


def test_non_lightrag_silo_counts_docs_unchanged():
    silo = SimpleNamespace(silo_id=1, vector_db_type="PGVECTOR")
    resource = _resource(silo)
    docs = [SimpleNamespace(page_content="a"), SimpleNamespace(page_content="b"), SimpleNamespace(page_content="c")]

    result = _run(resource, docs)

    assert result == 3


def test_extraction_error_returns_zero():
    silo = SimpleNamespace(
        silo_id=1, vector_db_type="LIGHTRAG",
        lightrag_chunk_token_size=100, lightrag_chunk_overlap_token_size=None,
    )
    resource = _resource(silo)

    with patch("services.silo_service.SessionLocal", return_value=_session(resource)), \
            patch.object(SiloService, "extract_documents_from_file", side_effect=RuntimeError("boom")):
        result = SiloService.count_resource_chunks(resource)

    assert result == 0

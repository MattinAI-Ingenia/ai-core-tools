"""Resolves a product's commercial name (e.g. 'TERMAT') to its resource_id(s)
via Resource.extra_metadata->>'DescArticulo' — a deterministic metadata match,
not a reranker. See the design doc's "reranker por nombre de documento"
section for why a semantic reranker doesn't apply here: file codes like
CDOC004043.pdf carry no semantic signal, but this metadata field does.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from services.silo_service import SiloService


@pytest.fixture()
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[
        Silo.__table__, Repository.__table__, Resource.__table__,
    ])
    Maker = sessionmaker(bind=engine)
    s = Maker()
    yield s
    s.close()
    engine.dispose()


def _seed(session, app_id=1):
    silo = Silo(silo_id=1, app_id=app_id, name="test")
    repo = Repository(repository_id=1, app_id=app_id, silo_id=1, name="repo")
    session.add_all([silo, repo])
    session.add(Resource(
        resource_id=271, repository_id=1, uri="CDOC004043.pdf",
        extra_metadata={"DescArticulo": "TERMAT 25", "document": "CDOC004043"},
    ))
    session.add(Resource(
        resource_id=275, repository_id=1, uri="CDOC004425.pdf",
        extra_metadata={"DescArticulo": "NANOCLIMA 4 HDX 135", "document": "CDOC004425"},
    ))
    session.commit()


def test_matches_by_commercial_name(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "TERMAT", 1, session) == [271]


def test_case_insensitive_and_partial(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "nanoclima", 1, session) == [275]


def test_no_match_returns_empty_list(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "THERMAPRO", 1, session) == []


def test_scoped_to_app_id(session):
    _seed(session, app_id=1)
    assert SiloService.resolve_document_by_name(2, "TERMAT", 1, session) == []


def test_scoped_to_silo_id(session):
    """A name match in a different silo's repository must not be returned."""
    _seed(session, app_id=1)
    assert SiloService.resolve_document_by_name(1, "TERMAT", 2, session) == []


def test_matches_by_document_code(session):
    _seed(session)
    assert SiloService.resolve_document_by_name(1, "CDOC004043", 1, session) == [271]


def test_matches_with_escaped_quotes(session):
    """Test that JSON string escaping is properly handled.

    Verifies the fix for the bug where cast(col, Text) preserved JSON
    escaping ("\"foo\"") instead of decoding it (foo), causing searches
    to fail for product names containing quotes.
    """
    silo = Silo(silo_id=1, app_id=1, name="test")
    repo = Repository(repository_id=1, app_id=1, silo_id=1, name="repo")
    session.add_all([silo, repo])
    session.add(Resource(
        resource_id=300, repository_id=1, uri="CDOC005000.pdf",
        extra_metadata={"DescArticulo": 'TERMAT "Special" 25', "document": "CDOC005000"},
    ))
    session.commit()

    # Search for substring spanning the quote should match
    assert SiloService.resolve_document_by_name(1, '"Special"', 1, session) == [300]
    # Partial match should also work
    assert SiloService.resolve_document_by_name(1, 'Special', 1, session) == [300]


def test_ambiguous_name_returns_every_matching_resource_not_just_the_first(session):
    """A generic name (a product family, not one specific model) legitimately
    matches several real documents. Picking only the first (as this function
    used to) silently drops the rest — confirmed live: 'Dual Clima' matches 3
    real manuals in the DOMUSA corpus, and a P20 search scoped to only the
    first missed the other two entirely."""
    silo = Silo(silo_id=1, app_id=1, name="test")
    repo = Repository(repository_id=1, app_id=1, silo_id=1, name="repo")
    session.add_all([silo, repo])
    session.add(Resource(
        resource_id=280, repository_id=1, uri="CDOC002464.pdf",
        extra_metadata={"DescArticulo": "DUAL CLIMA 16R EXPORT", "document": "CDOC002464"},
    ))
    session.add(Resource(
        resource_id=282, repository_id=1, uri="CDOC001744.pdf",
        extra_metadata={"DescArticulo": "DUAL CLIMA HYBRID GAZ 9R/50", "document": "CDOC001744"},
    ))
    session.add(Resource(
        resource_id=296, repository_id=1, uri="DSAT000120.pdf",
        extra_metadata={"DescArticulo": "DUAL CLIMA 12R NE DOMUSA iCONNECT", "document": "DSAT000120"},
    ))
    session.commit()

    assert SiloService.resolve_document_by_name(1, "Dual Clima", 1, session) == [280, 282, 296]


def test_unions_graph_entity_matches_with_metadata_matches():
    """A document can call itself something in its own text that never
    reaches the structured metadata (confirmed live: DSAT000120's DescArticulo
    is 'DUAL CLIMA 12R NE DOMUSA iCONNECT', but the manual's own text
    introduces itself as 'Dual Clima R' — a name the metadata match alone
    cannot find). The graph entity lookup is what catches this; the two
    sources are unioned, deduplicated."""
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.filter.return_value \
        .filter.return_value.order_by.return_value.all.return_value = [(271,)]

    with patch(
        "services.silo_service.SiloService._resolve_via_graph_entities",
        return_value=[271, 296],
    ) as graph_mock:
        result = SiloService.resolve_document_by_name(1, "Dual Clima R", 1, session)

    graph_mock.assert_called_once_with(1, "Dual Clima R")
    assert result == [271, 296]  # deduplicated union, sorted


def test_graph_entity_lookup_parses_resource_id_from_source_id():
    """The graph stores an entity's origin as chunk ids like
    'res296-p1-chunk-000<SEP>res296-p87-chunk-000' — confirmed live against
    Neo4j for the 'Dual Clima R' entity. Only the resource_id is needed."""
    fake_record = {"source_id": "res296-p1-chunk-000<SEP>res296-p87-chunk-000"}
    fake_session = MagicMock()
    fake_session.run.return_value = [fake_record]
    fake_driver = MagicMock()
    fake_driver.session.return_value.__enter__.return_value = fake_session

    with patch(
        "services.silo_graph_service.SiloGraphService._neo4j_driver",
        return_value=fake_driver,
    ):
        result = SiloService._resolve_via_graph_entities(37, "Dual Clima R")

    assert result == [296]


def test_graph_entity_lookup_degrades_to_empty_when_neo4j_unavailable():
    """Best-effort supplement, not a hard dependency — matches
    SiloGraphService's own existing pattern for optional graph access."""
    with patch(
        "services.silo_graph_service.SiloGraphService._neo4j_driver",
        side_effect=RuntimeError("NEO4J_URI not configured"),
    ):
        assert SiloService._resolve_via_graph_entities(37, "anything") == []

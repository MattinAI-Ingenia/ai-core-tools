"""``SiloGraphSourcesService.resolve_sources`` — chunk-id → source-document
resolver for the graph explorer's detail panels.

Parse-first: chunk ids shaped ``res{id}-p{page}-chunk-N`` yield (resource_id,
page) with a single ORM lookup for file names. Ids that don't parse (crawled
Domain pages, legacy indexes) fall back to the LightRAG KV table, whose
``file_path`` already carries the "(p. N)" label (see ``_page_label``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from models.resource import Resource
from services.silo_graph_sources_service import SiloGraphSourcesService


def _db_with_names(names: dict):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = list(names.items())
    return db


def test_res_ids_parsed_without_db():
    """Parse-first path: one ORM query, no LightRAG KV access."""
    db = _db_with_names({7: "manual.pdf", 9: "ficha.pdf"})

    result = SiloGraphSourcesService.resolve_sources(
        1, ["res7-p12-chunk-003", "res9-chunk-001"], db,
    )

    assert result == [
        {"chunk_id": "res7-p12-chunk-003", "resource_id": 7, "page": 12, "file_name": "manual.pdf"},
        {"chunk_id": "res9-chunk-001", "resource_id": 9, "page": None, "file_name": "ficha.pdf"},
    ]
    db.query.assert_called_once()
    db.execute.assert_not_called()


def test_legacy_id_uses_kv_fallback(monkeypatch):
    """Non-parsing ids (crawled pages, legacy indexes) resolve via the
    LightRAG KV table; its file_path carries the page label already."""
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = [
        {"id": "doc-abc123", "file_path": "instrucciones.html (p. 3)"},
    ]

    result = SiloGraphSourcesService.resolve_sources(1, ["doc-abc123"], db)

    assert result == [
        {"chunk_id": "doc-abc123", "resource_id": None, "page": None, "file_name": "instrucciones.html (p. 3)"},
    ]
    sql = db.execute.call_args.args[0].text
    assert "lightrag_doc_chunks" in sql
    assert db.execute.call_args.args[1] == {"ws": "silo_1", "ids": ["doc-abc123"]}


def test_unknown_id_yields_none():
    """KV lookup returns nothing → file_name None, never a crash."""
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []

    result = SiloGraphSourcesService.resolve_sources(1, ["doc-gone"], db)

    assert result == [
        {"chunk_id": "doc-gone", "resource_id": None, "page": None, "file_name": None},
    ]


def test_dedup_same_doc_page():
    """Two chunk ids from the same document page collapse to one chip."""
    db = _db_with_names({7: "manual.pdf"})

    result = SiloGraphSourcesService.resolve_sources(
        1, ["res7-p12-chunk-003", "res7-p12-chunk-004"], db,
    )

    assert len(result) == 1
    assert result[0]["file_name"] == "manual.pdf"
    assert result[0]["page"] == 12


def test_resource_name_missing_falls_back_to_none():
    """A resource deleted after indexing (or a stale graph) must not crash —
    the chip falls back to the frontend's 'Unknown source'."""
    db = _db_with_names({})

    result = SiloGraphSourcesService.resolve_sources(1, ["res7-p1-chunk-000"], db)

    assert result == [
        {"chunk_id": "res7-p1-chunk-000", "resource_id": 7, "page": 1, "file_name": None},
    ]

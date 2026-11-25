"""Unit tests for the graph explorer's display mapping.

The explorer must render the same DISPLAY names as the playground bubble
(canonical merge key → most-mentioned variant) or the two graph surfaces
diverge after reindex.

(The old node_label whitelist guard was removed with the label filter itself:
entity types are a Neo4j PROPERTY now, so the filter is parameterized and
cannot inject Cypher.)
"""

from services.silo_graph_service import SiloGraphService


class TestDisplayMap:
    def _canned_graph(self):
        return {
            "nodes": [
                {"id": "mcf 40", "labels": ["equipment"], "properties": {"entity_id": "mcf 40"}},
                {"id": "termico", "labels": [], "properties": {"entity_id": "termico"}},
            ],
            "edges": [
                {"id": "1", "source": "mcf 40", "target": "termico", "type": "PART_OF", "properties": {}},
            ],
            "node_count": 2,
            "edge_count": 1,
            "total_nodes": 2,
            "total_edges": 1,
        }

    def _install(self, monkeypatch, graph, name_map):
        monkeypatch.setattr(SiloGraphService, "_cypher_graph", classmethod(lambda cls, **kwargs: graph))
        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.best_variant_map",
            staticmethod(lambda silo_id, names, db: name_map),
        )

    def test_rewrites_nodes_and_edges_through_map(self, monkeypatch):
        self._install(
            monkeypatch,
            self._canned_graph(),
            {"mcf 40": "MCF-40", "termico": "Térmico"},
        )
        out = SiloGraphService.get_silo_graph(14, db=object())

        assert out["nodes"][0]["id"] == "MCF-40"
        assert out["nodes"][0]["properties"]["entity_id"] == "MCF-40"
        assert out["nodes"][1]["id"] == "Térmico"
        assert (out["edges"][0]["source"], out["edges"][0]["target"]) == ("MCF-40", "Térmico")

    def test_unknown_names_fall_back_to_canonical(self, monkeypatch):
        self._install(monkeypatch, self._canned_graph(), {"mcf 40": "MCF-40"})
        out = SiloGraphService.get_silo_graph(14, db=object())

        assert out["nodes"][0]["id"] == "MCF-40"
        assert out["nodes"][1]["id"] == "termico"  # not in the map → canonical
        assert (out["edges"][0]["source"], out["edges"][0]["target"]) == ("MCF-40", "termico")

    def test_no_db_keeps_canonical_names(self, monkeypatch):
        self._install(monkeypatch, self._canned_graph(), {})
        out = SiloGraphService.get_silo_graph(14, db=None)
        assert {n["id"] for n in out["nodes"]} == {"mcf 40", "termico"}

    def test_map_lookup_failure_degrades_to_canonical(self, monkeypatch):
        monkeypatch.setattr(SiloGraphService, "_cypher_graph", classmethod(lambda cls, **kwargs: self._canned_graph()))
        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.best_variant_map",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))),
        )
        out = SiloGraphService.get_silo_graph(14, db=object())
        assert {n["id"] for n in out["nodes"]} == {"mcf 40", "termico"}

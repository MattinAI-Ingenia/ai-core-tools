"""Unit tests for the graph explorer's search-clause normalization.

The graph explorer (SiloGraphService.get_silo_graph) searches nodes by
``entity_id``/description. Since LightRAG node names are now the canonical
merge key (case/accents/space/hyphen folded — see
``tools.vector_stores.lightrag.entity_name_normalization``), the clause
must ALSO match the folded search term, or a user-typed "MCF-40" misses the
canonical node "mcf 40". The raw-spelling conditions stay so prose
descriptions (kept with accents) and legacy node names keep matching.
"""

from services.silo_graph_service import SiloGraphService


class TestBuildSearchClause:
    def _clause(self, search: str):
        params = {}
        clause = SiloGraphService._build_search_clause(search, params)
        return clause, params

    def test_pretty_search_adds_canonical_term(self):
        clause, params = self._clause("MCF-40")
        assert params["canonical_search"] == "mcf 40"
        assert "n.entity_id CONTAINS $canonical_search" in clause
        # The raw conditions survive for descriptions and legacy nodes.
        assert "toLower(n.entity_id) CONTAINS toLower($search)" in clause
        assert "toLower(n.description) CONTAINS toLower($search)" in clause

    def test_accented_search_folds(self):
        clause, params = self._clause("Térmico")
        assert params["canonical_search"] == "termico"
        assert "n.entity_id CONTAINS $canonical_search" in clause

    def test_canonical_search_keeps_both_layers(self):
        clause, params = self._clause("mcf 40")
        assert params["canonical_search"] == "mcf 40"
        assert clause.count("CONTAINS") == 3

    def test_empty_fold_skips_canonical_condition(self):
        clause, params = self._clause("-")
        assert "canonical_search" not in params
        # Only the two raw conditions — CONTAINS "" would match everything.
        assert "CONTAINS" in clause
        assert "$canonical_search" not in clause

    def test_clause_is_valid_where_syntax(self):
        clause, _params = self._clause("mcf")
        assert clause.startswith("WHERE (")
        assert clause.endswith(") ")
        assert clause.count("(") == clause.count(")")

"""Unit tests for LightRAG entity-name canonicalization and variant tracking.

Covers the pieces described in docs/dependencies/lightrag.md §6.1:
- canonicalize_entity_name folding (case, accents, spaces, hyphens, NFKC),
- the per-run EntityNameVariantCollector,
- the monkeypatch of lightrag's normalize_entity_name (merge key + counts),
patched state restored after each test so other tests keep the original.
"""

import pytest

from tools.vector_stores.lightrag.entity_name_normalization import (
    EntityNameVariantCollector,
    canonicalize_entity_name,
    ensure_entity_name_normalization_patch,
    get_active_variant_collector,
    reset_active_variant_collector,
    set_active_variant_collector,
)


class TestCanonicalizeEntityName:
    def test_case_variants_converge(self):
        assert canonicalize_entity_name("MCF-40") == "mcf 40"
        assert canonicalize_entity_name("mcf 40") == "mcf 40"
        assert canonicalize_entity_name("Mcf 40") == "mcf 40"

    def test_accents_folded(self):
        assert canonicalize_entity_name("Térmico") == "termico"
        assert canonicalize_entity_name("Motor Térmico") == "motor termico"
        assert canonicalize_entity_name("ÂNGULO") == "angulo"

    def test_hyphen_like_separators_collapse_to_space(self):
        assert canonicalize_entity_name("500-1000") == "500 1000"
        assert canonicalize_entity_name("500\u20131000") == "500 1000"  # en dash
        assert canonicalize_entity_name("500\u20141000") == "500 1000"  # em dash
        assert canonicalize_entity_name("MCF\u00ad40") == "mcf 40"  # soft hyphen
        assert canonicalize_entity_name("MCF\u201040") == "mcf 40"  # unicode hyphen

    def test_no_separator_vs_separator_stays_distinct(self):
        # "MCF40" (no separator) is a different token than "MCF-40" — folding
        # only unifies variants of the SAME spelled name, not compact forms.
        assert canonicalize_entity_name("MCF40") == "mcf40"
        assert canonicalize_entity_name("MCF40") != canonicalize_entity_name("MCF-40")

    def test_nfkc_folds_compatibility_chars(self):
        assert canonicalize_entity_name("ＭＣＦ") == "mcf"
        assert canonicalize_entity_name("m²") == "m2"

    def test_whitespace_collapsed(self):
        assert canonicalize_entity_name("  Bt   Duo  ") == "bt duo"
        assert canonicalize_entity_name("Bt\tDuo\n") == "bt duo"

    def test_empty_and_edge_inputs(self):
        assert canonicalize_entity_name("") == ""
        assert canonicalize_entity_name(None) == ""
        assert canonicalize_entity_name("-") == ""


class TestEntityNameVariantCollector:
    def test_records_increment_counts(self):
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        collector.record("mcf 40", "MCF-40")
        collector.record("mcf 40", "mcf 40")
        assert collector.drain() == {"mcf 40": {"MCF-40": 2, "mcf 40": 1}}

    def test_drain_clears(self):
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        assert collector.drain() == {"mcf 40": {"MCF-40": 1}}
        assert not collector.has_data()
        assert collector.drain() == {}

    def test_ignores_empty_parts(self):
        collector = EntityNameVariantCollector()
        collector.record("", "MCF-40")
        collector.record("mcf 40", "")
        assert not collector.has_data()

    def test_contextvar_set_get_reset(self):
        collector = EntityNameVariantCollector()
        assert get_active_variant_collector() is None
        token = set_active_variant_collector(collector)
        try:
            assert get_active_variant_collector() is collector
        finally:
            reset_active_variant_collector(token)
        assert get_active_variant_collector() is None


class TestNormalizeEntityNamePatch:
    """Only meaningful with lightrag-hku installed (the store's own contract)."""

    @pytest.fixture(autouse=True)
    def _restore_lightrag_normalize(self):
        pytest.importorskip("lightrag")
        import lightrag.operate as operate_module
        import lightrag.utils as utils_module

        saved_utils = utils_module.normalize_entity_name
        saved_operate = operate_module.normalize_entity_name
        yield
        utils_module.normalize_entity_name = saved_utils
        operate_module.normalize_entity_name = saved_operate
        import tools.vector_stores.lightrag.entity_name_normalization as module
        module._PATCH_INSTALLED = False

    def _with_collector(self):
        collector = EntityNameVariantCollector()
        token = set_active_variant_collector(collector)
        return collector, token

    def test_patch_installs_and_is_idempotent(self):
        import lightrag.operate as operate_module
        import lightrag.utils as utils_module

        ensure_entity_name_normalization_patch()
        first = utils_module.normalize_entity_name
        ensure_entity_name_normalization_patch()
        assert utils_module.normalize_entity_name is first
        assert operate_module.normalize_entity_name is first
        assert getattr(first, "_mattin_canonicalizes", False)

    def test_variants_fold_to_canonical_and_are_counted(self):
        import lightrag.operate as operate_module

        ensure_entity_name_normalization_patch()
        collector, token = self._with_collector()
        try:
            assert operate_module.normalize_entity_name("MCF-40") == "mcf 40"
            assert operate_module.normalize_entity_name("Quemador MCF-40") == "quemador mcf 40"
            assert operate_module.normalize_entity_name("Térmico") == "termico"
        finally:
            reset_active_variant_collector(token)
        assert collector.drain() == {
            "mcf 40": {"MCF-40": 1},
            "quemador mcf 40": {"Quemador MCF-40": 1},
            "termico": {"Térmico": 1},
        }

    def test_no_collector_still_folds_key(self):
        import lightrag.operate as operate_module

        ensure_entity_name_normalization_patch()
        # Outside an indexing run the collector is None: the merge key still
        # folds (this is what makes extraction-then-query consistent), but no
        # counts are written.
        assert get_active_variant_collector() is None
        assert operate_module.normalize_entity_name("MCF-40") == "mcf 40"

    def test_identical_form_not_counted(self):
        import lightrag.operate as operate_module

        ensure_entity_name_normalization_patch()
        collector, token = self._with_collector()
        try:
            assert operate_module.normalize_entity_name("mcf 40") == "mcf 40"
        finally:
            reset_active_variant_collector(token)
        # canonical == cleaned → nothing to count, key unchanged
        assert collector.drain() == {}

    def test_light_rags_numeric_filter_contract_preserved(self):
        import lightrag.operate as operate_module

        ensure_entity_name_normalization_patch()
        # LightRAG drops short purely-numeric names ("" from its own cleanup):
        # the patch must return them untouched so the caller keeps dropping.
        assert operate_module.normalize_entity_name("42") == ""
        assert operate_module.normalize_entity_name("") == ""

    def test_non_ascii_name_not_canonicalized_when_cleaned_empty(self):
        import lightrag.operate as operate_module

        ensure_entity_name_normalization_patch()
        # Accent folding must not swallow real words: "Térmico" keeps letters.
        assert operate_module.normalize_entity_name("Térmico") == "termico"

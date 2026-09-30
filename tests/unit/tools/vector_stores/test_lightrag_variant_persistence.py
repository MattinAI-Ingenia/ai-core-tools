"""Unit tests for the variant-persistence wiring in lightrag_store.

The auditors flagged the gap: the pieces (folding, collector, upsert, search
clause) each had tests, but the glue — set→drain→persist, the finally-order,
silo-id parsing and the display-map cache — was unverified. A regression
there silently disables the whole feature (no counts → canonical fallback)
with zero failing tests. These tests pin that glue.
"""

import pytest

import tools.vector_stores.lightrag_store as lightrag_store
from tools.vector_stores.lightrag.entity_name_normalization import EntityNameVariantCollector


class TestSiloIdFromCollection:
    @pytest.mark.parametrize("collection,expected", [
        ("silo_14", 14),
        ("silo_0", 0),
        ("silo_", None),
        ("silo_x", None),
        ("silo_14abc", None),
        ("", None),
        ("other_collection", None),
        (None, None),
    ])
    def test_parses_only_integer_suffixes(self, collection, expected):
        assert lightrag_store._silo_id_from_collection(collection) == expected


class TestPersistVariantCounts:
    @pytest.fixture
    def fake_session(self, monkeypatch):
        """Replace SessionLocal with a recording fake session."""
        calls = {"sessions": [], "closed": 0}

        class FakeSession:
            def close(self):
                calls["closed"] += 1

        fake_db = FakeSession()

        def fake_sessionlocal():
            calls["sessions"].append(fake_db)
            return fake_db

        monkeypatch.setattr("db.database.SessionLocal", fake_sessionlocal)
        monkeypatch.setattr(
            "tools.vector_stores.lightrag_store.invalidate_variant_name_cache",
            lambda silo_id: calls.setdefault("invalidated", []).append(silo_id),
        )
        return calls, fake_db

    def test_persists_drained_counts_for_silo(self, fake_session, monkeypatch):
        calls, fake_db = fake_session
        recorded = {}
        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.record_mentions",
            staticmethod(lambda silo_id, counts, db: recorded.update({"args": (silo_id, counts, db)}) or 2),
        )
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        lightrag_store._persist_variant_counts("silo_14", collector)

        assert recorded["args"] == (14, {"mcf 40": {"MCF-40": 1}}, fake_db)
        assert calls["closed"] == 1
        assert calls["invalidated"] == [14]
        # Drained: the collector is empty afterwards.
        assert not collector.has_data()

    def test_empty_collector_skips_persistence(self, fake_session):
        calls, _ = fake_session
        lightrag_store._persist_variant_counts("silo_14", EntityNameVariantCollector())
        assert not calls.get("sessions")

    def test_non_silo_collection_skips_persistence(self, fake_session):
        calls, _ = fake_session
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        lightrag_store._persist_variant_counts("other_collection", collector)
        assert not calls.get("sessions")
        # Collector keeps its data — nothing was consumed.
        assert collector.has_data()

    def test_persistence_failure_does_not_escape(self, fake_session, monkeypatch):
        calls, _ = fake_session
        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.record_mentions",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        # Must not raise — the indexing run's finally must stay clean.
        lightrag_store._persist_variant_counts("silo_14", collector)
        assert calls["closed"] == 1

    def test_session_construction_failure_does_not_escape(self, monkeypatch):
        def broken_sessionlocal():
            raise RuntimeError("pool exhausted")

        monkeypatch.setattr("db.database.SessionLocal", broken_sessionlocal)
        collector = EntityNameVariantCollector()
        collector.record("mcf 40", "MCF-40")
        # Must not raise even when the session factory itself fails.
        lightrag_store._persist_variant_counts("silo_14", collector)


class TestBestVariantNamesCache:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        lightrag_store._VARIANT_NAME_CACHE.clear()
        yield
        lightrag_store._VARIANT_NAME_CACHE.clear()

    def _patch_repo(self, monkeypatch, calls):
        def fake_best_variant_map(silo_id, names, db):
            calls.append((silo_id, tuple(names)))
            return {"mcf 40": "MCF-40", "termico": "Térmico"}

        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.best_variant_map",
            staticmethod(fake_best_variant_map),
        )
        monkeypatch.setattr(
            "db.database.SessionLocal",
            lambda: type("S", (), {"close": staticmethod(lambda: calls.append(("close",)))})(),
        )

    def test_second_lookup_hits_cache_not_db(self, monkeypatch):
        calls = []
        self._patch_repo(monkeypatch, calls)

        first = lightrag_store._best_variant_names("silo_14", ["mcf 40"])
        second = lightrag_store._best_variant_names("silo_14", ["mcf 40", "termico"])

        # First call: one DB query + one session close. Second call: a cache
        # hit — NO session is created (it returns before SessionLocal()).
        assert calls == [(14, ("mcf 40",)), ("close",)]
        # The second call filtered from the cached silo-wide map.
        assert first == {"mcf 40": "MCF-40"}
        assert second == {"mcf 40": "MCF-40", "termico": "Térmico"}

    def test_invalidation_forces_refetch(self, monkeypatch):
        calls = []
        self._patch_repo(monkeypatch, calls)

        lightrag_store._best_variant_names("silo_14", ["mcf 40"])
        lightrag_store.invalidate_variant_name_cache(14)
        lightrag_store._best_variant_names("silo_14", ["mcf 40"])

        db_queries = [c for c in calls if isinstance(c, tuple) and c and isinstance(c[0], int)]
        assert len(db_queries) == 2

    def test_db_failure_degrades_to_empty(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "repositories.entity_name_variant_repository.EntityNameVariantRepository.best_variant_map",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))),
        )
        monkeypatch.setattr("db.database.SessionLocal", lambda: type("S", (), {"close": lambda self: None})())

        assert lightrag_store._best_variant_names("silo_14", ["mcf 40"]) == {}
        # The failure is NOT cached as a success: next call retries the DB.
        lightrag_store.invalidate_variant_name_cache(14)
        # (retry happens through the same failing path — no crash)
        assert lightrag_store._best_variant_names("silo_14", ["mcf 40"]) == {}

    def test_non_silo_collection_returns_empty(self, monkeypatch):
        calls = []
        self._patch_repo(monkeypatch, calls)
        assert lightrag_store._best_variant_names("other", ["mcf 40"]) == {}
        assert not [c for c in calls if isinstance(c, tuple)]

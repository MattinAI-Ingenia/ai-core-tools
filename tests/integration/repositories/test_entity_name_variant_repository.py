"""Integration tests for EntityNameVariantRepository.

Requires the test DB (docker compose --profile test, port 5433).

Covers the two queries the LightRAG integration needs:
- record_mentions composes across runs (ON CONFLICT increments, never resets),
- best_variant_map picks the most-mentioned variant per canonical key, with
  stable tie-breaking (first-seen wins).
"""

from repositories.entity_name_variant_repository import EntityNameVariantRepository


class TestRecordMentions:
    def test_inserts_new_variants(self, db, fake_silo):
        written = EntityNameVariantRepository.record_mentions(
            fake_silo.silo_id,
            {"mcf 40": {"MCF-40": 2, "mcf 40": 1}},
            db,
        )
        assert written == 2

    def test_conflict_increments_not_replaces(self, db, fake_silo):
        EntityNameVariantRepository.record_mentions(fake_silo.silo_id, {"mcf 40": {"MCF-40": 3}}, db)
        EntityNameVariantRepository.record_mentions(fake_silo.silo_id, {"mcf 40": {"MCF-40": 2}}, db)

        assert EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, db=db) == {
            "mcf 40": "MCF-40",
        }

    def test_empty_input_is_noop(self, db, fake_silo):
        assert EntityNameVariantRepository.record_mentions(fake_silo.silo_id, {}, db) == 0
        assert EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, db=db) == {}


class TestBestVariantMap:
    def test_picks_most_mentioned_variant(self, db, fake_silo):
        EntityNameVariantRepository.record_mentions(
            fake_silo.silo_id,
            {"mcf 40": {"MCF-40": 1, "Mcf 40": 5}},
            db,
        )
        assert EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, db=db) == {
            "mcf 40": "Mcf 40",
        }

    def test_scopes_to_requested_names_only(self, db, fake_silo):
        EntityNameVariantRepository.record_mentions(
            fake_silo.silo_id,
            {
                "mcf 40": {"MCF-40": 1},
                "tapa eliptica bt duo 500 1000": {"Tapa Elíptica BT Duo 500-1000": 4},
            },
            db,
        )
        result = EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, ["mcf 40"], db)
        assert result == {"mcf 40": "MCF-40"}

    def test_tie_breaks_on_first_inserted(self, db, fake_silo):
        EntityNameVariantRepository.record_mentions(
            fake_silo.silo_id,
            {"mcf 40": {"MCF-40": 2}},
            db,
        )
        EntityNameVariantRepository.record_mentions(
            fake_silo.silo_id,
            {"mcf 40": {"Mcf 40": 2}},
            db,
        )
        # Equal counts: first-seen (lower id) wins — rendering stays stable.
        assert EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, db=db) == {
            "mcf 40": "MCF-40",
        }

    def test_silos_are_isolated(self, db, fake_silo, fake_app):
        from models.silo import Silo

        EntityNameVariantRepository.record_mentions(fake_silo.silo_id, {"mcf 40": {"MCF-40": 1}}, db)
        other = Silo(name="other-silo", app_id=fake_app.app_id, embedding_service_id=None)
        db.add(other)
        db.commit()
        EntityNameVariantRepository.record_mentions(other.silo_id, {"mcf 40": {"MCF40": 1}}, db)

        assert EntityNameVariantRepository.best_variant_map(fake_silo.silo_id, db=db) == {
            "mcf 40": "MCF-40",
        }
        assert EntityNameVariantRepository.best_variant_map(other.silo_id, db=db) == {
            "mcf 40": "MCF40",
        }

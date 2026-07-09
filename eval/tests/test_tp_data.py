"""Tests for the TravelPlanner data loader."""

from __future__ import annotations

import pytest

# These tests download the dataset on first run; skipped if HF is unreachable.
pytest_plugins = []


def _can_load() -> bool:
    try:
        from datasets import load_dataset
        load_dataset("osunlp/TravelPlanner", "validation",
                     split="validation[:1]")
        return True
    except Exception:
        return False


_HF_AVAILABLE = _can_load()


@pytest.mark.skipif(not _HF_AVAILABLE,
                    reason="HuggingFace dataset not reachable")
class TestLoader:
    def test_load_returns_record_count(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        assert 100 < len(records) <= 200

    def test_record_has_required_fields(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        first = records[0]
        assert isinstance(first.org, str) and first.org
        assert isinstance(first.dest, str) and first.dest
        assert first.days in (3, 5, 7)
        assert isinstance(first.budget, int) and first.budget > 0
        assert first.level in ("easy", "medium", "hard")

    def test_date_parsed_to_list(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        first = records[0]
        assert isinstance(first.dates, list) and len(first.dates) == first.days

    def test_local_constraint_parsed_to_dict(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        first = records[0]
        assert isinstance(first.local_constraint, dict)
        # The four known keys should all be present (values may be None)
        for k in ("house rule", "cuisine", "room type", "transportation"):
            assert k in first.local_constraint

    def test_reference_information_parsed_to_list(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        first = records[0]
        assert isinstance(first.reference_information, list)
        assert all(isinstance(e, dict) and "Description" in e and "Content" in e
                   for e in first.reference_information)


@pytest.mark.skipif(not _HF_AVAILABLE,
                    reason="HuggingFace dataset not reachable")
class TestQueryIds:
    def test_query_ids_are_unique_and_stable(self):
        from benchmarks.travelplanner.data import load_validation_queries
        records = load_validation_queries()
        ids = [r.query_id for r in records]
        assert len(set(ids)) == len(ids)
        # Reload and confirm IDs are the same
        records2 = load_validation_queries()
        ids2 = [r.query_id for r in records2]
        assert ids == ids2

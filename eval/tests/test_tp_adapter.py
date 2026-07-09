"""Tests for TravelPlannerBenchmark adapter."""

from __future__ import annotations

import pytest


def _hf_ok() -> bool:
    try:
        from benchmarks.travelplanner.data import load_validation_queries
        load_validation_queries()
        return True
    except Exception:
        return False


_HF = _hf_ok()


@pytest.mark.skipif(not _HF, reason="HuggingFace dataset not reachable")
class TestAdapter:
    def test_load_queries_returns_180(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        queries = b.load_queries()
        assert len(queries) == 180

    def test_load_query_by_id(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        first_id = b.load_queries()[0].query_id
        q = b.load_query(first_id)
        assert q.query_id == first_id

    def test_load_queries_subset_filters_by_level(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        easy = b.load_queries(subset="easy")
        assert all(q.difficulty == "easy" for q in easy)
        assert len(easy) == 60

    def test_tools_for_returns_five(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        q = b.load_queries()[0]
        tools = b.tools_for(q)
        names = {t.name for t in tools}
        assert names == {"Flights", "Accommodations", "Restaurants",
                         "Attractions", "DistanceMatrix"}

    def test_query_payload_carries_task_text(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        q = b.load_queries()[0]
        assert "text" in q.payload
        assert isinstance(q.payload["text"], str) and q.payload["text"]
        # Should reference the natural-language query
        assert "<final_answer>" not in q.payload["text"]  # we'll add tag-instructions ourselves


@pytest.mark.skipif(not _HF, reason="HuggingFace dataset not reachable")
class TestScoring:
    def test_score_with_invalid_output_returns_failed(self):
        from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
        b = TravelPlannerBenchmark()
        q = b.load_queries()[0]
        result = b.score(q, "not json at all")
        assert result.completed is False

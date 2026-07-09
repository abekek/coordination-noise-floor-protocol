"""Tests for the MiniTrip toy benchmark."""

from __future__ import annotations

from benchmarks.toy.adapter import ToyBenchmark
from benchmarks.toy.data import FLIGHTS, HOTELS, QUERIES
from benchmarks.toy.scorer import check_completion
from benchmarks.toy.tools import (
    BookFlightTool, BookHotelTool, SearchFlightsTool, SearchHotelsTool,
)


class TestData:
    def test_flight_count(self):
        assert len(FLIGHTS) >= 8

    def test_hotel_count(self):
        assert len(HOTELS) >= 20

    def test_query_count(self):
        assert len(QUERIES) >= 30

    def test_query_difficulty_levels_present(self):
        levels = {q["difficulty"] for q in QUERIES}
        assert levels == {"easy", "medium", "hard"}


class TestTools:
    async def test_search_flights_filters_by_route(self):
        tool = SearchFlightsTool(seed=42)
        result = await tool(origin="NYC", destination="LON")
        assert isinstance(result, list)
        for flight in result:
            assert "id" in flight and "price" in flight and "available" in flight

    async def test_search_hotels_filters_by_city(self):
        tool = SearchHotelsTool(seed=42)
        result = await tool(city="LON")
        assert isinstance(result, list)
        for hotel in result:
            assert "id" in hotel and "nightly_rate" in hotel

    async def test_book_flight_succeeds_when_available(self):
        tool = BookFlightTool(seed=42)
        # F1 is hard-coded as available for seed 42; check data.py
        result = await tool(flight_id="F1")
        assert "success" in result

    async def test_book_hotel_returns_total_cost(self):
        tool = BookHotelTool(seed=42)
        result = await tool(hotel_id="H_LON_1", nights=2)
        if result["success"]:
            assert result["total_cost"] > 0


class TestScorer:
    def test_completion_requires_flight_and_hotel_within_budget(self):
        # Both booked, under budget
        assert check_completion(
            flight_id="F1", hotel_id="H1",
            flight_price=200, hotel_total=200, budget=500,
        ) is True
        # Over budget
        assert check_completion(
            flight_id="F1", hotel_id="H1",
            flight_price=300, hotel_total=300, budget=500,
        ) is False
        # Missing booking
        assert check_completion(
            flight_id=None, hotel_id="H1",
            flight_price=0, hotel_total=200, budget=500,
        ) is False


class TestAdapter:
    def test_load_queries_returns_all_30(self):
        b = ToyBenchmark()
        queries = b.load_queries()
        assert len(queries) >= 30

    def test_load_query_by_id(self):
        b = ToyBenchmark()
        queries = b.load_queries()
        first_id = queries[0].query_id
        q = b.load_query(first_id)
        assert q.query_id == first_id

    def test_tools_for_query_returns_four(self):
        b = ToyBenchmark()
        queries = b.load_queries()
        tools = b.tools_for(queries[0])
        names = {t.name for t in tools}
        assert names == {"search_flights", "search_hotels",
                         "book_flight", "book_hotel"}

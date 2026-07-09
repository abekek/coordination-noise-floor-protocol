"""Tests for the TravelPlanner sole-planning tools."""

from __future__ import annotations

import pytest

from benchmarks.travelplanner.tools import (
    AccommodationsTool,
    AttractionsTool,
    DistanceMatrixTool,
    FlightsTool,
    RestaurantsTool,
)


@pytest.fixture
def ref_info():
    return [
        {"Description": "Flight from Washington to Myrtle Beach on 2022-03-13",
         "Content": "Flight Number,Price,DepTime\nF001,$150,08:00\nF002,$180,12:00"},
        {"Description": "Self-driving from Washington to Myrtle Beach",
         "Content": "self-driving, duration: 6 hours, cost: $40"},
        {"Description": "Taxi from Washington to Myrtle Beach",
         "Content": "taxi, duration: 5 hours, cost: $200"},
        {"Description": "Restaurants in Myrtle Beach",
         "Content": "Name,Cuisine,Cost\nThe Sea House,Seafood,$30\nLa Bella,Italian,$25"},
        {"Description": "Accommodations in Myrtle Beach",
         "Content": "NAME,price,room type\nOceanside,$80,private room\nDowntown,$100,entire room"},
        {"Description": "Attractions in Myrtle Beach",
         "Content": "Name,Address\nBoardwalk,...\nAquarium,..."},
    ]


class TestFlights:
    async def test_finds_matching_flight(self, ref_info):
        tool = FlightsTool(reference_information=ref_info)
        out = await tool(origin="Washington", destination="Myrtle Beach",
                         departure_date="2022-03-13")
        assert "F001" in out["results"]
        assert "$150" in out["results"]

    async def test_returns_no_info_when_route_missing(self, ref_info):
        tool = FlightsTool(reference_information=ref_info)
        out = await tool(origin="NYC", destination="Tokyo",
                         departure_date="2022-03-13")
        assert out["results"] == "No information available."


class TestAccommodations:
    async def test_finds_city_listings(self, ref_info):
        tool = AccommodationsTool(reference_information=ref_info)
        out = await tool(city="Myrtle Beach")
        assert "Oceanside" in out["results"]
        assert "private room" in out["results"]

    async def test_unknown_city_no_info(self, ref_info):
        tool = AccommodationsTool(reference_information=ref_info)
        out = await tool(city="Tokyo")
        assert out["results"] == "No information available."


class TestRestaurants:
    async def test_finds_city_restaurants(self, ref_info):
        tool = RestaurantsTool(reference_information=ref_info)
        out = await tool(city="Myrtle Beach")
        assert "Seafood" in out["results"]


class TestAttractions:
    async def test_finds_city_attractions(self, ref_info):
        tool = AttractionsTool(reference_information=ref_info)
        out = await tool(city="Myrtle Beach")
        assert "Boardwalk" in out["results"]


class TestDistanceMatrix:
    async def test_self_driving_mode(self, ref_info):
        tool = DistanceMatrixTool(reference_information=ref_info)
        out = await tool(origin="Washington", destination="Myrtle Beach",
                         mode="self-driving")
        assert "$40" in out["results"] or "40" in out["results"]

    async def test_taxi_mode(self, ref_info):
        tool = DistanceMatrixTool(reference_information=ref_info)
        out = await tool(origin="Washington", destination="Myrtle Beach",
                         mode="taxi")
        assert "$200" in out["results"] or "200" in out["results"]

    async def test_unknown_mode(self, ref_info):
        tool = DistanceMatrixTool(reference_information=ref_info)
        out = await tool(origin="Washington", destination="Myrtle Beach",
                         mode="rocket")
        assert out["results"] == "No information available."

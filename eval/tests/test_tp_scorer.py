"""Tests for the TravelPlanner hard-constraint scorer."""

from __future__ import annotations

import json

import pytest

from benchmarks.travelplanner.scorer import ScoreReport, score_plan


def _plan_json(days_data: list[dict]) -> str:
    return json.dumps(days_data)


class TestDelivery:
    def test_valid_json_list_delivers(self):
        text = _plan_json([{"day": 1, "current_city": "Washington",
                            "transportation": "Flight F001, $150",
                            "breakfast": "Cafe Latte, $10",
                            "lunch": "La Bella Italian, $20",
                            "dinner": "Sea House Seafood, $30",
                            "accommodation": "Oceanside, $80, private room"}])
        report = score_plan(text, budget=1000, people_number=1,
                            local_constraint={})
        assert report.delivered is True

    def test_invalid_json_not_delivered(self):
        report = score_plan("not json", budget=1000, people_number=1,
                            local_constraint={})
        assert report.delivered is False

    def test_empty_list_not_delivered(self):
        report = score_plan("[]", budget=1000, people_number=1,
                            local_constraint={})
        assert report.delivered is False


class TestCost:
    def test_cost_under_budget_passes(self):
        text = _plan_json([{"day": 1, "current_city": "Washington",
                            "transportation": "Flight F001, $150",
                            "breakfast": "Cafe $10",
                            "lunch": "Bistro $20",
                            "dinner": "Sea House $30",
                            "accommodation": "Oceanside $80 per night"}])
        report = score_plan(text, budget=400, people_number=1,
                            local_constraint={})
        assert report.constraints["valid_cost"] == "pass"

    def test_cost_over_budget_fails(self):
        text = _plan_json([{"day": 1, "current_city": "Washington",
                            "transportation": "Flight F001, $500",
                            "breakfast": "Cafe $100",
                            "lunch": "Bistro $200",
                            "dinner": "Sea House $300",
                            "accommodation": "Oceanside $400"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={})
        assert report.constraints["valid_cost"] == "fail"


class TestCuisine:
    def test_required_cuisines_present(self):
        text = _plan_json([{"day": 1, "current_city": "Washington",
                            "transportation": "Flight $50",
                            "breakfast": "Pancake House",
                            "lunch": "Italian Bistro, Italian, $20",
                            "dinner": "Sushi Place, Japanese, $30",
                            "accommodation": "Oceanside $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"cuisine": ["Italian", "Japanese"]})
        assert report.constraints["valid_cuisine"] == "pass"

    def test_required_cuisine_missing_fails(self):
        text = _plan_json([{"day": 1, "current_city": "Washington",
                            "transportation": "Flight $50",
                            "breakfast": "Cafe",
                            "lunch": "Italian Bistro, Italian, $20",
                            "dinner": "Burger Joint, American, $30",
                            "accommodation": "Oceanside $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"cuisine": ["Italian", "Japanese"]})
        assert report.constraints["valid_cuisine"] == "fail"

    def test_no_cuisine_constraint_is_na(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "Cafe", "lunch": "B", "dinner": "C",
                            "accommodation": "Z $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"cuisine": None})
        assert report.constraints["valid_cuisine"] == "na"


class TestRoomType:
    def test_matching_room_type_passes(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Oceanside, private room, $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"room type": "private room"})
        assert report.constraints["valid_room_type"] == "pass"

    def test_wrong_room_type_fails(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Oceanside, shared room, $50"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"room type": "private room"})
        assert report.constraints["valid_room_type"] == "fail"


class TestTransportation:
    def test_no_flight_with_flight_fails(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight F001, $200",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Z $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"transportation": "no flight"})
        assert report.constraints["valid_transportation"] == "fail"

    def test_no_flight_without_flight_passes(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Self-driving, $40",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Z $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"transportation": "no flight"})
        assert report.constraints["valid_transportation"] == "pass"


class TestRoomRule:
    def test_no_parties_with_violation_fails(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Z, parties allowed, $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"house rule": "no parties"})
        assert report.constraints["valid_room_rule"] == "fail"

    def test_no_parties_without_violation_passes(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "C", "lunch": "L", "dinner": "D",
                            "accommodation": "Z, family-friendly, $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={"house rule": "no parties"})
        assert report.constraints["valid_room_rule"] == "pass"


class TestOverall:
    def test_overall_pass_requires_all_constraints(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $50",
                            "breakfast": "Cafe $10",
                            "lunch": "Bistro Italian $20",
                            "dinner": "Sea House $30",
                            "accommodation": "Z, private room, family-friendly, $80"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={
                                "cuisine": ["Italian"],
                                "room type": "private room",
                                "house rule": "no parties",
                                "transportation": None,
                            })
        assert report.completed is True

    def test_overall_fail_when_any_constraint_fails(self):
        text = _plan_json([{"day": 1, "current_city": "X",
                            "transportation": "Flight $500",
                            "breakfast": "Cafe $300",
                            "lunch": "Bistro Italian $300",
                            "dinner": "Sea House $300",
                            "accommodation": "Z, private room, $400"}])
        report = score_plan(text, budget=500, people_number=1,
                            local_constraint={
                                "cuisine": ["Italian"],
                                "room type": "private room",
                                "house rule": None,
                                "transportation": None,
                            })
        # Cost should fail
        assert report.completed is False

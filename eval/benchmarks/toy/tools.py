"""Four tools for the MiniTrip benchmark, bound to a per-trial seed.

The seed determines which flights/hotels are available — this is what
makes the same query exhibit different failure modes across trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.toy.data import FLIGHTS, HOTELS


@dataclass
class SearchFlightsTool:
    name: str = "search_flights"
    description: str = "Search for flights between two cities. Returns availability."
    seed: int = 42

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["origin", "destination"],
        }

    async def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin = kwargs["origin"]
        destination = kwargs["destination"]
        return [
            {"id": f["id"], "airline": f["airline"], "price": f["price"],
             "available": self.seed in f["available_in_seeds"]}
            for f in FLIGHTS
            if f["origin"] == origin and f["destination"] == destination
        ]


@dataclass
class SearchHotelsTool:
    name: str = "search_hotels"
    description: str = "Search for hotels in a city. Returns availability."
    seed: int = 42

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    async def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        city = kwargs["city"]
        return [
            {"id": h["id"], "name": h["name"], "nightly_rate": h["nightly_rate"],
             "available": self.seed in h["available_in_seeds"]}
            for h in HOTELS
            if h["city"] == city
        ]


@dataclass
class BookFlightTool:
    name: str = "book_flight"
    description: str = "Attempt to book a specific flight by id."
    seed: int = 42

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"flight_id": {"type": "string"}},
            "required": ["flight_id"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        fid = kwargs["flight_id"]
        for f in FLIGHTS:
            if f["id"] == fid:
                if self.seed in f["available_in_seeds"]:
                    return {"success": True, "flight_id": fid, "price": f["price"]}
                return {"success": False, "error": f"flight {fid} unavailable"}
        return {"success": False, "error": f"unknown flight_id {fid}"}


@dataclass
class BookHotelTool:
    name: str = "book_hotel"
    description: str = "Attempt to book a specific hotel by id for N nights."
    seed: int = 42

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "hotel_id": {"type": "string"},
                "nights": {"type": "integer", "minimum": 1},
            },
            "required": ["hotel_id", "nights"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        hid = kwargs["hotel_id"]
        nights = int(kwargs["nights"])
        for h in HOTELS:
            if h["id"] == hid:
                if self.seed in h["available_in_seeds"]:
                    total = h["nightly_rate"] * nights
                    return {"success": True, "hotel_id": hid, "total_cost": total}
                return {"success": False, "error": f"hotel {hid} unavailable"}
        return {"success": False, "error": f"unknown hotel_id {hid}"}

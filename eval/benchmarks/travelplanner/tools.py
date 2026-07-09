"""Sole-planning tools for TravelPlanner.

Each tool looks up data in the per-query `reference_information` list
instead of calling external APIs. The lookup is a case-insensitive
substring match on the entry's Description field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_NO_INFO = {"results": "No information available."}


def _find(ref_info: list[dict[str, str]], *needles: str) -> str | None:
    """Return Content of the first entry whose Description contains ALL needles (case-insensitive)."""
    lows = [n.lower() for n in needles]
    for entry in ref_info:
        desc = entry.get("Description", "").lower()
        if all(n in desc for n in lows):
            return entry.get("Content", "")
    return None


@dataclass
class FlightsTool:
    name: str = "Flights"
    description: str = (
        "Search for flights between two cities on a specific date. "
        "Args: origin (city), destination (city), departure_date (YYYY-MM-DD)."
    )
    reference_information: list[dict[str, str]] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "departure_date": {"type": "string"},
            },
            "required": ["origin", "destination", "departure_date"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        content = _find(self.reference_information,
                        "flight", "from", kwargs["origin"],
                        kwargs["destination"], kwargs["departure_date"])
        return {"results": content} if content else dict(_NO_INFO)


@dataclass
class AccommodationsTool:
    name: str = "Accommodations"
    description: str = "List accommodations in a city. Args: city."
    reference_information: list[dict[str, str]] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        content = _find(self.reference_information,
                        "accommodations", "in", kwargs["city"])
        return {"results": content} if content else dict(_NO_INFO)


@dataclass
class RestaurantsTool:
    name: str = "Restaurants"
    description: str = "List restaurants in a city. Args: city."
    reference_information: list[dict[str, str]] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        content = _find(self.reference_information,
                        "restaurants", "in", kwargs["city"])
        return {"results": content} if content else dict(_NO_INFO)


@dataclass
class AttractionsTool:
    name: str = "Attractions"
    description: str = "List attractions in a city. Args: city."
    reference_information: list[dict[str, str]] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        content = _find(self.reference_information,
                        "attractions", "in", kwargs["city"])
        return {"results": content} if content else dict(_NO_INFO)


@dataclass
class DistanceMatrixTool:
    name: str = "DistanceMatrix"
    description: str = (
        "Get travel info between two cities. Args: origin (city), "
        "destination (city), mode (one of: self-driving, taxi)."
    )
    reference_information: list[dict[str, str]] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"type": "string",
                         "enum": ["self-driving", "taxi"]},
            },
            "required": ["origin", "destination", "mode"],
        }

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs["mode"].lower()
        if mode not in ("self-driving", "taxi"):
            return dict(_NO_INFO)
        content = _find(self.reference_information,
                        mode, "from", kwargs["origin"], kwargs["destination"])
        return {"results": content} if content else dict(_NO_INFO)

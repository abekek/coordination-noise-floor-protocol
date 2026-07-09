"""ToyBenchmark: BenchmarkProtocol implementation for MiniTrip."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from benchmarks.base import BenchmarkProtocol, BenchmarkQuery, BenchmarkResult
from benchmarks.toy.data import FLIGHTS, HOTELS, QUERIES, get_query
from benchmarks.toy.scorer import check_completion
from benchmarks.toy.tools import (
    BookFlightTool, BookHotelTool, SearchFlightsTool, SearchHotelsTool,
)


@dataclass
class ToyBenchmark(BenchmarkProtocol):
    name: str = "toy"

    def load_queries(self, *, subset: str | None = None) -> list[BenchmarkQuery]:
        out: list[BenchmarkQuery] = []
        for q in QUERIES:
            if subset is not None and q["difficulty"] != subset:
                continue
            out.append(self._to_benchmark_query(q))
        return out

    def load_query(self, query_id: str) -> BenchmarkQuery:
        return self._to_benchmark_query(get_query(query_id))

    def tools_for(self, query: BenchmarkQuery) -> list[Any]:
        seed = int(query.payload.get("seed", 42))
        return [
            SearchFlightsTool(seed=seed),
            SearchHotelsTool(seed=seed),
            BookFlightTool(seed=seed),
            BookHotelTool(seed=seed),
        ]

    def score(self, query: BenchmarkQuery, output: Any) -> BenchmarkResult:
        text = str(output)
        flight_id, hotel_id = _extract_bookings(text)
        flight_price = _lookup_flight_price(flight_id)
        nights = int(query.payload.get("nights", 2))
        hotel_total = _lookup_hotel_total(hotel_id, nights)
        budget = float(query.payload["budget"])
        completed = check_completion(
            flight_id=flight_id, hotel_id=hotel_id,
            flight_price=flight_price, hotel_total=hotel_total, budget=budget,
        )
        return BenchmarkResult(
            query_id=query.query_id, completed=completed, raw_output=text,
        )

    def _to_benchmark_query(self, q: dict[str, Any]) -> BenchmarkQuery:
        text = (
            f"Book a flight + hotel for a 2-night trip from {q['origin']} to "
            f"{q['destination']}. Total cost must be ≤ ${q['budget']}. "
            f"Both bookings must succeed. When done, output your final "
            f"booking IDs inside <final_answer>...</final_answer> tags "
            f"with flight_id and hotel_id clearly labeled."
        )
        return BenchmarkQuery(
            query_id=q["id"],
            payload={**q, "text": text, "nights": 2},
            difficulty=q["difficulty"],
        )


_FLIGHT_RE = re.compile(r"\b(F\d+)\b")
_HOTEL_RE = re.compile(r"\b(H_[A-Z]+_\d+)\b")


def _extract_bookings(text: str) -> tuple[str | None, str | None]:
    f = _FLIGHT_RE.search(text)
    h = _HOTEL_RE.search(text)
    return (f.group(1) if f else None, h.group(1) if h else None)


def _lookup_flight_price(flight_id: str | None) -> float:
    if flight_id is None:
        return 0.0
    for f in FLIGHTS:
        if f["id"] == flight_id:
            return float(f["price"])
    return 0.0


def _lookup_hotel_total(hotel_id: str | None, nights: int) -> float:
    if hotel_id is None:
        return 0.0
    for h in HOTELS:
        if h["id"] == hotel_id:
            return float(h["nightly_rate"] * nights)
    return 0.0

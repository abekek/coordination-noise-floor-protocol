"""TravelPlannerBenchmark: BenchmarkProtocol implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.base import BenchmarkProtocol, BenchmarkQuery, BenchmarkResult
from benchmarks.travelplanner.data import (
    TpQuery,
    get_validation_query,
    load_validation_queries,
)
from benchmarks.travelplanner.scorer import score_plan
from benchmarks.travelplanner.tools import (
    AccommodationsTool,
    AttractionsTool,
    DistanceMatrixTool,
    FlightsTool,
    RestaurantsTool,
)


_TASK_PROMPT_TEMPLATE = """Travel-planning task (TravelPlanner benchmark).

User's request:
{user_query}

Constraints:
- Trip length: {days} days
- Departure city: {org}
- Destination(s): {dest}
- Travel dates: {dates}
- Budget: ${budget} total for {people_number} person(s)
- Local constraints: {local_constraint}

You have access to 5 tools: Flights, Accommodations, Restaurants, Attractions, DistanceMatrix.

Produce a JSON list of day-plans. Each day-plan is a dict with these keys:
  day: int (1-indexed)
  current_city: str
  transportation: str (include cost like "Flight F001, $150" or "Self-driving, $40")
  breakfast: str (include cuisine type and cost like "Cafe Latte, American, $10")
  lunch: str (include cuisine and cost)
  dinner: str (include cuisine and cost)
  accommodation: str (include room type and per-night cost like "Oceanside, private room, $80")

Include all dollar amounts inline so the scorer can total them. Stay within the budget."""


@dataclass
class TravelPlannerBenchmark(BenchmarkProtocol):
    name: str = "travelplanner"

    def load_queries(self, *, subset: str | None = None) -> list[BenchmarkQuery]:
        out: list[BenchmarkQuery] = []
        for q in load_validation_queries():
            if subset is not None and q.level != subset:
                continue
            out.append(self._to_benchmark_query(q))
        return out

    def load_query(self, query_id: str) -> BenchmarkQuery:
        return self._to_benchmark_query(get_validation_query(query_id))

    def tools_for(self, query: BenchmarkQuery) -> list[Any]:
        ref_info = query.payload["reference_information"]
        return [
            FlightsTool(reference_information=ref_info),
            AccommodationsTool(reference_information=ref_info),
            RestaurantsTool(reference_information=ref_info),
            AttractionsTool(reference_information=ref_info),
            DistanceMatrixTool(reference_information=ref_info),
        ]

    def score(self, query: BenchmarkQuery, output: Any) -> BenchmarkResult:
        report = score_plan(
            str(output),
            budget=query.payload["budget"],
            people_number=query.payload["people_number"],
            local_constraint=query.payload["local_constraint"],
        )
        return BenchmarkResult(
            query_id=query.query_id,
            completed=report.completed,
            raw_output={
                "text": str(output),
                "delivered": report.delivered,
                "constraints": report.constraints,
                "total_cost": report.total_cost,
                "error": report.error,
            },
        )

    def _to_benchmark_query(self, tp: TpQuery) -> BenchmarkQuery:
        text = _TASK_PROMPT_TEMPLATE.format(
            user_query=tp.query,
            days=tp.days,
            org=tp.org,
            dest=tp.dest,
            dates=", ".join(tp.dates),
            budget=tp.budget,
            people_number=tp.people_number,
            local_constraint=_render_local_constraint(tp.local_constraint),
        )
        return BenchmarkQuery(
            query_id=tp.query_id,
            payload={
                "text": text,
                "budget": tp.budget,
                "people_number": tp.people_number,
                "local_constraint": tp.local_constraint,
                "reference_information": tp.reference_information,
                "level": tp.level,
                "days": tp.days,
            },
            difficulty=tp.level,
        )


def _render_local_constraint(local: dict[str, Any]) -> str:
    active = [(k, v) for k, v in local.items() if v is not None]
    if not active:
        return "none"
    return "; ".join(f"{k}: {v}" for k, v in active)

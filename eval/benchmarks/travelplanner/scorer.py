"""TravelPlanner hard-constraint scorer for sole-planning mode.

The agent outputs a JSON list of day-plans inside <final_answer> tags
(extracted by the orchestration layer before this scorer runs). Each
day-plan has the keys: day, current_city, transportation, breakfast,
lunch, dinner, accommodation.

We check 5 hard constraints, each returning "pass", "fail", or "na".
Overall completion requires delivery + every non-na constraint to pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_COST_RE = re.compile(r"\$(\d+(?:\.\d+)?)")


@dataclass
class ScoreReport:
    delivered: bool
    completed: bool
    plan: list[dict[str, Any]]
    constraints: dict[str, str] = field(default_factory=dict)
    total_cost: float | None = None
    error: str | None = None


def score_plan(
    text: str, *, budget: int, people_number: int,
    local_constraint: dict[str, Any],
) -> ScoreReport:
    plan, parse_err = _parse_plan(text)
    if plan is None:
        return ScoreReport(delivered=False, completed=False, plan=[],
                           error=parse_err)
    if not plan:
        return ScoreReport(delivered=False, completed=False, plan=[],
                           error="empty plan list")

    constraints: dict[str, str] = {}
    total_cost = _sum_costs(plan)
    budget_total = budget * max(1, people_number)
    constraints["valid_cost"] = "pass" if total_cost <= budget_total else "fail"

    constraints["valid_cuisine"] = _check_cuisine(plan, local_constraint.get("cuisine"))
    constraints["valid_room_rule"] = _check_room_rule(plan, local_constraint.get("house rule"))
    constraints["valid_transportation"] = _check_transport(plan, local_constraint.get("transportation"))
    constraints["valid_room_type"] = _check_room_type(plan, local_constraint.get("room type"))

    completed = all(v in ("pass", "na") for v in constraints.values())
    return ScoreReport(
        delivered=True, completed=completed, plan=plan,
        constraints=constraints, total_cost=total_cost,
    )


def _parse_plan(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    text = text.strip()
    # Try to extract from possible code-fence wrapping
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json parse error: {exc}"
    if not isinstance(obj, list):
        return None, f"expected list, got {type(obj).__name__}"
    if not all(isinstance(d, dict) for d in obj):
        return None, "expected list of dicts"
    return obj, None


def _sum_costs(plan: list[dict[str, Any]]) -> float:
    total = 0.0
    for day in plan:
        for field_name in ("transportation", "breakfast", "lunch", "dinner",
                            "accommodation"):
            value = str(day.get(field_name, ""))
            for m in _COST_RE.findall(value):
                try:
                    total += float(m)
                except ValueError:
                    pass
    return total


def _check_cuisine(plan: list[dict[str, Any]], constraint) -> str:
    if not constraint:
        return "na"
    cuisines = [c.lower() for c in constraint] if isinstance(constraint, list) else [str(constraint).lower()]
    pooled = " ".join(
        str(day.get(meal, "")).lower()
        for day in plan for meal in ("breakfast", "lunch", "dinner")
    )
    return "pass" if all(c in pooled for c in cuisines) else "fail"


_ROOM_RULE_KEYWORDS = {
    "no parties": ["party", "parties"],
    "no smoking": ["smoking", "smoke"],
    "no children under 10": ["children", "kid"],
    "no visitors": ["visitor", "guest"],
    "no pets": ["pet", "dog", "cat"],
}


def _check_room_rule(plan: list[dict[str, Any]], constraint) -> str:
    if not constraint:
        return "na"
    keywords = _ROOM_RULE_KEYWORDS.get(str(constraint).lower())
    if not keywords:
        return "na"  # unknown rule, can't check
    # Fail if any accommodation mentions the rule violation alongside a
    # positive marker ("allowed", "permitted", etc.)
    pooled = " ".join(str(day.get("accommodation", "")).lower() for day in plan)
    for keyword in keywords:
        for positive in ("allowed", "permitted", "ok", "welcome"):
            if keyword in pooled and positive in pooled:
                return "fail"
    return "pass"


def _check_transport(plan: list[dict[str, Any]], constraint) -> str:
    if not constraint:
        return "na"
    constraint_lower = str(constraint).lower()
    if "no flight" in constraint_lower:
        for day in plan:
            if "flight" in str(day.get("transportation", "")).lower():
                return "fail"
        return "pass"
    if "no self-driving" in constraint_lower:
        for day in plan:
            if "self-driving" in str(day.get("transportation", "")).lower():
                return "fail"
        return "pass"
    return "na"


def _check_room_type(plan: list[dict[str, Any]], constraint) -> str:
    if not constraint:
        return "na"
    needle = str(constraint).lower()
    pooled = " ".join(str(day.get("accommodation", "")).lower() for day in plan)
    return "pass" if needle in pooled else "fail"

"""HuggingFace data loader for the TravelPlanner validation split.

Records have several string fields that are Python literals (lists,
dicts) — we parse them into native types with ast.literal_eval so
downstream code never deals with raw strings.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class TpQuery:
    """One parsed TravelPlanner validation query."""
    query_id: str
    org: str
    dest: str
    days: int
    visiting_city_number: int
    dates: list[str]
    local_constraint: dict[str, Any]
    budget: int
    query: str
    level: str
    reference_information: list[dict[str, str]]
    people_number: int


@lru_cache(maxsize=1)
def load_validation_queries() -> list[TpQuery]:
    """Load all validation queries from HuggingFace, cached after first call."""
    from datasets import load_dataset

    ds = load_dataset("osunlp/TravelPlanner", "validation",
                      split="validation")
    out: list[TpQuery] = []
    for idx, rec in enumerate(ds):
        out.append(_parse_record(rec, idx))
    return out


def get_validation_query(query_id: str) -> TpQuery:
    for q in load_validation_queries():
        if q.query_id == query_id:
            return q
    raise KeyError(f"unknown TravelPlanner query_id {query_id!r}")


def _parse_record(rec: dict[str, Any], idx: int) -> TpQuery:
    dates = _safe_literal_eval(rec["date"], default=[])
    local = _safe_literal_eval(rec["local_constraint"], default={})
    refinfo = _safe_literal_eval(rec["reference_information"], default=[])
    # Sanitize local_constraint: ensure all 4 keys exist
    for k in ("house rule", "cuisine", "room type", "transportation"):
        local.setdefault(k, None)
    return TpQuery(
        query_id=f"tp_val_{idx:03d}",
        org=rec["org"],
        dest=rec["dest"],
        days=int(rec["days"]),
        visiting_city_number=int(rec["visiting_city_number"]),
        dates=list(dates),
        local_constraint=dict(local),
        budget=int(rec["budget"]),
        query=rec["query"],
        level=rec["level"],
        reference_information=list(refinfo),
        people_number=int(rec.get("people_number", 1)),
    )


def _safe_literal_eval(s: str, *, default: Any) -> Any:
    if s is None:
        return default
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return default

"""Aggregate trial JSONL into per-cell and per-condition statistics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CellStats:
    """Stats for one (condition, benchmark, query_id) cell across seeds."""
    condition: str
    benchmark: str
    query_id: str
    n: int                           # number of trials (seeds)
    completion_rate: float           # fraction completed
    mean_input_tokens: float
    mean_output_tokens: float
    mean_total_tokens: float
    mean_redundant_call_rate: float
    mean_trace_events: float
    mean_wall_time_s: float
    error_rate: float                # fraction with non-null error


def load_trials(results_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Yield each trial row from results_dir/trials.jsonl."""
    path = Path(results_dir) / "trials.jsonl"
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_trials(results_dir: Path | str) -> list[CellStats]:
    """Group trials by (condition, benchmark, query_id), compute CellStats."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in load_trials(results_dir):
        key = (
            trial["spec"]["condition"],
            trial["spec"]["benchmark"],
            trial["spec"]["query_id"],
        )
        groups[key].append(trial)
    return [_cell_stats(k, trials) for k, trials in groups.items()]


def per_condition_summary(results_dir: Path | str) -> dict[str, dict[str, float]]:
    """Pool across all queries+seeds per condition. Returns
    {condition: {metric: value}}.
    """
    cells = aggregate_trials(results_dir)
    by_cond: dict[str, list[CellStats]] = defaultdict(list)
    for c in cells:
        by_cond[c.condition].append(c)
    out: dict[str, dict[str, float]] = {}
    for cond, cs in by_cond.items():
        # weight each cell equally (averages across queries first, then
        # across queries) — this matches the standard "averaged-then-averaged"
        # convention so a query with more seeds doesn't dominate.
        out[cond] = {
            "completion_rate": _mean([c.completion_rate for c in cs]),
            "mean_input_tokens": _mean([c.mean_input_tokens for c in cs]),
            "mean_output_tokens": _mean([c.mean_output_tokens for c in cs]),
            "mean_total_tokens": _mean([c.mean_total_tokens for c in cs]),
            "mean_redundant_call_rate": _mean([c.mean_redundant_call_rate for c in cs]),
            "mean_trace_events": _mean([c.mean_trace_events for c in cs]),
            "mean_wall_time_s": _mean([c.mean_wall_time_s for c in cs]),
            "n_queries": len(cs),
        }
    return out


def _cell_stats(key: tuple[str, str, str], trials: list[dict[str, Any]]) -> CellStats:
    cond, bench, qid = key
    n = len(trials)
    return CellStats(
        condition=cond, benchmark=bench, query_id=qid, n=n,
        completion_rate=_mean([1.0 if t["completed"] else 0.0 for t in trials]),
        mean_input_tokens=_mean([t["metrics"].get("input_tokens", 0) for t in trials]),
        mean_output_tokens=_mean([t["metrics"].get("output_tokens", 0) for t in trials]),
        mean_total_tokens=_mean([
            t["metrics"].get("input_tokens", 0) + t["metrics"].get("output_tokens", 0)
            for t in trials
        ]),
        mean_redundant_call_rate=_mean([
            t["metrics"].get("redundant_call_rate", 0.0) for t in trials
        ]),
        mean_trace_events=_mean([
            t["metrics"].get("trace_event_count", 0) for t in trials
        ]),
        mean_wall_time_s=_mean([t.get("wall_time_s", 0.0) for t in trials]),
        error_rate=_mean([1.0 if t.get("error") else 0.0 for t in trials]),
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

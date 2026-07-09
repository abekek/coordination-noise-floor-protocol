"""Tests for the JSONL → cell-statistics aggregator."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.aggregate import (
    CellStats,
    aggregate_trials,
    load_trials,
    per_condition_summary,
)


def _make_trials(tmp_path: Path, rows: list[dict]) -> Path:
    f = tmp_path / "trials.jsonl"
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def _row(condition: str, benchmark: str, query_id: str, seed: int,
         completed: bool, input_tokens: int, output_tokens: int,
         redundant: float = 0.0, trace_events: int = 0):
    return {
        "trial_id": f"{condition}_{query_id}_{seed}",
        "spec": {"condition": condition, "benchmark": benchmark,
                 "query_id": query_id, "seed": seed},
        "completed": completed,
        "metrics": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "redundant_call_rate": redundant,
            "trace_event_count": trace_events,
        },
        "wall_time_s": 10.0,
        "transcript_path": "x.json",
        "error": None,
        "timestamp": "2026-05-26T00:00:00Z",
    }


class TestLoad:
    def test_load_returns_list_of_dicts(self, tmp_path):
        _make_trials(tmp_path, [_row("B1", "toy", "q1", 42, True, 100, 20)])
        trials = list(load_trials(tmp_path))
        assert len(trials) == 1
        assert trials[0]["spec"]["condition"] == "B1"

    def test_load_skips_blank_lines(self, tmp_path):
        f = tmp_path / "trials.jsonl"
        f.write_text("\n" + json.dumps(_row("B1", "toy", "q1", 42, True, 100, 20)) + "\n\n")
        trials = list(load_trials(tmp_path))
        assert len(trials) == 1


class TestAggregate:
    def test_groups_by_condition_benchmark_query(self, tmp_path):
        rows = [
            _row("B1", "toy", "q1", 1, True, 100, 20),
            _row("B1", "toy", "q1", 2, False, 110, 25),
            _row("B1", "toy", "q1", 3, True, 105, 22),
            _row("B2", "toy", "q1", 1, True, 80, 15),
            _row("B1", "toy", "q2", 1, True, 200, 40),
        ]
        _make_trials(tmp_path, rows)
        cells = aggregate_trials(tmp_path)
        # 3 cells: (B1, toy, q1), (B2, toy, q1), (B1, toy, q2)
        assert len(cells) == 3
        b1_q1 = next(c for c in cells if c.condition == "B1" and c.query_id == "q1")
        assert b1_q1.n == 3
        assert b1_q1.completion_rate == 2 / 3
        assert b1_q1.mean_total_tokens == (120 + 135 + 127) / 3  # input+output sums

    def test_handles_failed_trials_in_aggregation(self, tmp_path):
        rows = [
            _row("B1", "toy", "q1", 1, True, 100, 20),
            _row("B1", "toy", "q1", 2, False, 0, 0),  # crashed; zero tokens
        ]
        _make_trials(tmp_path, rows)
        cells = aggregate_trials(tmp_path)
        b1 = cells[0]
        assert b1.n == 2
        assert b1.completion_rate == 0.5


class TestPerConditionSummary:
    def test_pools_across_queries(self, tmp_path):
        rows = [
            _row("B1", "toy", "q1", 1, True, 100, 20),
            _row("B1", "toy", "q2", 1, False, 110, 25),
            _row("B1", "toy", "q3", 1, True, 90, 15),
            _row("B2", "toy", "q1", 1, True, 200, 40),
        ]
        _make_trials(tmp_path, rows)
        summary = per_condition_summary(tmp_path)
        # 2 conditions
        assert set(summary.keys()) == {"B1", "B2"}
        assert summary["B1"]["completion_rate"] == 2 / 3
        assert summary["B2"]["completion_rate"] == 1.0

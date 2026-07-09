"""Tests for the analysis CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.__main__ import report_command


def _make_trials(tmp_path: Path, rows: list[dict]) -> Path:
    f = tmp_path / "trials.jsonl"
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def _row(condition: str, query_id: str, seed: int, completed: bool,
         input_tokens: int, output_tokens: int, trace_events: int = 0):
    return {
        "trial_id": f"{condition}_{query_id}_{seed}",
        "spec": {"condition": condition, "benchmark": "toy",
                 "query_id": query_id, "seed": seed},
        "completed": completed,
        "metrics": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                    "redundant_call_rate": 0.0,
                    "trace_event_count": trace_events},
        "wall_time_s": 10.0, "transcript_path": "x.json",
        "error": None, "timestamp": "2026-05-26T00:00:00Z",
    }


def test_report_writes_summary_md_and_tex(tmp_path):
    rows = [
        _row("B1_full_context", "q1", 1, True, 100, 20),
        _row("B2_summarization", "q1", 1, True, 80, 15),
        _row("B3_ca_mcp_style", "q1", 1, True, 90, 18),
        _row("ET_MCP_default", "q1", 1, True, 95, 22, trace_events=2),
    ]
    _make_trials(tmp_path, rows)
    exit_code = report_command(results_dir=tmp_path, output_dir=tmp_path / "report")
    assert exit_code == 0
    assert (tmp_path / "report" / "summary.md").exists()
    assert (tmp_path / "report" / "headline.tex").exists()
    assert (tmp_path / "report" / "pairwise_tokens.tex").exists()


def test_report_handles_missing_jsonl(tmp_path):
    exit_code = report_command(results_dir=tmp_path, output_dir=tmp_path / "out")
    # Should not crash, just produce empty report
    assert exit_code in (0, 1)


def test_report_summary_contains_per_condition_stats(tmp_path):
    rows = [
        _row("B1_full_context", "q1", 1, True, 100, 20),
        _row("ET_MCP_default", "q1", 1, True, 95, 22, trace_events=2),
    ]
    _make_trials(tmp_path, rows)
    report_command(results_dir=tmp_path, output_dir=tmp_path / "report")
    md = (tmp_path / "report" / "summary.md").read_text()
    assert "B1" in md or "B1_full_context" in md
    assert "ET" in md or "ET_MCP" in md

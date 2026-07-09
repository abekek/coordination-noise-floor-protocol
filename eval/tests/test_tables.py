"""Tests for LaTeX table rendering."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.tables import (
    headline_results_table,
    pairwise_comparison_table,
)


def _make_trials(tmp_path: Path, rows: list[dict]) -> Path:
    f = tmp_path / "trials.jsonl"
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def _row(condition: str, query_id: str, seed: int, completed: bool,
         input_tokens: int, output_tokens: int, redundant: float = 0.0,
         trace_events: int = 0):
    return {
        "trial_id": f"{condition}_{query_id}_{seed}",
        "spec": {"condition": condition, "benchmark": "toy",
                 "query_id": query_id, "seed": seed},
        "completed": completed,
        "metrics": {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "redundant_call_rate": redundant,
            "trace_event_count": trace_events,
        },
        "wall_time_s": 10.0, "transcript_path": "x.json",
        "error": None, "timestamp": "2026-05-26T00:00:00Z",
    }


class TestHeadlineTable:
    def test_renders_one_row_per_condition(self, tmp_path):
        rows = [
            _row("B1_full_context", "q1", 1, True, 100, 20),
            _row("B2_summarization", "q1", 1, True, 80, 15),
            _row("B3_ca_mcp_style", "q1", 1, True, 90, 18),
            _row("ET_MCP_default", "q1", 1, True, 95, 22, trace_events=2),
        ]
        _make_trials(tmp_path, rows)
        latex = headline_results_table(tmp_path)
        assert "\\begin{tabular}" in latex
        assert "B1_full_context" in latex or "B1 full context" in latex or "B1 Full context" in latex
        assert "ET-MCP" in latex or "ET\\_MCP" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_includes_caption_and_label(self, tmp_path):
        rows = [
            _row("B1_full_context", "q1", 1, True, 100, 20),
            _row("ET_MCP_default", "q1", 1, True, 95, 22),
        ]
        _make_trials(tmp_path, rows)
        latex = headline_results_table(tmp_path, caption="My caption",
                                        label="tab:my")
        assert "My caption" in latex
        assert "\\label{tab:my}" in latex

    def test_booktabs_midrule(self, tmp_path):
        rows = [_row("B1_full_context", "q1", 1, True, 100, 20)]
        _make_trials(tmp_path, rows)
        latex = headline_results_table(tmp_path)
        assert "\\midrule" in latex

    def test_empty_dir_returns_empty_table(self, tmp_path):
        latex = headline_results_table(tmp_path)
        assert "\\begin{table}" in latex
        assert "No results" in latex


class TestPairwiseTable:
    def test_renders_one_row_per_baseline(self, tmp_path):
        # 5 queries to give Wilcoxon something to work with
        rows = []
        for q in range(5):
            for cond, tokens in [
                ("B1_full_context", 1000),
                ("B2_summarization", 700),
                ("ET_MCP_default", 900),
            ]:
                rows.append(_row(cond, f"q{q}", 1, True, tokens, 100))
        _make_trials(tmp_path, rows)
        latex = pairwise_comparison_table(
            tmp_path, condition="ET_MCP_default",
            baselines=["B1_full_context", "B2_summarization"],
            metric="mean_total_tokens",
        )
        assert "\\begin{tabular}" in latex
        # Both baselines should appear
        assert "B1_full_context" in latex or "B1" in latex
        assert "B2_summarization" in latex or "B2" in latex
        # p_value and cliffs_delta columns
        assert "p" in latex
        assert "delta" in latex.lower() or "\\delta" in latex

    def test_pairwise_includes_caption_and_label(self, tmp_path):
        rows = []
        for q in range(3):
            for cond, tokens in [("B1_full_context", 1000), ("ET_MCP_default", 900)]:
                rows.append(_row(cond, f"q{q}", 1, True, tokens, 100))
        _make_trials(tmp_path, rows)
        latex = pairwise_comparison_table(
            tmp_path, condition="ET_MCP_default",
            baselines=["B1_full_context"],
            caption="Custom caption",
            label="tab:custom",
        )
        assert "Custom caption" in latex
        assert "\\label{tab:custom}" in latex

    def test_pairwise_missing_condition_returns_empty(self, tmp_path):
        rows = [_row("B1_full_context", "q1", 1, True, 100, 20)]
        _make_trials(tmp_path, rows)
        latex = pairwise_comparison_table(
            tmp_path, condition="ET_MCP_default",
            baselines=["B1_full_context"],
        )
        assert "\\begin{table}" in latex

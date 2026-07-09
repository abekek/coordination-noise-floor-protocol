"""Tests for the JSONL results writer/reader."""

from __future__ import annotations

import json
from pathlib import Path

from harness.results import read_trial_results, write_trial_result
from harness.runner import TrialResult, TrialSpec
from harness.transcript import AgentStepResult, Transcript


def _trial_result(tmp_path: Path) -> tuple[TrialResult, Transcript]:
    transcript = Transcript()
    transcript.append(AgentStepResult(
        agent_id="planner", final_text="<final_answer>booked</final_answer>",
        tool_calls=[], llm_calls=2,
        input_tokens=500, output_tokens=80, cache_read_tokens=100,
        raw_messages=[], hit_max_iterations=False,
    ))
    transcript.finalize("booked")
    result = TrialResult(
        spec=TrialSpec(condition="B1_full_context", benchmark="toy",
                       query_id="q_easy_001", seed=42),
        trial_id="01HMX_test",
        completed=True,
        metrics={"llm_calls": 2, "input_tokens": 500},
        transcript_path="",  # set by writer
        wall_time_s=5.0,
        error=None,
    )
    return result, transcript


def test_write_appends_jsonl_line(tmp_path):
    result, transcript = _trial_result(tmp_path)
    write_trial_result(result, transcript, tmp_path)
    jsonl = (tmp_path / "trials.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1
    row = json.loads(jsonl[0])
    assert row["trial_id"] == "01HMX_test"
    assert row["completed"] is True
    assert row["metrics"]["llm_calls"] == 2


def test_write_emits_transcript_file(tmp_path):
    result, transcript = _trial_result(tmp_path)
    write_trial_result(result, transcript, tmp_path)
    transcript_files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(transcript_files) == 1


def test_read_returns_iterator(tmp_path):
    result, transcript = _trial_result(tmp_path)
    write_trial_result(result, transcript, tmp_path)
    rows = list(read_trial_results(tmp_path))
    assert len(rows) == 1
    assert rows[0]["trial_id"] == "01HMX_test"

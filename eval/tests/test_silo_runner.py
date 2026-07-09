"""Tests for the Silo-Bench runner.

These tests exercise the pure-Python helpers (list_tasks, write_trial_result)
that don't require LLM calls or the full vendored engine.  They run offline
and should complete in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.silo_bench.runner import (
    SiloTrialResult,
    SiloTrialSpec,
    existing_trial_keys,
    list_tasks,
    run_sweep,
    trial_cost_usd,
    write_trial_result,
)


def test_list_tasks_returns_30():
    """All 30 task JSONs (I, II, III) should be discoverable."""
    tasks = list_tasks(n=5)
    assert len(tasks) >= 30, f"Expected >=30 tasks, got {len(tasks)}"
    assert all(t.endswith("_n5.json") for t in tasks), "All filenames must end in _n5.json"


def test_list_tasks_sorted():
    tasks = list_tasks(n=5)
    assert tasks == sorted(tasks), "list_tasks() must return a sorted list"


def test_write_trial_result_emits_jsonl(tmp_path: Path):
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="etmcp", seed=42)
    result = SiloTrialResult(
        spec=spec,
        trial_id="01HMX_test",
        completed=True,
        s_score=1.0,
        p_score=1.0,
        c_score=100.0,
        d_score=0.2,
        total_input_tokens=500,
        total_output_tokens=80,
        wall_time_s=10.0,
        error=None,
    )
    write_trial_result(result, tmp_path)
    jsonl_path = tmp_path / "trials.jsonl"
    assert jsonl_path.exists(), "trials.jsonl must be created"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1, f"Expected 1 JSONL line, got {len(lines)}"
    row = json.loads(lines[0])

    # --- spec block ---
    assert row["spec"]["condition"] == "etmcp"
    assert row["spec"]["benchmark"] == "silo_bench"
    assert row["spec"]["query_id"] == "I-01_n5"
    assert row["spec"]["seed"] == 42

    # --- top-level fields ---
    assert row["completed"] is True
    assert row["error"] is None
    assert isinstance(row["trial_id"], str)
    assert isinstance(row["wall_time_s"], float)
    assert "timestamp" in row

    # --- metrics block ---
    m = row["metrics"]
    assert m["s_score"] == 1.0
    assert m["p_score"] == 1.0
    assert m["comm_reasoning_gap"] == pytest.approx(0.0)
    assert m["input_tokens"] == 500
    assert m["output_tokens"] == 80
    assert "redundant_call_rate" in m
    assert "trace_event_count" in m


def test_write_trial_result_appends(tmp_path: Path):
    """Multiple calls must append lines, not overwrite."""
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="msg", seed=42)
    for i in range(3):
        result = SiloTrialResult(
            spec=spec,
            trial_id=f"trial-{i}",
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            wall_time_s=1.0,
            error=None,
        )
        write_trial_result(result, tmp_path)

    lines = (tmp_path / "trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3, f"Expected 3 lines after 3 writes, got {len(lines)}"


def test_write_trial_result_error_field(tmp_path: Path):
    """Error field must round-trip through JSONL."""
    spec = SiloTrialSpec(task_file="I-02_n5.json", protocol="broadcast", seed=1)
    result = SiloTrialResult(
        spec=spec,
        trial_id="err-trial",
        completed=False,
        s_score=0.0,
        p_score=0.0,
        c_score=0.0,
        d_score=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        wall_time_s=0.5,
        error="RuntimeError: something broke",
    )
    write_trial_result(result, tmp_path)
    row = json.loads((tmp_path / "trials.jsonl").read_text().strip())
    assert row["error"] == "RuntimeError: something broke"
    assert row["completed"] is False


def test_trial_cost_usd_uses_haiku_pricing():
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="msg", seed=42)
    result = SiloTrialResult(
        spec=spec,
        trial_id="cost-trial",
        completed=True,
        s_score=1.0,
        p_score=1.0,
        c_score=0.0,
        d_score=0.0,
        total_input_tokens=1_000_000,
        total_output_tokens=1_000_000,
        wall_time_s=1.0,
        error=None,
    )
    assert trial_cost_usd(result, "claude-haiku-4-5") == pytest.approx(4.8)


def test_existing_trial_keys_reads_jsonl(tmp_path: Path):
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="etmcp", seed=42)
    result = SiloTrialResult(
        spec=spec,
        trial_id="resume-trial",
        completed=True,
        s_score=1.0,
        p_score=1.0,
        c_score=0.0,
        d_score=0.0,
        total_input_tokens=10,
        total_output_tokens=5,
        wall_time_s=1.0,
        error=None,
    )
    write_trial_result(result, tmp_path)
    assert existing_trial_keys(tmp_path) == {("I-01_n5", "etmcp", 42)}


def test_existing_trial_keys_ignores_zero_token_errors(tmp_path: Path):
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="msg", seed=42)
    result = SiloTrialResult(
        spec=spec,
        trial_id="auth-error",
        completed=False,
        s_score=0.0,
        p_score=0.0,
        c_score=0.0,
        d_score=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        wall_time_s=0.1,
        error="TypeError: missing api key",
    )
    write_trial_result(result, tmp_path)
    assert existing_trial_keys(tmp_path) == set()


def test_existing_trial_keys_ignores_rate_limit_errors_with_tokens(tmp_path: Path):
    spec = SiloTrialSpec(task_file="I-01_n5.json", protocol="msg", seed=42)
    result = SiloTrialResult(
        spec=spec,
        trial_id="rate-limit-error",
        completed=False,
        s_score=0.0,
        p_score=0.0,
        c_score=0.0,
        d_score=0.0,
        total_input_tokens=100,
        total_output_tokens=50,
        wall_time_s=0.1,
        error="RateLimitError: Quota exceeded",
    )
    write_trial_result(result, tmp_path)
    assert existing_trial_keys(tmp_path) == set()


def test_run_sweep_dry_run_does_not_write(tmp_path: Path):
    results = run_sweep(
        tasks=["I-01_n5.json"],
        protocols=["msg", "etmcp"],
        seeds=[42, 43],
        results_dir=tmp_path,
        dry_run=True,
    )
    assert results == []
    assert not (tmp_path / "trials.jsonl").exists()


def test_run_sweep_respects_max_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_trial(spec: SiloTrialSpec, results_dir: Path, **kwargs) -> SiloTrialResult:
        return SiloTrialResult(
            spec=spec,
            trial_id=f"{spec.protocol}-{spec.seed}",
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=10,
            total_output_tokens=5,
            wall_time_s=0.1,
            error=None,
        )

    monkeypatch.setattr("benchmarks.silo_bench.runner.run_silo_trial", fake_trial)
    results = run_sweep(
        tasks=["I-01_n5.json", "I-02_n5.json"],
        protocols=["msg", "etmcp"],
        seeds=[42, 43],
        results_dir=tmp_path,
        max_trials=3,
    )
    assert len(results) == 3
    assert len((tmp_path / "trials.jsonl").read_text().strip().splitlines()) == 3


def test_run_sweep_does_not_write_retryable_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_trial(spec: SiloTrialSpec, results_dir: Path, **kwargs) -> SiloTrialResult:
        return SiloTrialResult(
            spec=spec,
            trial_id="quota-error",
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=100,
            total_output_tokens=50,
            wall_time_s=0.1,
            error="RateLimitError: Quota exceeded",
        )

    monkeypatch.setattr("benchmarks.silo_bench.runner.run_silo_trial", fake_trial)
    results = run_sweep(
        tasks=["I-01_n5.json"],
        protocols=["msg"],
        seeds=[42],
        results_dir=tmp_path,
    )
    assert len(results) == 1
    assert not (tmp_path / "trials.jsonl").exists()


def test_run_sweep_retries_zero_token_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = 0
    sleeps: list[float] = []

    def fake_trial(spec: SiloTrialSpec, results_dir: Path, **kwargs) -> SiloTrialResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SiloTrialResult(
                spec=spec,
                trial_id="quota-error",
                completed=False,
                s_score=0.0,
                p_score=0.0,
                c_score=0.0,
                d_score=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                wall_time_s=0.1,
                error="RateLimitError: quota exceeded",
            )
        return SiloTrialResult(
            spec=spec,
            trial_id="retry-ok",
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=10,
            total_output_tokens=5,
            wall_time_s=0.1,
            error=None,
        )

    monkeypatch.setattr("benchmarks.silo_bench.runner.run_silo_trial", fake_trial)
    monkeypatch.setattr("benchmarks.silo_bench.runner.time.sleep", sleeps.append)
    results = run_sweep(
        tasks=["I-01_n5.json"],
        protocols=["msg"],
        seeds=[42],
        results_dir=tmp_path,
        cooldown_seconds=7,
        zero_token_retries=1,
    )
    assert calls == 2
    assert len(results) == 1
    assert results[0].trial_id == "retry-ok"
    assert sleeps == [7, 7]
    assert len((tmp_path / "trials.jsonl").read_text().strip().splitlines()) == 1


def test_run_sweep_initial_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []

    def fake_trial(spec: SiloTrialSpec, results_dir: Path, **kwargs) -> SiloTrialResult:
        return SiloTrialResult(
            spec=spec,
            trial_id="ok",
            completed=False,
            s_score=0.0,
            p_score=0.0,
            c_score=0.0,
            d_score=0.0,
            total_input_tokens=1,
            total_output_tokens=1,
            wall_time_s=0.1,
            error=None,
        )

    monkeypatch.setattr("benchmarks.silo_bench.runner.run_silo_trial", fake_trial)
    monkeypatch.setattr("benchmarks.silo_bench.runner.time.sleep", sleeps.append)
    run_sweep(
        tasks=["I-01_n5.json"],
        protocols=["msg"],
        seeds=[42],
        results_dir=tmp_path,
        initial_cooldown_seconds=11,
    )
    assert sleeps == [11]

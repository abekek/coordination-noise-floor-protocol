"""Tests for the mid-run token ceiling enforcement in TrialRunner.run_many."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from baselines.full_context import FullContextBaseline
from harness.runner import TokenBudgetExceeded, TrialResult, TrialRunner, TrialSpec


class _ScriptedLLM:
    """LLM stub that always errors so each trial's metrics dict is empty.

    Sufficient for testing that ceiling logic works on the recorded
    metrics; we'll override the metrics directly via a subclass below.
    """
    async def call(self, **kwargs):
        raise RuntimeError("never called in this test")


class _FixedTokenRunner(TrialRunner):
    """Test variant of TrialRunner that produces fake results with fixed
    token counts instead of actually running trials."""
    def __init__(self, tokens_per_trial: int, results_dir: Path) -> None:
        super().__init__(
            llm=_ScriptedLLM(),
            results_dir=results_dir,
            benchmarks={},
            baseline_factory=lambda c, t, l: FullContextBaseline(),
        )
        self._tokens_per_trial = tokens_per_trial
        self._calls = 0

    async def run_trial(self, spec):
        self._calls += 1
        return TrialResult(
            spec=spec, trial_id=f"trial_{self._calls}",
            completed=True,
            metrics={"input_tokens": self._tokens_per_trial // 2,
                     "output_tokens": self._tokens_per_trial // 2},
            transcript_path="",
            wall_time_s=0.01,
            error=None,
        )


@pytest.fixture
def specs():
    return [TrialSpec(condition="B1_full_context", benchmark="toy",
                      query_id=f"q{i}", seed=42) for i in range(10)]


async def test_run_many_completes_when_under_ceiling(tmp_path, specs):
    runner = _FixedTokenRunner(tokens_per_trial=100, results_dir=tmp_path)
    results = await runner.run_many(specs, max_concurrency=1, max_total_tokens=10_000)
    assert len(results) == 10  # all 10 completed
    assert all(r.error is None for r in results)


async def test_run_many_aborts_when_ceiling_exceeded(tmp_path, specs):
    """With 1000 tokens/trial and 5500 cap, the 6th trial pushes total over
    the ceiling, and trials 7-10 must NOT run."""
    runner = _FixedTokenRunner(tokens_per_trial=1000, results_dir=tmp_path)
    with pytest.raises(TokenBudgetExceeded):
        await runner.run_many(specs, max_concurrency=1, max_total_tokens=5500)
    # The 6th trial was the one that crossed; expect at most 6 calls
    assert runner._calls <= 6


async def test_run_many_no_ceiling_runs_all(tmp_path, specs):
    """max_total_tokens=None should disable the ceiling (default behavior)."""
    runner = _FixedTokenRunner(tokens_per_trial=1000, results_dir=tmp_path)
    results = await runner.run_many(specs, max_concurrency=1)
    assert len(results) == 10

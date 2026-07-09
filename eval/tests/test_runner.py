"""Integration test for TrialRunner using a stub LLM and the toy benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from baselines.full_context import FullContextBaseline
from benchmarks.toy.adapter import ToyBenchmark
from harness.llm import LLMResponse, ToolUseBlock
from harness.runner import TrialRunner, TrialSpec


@dataclass
class _ScriptedLLM:
    """LLM that yields a deterministic 4-step success trial."""
    script: list[LLMResponse] = field(default_factory=list)

    async def call(self, *, system, messages, tools=None,
                   temperature=0.0, max_tokens=4096):
        return self.script.pop(0)


def _r(text="", tool_uses=None, inp=200, out=40):
    return LLMResponse(
        text=text, tool_uses=tool_uses or [],
        input_tokens=inp, output_tokens=out, cache_read_tokens=0,
        raw_response=None,
    )


@pytest.mark.skip(reason="Full integration test runs in smoke command")
async def test_runner_end_to_end_with_stub_llm(tmp_path):
    """Placeholder for a heavier integration test that scripts a full
    successful trial. Skipped by default because it requires scripting
    every LLM call across two agents and ~6 steps. The actual end-to-end
    coverage comes from `python -m harness smoke`, which exercises the
    real LLM."""
    pass


async def test_runner_writes_jsonl_on_error(tmp_path):
    """When the trial crashes, the runner still writes a JSONL line with
    error populated."""
    from harness.llm import AnthropicClient
    # We do NOT call the real Anthropic; we install a stub that always raises
    bench = ToyBenchmark()

    class _BoomLLM:
        async def call(self, **kwargs):
            raise RuntimeError("intentional test failure")

    runner = TrialRunner(
        llm=_BoomLLM(),  # type: ignore[arg-type]
        results_dir=tmp_path,
        benchmarks={"toy": bench},
        baseline_factory=lambda condition, task_id, llm: FullContextBaseline(),
    )
    spec = TrialSpec(condition="B1_full_context", benchmark="toy",
                     query_id="q_easy_001", seed=42)
    result = await runner.run_trial(spec)
    assert result.completed is False
    assert result.error is not None
    assert (tmp_path / "trials.jsonl").exists()

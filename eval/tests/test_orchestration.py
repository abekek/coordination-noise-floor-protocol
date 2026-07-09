"""Tests for run_two_agent_trial. Uses stub agents + fake baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from baselines.base import BaselineProtocol, HandoffContext
from benchmarks.base import BenchmarkQuery
from harness.orchestration import run_two_agent_trial
from harness.transcript import AgentStepResult, ToolCall, Transcript


@dataclass
class _ScriptedAgent:
    agent_id: str
    role: str
    script: list[AgentStepResult] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def step(self, *, task: str, handoff_context, peer_notes):
        self.calls.append({"task": task, "handoff": handoff_context, "peer_notes": peer_notes})
        return self.script.pop(0)


class _FakeBaseline(BaselineProtocol):
    name = "fake"

    def __init__(self) -> None:
        self.prepare_calls: list[str] = []
        self.consume_calls: list[str] = []
        self.query_calls: list[str] = []
        self._next_peer_note: str | None = None

    async def prepare_handoff(self, agent_id, agent_state):
        self.prepare_calls.append(agent_id)
        return HandoffContext(payload={"from": agent_id}, token_cost=10)

    async def consume_handoff(self, agent_id, context):
        self.consume_calls.append(agent_id)

    async def query_during_step(self, agent_id, question):
        self.query_calls.append(agent_id)
        return self._next_peer_note


def _step(final: str = "", agent_id: str = "p") -> AgentStepResult:
    return AgentStepResult(
        agent_id=agent_id, final_text=final,
        tool_calls=[], llm_calls=1,
        input_tokens=100, output_tokens=20, cache_read_tokens=0,
        raw_messages=[], hit_max_iterations=False,
    )


def _query(qid: str = "q1") -> BenchmarkQuery:
    return BenchmarkQuery(query_id=qid, payload={"text": "do the thing"},
                          difficulty="easy")


class TestEarlyTermination:
    async def test_planner_done_returns_immediately(self):
        planner = _ScriptedAgent("p", "planner",
            script=[_step("<final_answer>direct</final_answer>", "p")])
        executor = _ScriptedAgent("e", "executor", script=[])
        baseline = _FakeBaseline()
        transcript = await run_two_agent_trial(
            planner=planner, executor=executor, query=_query(), baseline=baseline,
        )
        assert "direct" in transcript.final_output
        assert len(planner.calls) == 1
        assert len(executor.calls) == 0

    async def test_executor_done_after_one_handoff(self):
        planner = _ScriptedAgent("p", "planner",
            script=[_step("Need executor help", "p")])
        executor = _ScriptedAgent("e", "executor",
            script=[_step("<final_answer>executor solved it</final_answer>", "e")])
        baseline = _FakeBaseline()
        transcript = await run_two_agent_trial(
            planner=planner, executor=executor, query=_query(), baseline=baseline,
        )
        assert "executor solved" in transcript.final_output
        assert len(planner.calls) == 1
        assert len(executor.calls) == 1


class TestHandoffOrdering:
    async def test_prepare_then_consume_then_query_then_step(self):
        planner = _ScriptedAgent("p", "planner",
            script=[_step("need help", "p"),
                    _step("<final_answer>done</final_answer>", "p")])
        executor = _ScriptedAgent("e", "executor",
            script=[_step("did some work", "e")])
        baseline = _FakeBaseline()
        await run_two_agent_trial(
            planner=planner, executor=executor, query=_query(), baseline=baseline,
        )
        # prepare_handoff called after EVERY step (including the final planner step)
        # so ET-MCP can record events from completing steps
        assert baseline.prepare_calls == ["planner", "executor", "planner"]
        assert baseline.consume_calls == ["executor", "planner"]
        # query_during_step called before EVERY agent step (3 calls)
        assert baseline.query_calls == ["planner", "executor", "planner"]


class TestMaxHandoffs:
    async def test_hits_max_handoffs_when_no_final_answer(self):
        # Both agents loop forever without emitting <final_answer>
        planner = _ScriptedAgent("p", "planner",
            script=[_step("step", "p")] * 20)
        executor = _ScriptedAgent("e", "executor",
            script=[_step("step", "e")] * 20)
        baseline = _FakeBaseline()
        transcript = await run_two_agent_trial(
            planner=planner, executor=executor, query=_query(),
            baseline=baseline, max_handoffs=2,
        )
        assert transcript.hit_max_handoffs is True

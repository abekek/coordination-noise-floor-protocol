"""Tests for the four baseline implementations (B1, B2, B3, ET-MCP)."""

from __future__ import annotations

import pytest

from baselines.full_context import FullContextBaseline


class TestFullContext:
    async def test_prepare_serializes_state(self):
        b = FullContextBaseline()
        h = await b.prepare_handoff(
            agent_id="planner",
            agent_state={"transcript": "did A; did B; got result R"},
        )
        assert "did A" in h.payload_as_text()

    async def test_consume_stores_per_agent(self):
        b = FullContextBaseline()
        h = await b.prepare_handoff(
            agent_id="planner", agent_state={"transcript": "X"},
        )
        await b.consume_handoff(agent_id="executor", context=h)
        # No exception is the success contract for consume.

    async def test_query_during_step_returns_none(self):
        b = FullContextBaseline()
        assert await b.query_during_step(agent_id="executor", question="?") is None


from baselines.summarization import SummarizationBaseline


class _StubLLM:
    """Returns a fixed summary regardless of input."""
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = 0

    async def call(self, *, system, messages, tools=None,
                   temperature=0.0, max_tokens=4096):
        from harness.llm import LLMResponse
        self.calls += 1
        return LLMResponse(
            text=self.summary, tool_uses=[],
            input_tokens=200, output_tokens=80,
            cache_read_tokens=0, raw_response=None,
        )


class TestSummarization:
    async def test_prepare_calls_llm_and_stores_summary(self):
        llm = _StubLLM(summary="planner did A then B")
        b = SummarizationBaseline(llm=llm)
        h = await b.prepare_handoff(
            agent_id="planner",
            agent_state={"transcript": "long transcript ..." * 100},
        )
        assert llm.calls == 1
        assert "planner did A" in h.payload_as_text()
        # Summary is shorter than original
        assert len(h.payload_as_text()) < len("long transcript ..." * 100)

    async def test_query_during_step_returns_none(self):
        llm = _StubLLM(summary="x")
        b = SummarizationBaseline(llm=llm)
        assert await b.query_during_step(agent_id="e", question="?") is None


from baselines.ca_mcp_style import CaMcpStyleBaseline


class TestCaMcpStyle:
    async def test_prepare_pushes_structured_state(self):
        b = CaMcpStyleBaseline(task_id="t1")
        await b.prepare_handoff(
            agent_id="planner",
            agent_state={"tools_called": ["search"], "last_results": [{"r": 1}],
                         "current_step": "planning hotel"},
        )
        snapshot = b.state_store["t1"]
        assert snapshot["tools_called"] == ["search"]
        assert snapshot["current_step"] == "planning hotel"

    async def test_consume_reads_shared_snapshot(self):
        b = CaMcpStyleBaseline(task_id="t1")
        await b.prepare_handoff(
            agent_id="planner",
            agent_state={"tools_called": ["a"], "last_results": [],
                         "current_step": "step_x"},
        )
        h = await b._build_consume_context()  # construct what consume sees
        assert "tools_called" in h.payload_as_text()
        assert "step_x" in h.payload_as_text()

    async def test_query_during_step_returns_none(self):
        b = CaMcpStyleBaseline(task_id="t1")
        assert await b.query_during_step(agent_id="e", question="?") is None


from baselines.et_mcp import EtMcpBaseline
from et_mcp.client.policy import AgentStep, Policy
from et_mcp.events import FailedPath


class TestEtMcp:
    async def test_setup_initializes_task_namespace(self):
        b = EtMcpBaseline(task_id="t_eval_1")
        await b.setup()
        assert "t_eval_1" in b.server.store.registered_tasks()
        await b.teardown()

    async def test_record_via_node_writes_event(self):
        b = EtMcpBaseline(task_id="t1")
        await b.setup()
        node = b.node_for("planner")
        step = AgentStep(
            outcome="failed",
            failed_path=FailedPath(
                approach="book flight X", reason="unavailable",
                constraints_hit=[], steps_taken=[],
            ),
        )
        await node.record(step)
        events = await b.server.store.list_for_task("t1")
        assert len(events) == 1
        await b.teardown()

    async def test_query_during_step_returns_peer_summary(self):
        b = EtMcpBaseline(task_id="t1")
        await b.setup()
        planner = b.node_for("planner")
        await planner.record(AgentStep(
            outcome="failed",
            failed_path=FailedPath(
                approach="route via Eurostar", reason="no weekend service",
                constraints_hit=[], steps_taken=[],
            ),
        ))
        peer_notes = await b.query_during_step(
            agent_id="executor", question="what paths failed?",
        )
        assert peer_notes is not None
        assert peer_notes != ""
        assert "Eurostar" in peer_notes or "FAILED_PATH" in peer_notes
        await b.teardown()

    async def test_policy_default_is_failure_only(self):
        b = EtMcpBaseline(task_id="t1")
        assert b.policy == Policy.FAILURE_ONLY
        await b.setup()
        await b.teardown()

"""Tests for the transcript dataclasses."""

from __future__ import annotations

from harness.transcript import (
    AgentStepResult,
    HandoffContext,
    ToolCall,
    Transcript,
)


def _step(agent_id: str = "planner", text: str = "ok",
          input_tokens: int = 100, output_tokens: int = 20,
          tool_calls: list[ToolCall] | None = None,
          llm_calls: int = 1) -> AgentStepResult:
    return AgentStepResult(
        agent_id=agent_id,
        final_text=text,
        tool_calls=tool_calls or [],
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        raw_messages=[],
        hit_max_iterations=False,
    )


class TestToolCall:
    def test_minimal_construction(self):
        tc = ToolCall(tool_name="search", input={"q": "x"}, output={"r": 1},
                      errored=False, latency_ms=42.0)
        assert tc.tool_name == "search"
        assert tc.input == {"q": "x"}


class TestHandoffContext:
    def test_empty_payload(self):
        h = HandoffContext(payload={}, token_cost=0)
        assert h.token_cost == 0

    def test_payload_as_text_returns_string(self):
        h = HandoffContext(payload={"summary": "hello"}, token_cost=50)
        s = h.payload_as_text()
        assert isinstance(s, str)
        assert "summary" in s and "hello" in s


class TestTranscript:
    def test_append_and_totals(self):
        t = Transcript()
        t.append(_step(input_tokens=100, output_tokens=20, llm_calls=1))
        t.append(_step(input_tokens=200, output_tokens=30, llm_calls=2))
        assert t.input_tokens_total() == 300
        assert t.output_tokens_total() == 50
        assert t.llm_calls_total() == 3

    def test_tool_calls_flat_list(self):
        t = Transcript()
        tc_a = ToolCall(tool_name="a", input={}, output={}, errored=False, latency_ms=1.0)
        tc_b = ToolCall(tool_name="b", input={}, output={}, errored=True, latency_ms=2.0)
        t.append(_step(tool_calls=[tc_a]))
        t.append(_step(tool_calls=[tc_b]))
        flat = t.tool_calls()
        assert len(flat) == 2
        assert flat[0].tool_name == "a"
        assert flat[1].errored is True

    def test_finalize_sets_final_output(self):
        t = Transcript()
        t.append(_step(text="step1"))
        t.finalize("the answer", hit_max_handoffs=False)
        assert t.final_output == "the answer"
        assert t.hit_max_handoffs is False

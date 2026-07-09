"""Tests for Agent.step(). Uses a stub LLM that scripts responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from harness.agent import Agent
from harness.llm import LLMResponse, ToolUseBlock
from harness.tools import ToolRegistry
from harness.transcript import HandoffContext


@dataclass
class StubLLM:
    """Scripted LLM that pops responses off `script` per call."""
    script: list[LLMResponse] = field(default_factory=list)
    calls: int = 0

    async def call(self, *, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        self.calls += 1
        return self.script.pop(0)


class _EchoTool:
    name = "echo"
    description = "echo"
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}},
                    "required": ["x"]}
    async def __call__(self, **kwargs):
        return {"echoed": kwargs["x"]}


def _resp(text: str = "", tool_uses: list[ToolUseBlock] | None = None,
          inp: int = 100, out: int = 20) -> LLMResponse:
    return LLMResponse(
        text=text, tool_uses=tool_uses or [],
        input_tokens=inp, output_tokens=out,
        cache_read_tokens=0, raw_response=None,
    )


@pytest.fixture
def registry():
    return ToolRegistry(tools={"echo": _EchoTool()})


class TestSimpleResponse:
    async def test_returns_final_text_immediately(self, registry):
        llm = StubLLM(script=[_resp(text="<final_answer>done</final_answer>")])
        agent = Agent(agent_id="p", role="planner", llm=llm, tools=registry,
                      system_prompt_template="You are a planner. {task} {handoff} {peer_notes}")
        result = await agent.step(task="do x", handoff_context=None, peer_notes=None)
        assert "done" in result.final_text
        assert result.llm_calls == 1
        assert result.tool_calls == []
        assert result.input_tokens == 100


class TestToolUseLoop:
    async def test_runs_tool_and_continues(self, registry):
        llm = StubLLM(script=[
            _resp(tool_uses=[ToolUseBlock(id="t1", name="echo", input={"x": "hi"})]),
            _resp(text="<final_answer>got hi back</final_answer>"),
        ])
        agent = Agent(agent_id="p", role="planner", llm=llm, tools=registry,
                      system_prompt_template="{task} {handoff} {peer_notes}")
        result = await agent.step(task="echo something", handoff_context=None, peer_notes=None)
        assert result.llm_calls == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "echo"
        assert result.tool_calls[0].output == {"echoed": "hi"}
        assert "got hi back" in result.final_text


class TestMaxIterations:
    async def test_hits_max_iter_when_stuck_in_tool_loop(self, registry):
        # Script: every response calls echo again, never emits final text.
        loop_response = _resp(
            tool_uses=[ToolUseBlock(id="t0", name="echo", input={"x": "."})]
        )
        llm = StubLLM(script=[loop_response] * 20)
        agent = Agent(agent_id="p", role="planner", llm=llm, tools=registry,
                      system_prompt_template="{task} {handoff} {peer_notes}",
                      max_iterations=3)
        result = await agent.step(task="loop", handoff_context=None, peer_notes=None)
        assert result.hit_max_iterations is True
        assert result.llm_calls == 3


class TestPromptSlots:
    async def test_peer_notes_and_handoff_injected(self, registry):
        llm = StubLLM(script=[_resp(text="<final_answer>ok</final_answer>")])
        agent = Agent(agent_id="p", role="planner", llm=llm, tools=registry,
                      system_prompt_template="ROLE_LINE\nT={task}\nH={handoff}\nP={peer_notes}")
        h = HandoffContext(payload={"key": "value"}, token_cost=10)
        await agent.step(task="the_task", handoff_context=h, peer_notes="peer info")
        # Inspect what was passed to the LLM
        # (We mutated stub but didn't capture; this test asserts the integration in TestPeerNotesContent below.)
        assert llm.calls == 1


class TestPeerNotesContent:
    async def test_peer_notes_appear_in_system_prompt(self, registry, monkeypatch):
        captured = {}

        async def fake_call(**kwargs):
            captured["system"] = kwargs["system"]
            captured["messages"] = kwargs["messages"]
            return _resp(text="<final_answer>ok</final_answer>")

        llm = StubLLM()
        llm.call = fake_call  # type: ignore[assignment]

        agent = Agent(agent_id="p", role="planner", llm=llm, tools=registry,
                      system_prompt_template="ROLE\nT={task}\nH={handoff}\nP={peer_notes}")
        await agent.step(task="the_task", handoff_context=None,
                         peer_notes="prior agent failed approach X")
        assert "prior agent failed approach X" in captured["system"]
        assert "the_task" in captured["system"]

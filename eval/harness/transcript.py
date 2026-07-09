"""Shared dataclasses for the eval harness.

These are the vocabulary every other harness module uses: ToolCall
captures one tool invocation, AgentStepResult captures one agent step
(one call to Agent.step()), Transcript accumulates step results across
a whole trial, and HandoffContext is the payload baselines pass between
agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any]
    errored: bool
    latency_ms: float


@dataclass
class AgentStepResult:
    agent_id: str
    final_text: str
    tool_calls: list[ToolCall]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    raw_messages: list[dict[str, Any]]
    hit_max_iterations: bool


@dataclass
class HandoffContext:
    payload: dict[str, Any]
    token_cost: int

    def payload_as_text(self) -> str:
        """Render the payload as a human-readable string for prompt injection."""
        return json.dumps(self.payload, indent=2, sort_keys=True)


@dataclass
class Transcript:
    steps: list[AgentStepResult] = field(default_factory=list)
    final_output: str = ""
    hit_max_handoffs: bool = False

    def append(self, step: AgentStepResult) -> None:
        self.steps.append(step)

    def tool_calls(self) -> list[ToolCall]:
        flat: list[ToolCall] = []
        for s in self.steps:
            flat.extend(s.tool_calls)
        return flat

    def input_tokens_total(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    def output_tokens_total(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    def cache_read_tokens_total(self) -> int:
        return sum(s.cache_read_tokens for s in self.steps)

    def llm_calls_total(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    def finalize(self, final_output: str, *, hit_max_handoffs: bool = False) -> None:
        self.final_output = final_output
        self.hit_max_handoffs = hit_max_handoffs

"""Single-agent tool-use loop wrapping AnthropicClient.

One Agent.step() call runs one full tool-use loop: send task → if the
model emits tool_use blocks, run them and feed results back → continue
until the model emits a final text response (no tool_use blocks) OR
max_iterations is hit.

The agent does NOT do its own handoff or peer-query logic — those are
the orchestration layer's job. The agent just expects them as parameters
to step() and injects them into the system prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from harness.llm import LLMResponse
from harness.tools import ToolRegistry
from harness.transcript import AgentStepResult, HandoffContext, ToolCall


class _LLMLike(Protocol):
    async def call(
        self, *, system: str, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0, max_tokens: int = 4096,
    ) -> LLMResponse: ...


@dataclass
class Agent:
    agent_id: str
    role: str
    llm: _LLMLike
    tools: ToolRegistry
    system_prompt_template: str
    max_iterations: int = 10
    _temperature: float = 0.0

    async def step(
        self,
        *,
        task: str = "",
        handoff_context: HandoffContext | None = None,
        peer_notes: str | None = None,
    ) -> AgentStepResult:
        system = self.system_prompt_template.format(
            task=task,
            handoff=handoff_context.payload_as_text() if handoff_context
            else "None - you are starting from scratch.",
            peer_notes=peer_notes if peer_notes is not None else "None.",
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": task or "Begin."}]
        tool_defs = self.tools.anthropic_format()

        tool_calls: list[ToolCall] = []
        llm_calls = 0
        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        final_text = ""
        hit_max = False

        for _ in range(self.max_iterations):
            response = await self.llm.call(
                system=system,
                messages=messages,
                tools=tool_defs if tool_defs else None,
                temperature=self._temperature,
            )
            llm_calls += 1
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cache_read += response.cache_read_tokens

            if not response.tool_uses:
                final_text = response.text
                break

            assistant_blocks: list[dict[str, Any]] = []
            if response.text:
                assistant_blocks.append({"type": "text", "text": response.text})
            for tu in response.tool_uses:
                assistant_blocks.append({
                    "type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input,
                })
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results: list[dict[str, Any]] = []
            for tu in response.tool_uses:
                start = time.perf_counter()
                errored = False
                try:
                    output = await self.tools.invoke(tu.name, tu.input)
                except Exception as exc:
                    output = {"error": str(exc)}
                    errored = True
                latency_ms = (time.perf_counter() - start) * 1000.0
                tool_calls.append(ToolCall(
                    tool_name=tu.name, input=dict(tu.input), output=output,
                    errored=errored, latency_ms=latency_ms,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": _serialize_tool_output(output),
                    "is_error": errored,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            hit_max = True

        return AgentStepResult(
            agent_id=self.agent_id,
            final_text=final_text,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            raw_messages=messages,
            hit_max_iterations=hit_max,
        )


def _serialize_tool_output(output: dict[str, Any]) -> str:
    import json
    return json.dumps(output)

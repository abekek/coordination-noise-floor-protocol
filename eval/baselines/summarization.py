"""B2: Summarization handoff baseline.

Runs one extra LLM call to summarize the previous agent's state in
~300 tokens before passing it to the next agent.
"""

from __future__ import annotations

from typing import Any, Protocol

from baselines.base import BaselineProtocol, HandoffContext


class _LLMLike(Protocol):
    async def call(self, *, system: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None,
                   temperature: float = 0.0,
                   max_tokens: int = 4096): ...


_SUMMARIZE_SYSTEM = (
    "You summarize an agent's working state for handoff to the next "
    "agent in a multi-agent task. Keep the summary under 300 tokens. "
    "Focus on: what was attempted, what succeeded, what failed and why, "
    "and what the next agent must know to continue."
)


class SummarizationBaseline(BaselineProtocol):
    name = "B2_summarization"

    def __init__(self, *, llm: _LLMLike) -> None:
        self._llm = llm
        self._pending: dict[str, HandoffContext] = {}

    async def prepare_handoff(
        self, agent_id: str, agent_state: dict[str, Any],
    ) -> HandoffContext:
        transcript = str(agent_state.get("transcript", ""))
        response = await self._llm.call(
            system=_SUMMARIZE_SYSTEM,
            messages=[{"role": "user", "content": f"Summarize:\n\n{transcript}"}],
            temperature=0.0,
            max_tokens=400,
        )
        summary = response.text
        token_cost = response.input_tokens + response.output_tokens
        return HandoffContext(payload={"summary": summary}, token_cost=token_cost)

    async def consume_handoff(
        self, agent_id: str, context: HandoffContext,
    ) -> None:
        self._pending[agent_id] = context

    def pop_pending(self, agent_id: str) -> HandoffContext | None:
        return self._pending.pop(agent_id, None)

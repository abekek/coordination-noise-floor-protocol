"""B1: Full-context forwarding baseline.

Passes the entire previous-agent transcript text to the next agent.
"""

from __future__ import annotations

from typing import Any

from baselines.base import BaselineProtocol, HandoffContext


class FullContextBaseline(BaselineProtocol):
    name = "B1_full_context"

    def __init__(self) -> None:
        self._pending: dict[str, HandoffContext] = {}

    async def prepare_handoff(
        self, agent_id: str, agent_state: dict[str, Any],
    ) -> HandoffContext:
        # agent_state expected to carry a "transcript" string serialized
        # by the orchestration layer.
        text = str(agent_state.get("transcript", ""))
        token_cost = max(1, len(text) // 4)  # rough estimate, 1 token ~= 4 chars
        return HandoffContext(payload={"transcript": text}, token_cost=token_cost)

    async def consume_handoff(
        self, agent_id: str, context: HandoffContext,
    ) -> None:
        self._pending[agent_id] = context

    def pop_pending(self, agent_id: str) -> HandoffContext | None:
        return self._pending.pop(agent_id, None)

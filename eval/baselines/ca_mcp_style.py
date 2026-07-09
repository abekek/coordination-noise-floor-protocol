"""B3: CA-MCP-style shared state baseline.

In-memory shared dict keyed by task_id. The writer pushes a structured
snapshot; the reader consumes the entire snapshot. Push semantics
emulate the pre-2026-07-28 stateful CA-MCP design.
"""

from __future__ import annotations

from typing import Any

from baselines.base import BaselineProtocol, HandoffContext


class CaMcpStyleBaseline(BaselineProtocol):
    name = "B3_ca_mcp_style"

    def __init__(self, *, task_id: str) -> None:
        self._task_id = task_id
        self.state_store: dict[str, dict[str, Any]] = {}

    async def prepare_handoff(
        self, agent_id: str, agent_state: dict[str, Any],
    ) -> HandoffContext:
        snapshot = {
            "tools_called": list(agent_state.get("tools_called", [])),
            "last_results": list(agent_state.get("last_results", [])),
            "current_step": str(agent_state.get("current_step", "")),
        }
        self.state_store[self._task_id] = snapshot
        token_cost = len(str(snapshot)) // 4
        return HandoffContext(payload={"ack": True}, token_cost=token_cost)

    async def consume_handoff(
        self, agent_id: str, context: HandoffContext,
    ) -> None:
        # In CA-MCP-style semantics the consumer reads from the shared
        # store; the context payload itself is a no-op marker.
        return None

    async def _build_consume_context(self) -> HandoffContext:
        """What an agent would see when reading the shared state."""
        snapshot = self.state_store.get(self._task_id, {})
        return HandoffContext(payload=snapshot, token_cost=len(str(snapshot)) // 4)

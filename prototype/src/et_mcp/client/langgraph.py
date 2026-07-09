"""A thin, LangGraph-friendly wrapper.

Wraps an EtMcpServer reference with task_id + agent_id baked in. Any
LangGraph node can hold an `EtMcpNode` and call `.record(...)` /
`.query(...)` without needing to know about the underlying MCP protocol.

This is a *prototype* convenience for the eval harness — for real
deployments, agents call the MCP server over HTTP via the standard MCP
client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from et_mcp.client.policy import AgentStep, Policy, should_write
from et_mcp.events import EventType
from et_mcp.server import EtMcpServer


_PAYLOAD_TO_EVENT_TYPE = {
    "FailedPath": EventType.FAILED_PATH,
    "ConstraintViolation": EventType.CONSTRAINT_VIOLATION,
    "AbandonedApproach": EventType.ABANDONED_APPROACH,
    "IntermediateDecision": EventType.INTERMEDIATE_DECISION,
    "ToolError": EventType.TOOL_ERROR,
}


@dataclass
class EtMcpNode:
    server: EtMcpServer
    task_id: str
    agent_id: str
    policy: Policy = Policy.FAILURE_ONLY

    async def record(self, step: AgentStep) -> dict[str, Any] | None:
        payload = should_write(step, self.policy)
        if payload is None:
            return None
        event_type = _PAYLOAD_TO_EVENT_TYPE[type(payload).__name__]
        return await self.server.trace_write(
            task_id=self.task_id,
            event_type=event_type.value,
            agent_id=self.agent_id,
            payload=payload.model_dump(),
        )

    async def query(
        self, question: str, *,
        event_types: list[EventType] | None = None,
        peer_only: bool = False, limit: int = 10,
    ) -> dict[str, Any]:
        """Query the task's trace store.

        peer_only=True excludes events authored by self.agent_id from results.
        We over-fetch (limit*2) when filtering to preserve the requested limit
        after self-events are dropped.
        """
        fetch_limit = limit * 2 if peer_only else limit
        result = await self.server.trace_query(
            task_id=self.task_id,
            question=question,
            event_types=[et.value for et in event_types] if event_types else None,
            agent_id=None,
            limit=fetch_limit,
        )
        if not peer_only:
            return result
        # Filter out events authored by this agent (parallel arrays must
        # stay aligned, so filter all three).
        keep_indices = [
            i for i, ev in enumerate(result["events"])
            if ev["agent_id"] != self.agent_id
        ][:limit]
        return {
            "events": [result["events"][i] for i in keep_indices],
            "summaries": [result["summaries"][i] for i in keep_indices],
            "scores": [result["scores"][i] for i in keep_indices],
        }

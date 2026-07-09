"""ET-MCP baseline — pull-based, this work's contribution.

Holds an in-process EtMcpServer and per-agent EtMcpNode instances.
prepare_handoff translates AgentStepResult into AgentStep events via a
coarse heuristic (TOOL_ERROR + FAILED_PATH only); the selection policy
then decides which to write to the trace store. consume_handoff is a
no-op (pull semantics). query_during_step performs a trace.query and
returns formatted summary lines for prompt injection.
"""

from __future__ import annotations

from typing import Any

from et_mcp.client.langgraph import EtMcpNode
from et_mcp.client.policy import AgentStep, Policy
from et_mcp.events import AbandonedApproach, FailedPath, IntermediateDecision, ToolError
from et_mcp.server import build_server

from baselines.base import BaselineProtocol, HandoffContext


class EtMcpBaseline(BaselineProtocol):
    name = "ET_MCP_default"

    def __init__(
        self, *, task_id: str, policy: Policy = Policy.FAILURE_ONLY,
    ) -> None:
        self._task_id = task_id
        self.policy = policy
        self.server = build_server()
        self._nodes: dict[str, EtMcpNode] = {}

    async def setup(self) -> None:
        await self.server.lifecycle.init(self._task_id, owner="trial")

    async def teardown(self) -> None:
        await self.server.lifecycle.complete(self._task_id)

    def node_for(self, agent_id: str) -> EtMcpNode:
        if agent_id not in self._nodes:
            self._nodes[agent_id] = EtMcpNode(
                server=self.server,
                task_id=self._task_id,
                agent_id=agent_id,
                policy=self.policy,
            )
        return self._nodes[agent_id]

    async def prepare_handoff(
        self, agent_id: str, agent_state: dict[str, Any],
    ) -> HandoffContext:
        node = self.node_for(agent_id)
        # Translate raw AgentStepResult into AgentStep events
        step_result = agent_state.get("agent_step_result")
        if step_result is not None:
            for agent_step in _translate_step_result(step_result):
                await node.record(agent_step)
        # Legacy: also accept pre-built AgentStep list
        for step in agent_state.get("steps", []):
            if isinstance(step, AgentStep):
                await node.record(step)
        # Include a minimal transcript summary in the payload so receiving agents
        # have forward-facing context even when the trace store has no events yet.
        transcript_summary = agent_state.get("transcript", "")
        if transcript_summary:
            # Truncate to keep it short — ET-MCP agents primarily rely on
            # query_during_step; this is only a fallback when the store is empty.
            transcript_summary = transcript_summary[:800]
        return HandoffContext(
            payload={"ack": True, "step_summary": transcript_summary},
            token_cost=len(transcript_summary) // 4,
        )

    async def consume_handoff(
        self, agent_id: str, context: HandoffContext,
    ) -> None:
        return None  # pull semantics

    async def query_during_step(
        self, agent_id: str, question: str,
    ) -> str | None:
        node = self.node_for(agent_id)
        try:
            result = await node.query(question)
        except KeyError:
            return None
        summaries = result.get("summaries", [])
        if not summaries:
            return ""
        return "\n".join(f"- {s}" for s in summaries)


def _translate_step_result(step) -> list[AgentStep]:
    """Coarse step-to-event translation (spec §5.6).

    Maps:
    - Any errored tool call → TOOL_ERROR event
    - book_* tool returning success=False → FAILED_PATH event
    - book_* tool returning success=True → IntermediateDecision(reversible=False)
    - Any tool returning "no information available" → ABANDONED_APPROACH event
      (a TravelPlanner-friendly signal — search tools returning empty
       results are tried-and-rejected paths)
    """
    import json
    events: list[AgentStep] = []
    for tc in step.tool_calls:
        if tc.errored:
            events.append(AgentStep(
                outcome="ongoing",
                tool_error=ToolError(
                    tool_name=tc.tool_name,
                    input=dict(tc.input),
                    error=str(tc.output.get("error", "unknown")) if isinstance(tc.output, dict) else str(tc.output),
                    retry_count=0,
                    recovered=False,
                ),
            ))
        # KEEP: book_* failure → FAILED_PATH
        if (tc.tool_name.startswith("book_")
                and isinstance(tc.output, dict)
                and not tc.output.get("success", True)):
            events.append(AgentStep(
                outcome="failed",
                failed_path=FailedPath(
                    approach=f"{tc.tool_name}({json.dumps(tc.input)})",
                    reason=str(tc.output.get("error", "booking failed")),
                    constraints_hit=[],
                    steps_taken=[],
                ),
            ))
        # NEW: tool returning "no information available" → ABANDONED_APPROACH
        if isinstance(tc.output, dict) and _no_information(tc.output):
            events.append(AgentStep(
                outcome="ongoing",
                abandoned=AbandonedApproach(
                    description=f"{tc.tool_name}({json.dumps(tc.input)})",
                    why_abandoned="tool returned no matching information",
                    alternatives_considered=[],
                ),
            ))
        # KEEP: book_* success → IntermediateDecision(reversible=False) (still recorded)
        if (tc.tool_name.startswith("book_")
                and isinstance(tc.output, dict)
                and tc.output.get("success", False)):
            events.append(AgentStep(
                outcome="ongoing",
                decision=IntermediateDecision(
                    decision=f"{tc.tool_name}({json.dumps(tc.input)})",
                    reasoning="booking committed; irreversible",
                    confidence=1.0,
                    reversible=False,
                ),
            ))
    return events


def _no_information(output: dict) -> bool:
    """Detect 'no information available' tool responses."""
    for value in output.values():
        if isinstance(value, str) and "no information available" in value.lower():
            return True
    return False

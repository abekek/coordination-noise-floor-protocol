"""ET-MCP protocol tool implementations.

Agents interact with an in-process EtMcpServer instead of message-passing.
The server is initialised once per trial (in engine.init_case) and stored in
the module-level _SERVERS dict keyed by case_dir string so that the stateless
execute_tool() dispatcher can reach it across invocations.

Pull-based mechanic:
  trace_write  – publish a typed fact to the shared trace store
  trace_query  – retrieve relevant peer events by natural-language question
  wait         – no-op synchronisation turn-ender (mirrors msg/sfs)
  submit_result – finalise this agent's answer (same as msg/sfs)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.models import AgentState, ToolCall, ToolResult
from src.utils.persistence import write_json

# ---------------------------------------------------------------------------
# Module-level server registry
# key: str(case_dir), value: EtMcpServer instance
# ---------------------------------------------------------------------------
_SERVERS: dict[str, Any] = {}


def register_server(case_dir: Path, server: Any) -> None:
    """Called by engine.init_case to store the per-trial ET-MCP server."""
    _SERVERS[str(case_dir)] = server


def get_server(case_dir: Path) -> Any | None:
    """Return the server for this trial, or None if not yet registered."""
    return _SERVERS.get(str(case_dir))


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running loop (shouldn't normally happen
            # in the silo-bench engine, but guard for safety).
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def tool_trace_write(
    agent_id: int,
    case_dir: Path,
    task_id: str,
    event_type: str,
    payload: str,
) -> dict[str, Any]:
    """Publish a typed trace event to the ET-MCP store.

    event_type must be one of:
        FAILED_PATH, CONSTRAINT_VIOLATION, ABANDONED_APPROACH,
        INTERMEDIATE_DECISION, TOOL_ERROR

    payload is free-form text describing the event.  Internally it is wrapped
    in an IntermediateDecision envelope so every write is uniformly stored
    without requiring agents to produce structured JSON.
    """
    server = get_server(case_dir)
    if server is None:
        return {"success": False, "error": "ET-MCP server not initialised for this trial"}

    # Normalise event_type to upper-case
    event_type_upper = event_type.strip().upper()

    # Allowed event types
    ALLOWED = {
        "FAILED_PATH",
        "CONSTRAINT_VIOLATION",
        "ABANDONED_APPROACH",
        "INTERMEDIATE_DECISION",
        "TOOL_ERROR",
    }
    if event_type_upper not in ALLOWED:
        event_type_upper = "INTERMEDIATE_DECISION"

    # Build the typed payload dict for the chosen envelope
    if event_type_upper == "FAILED_PATH":
        typed_payload = {
            "approach": event_type,
            "reason": payload,
            "constraints_hit": [],
            "steps_taken": [],
            "evidence": None,
        }
    elif event_type_upper == "CONSTRAINT_VIOLATION":
        typed_payload = {
            "constraint": payload,
            "value_attempted": None,
            "threshold": None,
            "context": payload,
        }
    elif event_type_upper == "ABANDONED_APPROACH":
        typed_payload = {
            "description": payload,
            "why_abandoned": payload,
            "alternatives_considered": [],
        }
    elif event_type_upper == "TOOL_ERROR":
        typed_payload = {
            "tool_name": "unknown",
            "input": {},
            "error": payload,
            "retry_count": 0,
            "recovered": False,
        }
    else:
        # Default: INTERMEDIATE_DECISION — most flexible
        typed_payload = {
            "decision": event_type,
            "reasoning": payload,
            "confidence": 0.5,
            "reversible": True,
        }

    try:
        result = _run(
            server.trace_write(
                task_id=task_id,
                event_type=event_type_upper,
                agent_id=f"agent_{agent_id}",
                payload=typed_payload,
            )
        )
        return {"success": True, "event_id": result.get("event_id"), "version": result.get("version")}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_trace_query(
    agent_id: int,
    case_dir: Path,
    task_id: str,
    question: str,
) -> dict[str, Any]:
    """Pull relevant peer trace events by natural-language question.

    Returns a list of formatted event summaries from all agents in this trial.
    """
    server = get_server(case_dir)
    if server is None:
        return {"events": [], "error": "ET-MCP server not initialised for this trial"}

    try:
        result = _run(
            server.trace_query(
                task_id=task_id,
                question=question,
                limit=20,
            )
        )
        # Format events into a readable list for the agent
        events = result.get("events", [])
        summaries = result.get("summaries", [])
        formatted = []
        for i, ev in enumerate(events):
            summary = summaries[i] if i < len(summaries) else ""
            formatted.append({
                "event_id": ev.get("event_id"),
                "agent_id": ev.get("agent_id"),
                "event_type": ev.get("event_type"),
                "summary": summary,
                "payload": ev.get("payload"),
                "timestamp": ev.get("timestamp"),
            })
        return {"events": formatted, "count": len(formatted)}
    except Exception as exc:
        return {"events": [], "error": str(exc)}


def tool_wait() -> dict[str, Any]:
    """Wait for other agents to act."""
    return {"status": "waiting"}


def tool_submit_result(
    agent_id: int,
    answer: Any,
    round_dir: Path,
    current_round: int,
    state: AgentState,
) -> dict[str, Any]:
    """Submit the final answer."""
    if state.submitted:
        return {"status": "already_submitted"}

    submission = {
        "agent_id": agent_id,
        "answer": answer,
        "round": current_round,
    }

    agent_dir = round_dir / f"agent-{agent_id:03d}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    write_json(agent_dir / "submission.json", submission)

    state.submitted = True
    state.submission_round = current_round
    return {"status": "submitted"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def execute_tool(
    tool_call: ToolCall,
    agent_id: int,
    case_dir: Path,
    round_dir: Path,
    current_round: int,
    state: AgentState,
    num_agents: int,
) -> ToolResult:
    """Dispatch and execute a tool call for the ET-MCP protocol."""
    name = tool_call.tool
    params = tool_call.parameters

    # Derive the stable task_id from the case directory name
    task_id = case_dir.name

    try:
        if name == "trace_write":
            result = tool_trace_write(
                agent_id=agent_id,
                case_dir=case_dir,
                task_id=task_id,
                event_type=str(params.get("event_type", "INTERMEDIATE_DECISION")),
                payload=str(params.get("payload", "")),
            )
        elif name == "trace_query":
            result = tool_trace_query(
                agent_id=agent_id,
                case_dir=case_dir,
                task_id=task_id,
                question=str(params.get("question", "")),
            )
        elif name == "wait":
            result = tool_wait()
        elif name == "submit_result":
            result = tool_submit_result(
                agent_id=agent_id,
                answer=params.get("answer"),
                round_dir=round_dir,
                current_round=current_round,
                state=state,
            )
        else:
            return ToolResult(
                tool=name,
                parameters=params,
                result={"error": f"Unknown tool: {name}"},
                success=False,
                error=f"Unknown tool: {name}",
            )

        return ToolResult(tool=name, parameters=params, result=result, success=True)

    except Exception as e:
        return ToolResult(
            tool=name,
            parameters=params,
            result={"error": str(e)},
            success=False,
            error=str(e),
        )

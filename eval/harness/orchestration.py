"""Two-agent planner+executor trial loop.

Same loop for every baseline — only the baseline's
prepare_handoff / consume_handoff / query_during_step hooks differ.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from baselines.base import BaselineProtocol
from benchmarks.base import BenchmarkQuery
from harness.transcript import AgentStepResult, Transcript


_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


class _AgentLike(Protocol):
    agent_id: str
    role: str

    async def step(
        self, *, task: str, handoff_context, peer_notes,
    ) -> AgentStepResult: ...


def _is_done(step: AgentStepResult) -> bool:
    return bool(_FINAL_ANSWER_RE.search(step.final_text))


def _extract_final(step: AgentStepResult) -> str:
    match = _FINAL_ANSWER_RE.search(step.final_text)
    return match.group(1).strip() if match else step.final_text


def _agent_state_from_step(step: AgentStepResult) -> dict[str, Any]:
    """Bundle what a baseline needs to build a handoff payload."""
    return {
        "transcript": _serialize_transcript(step),
        "tools_called": [tc.tool_name for tc in step.tool_calls],
        "last_results": [tc.output for tc in step.tool_calls],
        "current_step": step.final_text,
        "agent_step_result": step,  # NEW: raw step for ET-MCP to translate
        "steps": [],  # kept for backward compat
    }


def _serialize_transcript(step: AgentStepResult) -> str:
    parts = []
    for tc in step.tool_calls:
        parts.append(f"{tc.tool_name}({json.dumps(tc.input)}) -> {json.dumps(tc.output)}")
    if step.final_text:
        parts.append(f"text: {step.final_text}")
    return "\n".join(parts)


async def run_two_agent_trial(
    *,
    planner: _AgentLike,
    executor: _AgentLike,
    query: BenchmarkQuery,
    baseline: BaselineProtocol,
    max_handoffs: int = 4,
) -> Transcript:
    transcript = Transcript()
    task = str(query.payload.get("text", ""))

    # Planner's first step (no incoming handoff)
    peer = await baseline.query_during_step("planner", task)
    p_step = await planner.step(task=task, handoff_context=None, peer_notes=peer)
    transcript.append(p_step)
    # prepare_handoff both records events (ET-MCP) and builds the outgoing payload (B1/B2)
    p_to_e_handoff = await baseline.prepare_handoff("planner", _agent_state_from_step(p_step))
    if _is_done(p_step):
        transcript.finalize(_extract_final(p_step))
        return transcript

    last_planner_step = p_step
    outgoing_handoff = p_to_e_handoff  # carry forward the prepared context
    for _ in range(max_handoffs):
        # Planner → Executor handoff
        handoff = await _resolve_consume_context(baseline, outgoing_handoff)
        await baseline.consume_handoff("executor", handoff)

        subtask = last_planner_step.final_text or "Continue the work."
        peer = await baseline.query_during_step("executor", subtask)
        e_step = await executor.step(task=subtask, handoff_context=handoff, peer_notes=peer)
        transcript.append(e_step)
        # prepare_handoff both records events and builds outgoing payload
        e_to_p_handoff = await baseline.prepare_handoff("executor", _agent_state_from_step(e_step))
        if _is_done(e_step):
            transcript.finalize(_extract_final(e_step))
            return transcript

        # Executor → Planner handoff
        handoff = await _resolve_consume_context(baseline, e_to_p_handoff)
        await baseline.consume_handoff("planner", handoff)

        question = task  # keep original task so planner retains full context
        peer = await baseline.query_during_step("planner", question)
        p_step = await planner.step(task=question, handoff_context=handoff, peer_notes=peer)
        transcript.append(p_step)
        last_planner_step = p_step
        # prepare_handoff both records events and builds outgoing payload
        outgoing_handoff = await baseline.prepare_handoff("planner", _agent_state_from_step(p_step))
        if _is_done(p_step):
            transcript.finalize(_extract_final(p_step))
            return transcript

    transcript.finalize(last_planner_step.final_text, hit_max_handoffs=True)
    return transcript


async def _resolve_consume_context(baseline: BaselineProtocol, prepared: Any):
    """Turn the result of prepare_handoff into a HandoffContext for consume_handoff.

    - For baselines with _build_consume_context (e.g. CA-MCP-style), call that.
    - For baselines that return a meaningful HandoffContext from prepare_handoff
      (e.g. B1 full-context, B2 summarization), use `prepared` directly.
    - Otherwise fall back to empty context.
    """
    from baselines.base import HandoffContext
    if hasattr(baseline, "_build_consume_context"):
        return await baseline._build_consume_context()
    if prepared is not None and isinstance(prepared, HandoffContext):
        if prepared.payload:
            return prepared
    return HandoffContext(payload={}, token_cost=0)

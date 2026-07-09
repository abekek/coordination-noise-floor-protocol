"""ET-MCP agent for tau2-bench.

A LLMAgent subclass that holds a per-task TraceStore shared across the N
trials of that task, and exposes four coordination protocols for the
ablation in §6 of the paper.

The setup leverages tau2-bench's pass^k metric: with k independent trials
per task, ET-MCP's cross-trial trace store gives later trials access to
earlier trials' failures. The 4-protocol comparison is:

- no_coord:        each trial gets a fresh agent with no shared state
- push_scratchpad: cross-trial shared dict; every prior trial's failure
                   dump is wholesale prepended to the system prompt
                   (CA-MCP-style push, unfiltered, no schema)
- message_passing: cross-trial fixed-format summary message of prior
                   trials, attached as a single system note (push, fixed
                   schema)
- et_mcp:          cross-trial typed event store; the agent queries it
                   on demand via TraceStore.query() and the top-k
                   matching events are injected as <peer_warnings>

In all three coordinated protocols, the writer side is the same: on a
ToolMessage marked error, a FAILED_PATH event is recorded with the
tool name + error text. The protocols differ in how the *reader* gets
to those records.
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from .trace_store import TraceStore


SUPPORTED_PROTOCOLS = {
    "no_coord",
    "push_scratchpad",
    "message_passing",
    "et_mcp",
}


ET_MCP_AGENT_INSTRUCTION = (
    AGENT_INSTRUCTION
    + "\n\n"
    + (
        "Where present, a <peer_warnings> block surfaces negative knowledge "
        "(failed paths, constraint violations, abandoned approaches) "
        "produced by peer agents on this task. Treat warnings as advisory: "
        "use them to avoid known dead-ends, but defer to the current user "
        "request when it conflicts."
    )
)


# Module-level cache so the *same* TraceStore is reused across the N trials
# of a single task. tau2 invokes our factory once per (task, trial), so this
# is the only way to give ET-MCP cross-trial access. Keyed by task_id; reset
# at the boundary of a new run via reset_task_stores().
_TASK_STORES: dict[str, TraceStore] = {}


def get_or_create_task_store(task_id: str) -> TraceStore:
    """Reuse the same TraceStore across trials of one task."""
    if task_id not in _TASK_STORES:
        _TASK_STORES[task_id] = TraceStore(task_id=task_id)
    return _TASK_STORES[task_id]


def reset_task_stores() -> None:
    """Clear the module-level store cache (test fixtures / run boundaries)."""
    _TASK_STORES.clear()


class ETMCPAgent(LLMAgent):
    """tau2 LLMAgent with a cross-trial coordination layer.

    The agent records FAILED_PATH events on every errored tool response (the
    failure_only selection policy). The reader side is switched on by
    `coord_protocol`:

    - no_coord:        no reads, no writes (and the store is per-trial fresh)
    - push_scratchpad: all prior events dumped wholesale
    - message_passing: prior events summarized into a fixed-format message
    - et_mcp:          TF-IDF top-k query per turn, returned as warnings
    """

    coord_protocol: str = "et_mcp"
    query_top_k: int = 3

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        coord_protocol: str = "et_mcp",
        query_top_k: int = 3,
        task_id: str = "unknown_task",
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        if coord_protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unknown coord_protocol={coord_protocol!r}. "
                f"Supported: {sorted(SUPPORTED_PROTOCOLS)}"
            )
        self.coord_protocol = coord_protocol
        self.query_top_k = query_top_k

        # no_coord uses a private, per-trial store (no cache lookup); the
        # other three protocols share the per-task store across trials.
        if coord_protocol == "no_coord":
            self.trace_store = TraceStore(task_id=f"{task_id}::no_coord_isolated")
        else:
            self.trace_store = get_or_create_task_store(task_id)

        logger.debug(
            f"ETMCPAgent initialized: protocol={coord_protocol}, "
            f"task_id={task_id}, store_events_at_init={len(self.trace_store)}"
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=ET_MCP_AGENT_INSTRUCTION,
        )

    # -----------------------------------------------------------------
    # Writer side: selection policy = failure_only
    # -----------------------------------------------------------------

    def _record_tool_failure(self, tool_msg: ToolMessage) -> None:
        if self.coord_protocol == "no_coord":
            # Still record into the isolated store so within-trial reads
            # could in principle work; but we never read in no_coord.
            return
        if not getattr(tool_msg, "error", False):
            return
        payload = {
            "tool_call_id": getattr(tool_msg, "tool_call_id", None),
            "tool_name": getattr(tool_msg, "requestor", "unknown"),
            "error_text": str(getattr(tool_msg, "content", ""))[:500],
        }
        self.trace_store.write(
            event_type="FAILED_PATH",
            agent_id="executor",
            payload=payload,
        )

    # -----------------------------------------------------------------
    # Reader side: per-protocol injection into the per-turn system prompt
    # -----------------------------------------------------------------

    def _last_user_or_tool_text(self, state: LLMAgentState) -> str:
        for msg in reversed(state.messages):
            if isinstance(msg, (UserMessage, ToolMessage)) and getattr(
                msg, "content", None
            ):
                return str(msg.content)[:500]
        return self.domain_policy[:500]

    def _build_et_mcp_block(self, state: LLMAgentState) -> Optional[str]:
        """Pull-based: query the store for events relevant to the current
        question, return top-k as a <peer_warnings> block. None if empty."""
        if len(self.trace_store) == 0:
            return None
        question = self._last_user_or_tool_text(state)
        events = self.trace_store.query(
            question, event_types=["FAILED_PATH"], limit=self.query_top_k
        )
        if not events:
            return None
        lines = [
            f"- [{e.event_type}] {json.dumps(e.payload, separators=(',', ':'))}"
            for e in events
        ]
        return "<peer_warnings>\n" + "\n".join(lines) + "\n</peer_warnings>"

    def _build_push_scratchpad_block(self) -> Optional[str]:
        """Push-all: dump every prior event verbatim. No filtering, no
        ranking, no schema beyond the event-type tag. Closest analogue to
        a CA-MCP-style shared dict."""
        if len(self.trace_store) == 0:
            return None
        lines = [
            f"- [{e.event_type}] agent={e.agent_id} "
            f"payload={json.dumps(e.payload, separators=(',', ':'))}"
            for e in self.trace_store.events
        ]
        return "<prior_trial_dump>\n" + "\n".join(lines) + "\n</prior_trial_dump>"

    def _build_message_passing_block(self) -> Optional[str]:
        """Push-fixed-schema: one summary line per prior trial's failure
        count, plus the list of distinct tool names that have failed.
        Mimics what a planner→executor handoff message would carry."""
        if len(self.trace_store) == 0:
            return None
        n = len(self.trace_store)
        tools_failed = sorted(
            {e.payload.get("tool_name", "?") for e in self.trace_store.events}
        )
        summary = (
            f"Prior trials of this task recorded {n} failure event(s). "
            f"Tools that previously failed: {', '.join(tools_failed)}."
        )
        return "<peer_handoff>\n" + summary + "\n</peer_handoff>"

    def _build_warning_block(self, state: LLMAgentState) -> Optional[str]:
        if self.coord_protocol == "no_coord":
            return None
        if self.coord_protocol == "et_mcp":
            return self._build_et_mcp_block(state)
        if self.coord_protocol == "push_scratchpad":
            return self._build_push_scratchpad_block()
        if self.coord_protocol == "message_passing":
            return self._build_message_passing_block()
        return None

    def _maybe_inject_warnings(self, state: LLMAgentState) -> LLMAgentState:
        block = self._build_warning_block(state)
        if not block:
            return state
        augmented_sys = SystemMessage(
            role="system",
            content=self.system_prompt + "\n\n" + block,
        )
        return LLMAgentState(
            system_messages=[augmented_sys],
            messages=state.messages,
        )

    # -----------------------------------------------------------------
    # Orchestrator entry point
    # -----------------------------------------------------------------

    def generate_next_message(
        self, message, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        # Writer side: record any incoming tool failure before generating.
        if isinstance(message, ToolMessage):
            self._record_tool_failure(message)
        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                self._record_tool_failure(tm)

        # Reader side: per-protocol injection into a per-turn system prompt.
        # augmented_state.messages aliases state.messages (same list reference),
        # so the appends super() performs on augmented_state mutate state too.
        augmented_state = self._maybe_inject_warnings(state)

        # Delegate to LLMAgent for the actual generation. super() appends both
        # the incoming `message` AND the generated `assistant_message` to
        # augmented_state.messages, which IS state.messages by reference —
        # so we must NOT append again here, or the assistant message ends up
        # doubled and the next turn's API call fails with
        # "tool_use ids were found without tool_result blocks".
        assistant_message, _ = super().generate_next_message(
            message, augmented_state
        )
        return assistant_message, state

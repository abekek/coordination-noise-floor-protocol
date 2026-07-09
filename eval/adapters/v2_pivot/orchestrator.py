"""Custom Anthropic-native orchestrator for tau2 task data.

The orchestrator runs an assistant ↔ user-simulator dialogue against a
tau2 Environment, calling tau2 tools directly via ``env.use_tool``.
Coordination protocols are switchable; the default is ``no_coord``
(vanilla agent loop).

The whole orchestrator avoids litellm — it talks directly to the
Anthropic Messages API. The Conversation helper guarantees that the
messages array never violates Anthropic's tool_use/tool_result pairing
invariants (which is the exact bug that broke our tau2 sweeps).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import anthropic
from loguru import logger

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment

from .conversation import Conversation
from .tools_adapter import all_tools_to_anthropic


# ---- system prompts (lifted from tau2 with light editing) -----------

AGENT_SYSTEM_TEMPLATE = """\
You are a customer service agent that helps the user according to the
<policy> provided below.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Always be helpful, follow the policy precisely, and prefer reading the
state of the world with tools before mutating it.

<policy>
{policy}
</policy>
"""


USER_SYSTEM_TEMPLATE = """\
You are role-playing a customer contacting a customer service
representative. Your goal is to simulate a realistic customer
interaction while strictly following the scenario instructions below.

## Rules
- Generate exactly one message at a time, in the customer's voice.
- Never make up information not provided in the scenario.
- Disclose information progressively. Wait for the agent to ask.
- Do not repeat the scenario verbatim; paraphrase.

## Termination
- When the scenario goal is satisfied, end your message with the
  literal token ###STOP### on its own.
- If transferred to another agent, end with ###TRANSFER###.
- If the scenario does not give you enough information to continue,
  end with ###OUT-OF-SCOPE###.

<scenario>
{scenario}
</scenario>
"""


# ---- protocol hook (intercept point) --------------------------------

InterceptHook = Callable[
    [
        # tool_name, tool_args (dict), tool_result (str), is_error (bool),
        # trial_state (dict)
        str,
        dict,
        str,
        bool,
        dict,
    ],
    str,  # returns the (possibly augmented) tool result string
]


def no_intercept(
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    is_error: bool,
    trial_state: dict,
) -> str:
    """Default intercept: pass tool result through unchanged."""
    return tool_result


# ---- core trial loop ------------------------------------------------

@dataclass
class TrialResult:
    task_id: str
    trial: int
    completed: bool
    n_assistant_turns: int
    n_tool_calls: int
    n_errored_calls: int
    n_redundant_calls: int
    termination_reason: str
    agent_messages: list[dict]
    duration_s: float
    input_tokens_agent: int = 0
    output_tokens_agent: int = 0
    input_tokens_user: int = 0
    output_tokens_user: int = 0


def run_trial(
    task: Task,
    env: Environment,
    *,
    agent_model: str = "claude-haiku-4-5",
    user_model: str = "claude-haiku-4-5",
    max_assistant_turns: int = 30,
    temperature: float = 0.0,
    intercept_hook: InterceptHook = no_intercept,
    system_augmenter: Optional[Callable[[str, Any], str]] = None,
    trial: int = 0,
    api_key: Optional[str] = None,
    trace_store: Optional[Any] = None,  # opaque; passed to intercept_hook via trial_state
) -> TrialResult:
    """Run one (task × trial) and return its metrics + transcript."""
    start = time.time()
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Build conversations. The system_augmenter (for the `pull` protocol)
    # gets a chance to inject a <peer_warnings> block based on the store.
    base_system = AGENT_SYSTEM_TEMPLATE.format(policy=env.get_policy())
    if system_augmenter is not None and trace_store is not None:
        agent_system = system_augmenter(base_system, trace_store)
    else:
        agent_system = base_system
    anth_tools = all_tools_to_anthropic(env.get_tools())
    agent_conv = Conversation(
        system=agent_system,
        tools=anth_tools,
    )
    user_conv = Conversation(
        system=USER_SYSTEM_TEMPLATE.format(scenario=str(task.user_scenario)),
        tools=[],
    )

    # Per-trial state for the intercept hook + redundancy metric
    failed_tool_calls: set[tuple[str, str]] = set()
    n_tool_calls = 0
    n_errored = 0
    n_redundant = 0
    n_turns = 0
    input_tok_a = output_tok_a = 0
    input_tok_u = output_tok_u = 0

    # User opens the conversation. Anthropic requires ≥1 message, so we
    # prime the user-sim with a bootstrap "Begin." prompt that won't
    # appear in the dialogue with the agent.
    user_conv.append_user_text(
        "Begin the conversation by greeting the customer service "
        "representative and stating your reason for contacting them."
    )
    set_hash_audit_ctx(
        task_id=task.id, trial=trial, turn=-1, role="user_sim_open"
    )
    user_msg_blocks, ui, uo = _generate_anthropic(
        client, user_conv, user_model, temperature
    )
    input_tok_u += ui
    output_tok_u += uo
    user_conv.append_assistant(user_msg_blocks)
    user_text = _extract_text(user_msg_blocks)
    termination = "completed"

    if not user_text:
        return TrialResult(
            task_id=task.id, trial=trial, completed=False,
            n_assistant_turns=0, n_tool_calls=0, n_errored_calls=0,
            n_redundant_calls=0, termination_reason="empty_user_open",
            agent_messages=agent_conv.messages, duration_s=time.time() - start,
        )

    for turn in range(max_assistant_turns):
        # Pass user text to agent
        agent_conv.append_user_text(user_text)

        # Validate before sending — catches any append-order bug fast
        validation_errors = agent_conv.validate()
        if validation_errors:
            logger.error(f"trial {task.id}.{trial}: agent conv invalid: {validation_errors}")
            termination = f"invariant_violation:{validation_errors[0]}"
            break

        # Agent's turn
        set_hash_audit_ctx(
            task_id=task.id, trial=trial, turn=turn, role="agent"
        )
        agent_msg_blocks, ai, ao = _generate_anthropic(
            client, agent_conv, agent_model, temperature
        )
        input_tok_a += ai
        output_tok_a += ao
        agent_conv.append_assistant(agent_msg_blocks)
        n_turns += 1

        # Execute any tool_use blocks
        tool_uses = [b for b in agent_msg_blocks if b.get("type") == "tool_use"]
        if tool_uses:
            results: list[tuple[str, str, bool]] = []
            for tu in tool_uses:
                tool_name = tu["name"]
                tool_args = tu.get("input", {}) or {}
                key = (tool_name, _normalize_args(tool_args))
                try:
                    raw = env.make_tool_call(tool_name, requestor="assistant", **tool_args)
                    out_str = str(raw)
                    is_err = False
                except Exception as exc:
                    out_str = f"ERROR: {type(exc).__name__}: {exc}"
                    is_err = True

                n_tool_calls += 1
                if is_err:
                    n_errored += 1
                    failed_tool_calls.add(key)
                elif key in failed_tool_calls:
                    n_redundant += 1

                # Pass through the intercept hook — this is where the v2
                # "tool-call interception" protocol injects peer-warning
                # context into the tool response.
                trial_state = {
                    "failed_tool_calls": failed_tool_calls,
                    "trace_store": trace_store,
                    "task_id": task.id,
                    "trial": trial,
                    "turn": turn,
                }
                augmented = intercept_hook(
                    tool_name, tool_args, out_str, is_err, trial_state
                )
                results.append((tu["id"], augmented, is_err))

            agent_conv.append_tool_results(results)
            # Loop back: agent gets the tool results and produces its next message
            continue

        # Agent produced text only — pass it to the user simulator
        agent_text = _extract_text(agent_msg_blocks)
        if not agent_text:
            termination = "agent_empty"
            break

        user_conv.append_user_text(agent_text)
        set_hash_audit_ctx(
            task_id=task.id, trial=trial, turn=turn, role="user_sim"
        )
        user_msg_blocks, ui, uo = _generate_anthropic(
            client, user_conv, user_model, temperature
        )
        input_tok_u += ui
        output_tok_u += uo
        user_conv.append_assistant(user_msg_blocks)
        user_text = _extract_text(user_msg_blocks)

        if "###STOP###" in user_text:
            termination = "user_stop"
            break
        if "###TRANSFER###" in user_text:
            termination = "user_transfer"
            break
        if "###OUT-OF-SCOPE###" in user_text:
            termination = "user_out_of_scope"
            break

    else:
        termination = "max_turns"

    return TrialResult(
        task_id=task.id,
        trial=trial,
        completed=(termination in {"user_stop", "user_transfer"}),
        n_assistant_turns=n_turns,
        n_tool_calls=n_tool_calls,
        n_errored_calls=n_errored,
        n_redundant_calls=n_redundant,
        termination_reason=termination,
        agent_messages=agent_conv.messages,
        duration_s=time.time() - start,
        input_tokens_agent=input_tok_a,
        output_tokens_agent=output_tok_a,
        input_tokens_user=input_tok_u,
        output_tokens_user=output_tok_u,
    )


# ---- helpers --------------------------------------------------------

# ---- request-equivalence hash audit (opt-in via env var) -----------
#
# When ETMCP_HASH_AUDIT=1, _generate_anthropic appends a JSONL line to
# /tmp/hash_audit.jsonl with a SHA-256 of the serialized request payload
# for every Messages API call. The audit context (task_id, trial,
# protocol, role) is passed through process-global state mutated by the
# orchestration layer (run_matrix / run_trial). This lets us empirically
# verify the §6.3 request-equivalence claim without changing the
# orchestrator's call signatures.

_HASH_AUDIT_CTX: dict[str, Any] = {
    "task_id": None,
    "trial": None,
    "protocol": None,
    "role": None,
    "turn": None,
}


def set_hash_audit_ctx(**kwargs: Any) -> None:
    """Update the process-global hash-audit context."""
    for k, v in kwargs.items():
        _HASH_AUDIT_CTX[k] = v


def _hash_payload(system: str, tools: list[dict], messages: list[dict]) -> str:
    import hashlib
    import json

    payload = {"system": system, "tools": tools, "messages": messages}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _write_hash_audit_record(
    system: str,
    tools: list[dict],
    messages: list[dict],
    input_tokens: int | None = None,
) -> None:
    if os.environ.get("ETMCP_HASH_AUDIT") != "1":
        return
    import json

    path = os.environ.get("ETMCP_HASH_AUDIT_PATH", "/tmp/hash_audit.jsonl")
    rec = {
        "task_id": _HASH_AUDIT_CTX.get("task_id"),
        "trial": _HASH_AUDIT_CTX.get("trial"),
        "turn": _HASH_AUDIT_CTX.get("turn"),
        "protocol": _HASH_AUDIT_CTX.get("protocol"),
        "role": _HASH_AUDIT_CTX.get("role"),
        "sha256": _hash_payload(system, tools, messages),
        "input_token_count": input_tokens,
        "n_messages": len(messages),
        "system_len": len(system),
        "n_tools": len(tools),
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        # Audit must never break the run; swallow IO errors.
        pass


def _generate_anthropic(
    client: anthropic.Anthropic,
    conv: Conversation,
    model: str,
    temperature: float,
    max_tokens: int = 4096,
) -> tuple[list[dict], int, int]:
    """Make one Anthropic Messages API call. Returns (content_blocks, in_tok, out_tok)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "system": conv.system,
        "messages": conv.messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if conv.tools:
        kwargs["tools"] = conv.tools
    # Hash audit (pre-call): record the exact serialized request bytes.
    _write_hash_audit_record(
        system=conv.system,
        tools=conv.tools,
        messages=conv.messages,
    )
    resp = client.messages.create(**kwargs)
    content_blocks = [_block_to_dict(b) for b in resp.content]
    return (
        content_blocks,
        resp.usage.input_tokens,
        resp.usage.output_tokens,
    )


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK ContentBlock to a pure dict for storage in messages."""
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    # Fallback for older SDK shapes
    return dict(block)


def _extract_text(blocks: list[dict]) -> str:
    """Concatenate text blocks, ignoring tool_use."""
    parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def _normalize_args(args: dict) -> str:
    """Canonical hashable form for the redundancy metric."""
    import json
    return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)

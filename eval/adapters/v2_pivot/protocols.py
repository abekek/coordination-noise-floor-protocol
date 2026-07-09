"""The three v2 coordination protocols.

All protocols write FAILED_PATH events on errored tool calls (this is
the writer-side default — same as the v1 failure_only selection policy).
They differ on the READER side:

- no_coord:  reader sees nothing. Each trial is independent.
- pull:      reader sees a <peer_warnings> block injected into the
             system prompt at the start of the trial. Static; no
             per-turn LLM cost beyond the longer system prompt.
- intercept: reader sees no warning block. Instead, when the agent
             calls a tool whose (name, args) was previously failed by a
             peer trial, the orchestrator prepends a structured warning
             to that tool's RESPONSE before handing it to the agent.
             The agent never has to "decide" to ask for warnings —
             they appear exactly when they are relevant. This is the
             v2 architectural pivot.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from .orchestrator import InterceptHook, no_intercept
from .trace_store import TraceStore


# Type alias for the system-prompt augmenter (called once at trial start,
# returns either the original system prompt or an augmented one).
SystemPromptAugmenter = Callable[[str, TraceStore], str]


def _no_augment(system: str, store: TraceStore) -> str:
    return system


def pull_augmenter(system: str, store: TraceStore) -> str:
    """Inject a <peer_warnings> block summarizing prior-trial failures."""
    warnings = store.warnings_block()
    if not warnings:
        return system
    return system + "\n\n" + warnings


def make_intercept_hook(store: Optional[TraceStore]) -> InterceptHook:
    """Build the intercept hook for the `intercept` protocol.

    When the agent calls a tool whose (tool_name, normalized_args)
    appears in the trace store as a prior failure, prepend a
    structured `[PEER-WARNING]` line to the tool's response. The agent
    sees the warning attached to the result of THIS call and can react
    to it on the next turn without an extra LLM round-trip.
    """
    def _intercept(
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        is_error: bool,
        trial_state: dict,
    ) -> str:
        # Always write failures (writer side, same for all coord protocols).
        if is_error and store is not None:
            store.write(
                trial=trial_state.get("trial", 0),
                tool_name=tool_name,
                tool_args=tool_args,
                error_text=tool_result,
            )
        # Now, on the READER side, augment the result with any peer
        # warnings that match THIS call's key.
        if store is not None:
            hits = store.has_event_for(tool_name, tool_args)
            current_trial = trial_state.get("trial", 0)
            cross_trial = [h for h in hits if h.trial != current_trial]
            if cross_trial:
                ev = cross_trial[-1]
                if ev.event_type == "FAILED_PATH":
                    body = (
                        f"A previous trial got an error from this exact "
                        f"call: {ev.error_text[:200]}. Consider an "
                        f"alternative approach."
                    )
                elif ev.event_type == "FAILED_TRIAL_ACTION":
                    body = (
                        f"A previous trial of this task made this exact "
                        f"call as part of an unsuccessful attempt "
                        f"(trial {ev.trial} ended with reward<1). "
                        f"Reconsider whether this call is on the right "
                        f"path before relying on its result."
                    )
                else:
                    body = "A previous trial flagged this call."
                warning = f"[PEER-WARNING] {body}"
                return warning + "\n\n" + tool_result
        return tool_result
    return _intercept


def make_pull_writer_hook(store: Optional[TraceStore]) -> InterceptHook:
    """Writer-only intercept for the `pull` protocol: records failures
    so trials i+1..k can see them, but does NOT modify tool results
    (the agent sees warnings via the system-prompt augment instead)."""
    def _writer_only(
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        is_error: bool,
        trial_state: dict,
    ) -> str:
        if is_error and store is not None:
            store.write(
                trial=trial_state.get("trial", 0),
                tool_name=tool_name,
                tool_args=tool_args,
                error_text=tool_result,
            )
        return tool_result
    return _writer_only


def make_selective_intercept_hook(
    store: Optional[TraceStore],
    theta: float = 0.5,
    alpha: float = 0.4,  # recency weight
    beta: float = 0.4,   # in-trial frequency penalty
    gamma: float = 0.2,  # argument-distance weight (1.0 for exact match)
) -> InterceptHook:
    """P2: Selective intercept (validates the M2 diagnostic).

    Gates injection by confidence score:
        c(e, t) = α · r(e) + β · (1 - f(e)) + γ · d(e, t)

    where:
      r(e) ∈ [0, 1]   = recency (newest event → 1.0)
      f(e) ∈ [0, 1]   = fraction of times event e has fired in this trial
                         (1.0 = capped at threshold; diminishing returns)
      d(e, t) ∈ [0, 1] = argument-key equality (1.0 if (name, args) match
                          exactly, lower under partial match)

    Fires only when c > theta. Pre-registered hypothesis (this paper):
    if M2 (per-match injection noise) is the diagnosed failure mode,
    selective intercept with theta=0.5 should recover >=5pp on the
    trial-1-given-trial-0-failed subset vs naive intercept.

    Defaults: α=β=0.4, γ=0.2; threshold theta=0.5 means the warning
    fires only on the most-recent matching event for an exact-key
    call that has not already fired more than ~1.5x in this trial.
    """
    # Per-trial firing counts: {(tool_name, args_norm) -> count}
    firing_counts: dict = {}

    def _selective(
        tool_name: str,
        tool_args: dict,
        tool_result: str,
        is_error: bool,
        trial_state: dict,
    ) -> str:
        # Writer side: identical to naive intercept (matched payload).
        if is_error and store is not None:
            store.write(
                trial=trial_state.get("trial", 0),
                tool_name=tool_name,
                tool_args=tool_args,
                error_text=tool_result,
            )

        if store is None:
            return tool_result

        hits = store.has_event_for(tool_name, tool_args)
        current_trial = trial_state.get("trial", 0)
        cross_trial = [h for h in hits if h.trial != current_trial]
        if not cross_trial:
            return tool_result

        # Confidence over candidate events; pick the highest-c hit.
        max_c = -1.0
        chosen = None
        from .trace_store import normalize_args
        key = (tool_name, normalize_args(tool_args))
        firing_counts.setdefault(key, 0)
        f_e = min(firing_counts[key] / 2.0, 1.0)  # cap at 2 fires
        for i, ev in enumerate(reversed(cross_trial)):
            r_e = 1.0 / (1 + i)  # newest → 1.0, decay
            d_e = 1.0  # exact (name, args) match → 1.0
            c = alpha * r_e + beta * (1.0 - f_e) + gamma * d_e
            if c > max_c:
                max_c = c
                chosen = ev
        if max_c <= theta or chosen is None:
            return tool_result
        firing_counts[key] = firing_counts.get(key, 0) + 1

        if chosen.event_type == "FAILED_PATH":
            body = (
                f"A previous trial got an error from this exact "
                f"call: {chosen.error_text[:200]}. Consider an "
                f"alternative approach."
            )
        elif chosen.event_type == "FAILED_TRIAL_ACTION":
            body = (
                f"A previous trial of this task made this exact "
                f"call as part of an unsuccessful attempt "
                f"(trial {chosen.trial} ended with reward<1). "
                f"Reconsider whether this call is on the right "
                f"path before relying on its result."
            )
        else:
            body = "A previous trial flagged this call."
        warning = f"[PEER-WARNING] {body}"
        return warning + "\n\n" + tool_result

    return _selective


def oracle_guidance_block(task: Any) -> str:
    """Format the task's golden action sequence as ground-truth guidance.

    Duck-typed against tau2's Task: reads
    ``task.evaluation_criteria.actions`` (each action has ``name``,
    ``arguments``, ``requestor``). Returns "" when the task has no
    golden actions (those tasks score 1.0 unconditionally, so guidance
    is moot).
    """
    ec = getattr(task, "evaluation_criteria", None)
    actions = (getattr(ec, "actions", None) if ec is not None else None) or []
    if not actions:
        return ""
    lines = [
        "<oracle_guidance>",
        "A prior attempt at this task did not resolve it. Ground-truth",
        "guidance: the correct resolution consists of exactly these",
        "actions, in this order (actions marked (user) are performed by",
        "the customer, not you — guide them through those):",
    ]
    for a in actions:
        args_compact = json.dumps(
            getattr(a, "arguments", {}) or {}, separators=(",", ":"), default=str
        )[:200]
        who = getattr(a, "requestor", "assistant")
        lines.append(f"- ({who}) {getattr(a, 'name', '?')}({args_compact})")
    lines.append(
        "Perform the assistant actions exactly as specified once the"
        " policy's confirmation requirements are met; do not perform"
        " actions outside this list that modify state."
    )
    lines.append("</oracle_guidance>")
    return "\n".join(lines)


def make_oracle_augmenter(task: Any) -> SystemPromptAugmenter:
    """Positive control: ground-truth guidance on the pull surface.

    Reviewer-requested oracle condition (KDD'26 workshop review): the
    paired protocol shows naive coordination does not clear the noise
    floor, but never demonstrates the measurement could detect a real
    gain if one existed. This condition injects the task's golden
    action sequence — perfect coordination content — through the same
    activation semantics as ``pull``:

    - trial 0 (empty store): returns the system prompt unchanged, so
      oracle remains configuration-equivalent with the other arms at
      trial 0 and the trial-0 floor measurement is undisturbed;
    - trial >0 with a non-empty store (i.e., a prior trial failed):
      appends an <oracle_guidance> block with the golden actions.

    If coordination-active pass^k cannot separate THIS from no_coord,
    the metric is insensitive; the measured gap upper-bounds what any
    real reader-side mechanism could show on this benchmark.
    """
    def _oracle(system: str, store: TraceStore) -> str:
        if len(store) == 0:
            return system
        block = oracle_guidance_block(task)
        if not block:
            return system
        return system + "\n\n" + block
    return _oracle


# Protocol configurations: (system_augmenter, intercept_hook_factory)
PROTOCOLS = {
    "no_coord": (_no_augment, lambda store: no_intercept),
    "pull": (pull_augmenter, make_pull_writer_hook),
    "intercept": (_no_augment, make_intercept_hook),
    "selective_intercept": (_no_augment, make_selective_intercept_hook),
    # Positive control (oracle): pull-surface writer hook (matched
    # payload with pull) + per-task ground-truth augmenter, resolved in
    # run_matrix via PER_TASK_AUGMENTERS.
    "oracle": (_no_augment, make_pull_writer_hook),
}

# Protocols whose system-prompt augmenter must be bound to the current
# task (the plain SystemPromptAugmenter signature has no task access).
PER_TASK_AUGMENTERS: dict[str, Callable[[Any], SystemPromptAugmenter]] = {
    "oracle": make_oracle_augmenter,
}

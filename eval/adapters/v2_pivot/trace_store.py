"""Cross-trial trace store for the v2-pivot coordination protocols.

Task-scoped: one store per task, persisted across the N trials of that
task. Failures from trial i are visible to trials i+1..k.

The store carries typed events biased toward negative knowledge
(FAILED_PATH, TOOL_ERROR). Each event records what was attempted and
what went wrong, keyed by (tool_name, normalized_args) so the
intercept protocol can hit-test future calls against past failures
in O(1).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


EventType = Literal[
    "FAILED_PATH",            # tool returned an error mid-trial
    "TOOL_ERROR",             # legacy alias
    "FAILED_TRIAL_ACTION",    # tool call made in a trial that ended reward<1
]


@dataclass
class TraceEvent:
    event_id: str
    event_type: EventType
    trial: int
    timestamp: float
    tool_name: str
    tool_args_norm: str
    error_text: str  # truncated
    raw_args: dict


def normalize_args(args: dict[str, Any]) -> str:
    """Hashable canonical form."""
    return json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class TraceStore:
    task_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def write(
        self,
        trial: int,
        tool_name: str,
        tool_args: dict,
        error_text: str,
        event_type: EventType = "FAILED_PATH",
    ) -> TraceEvent:
        evt = TraceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            trial=trial,
            timestamp=time.time(),
            tool_name=tool_name,
            tool_args_norm=normalize_args(tool_args),
            error_text=str(error_text)[:500],
            raw_args=tool_args or {},
        )
        self.events.append(evt)
        return evt

    def has_event_for(self, tool_name: str, tool_args: dict) -> list[TraceEvent]:
        """Return events whose (tool_name, normalized_args) match this call.

        Used by the `intercept` protocol to hit-test a pending call.
        """
        key = normalize_args(tool_args)
        return [
            e for e in self.events
            if e.tool_name == tool_name and e.tool_args_norm == key
        ]

    def warnings_block(self, max_events: int = 8) -> str:
        """Produce a compact <peer_warnings> block summarizing prior-trial
        failures, used by the `pull` protocol.

        Groups FAILED_TRIAL_ACTION events by trial and presents one
        bulleted summary per failed trial — this is more useful for the
        agent than a flat list of N tool calls, because the structure
        signals "trial K tried this sequence and failed".
        """
        if not self.events:
            return ""
        # Group action-type events by trial index
        by_trial: dict[int, list[TraceEvent]] = {}
        path_events = [e for e in self.events if e.event_type == "FAILED_PATH"]
        for e in self.events:
            if e.event_type == "FAILED_TRIAL_ACTION":
                by_trial.setdefault(e.trial, []).append(e)
        lines = ["<peer_warnings>"]
        if by_trial:
            lines.append(
                "Prior trials of this task that did NOT successfully complete:"
            )
            for trial_idx in sorted(by_trial.keys()):
                evs = by_trial[trial_idx]
                lines.append(f"- Trial {trial_idx} (failed) made these tool calls:")
                for e in evs[:max_events]:
                    args_compact = json.dumps(
                        e.raw_args, separators=(",", ":"), default=str
                    )[:200]
                    lines.append(f"    * {e.tool_name}({args_compact})")
        if path_events:
            lines.append(
                "Prior trials also recorded these in-flight tool errors:"
            )
            for e in path_events[-max_events:]:
                args_compact = json.dumps(
                    e.raw_args, separators=(",", ":"), default=str
                )[:200]
                lines.append(
                    f"- {e.tool_name}({args_compact}): {e.error_text[:150]}"
                )
        lines.append("</peer_warnings>")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.events)


# ---- module-level cache for cross-trial sharing ----------------------

_TASK_STORES: dict[str, TraceStore] = {}


def get_or_create(task_id: str) -> TraceStore:
    if task_id not in _TASK_STORES:
        _TASK_STORES[task_id] = TraceStore(task_id=task_id)
    return _TASK_STORES[task_id]


def reset_stores() -> None:
    _TASK_STORES.clear()

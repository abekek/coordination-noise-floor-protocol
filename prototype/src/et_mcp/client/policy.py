"""Client-side selection policy: what to write to the trace store, when.

The store accepts whatever the client writes. The selection policy is
client-side on purpose — different agent types should be able to pick
different policies, and the ablation in the paper varies this single
function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from et_mcp.events import (
    AbandonedApproach,
    ConstraintViolation,
    FailedPath,
    IntermediateDecision,
    ToolError,
)


class Policy(str, Enum):
    FAILURE_ONLY = "failure_only"
    WRITE_EVERYTHING = "write_everything"
    FAILURE_STRICT = "failure_strict"


@dataclass
class AgentStep:
    """The agent's per-step bundle. Any of the typed fields may be present."""

    outcome: str   # "succeeded" | "failed" | "ongoing"
    failed_path: FailedPath | None = None
    constraint_violation: ConstraintViolation | None = None
    abandoned: AbandonedApproach | None = None
    decision: IntermediateDecision | None = None
    tool_error: ToolError | None = None


# Each entry returns the payload to write, or None to skip.
def should_write(
    step: AgentStep, policy: Policy,
) -> (
    FailedPath | ConstraintViolation | AbandonedApproach
    | IntermediateDecision | ToolError | None
):
    if policy == Policy.FAILURE_STRICT:
        return step.failed_path or step.constraint_violation

    if policy == Policy.FAILURE_ONLY:
        if step.failed_path:
            return step.failed_path
        if step.constraint_violation:
            return step.constraint_violation
        if step.abandoned:
            return step.abandoned
        if step.tool_error:
            return step.tool_error
        if step.decision and not step.decision.reversible:
            return step.decision
        return None

    # WRITE_EVERYTHING
    return (
        step.failed_path
        or step.constraint_violation
        or step.abandoned
        or step.tool_error
        or step.decision
    )

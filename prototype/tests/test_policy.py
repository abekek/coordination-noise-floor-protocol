"""Tests for the selection-policy helper."""

from __future__ import annotations

import pytest

from et_mcp.client.policy import AgentStep, Policy, should_write
from et_mcp.events import (
    AbandonedApproach,
    ConstraintViolation,
    FailedPath,
    IntermediateDecision,
    ToolError,
)


def _step_failed():
    return AgentStep(
        outcome="failed",
        failed_path=FailedPath(
            approach="a", reason="r", constraints_hit=[], steps_taken=[],
        ),
    )


def _step_constraint():
    return AgentStep(
        outcome="ongoing",
        constraint_violation=ConstraintViolation(
            constraint="c", value_attempted=1, threshold=0,
        ),
    )


def _step_abandoned():
    return AgentStep(
        outcome="ongoing",
        abandoned=AbandonedApproach(
            description="d", why_abandoned="w", alternatives_considered=[],
        ),
    )


def _step_decision(reversible: bool):
    return AgentStep(
        outcome="ongoing",
        decision=IntermediateDecision(
            decision="d", reasoning="r", confidence=0.5, reversible=reversible,
        ),
    )


def _step_tool_error(recovered: bool):
    return AgentStep(
        outcome="ongoing",
        tool_error=ToolError(
            tool_name="t", input={}, error="e",
            retry_count=0, recovered=recovered,
        ),
    )


def _step_success():
    return AgentStep(outcome="succeeded")


class TestFailureOnly:
    def test_writes_failed_path(self):
        assert should_write(_step_failed(), Policy.FAILURE_ONLY) is not None

    def test_writes_constraint(self):
        assert should_write(_step_constraint(), Policy.FAILURE_ONLY) is not None

    def test_writes_abandoned(self):
        assert should_write(_step_abandoned(), Policy.FAILURE_ONLY) is not None

    def test_writes_tool_error(self):
        assert should_write(_step_tool_error(False), Policy.FAILURE_ONLY) is not None

    def test_writes_irreversible_decision(self):
        assert should_write(_step_decision(reversible=False),
                            Policy.FAILURE_ONLY) is not None

    def test_skips_reversible_decision(self):
        assert should_write(_step_decision(reversible=True),
                            Policy.FAILURE_ONLY) is None

    def test_skips_success(self):
        assert should_write(_step_success(), Policy.FAILURE_ONLY) is None


class TestWriteEverything:
    def test_writes_reversible_decision(self):
        assert should_write(_step_decision(reversible=True),
                            Policy.WRITE_EVERYTHING) is not None


class TestFailureStrict:
    def test_only_failed_and_constraint(self):
        assert should_write(_step_failed(), Policy.FAILURE_STRICT) is not None
        assert should_write(_step_constraint(), Policy.FAILURE_STRICT) is not None
        assert should_write(_step_abandoned(), Policy.FAILURE_STRICT) is None
        assert should_write(_step_tool_error(False),
                            Policy.FAILURE_STRICT) is None
        assert should_write(_step_decision(reversible=False),
                            Policy.FAILURE_STRICT) is None

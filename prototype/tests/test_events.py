"""Tests for the event envelope and the 5 event-type payload schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from et_mcp.events import (
    AbandonedApproach,
    ConstraintViolation,
    Event,
    EventType,
    FailedPath,
    IntermediateDecision,
    ToolError,
    event_type_schemas,
    new_event,
)


class TestFailedPath:
    def test_minimal_valid(self):
        FailedPath(
            approach="book via API X",
            reason="airline unavailable",
            constraints_hit=["budget"],
            steps_taken=["call X", "retry"],
        )

    def test_missing_required_rejected(self):
        with pytest.raises(ValidationError):
            FailedPath(approach="x")  # type: ignore[call-arg]


class TestConstraintViolation:
    def test_minimal_valid(self):
        ConstraintViolation(
            constraint="budget_per_night <= 200",
            value_attempted=250,
            threshold=200,
        )


class TestAbandonedApproach:
    def test_minimal_valid(self):
        AbandonedApproach(
            description="train route",
            why_abandoned="no service Sundays",
            alternatives_considered=["bus", "flight"],
        )


class TestIntermediateDecision:
    def test_confidence_bounds(self):
        IntermediateDecision(
            decision="use OpenAI for ranking",
            reasoning="latency budget",
            confidence=0.8,
            reversible=True,
        )
        with pytest.raises(ValidationError):
            IntermediateDecision(
                decision="d", reasoning="r", confidence=1.5, reversible=True
            )


class TestToolError:
    def test_minimal_valid(self):
        ToolError(
            tool_name="search_api",
            input={"q": "ny"},
            error="HTTP 500",
            retry_count=2,
            recovered=False,
        )


class TestEnvelope:
    def test_new_event_assigns_envelope_fields(self, task_id, agent_id):
        ev = new_event(
            task_id=task_id,
            agent_id=agent_id,
            payload=FailedPath(
                approach="x", reason="y", constraints_hit=[], steps_taken=[]
            ),
        )
        assert ev.task_id == task_id
        assert ev.agent_id == agent_id
        assert ev.event_type == EventType.FAILED_PATH
        assert ev.event_id  # non-empty
        assert ev.version == 1
        assert ev.timestamp.tzinfo is not None  # tz-aware

    def test_envelope_roundtrip_json(self, task_id, agent_id):
        ev = new_event(
            task_id=task_id,
            agent_id=agent_id,
            payload=ConstraintViolation(
                constraint="c", value_attempted=1, threshold=0
            ),
        )
        roundtripped = Event.model_validate_json(ev.model_dump_json())
        assert roundtripped == ev


class TestSchemaExport:
    def test_all_five_schemas_exported(self):
        schemas = event_type_schemas()
        assert set(schemas.keys()) == {
            EventType.FAILED_PATH,
            EventType.CONSTRAINT_VIOLATION,
            EventType.ABANDONED_APPROACH,
            EventType.INTERMEDIATE_DECISION,
            EventType.TOOL_ERROR,
        }
        for schema in schemas.values():
            assert schema["$schema"].endswith("2020-12/schema")

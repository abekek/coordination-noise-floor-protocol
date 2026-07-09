"""Event envelope and the 5 trace event-type payload schemas.

JSON Schema 2020-12 dialect is requested explicitly for the exported
schemas so that `trace.discover` returns spec-conformant artifacts.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    FAILED_PATH = "FAILED_PATH"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    ABANDONED_APPROACH = "ABANDONED_APPROACH"
    INTERMEDIATE_DECISION = "INTERMEDIATE_DECISION"
    TOOL_ERROR = "TOOL_ERROR"


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailedPath(_Payload):
    approach: str
    reason: str
    constraints_hit: list[str]
    steps_taken: list[str]
    evidence: str | None = None


class ConstraintViolation(_Payload):
    constraint: str
    value_attempted: Any
    threshold: Any
    context: str | None = None


class AbandonedApproach(_Payload):
    description: str
    why_abandoned: str
    alternatives_considered: list[str]


class IntermediateDecision(_Payload):
    decision: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    reversible: bool


class ToolError(_Payload):
    tool_name: str
    input: dict[str, Any]
    error: str
    retry_count: int = Field(ge=0)
    recovered: bool


_PAYLOAD_FOR_TYPE: dict[EventType, type[_Payload]] = {
    EventType.FAILED_PATH: FailedPath,
    EventType.CONSTRAINT_VIOLATION: ConstraintViolation,
    EventType.ABANDONED_APPROACH: AbandonedApproach,
    EventType.INTERMEDIATE_DECISION: IntermediateDecision,
    EventType.TOOL_ERROR: ToolError,
}


_TYPE_FOR_PAYLOAD: dict[type[_Payload], EventType] = {
    v: k for k, v in _PAYLOAD_FOR_TYPE.items()
}


class Event(BaseModel):
    """The wire envelope. `payload` is one of the 5 typed models above."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    task_id: str
    event_type: EventType
    agent_id: str
    timestamp: datetime
    version: int = 1
    payload: dict[str, Any]


def new_event(
    *,
    task_id: str,
    agent_id: str,
    payload: _Payload,
    version: int = 1,
) -> Event:
    event_type = _TYPE_FOR_PAYLOAD[type(payload)]
    return Event(
        event_id=str(uuid.uuid4()),
        task_id=task_id,
        event_type=event_type,
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        version=version,
        payload=payload.model_dump(),
    )


def event_type_schemas() -> dict[EventType, dict[str, Any]]:
    """Return JSON Schema 2020-12 for each payload type, keyed by EventType."""
    schemas: dict[EventType, dict[str, Any]] = {}
    for ev_type, model in _PAYLOAD_FOR_TYPE.items():
        schema = model.model_json_schema(mode="serialization")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schemas[ev_type] = schema
    return schemas


def payload_model_for(event_type: EventType) -> type[_Payload]:
    return _PAYLOAD_FOR_TYPE[event_type]

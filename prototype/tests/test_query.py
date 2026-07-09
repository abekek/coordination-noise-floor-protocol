"""Tests for the query pipeline (filter, rank, project)."""

from __future__ import annotations

import pytest

from et_mcp.events import (
    AbandonedApproach,
    ConstraintViolation,
    EventType,
    FailedPath,
    new_event,
)
from et_mcp.query import QueryResult, run_query
from et_mcp.store import TraceStore


async def _seed(store: TraceStore, task_id: str) -> None:
    await store.register_task(task_id)
    await store.append(new_event(
        task_id=task_id, agent_id="planner",
        payload=FailedPath(
            approach="book flight via Skyscanner",
            reason="API timeout",
            constraints_hit=["latency"],
            steps_taken=["GET /flights", "retry"],
        ),
    ))
    await store.append(new_event(
        task_id=task_id, agent_id="planner",
        payload=ConstraintViolation(
            constraint="budget_per_night <= 200",
            value_attempted=350,
            threshold=200,
        ),
    ))
    await store.append(new_event(
        task_id=task_id, agent_id="researcher",
        payload=AbandonedApproach(
            description="train route via Eurostar",
            why_abandoned="no weekend service",
            alternatives_considered=["bus", "flight"],
        ),
    ))


class TestFiltering:
    async def test_filter_by_event_type(self, task_id):
        store = TraceStore()
        await _seed(store, task_id)
        result = await run_query(
            store, task_id=task_id, question="anything",
            event_types=[EventType.FAILED_PATH],
        )
        assert all(e.event_type == EventType.FAILED_PATH for e in result.events)
        assert len(result.events) == 1

    async def test_filter_by_agent_id(self, task_id):
        store = TraceStore()
        await _seed(store, task_id)
        result = await run_query(
            store, task_id=task_id, question="anything",
            agent_id="researcher",
        )
        assert all(e.agent_id == "researcher" for e in result.events)


class TestRanking:
    async def test_question_keywords_boost_relevant_events(self, task_id):
        store = TraceStore()
        await _seed(store, task_id)
        result = await run_query(
            store, task_id=task_id,
            question="what paths failed for booking flights via API?",
        )
        assert isinstance(result, QueryResult)
        # The Skyscanner FAILED_PATH event should be top-ranked.
        assert result.events[0].event_type == EventType.FAILED_PATH


class TestProjection:
    async def test_limit_respected(self, task_id):
        store = TraceStore()
        await _seed(store, task_id)
        result = await run_query(
            store, task_id=task_id, question="anything", limit=2,
        )
        assert len(result.events) <= 2

    async def test_summary_field_populated(self, task_id):
        store = TraceStore()
        await _seed(store, task_id)
        result = await run_query(
            store, task_id=task_id, question="anything",
        )
        assert all(isinstance(s, str) and s for s in result.summaries)


class TestEmpty:
    async def test_unknown_task_raises(self):
        store = TraceStore()
        with pytest.raises(KeyError):
            await run_query(store, task_id="missing", question="?")

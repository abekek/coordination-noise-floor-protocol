"""Tests for the FastMCP server: the 4 MCP tools + host-only endpoints."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from et_mcp.events import EventType
from et_mcp.server import build_server


@pytest.fixture
def server():
    return build_server(default_ttl_s=60.0)


class TestHostEndpoints:
    def test_init_then_complete(self, server):
        client = TestClient(server.http_app())
        r = client.post("/tasks/t1/init", json={"owner": "orchestrator"})
        assert r.status_code == 200
        assert r.json()["task_id"] == "t1"
        r = client.post("/tasks/t1/complete")
        assert r.status_code == 200

    def test_init_requires_owner(self, server):
        client = TestClient(server.http_app())
        r = client.post("/tasks/t1/init", json={})
        assert r.status_code == 422

    def test_complete_unknown_task_is_204(self, server):
        client = TestClient(server.http_app())
        r = client.post("/tasks/never/complete")
        assert r.status_code in (200, 204)


class TestTraceWrite:
    async def test_write_valid_event(self, server):
        await server.lifecycle.init("t1", owner="o")
        result = await server.trace_write(
            task_id="t1",
            event_type="FAILED_PATH",
            agent_id="agent_a",
            payload={
                "approach": "x", "reason": "y",
                "constraints_hit": [], "steps_taken": [],
            },
        )
        assert result["event_id"]
        assert result["version"] == 1

    async def test_write_to_unknown_task_raises(self, server):
        with pytest.raises(KeyError):
            await server.trace_write(
                task_id="missing",
                event_type="FAILED_PATH",
                agent_id="a",
                payload={
                    "approach": "x", "reason": "y",
                    "constraints_hit": [], "steps_taken": [],
                },
            )

    async def test_write_invalid_payload_raises(self, server):
        await server.lifecycle.init("t1", owner="o")
        with pytest.raises(ValueError):
            await server.trace_write(
                task_id="t1",
                event_type="FAILED_PATH",
                agent_id="a",
                payload={"approach": "only this field"},
            )


class TestTraceQuery:
    async def test_query_returns_matches(self, server):
        await server.lifecycle.init("t1", owner="o")
        await server.trace_write(
            task_id="t1", event_type="FAILED_PATH", agent_id="a",
            payload={"approach": "book flight", "reason": "timeout",
                     "constraints_hit": [], "steps_taken": []},
        )
        result = await server.trace_query(
            task_id="t1", question="what failed?",
        )
        assert len(result["events"]) == 1
        assert result["events"][0]["event_type"] == "FAILED_PATH"

    async def test_query_filter_by_event_type(self, server):
        await server.lifecycle.init("t1", owner="o")
        await server.trace_write(
            task_id="t1", event_type="FAILED_PATH", agent_id="a",
            payload={"approach": "x", "reason": "y",
                     "constraints_hit": [], "steps_taken": []},
        )
        await server.trace_write(
            task_id="t1", event_type="TOOL_ERROR", agent_id="a",
            payload={"tool_name": "t", "input": {}, "error": "e",
                     "retry_count": 0, "recovered": False},
        )
        result = await server.trace_query(
            task_id="t1", question="anything",
            event_types=["FAILED_PATH"],
        )
        assert len(result["events"]) == 1
        assert result["events"][0]["event_type"] == "FAILED_PATH"


class TestTraceCas:
    async def test_initial_set(self, server):
        await server.lifecycle.init("t1", owner="o")
        r = await server.trace_cas(
            task_id="t1", key="approach:x",
            expected_version=None, new_payload={"status": "claimed"},
        )
        assert r["ok"] is True
        assert r["version"] == 1

    async def test_conflict(self, server):
        await server.lifecycle.init("t1", owner="o")
        await server.trace_cas(
            task_id="t1", key="k", expected_version=None,
            new_payload={"v": 1},
        )
        r = await server.trace_cas(
            task_id="t1", key="k", expected_version=42,
            new_payload={"v": 2},
        )
        assert r["ok"] is False


class TestTraceDiscover:
    async def test_returns_five_event_types(self, server):
        await server.lifecycle.init("t1", owner="o")
        r = await server.trace_discover(task_id="t1")
        names = {et["name"] for et in r["event_types"]}
        assert names == {
            EventType.FAILED_PATH.value,
            EventType.CONSTRAINT_VIOLATION.value,
            EventType.ABANDONED_APPROACH.value,
            EventType.INTERMEDIATE_DECISION.value,
            EventType.TOOL_ERROR.value,
        }
        assert r["task_meta"]["task_id"] == "t1"
        assert r["query_examples"]

    async def test_discover_without_task(self, server):
        r = await server.trace_discover(task_id=None)
        assert r["task_meta"] is None
        assert len(r["event_types"]) == 5

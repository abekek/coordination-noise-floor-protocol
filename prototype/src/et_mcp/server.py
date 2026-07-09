"""FastMCP server exposing the 4 ET-MCP tools + host-only lifecycle routes.

Tools (agent-callable, registered with FastMCP):
    trace.write, trace.query, trace.cas, trace.discover

Host endpoints (NOT exposed as MCP tools, mounted as Starlette routes):
    POST /tasks/{task_id}/init     {owner: str, ttl_s?: float}
    POST /tasks/{task_id}/complete

The split prevents agents from initializing or tearing down other agents'
task namespaces.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from et_mcp.events import EventType, event_type_schemas, new_event, payload_model_for
from et_mcp.lifecycle import TaskLifecycleManager
from et_mcp.query import run_query
from et_mcp.store import TraceStore


_QUERY_EXAMPLES = [
    "what paths failed for booking flights?",
    "what constraints did agent_planner hit?",
    "what approaches were considered and abandoned?",
    "what irreversible decisions has the task already made?",
    "which tools are flaky for this input shape?",
]


def _query_hint(event_type: EventType) -> str:
    return {
        EventType.FAILED_PATH: "ask this when planning a new approach",
        EventType.CONSTRAINT_VIOLATION: "ask this when proposing a value",
        EventType.ABANDONED_APPROACH: "ask this when ranking alternatives",
        EventType.INTERMEDIATE_DECISION: "ask this before reversing a commitment",
        EventType.TOOL_ERROR: "ask this before retrying a flaky tool",
    }[event_type]


class _InitBody(BaseModel):
    owner: str
    ttl_s: float | None = None


@dataclass
class EtMcpServer:
    """Container holding the FastMCP instance + lifecycle + store + HTTP app."""

    mcp: FastMCP
    store: TraceStore
    lifecycle: TaskLifecycleManager

    def http_app(self) -> Starlette:
        return _build_http_app(self)

    async def trace_write(
        self,
        *,
        task_id: str,
        event_type: str,
        agent_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        et = EventType(event_type)
        try:
            typed_payload = payload_model_for(et)(**payload)
        except ValidationError as exc:
            raise ValueError(f"invalid payload for {event_type}: {exc}") from exc
        ev = new_event(
            task_id=task_id,
            agent_id=agent_id,
            payload=typed_payload,
        )
        await self.store.append(ev)
        return {"event_id": ev.event_id, "version": ev.version}

    async def trace_query(
        self,
        *,
        task_id: str,
        question: str,
        event_types: list[str] | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        ets = [EventType(t) for t in event_types] if event_types else None
        result = await run_query(
            self.store,
            task_id=task_id,
            question=question,
            event_types=ets,
            agent_id=agent_id,
            limit=limit,
        )
        return {
            "events": [e.model_dump(mode="json") for e in result.events],
            "summaries": result.summaries,
            "scores": result.scores,
        }

    async def trace_cas(
        self,
        *,
        task_id: str,
        key: str,
        expected_version: int | None,
        new_payload: dict[str, Any],
    ) -> dict[str, Any]:
        r = await self.store.cas(
            task_id=task_id,
            key=key,
            expected_version=expected_version,
            new_payload=new_payload,
        )
        return {"ok": r.ok, "current": r.current, "version": r.version}

    async def trace_discover(self, *, task_id: str | None = None) -> dict[str, Any]:
        schemas = event_type_schemas()
        event_types = [
            {
                "name": et.value,
                "schema": schemas[et],
                "query_hint": _query_hint(et),
            }
            for et in EventType
        ]
        task_meta = None
        if task_id is not None and (meta := self.lifecycle.meta(task_id)):
            task_meta = {
                "task_id": meta.task_id,
                "owner": meta.owner,
                "started_at": meta.started_at,
                "ttl_s": meta.ttl_s,
            }
        return {
            "event_types": event_types,
            "task_meta": task_meta,
            "query_examples": list(_QUERY_EXAMPLES),
        }


def build_server(*, default_ttl_s: float = 3600.0) -> EtMcpServer:
    mcp = FastMCP("et-mcp")
    store = TraceStore()
    lifecycle = TaskLifecycleManager(store=store, default_ttl_s=default_ttl_s)
    server = EtMcpServer(mcp=mcp, store=store, lifecycle=lifecycle)

    @mcp.tool(name="trace.write", description="Append a typed trace event to this task's store.")
    async def _w(
        task_id: str,
        event_type: str,
        agent_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await server.trace_write(
            task_id=task_id,
            event_type=event_type,
            agent_id=agent_id,
            payload=payload,
        )

    @mcp.tool(
        name="trace.query",
        description="Pull-based peer query over this task's trace events.",
    )
    async def _q(
        task_id: str,
        question: str,
        event_types: list[str] | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return await server.trace_query(
            task_id=task_id,
            question=question,
            event_types=event_types,
            agent_id=agent_id,
            limit=limit,
        )

    @mcp.tool(
        name="trace.cas",
        description="Compare-and-swap for cooperative shared-key writes.",
    )
    async def _c(
        task_id: str,
        key: str,
        expected_version: int | None,
        new_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await server.trace_cas(
            task_id=task_id,
            key=key,
            expected_version=expected_version,
            new_payload=new_payload,
        )

    @mcp.tool(
        name="trace.discover",
        description="List available event types, their JSON Schemas, and task metadata.",
    )
    async def _d(task_id: str | None = None) -> dict[str, Any]:
        return await server.trace_discover(task_id=task_id)

    return server


def _build_http_app(server: EtMcpServer) -> Starlette:
    async def _init(request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        try:
            body = _InitBody.model_validate(await request.json())
        except (ValidationError, ValueError):
            return JSONResponse({"error": "owner required"}, status_code=422)
        try:
            meta = await server.lifecycle.init(
                task_id,
                owner=body.owner,
                ttl_s=body.ttl_s,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(
            {
                "task_id": meta.task_id,
                "owner": meta.owner,
                "started_at": meta.started_at,
                "ttl_s": meta.ttl_s,
            }
        )

    async def _complete(request: Request) -> JSONResponse:
        task_id = request.path_params["task_id"]
        await server.lifecycle.complete(task_id)
        return JSONResponse({"task_id": task_id, "status": "completed"})

    # Mount FastMCP's streamable HTTP transport at /mcp so trace.* tools are reachable.
    # Pass mcp_app.router.lifespan_context to the outer Starlette so FastMCP's
    # startup/shutdown hooks (session manager, etc.) run correctly.
    mcp_app = server.mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def _lifespan(app):
        sweeper = asyncio.create_task(_sweep_loop(server.lifecycle))
        try:
            async with mcp_app.router.lifespan_context(app):
                yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper

    return Starlette(
        routes=[
            Route("/tasks/{task_id}/init", _init, methods=["POST"]),
            Route("/tasks/{task_id}/complete", _complete, methods=["POST"]),
            Mount("/mcp", app=mcp_app),
        ],
        lifespan=_lifespan,
    )


async def _sweep_loop(
    lifecycle: TaskLifecycleManager,
    *,
    interval_s: float = 60.0,
) -> None:
    """Background task: periodically evict expired task namespaces.

    Started from the Starlette lifespan in `main`; cancelled on shutdown.
    """
    while True:
        await asyncio.sleep(interval_s)
        await lifecycle.sweep_expired()


def main() -> None:
    """Entry point for `python -m et_mcp.server`."""
    import uvicorn

    server = build_server()
    app = server.http_app()
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()

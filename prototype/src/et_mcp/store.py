"""In-memory trace store, task-scoped, with per-task locks and CAS keys."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from et_mcp.events import Event


@dataclass
class _CasEntry:
    payload: dict[str, Any]
    version: int


@dataclass
class _TaskState:
    events: list[Event] = field(default_factory=list)
    cas: dict[str, _CasEntry] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class CasResult:
    ok: bool
    current: dict[str, Any]
    version: int


class TraceStore:
    """Multi-tenant in-memory store; one namespace per task_id."""

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskState] = {}
        self._registry_lock = asyncio.Lock()

    async def register_task(self, task_id: str) -> None:
        async with self._registry_lock:
            if task_id in self._tasks:
                raise ValueError(f"task {task_id!r} already registered")
            self._tasks[task_id] = _TaskState()

    async def unregister_task(self, task_id: str) -> None:
        async with self._registry_lock:
            self._tasks.pop(task_id, None)

    def _state(self, task_id: str) -> _TaskState:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"task {task_id!r} not registered") from exc

    async def append(self, event: Event) -> None:
        state = self._state(event.task_id)
        async with state.lock:
            state.events.append(event)

    async def list_for_task(self, task_id: str) -> list[Event]:
        state = self._state(task_id)
        async with state.lock:
            return list(state.events)

    async def cas(
        self,
        task_id: str,
        key: str,
        expected_version: int | None,
        new_payload: dict[str, Any],
    ) -> CasResult:
        state = self._state(task_id)
        async with state.lock:
            current = state.cas.get(key)
            if current is None:
                if expected_version not in (None, 0):
                    return CasResult(ok=False, current={}, version=0)
                state.cas[key] = _CasEntry(payload=dict(new_payload), version=1)
                return CasResult(ok=True, current=dict(new_payload), version=1)
            if expected_version != current.version:
                return CasResult(
                    ok=False,
                    current=dict(current.payload),
                    version=current.version,
                )
            current.payload = dict(new_payload)
            current.version += 1
            return CasResult(
                ok=True,
                current=dict(current.payload),
                version=current.version,
            )

    def registered_tasks(self) -> list[str]:
        return list(self._tasks.keys())

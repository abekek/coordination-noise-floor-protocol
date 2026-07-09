"""Task lifecycle: init, complete, TTL eviction.

This module owns the *namespace* portion of ET-MCP. The trace store
holds the data; the lifecycle manager decides when a task's namespace
exists and when it should be torn down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from et_mcp.store import TraceStore


@dataclass
class TaskMeta:
    task_id: str
    owner: str
    started_at: float       # unix seconds
    ttl_s: float
    completed_at: float | None = None


class TaskLifecycleManager:
    def __init__(self, *, store: TraceStore, default_ttl_s: float = 3600.0) -> None:
        self._store = store
        self._default_ttl = default_ttl_s
        self._meta: dict[str, TaskMeta] = {}

    async def init(self, task_id: str, *, owner: str,
                   ttl_s: float | None = None) -> TaskMeta:
        if task_id in self._meta:
            raise ValueError(f"task {task_id!r} already initialized")
        await self._store.register_task(task_id)
        meta = TaskMeta(
            task_id=task_id,
            owner=owner,
            started_at=time.time(),
            ttl_s=ttl_s if ttl_s is not None else self._default_ttl,
        )
        self._meta[task_id] = meta
        return meta

    async def complete(self, task_id: str) -> None:
        if task_id not in self._meta:
            return
        await self._store.unregister_task(task_id)
        self._meta.pop(task_id, None)

    def meta(self, task_id: str) -> TaskMeta | None:
        return self._meta.get(task_id)

    def all_meta(self) -> list[TaskMeta]:
        return list(self._meta.values())

    async def sweep_expired(self, *, now: float | None = None) -> list[str]:
        cutoff_now = now if now is not None else time.time()
        expired = [
            tid for tid, m in self._meta.items()
            if (cutoff_now - m.started_at) >= m.ttl_s
        ]
        for tid in expired:
            await self.complete(tid)
        return expired

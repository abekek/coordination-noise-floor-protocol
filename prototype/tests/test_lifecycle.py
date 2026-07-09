"""Tests for TaskLifecycleManager: init, complete, TTL."""

from __future__ import annotations

import asyncio

import pytest

from et_mcp.lifecycle import TaskLifecycleManager, TaskMeta
from et_mcp.store import TraceStore


class TestInitAndComplete:
    async def test_init_registers_in_store(self):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=60.0)
        await mgr.init("t1", owner="orchestrator_a")
        assert "t1" in store.registered_tasks()
        meta = mgr.meta("t1")
        assert isinstance(meta, TaskMeta) and meta.owner == "orchestrator_a"

    async def test_double_init_rejected(self):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=60.0)
        await mgr.init("t1", owner="o")
        with pytest.raises(ValueError):
            await mgr.init("t1", owner="o")

    async def test_complete_unregisters(self):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=60.0)
        await mgr.init("t1", owner="o")
        await mgr.complete("t1")
        assert "t1" not in store.registered_tasks()
        assert mgr.meta("t1") is None

    async def test_complete_unknown_task_is_noop(self):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=60.0)
        await mgr.complete("never_existed")  # must not raise


class TestTtl:
    async def test_ttl_evicts(self, monkeypatch):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=0.05)
        await mgr.init("t1", owner="o")
        await asyncio.sleep(0.1)
        await mgr.sweep_expired()
        assert "t1" not in store.registered_tasks()

    async def test_explicit_complete_before_ttl(self):
        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=10.0)
        await mgr.init("t1", owner="o")
        await mgr.complete("t1")
        # Sweep should be a no-op now
        await mgr.sweep_expired()
        assert "t1" not in store.registered_tasks()


class TestSweepLoop:
    async def test_loop_evicts_expired_tasks_periodically(self):
        import asyncio
        import contextlib

        from et_mcp.server import _sweep_loop

        store = TraceStore()
        mgr = TaskLifecycleManager(store=store, default_ttl_s=0.05)
        await mgr.init("t1", owner="o")

        loop_task = asyncio.create_task(_sweep_loop(mgr, interval_s=0.02))
        await asyncio.sleep(0.15)
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        assert "t1" not in store.registered_tasks()

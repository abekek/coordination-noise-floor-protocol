"""Tests for Silo-Bench LLM bridge helpers."""

from __future__ import annotations

from benchmarks.silo_bench import llm_bridge


def test_maybe_throttle_sleeps_between_calls(monkeypatch):
    sleeps: list[float] = []
    ticks = iter([100.0, 102.0, 102.0])

    monkeypatch.setenv("SILO_LLM_CALL_COOLDOWN_SECONDS", "10")
    monkeypatch.setattr(llm_bridge.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(llm_bridge.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_bridge, "_last_call_at", None)

    llm_bridge._maybe_throttle()
    llm_bridge._maybe_throttle()

    assert sleeps == [8.0]


def test_maybe_throttle_uses_token_budget(monkeypatch):
    sleeps: list[float] = []
    ticks = iter([100.0, 105.0, 115.0])

    monkeypatch.setenv("SILO_LLM_CALL_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("SILO_LLM_INPUT_TOKENS_PER_MINUTE", "30000")
    monkeypatch.setattr(llm_bridge.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(llm_bridge.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_bridge, "_last_call_at", None)
    monkeypatch.setattr(llm_bridge, "_last_input_tokens", None)

    llm_bridge._maybe_throttle()
    monkeypatch.setattr(llm_bridge, "_last_input_tokens", 15000)
    llm_bridge._maybe_throttle()

    assert sleeps == [25.0]


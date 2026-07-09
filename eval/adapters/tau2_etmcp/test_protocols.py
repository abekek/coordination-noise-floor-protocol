"""Tests for the 4 coordination protocols.

These tests verify writer/reader behavior at the TraceStore + warning-block
layer; they do NOT call an LLM. End-to-end LLM tests live in the Day 3
smoke harness.

Run from the tau2 venv:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        -m pytest eval/adapters/tau2_etmcp/test_protocols.py -v
or, since we may not have pytest in the tau2 venv:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        eval/adapters/tau2_etmcp/test_protocols.py
"""

from __future__ import annotations

import pathlib
import sys


def _setup_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_setup_path()


from eval.adapters.tau2_etmcp.et_mcp_agent import (  # noqa: E402
    SUPPORTED_PROTOCOLS,
    ETMCPAgent,
    get_or_create_task_store,
    reset_task_stores,
)
from eval.adapters.tau2_etmcp.trace_store import TraceStore  # noqa: E402


# Minimal ToolMessage shim — tau2's actual ToolMessage uses Pydantic with
# extra fields we don't need to construct here. Duck-typed against
# _record_tool_failure's getattr-based access.
class _ToolMsg:
    def __init__(self, error: bool, content: str, requestor: str, tool_call_id: str = "tc1"):
        self.error = error
        self.content = content
        self.requestor = requestor
        self.tool_call_id = tool_call_id


def _make_agent(protocol: str, task_id: str = "T1") -> ETMCPAgent:
    """Build an agent without touching LLMConfigMixin's litellm probe."""
    # ETMCPAgent.__init__ calls super().__init__ → LLMAgent.__init__ →
    # LLMConfigMixin.__init__, which validates the llm string. We pass a
    # safe-looking one and never call generate_next_message in these tests.
    return ETMCPAgent(
        tools=[],
        domain_policy="test policy",
        llm="anthropic/claude-haiku-4-5",
        coord_protocol=protocol,
        task_id=task_id,
    )


def test_task_store_caches_across_trials() -> None:
    reset_task_stores()
    s1 = get_or_create_task_store("task42")
    s2 = get_or_create_task_store("task42")
    assert s1 is s2, "expected same store instance across calls"
    s3 = get_or_create_task_store("task99")
    assert s3 is not s1
    print("PASS test_task_store_caches_across_trials")


def test_no_coord_uses_isolated_store() -> None:
    reset_task_stores()
    a = _make_agent("no_coord", task_id="T1")
    b = _make_agent("no_coord", task_id="T1")
    assert a.trace_store is not b.trace_store, (
        "no_coord must give each agent its own store"
    )
    print("PASS test_no_coord_uses_isolated_store")


def test_shared_protocols_share_store_across_trials() -> None:
    for proto in ["push_scratchpad", "message_passing", "et_mcp"]:
        reset_task_stores()
        a = _make_agent(proto, task_id="T1")
        b = _make_agent(proto, task_id="T1")
        assert a.trace_store is b.trace_store, (
            f"protocol={proto} should share store across trials of same task"
        )
        print(f"PASS test_shared_protocols_share_store_across_trials[{proto}]")


def test_record_tool_failure_writes_on_error() -> None:
    reset_task_stores()
    a = _make_agent("et_mcp", task_id="T2")
    a._record_tool_failure(_ToolMsg(error=True, content="boom", requestor="search"))
    assert len(a.trace_store) == 1
    e = a.trace_store.events[0]
    assert e.event_type == "FAILED_PATH"
    assert e.payload["tool_name"] == "search"
    assert "boom" in e.payload["error_text"]
    print("PASS test_record_tool_failure_writes_on_error")


def test_record_tool_failure_skips_on_success() -> None:
    reset_task_stores()
    a = _make_agent("et_mcp", task_id="T3")
    a._record_tool_failure(_ToolMsg(error=False, content="ok", requestor="search"))
    assert len(a.trace_store) == 0
    print("PASS test_record_tool_failure_skips_on_success")


def test_record_tool_failure_skips_in_no_coord() -> None:
    reset_task_stores()
    a = _make_agent("no_coord", task_id="T4")
    a._record_tool_failure(_ToolMsg(error=True, content="boom", requestor="search"))
    assert len(a.trace_store) == 0
    print("PASS test_record_tool_failure_skips_in_no_coord")


def test_et_mcp_block_returns_warnings_when_relevant() -> None:
    reset_task_stores()
    store = get_or_create_task_store("T5")
    store.write(
        "FAILED_PATH",
        agent_id="executor",
        payload={"tool_name": "search_flights", "error_text": "no inventory"},
    )
    a = _make_agent("et_mcp", task_id="T5")
    # Synthesize a minimal state with a relevant question
    from tau2.data_model.message import SystemMessage, UserMessage
    state_messages = [UserMessage(role="user", content="search flights")]
    state = type(
        "S", (), {"messages": state_messages, "system_messages": [SystemMessage(role="system", content="x")]}
    )()
    block = a._build_et_mcp_block(state)
    assert block is not None and "peer_warnings" in block
    assert "search_flights" in block
    print("PASS test_et_mcp_block_returns_warnings_when_relevant")


def test_push_scratchpad_dumps_all_events() -> None:
    reset_task_stores()
    store = get_or_create_task_store("T6")
    store.write("FAILED_PATH", agent_id="e1", payload={"tool_name": "a", "error_text": "x"})
    store.write("FAILED_PATH", agent_id="e2", payload={"tool_name": "b", "error_text": "y"})
    a = _make_agent("push_scratchpad", task_id="T6")
    block = a._build_push_scratchpad_block()
    assert block is not None
    assert "prior_trial_dump" in block
    assert "agent=e1" in block and "agent=e2" in block
    print("PASS test_push_scratchpad_dumps_all_events")


def test_message_passing_summarizes_failures() -> None:
    reset_task_stores()
    store = get_or_create_task_store("T7")
    store.write("FAILED_PATH", agent_id="e1", payload={"tool_name": "a", "error_text": "x"})
    store.write("FAILED_PATH", agent_id="e1", payload={"tool_name": "b", "error_text": "y"})
    a = _make_agent("message_passing", task_id="T7")
    block = a._build_message_passing_block()
    assert block is not None
    assert "peer_handoff" in block
    assert "2 failure event" in block
    assert "a" in block and "b" in block
    print("PASS test_message_passing_summarizes_failures")


def test_no_coord_never_emits_warning_block() -> None:
    reset_task_stores()
    a = _make_agent("no_coord", task_id="T8")
    # Even if a store happened to have events, no_coord should never read.
    a.trace_store.write("FAILED_PATH", agent_id="e", payload={"tool_name": "a", "error_text": "x"})
    from tau2.data_model.message import SystemMessage, UserMessage
    state = type(
        "S", (), {"messages": [UserMessage(role="user", content="hi")], "system_messages": [SystemMessage(role="system", content="x")]}
    )()
    assert a._build_warning_block(state) is None
    print("PASS test_no_coord_never_emits_warning_block")


def test_empty_store_returns_none_for_all_protocols() -> None:
    for proto in SUPPORTED_PROTOCOLS - {"no_coord"}:
        reset_task_stores()
        a = _make_agent(proto, task_id=f"empty_{proto}")
        from tau2.data_model.message import SystemMessage, UserMessage
        state = type(
            "S", (), {"messages": [UserMessage(role="user", content="hi")], "system_messages": [SystemMessage(role="system", content="x")]}
        )()
        assert a._build_warning_block(state) is None, (
            f"protocol={proto} should return None when store is empty"
        )
        print(f"PASS test_empty_store_returns_none_for_all_protocols[{proto}]")


def test_unknown_protocol_raises() -> None:
    reset_task_stores()
    try:
        _make_agent("bogus", task_id="T9")
    except ValueError as exc:
        assert "bogus" in str(exc)
        print("PASS test_unknown_protocol_raises")
        return
    raise AssertionError("expected ValueError for unknown protocol")


if __name__ == "__main__":
    tests = [
        test_task_store_caches_across_trials,
        test_no_coord_uses_isolated_store,
        test_shared_protocols_share_store_across_trials,
        test_record_tool_failure_writes_on_error,
        test_record_tool_failure_skips_on_success,
        test_record_tool_failure_skips_in_no_coord,
        test_et_mcp_block_returns_warnings_when_relevant,
        test_push_scratchpad_dumps_all_events,
        test_message_passing_summarizes_failures,
        test_no_coord_never_emits_warning_block,
        test_empty_store_returns_none_for_all_protocols,
        test_unknown_protocol_raises,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        sys.exit(1)
    print(f"\nAll {len(tests)} test functions passed.")

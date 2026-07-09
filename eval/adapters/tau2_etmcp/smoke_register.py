"""Smoke check: import our adapter and verify tau2 sees the agent.

Run from the tau2 venv:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        eval/adapters/tau2_etmcp/smoke_register.py

Should print: "Registered: True; available agents: [...et_mcp_agent...]"
and exit 0.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Ensure adapter package is importable when running this script directly
    # against the tau2 venv (which doesn't have our repo on sys.path by default).
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    # Importing factory triggers registration as a side effect.
    from eval.adapters.tau2_etmcp import factory  # noqa: F401

    from tau2.registry import registry

    f = registry.get_agent_factory("et_mcp_agent")
    if f is None:
        print("FAIL: et_mcp_agent not registered.")
        return 1

    info = registry.get_info()
    # RegistryInfo is a Pydantic model in newer tau2; fall back to attribute access.
    agents = (
        info.get("agents", []) if isinstance(info, dict) else getattr(info, "agents", [])
    )
    print(f"Registered: True; available agents: {agents}")

    # Sanity-check the trace store + agent class import cleanly
    from eval.adapters.tau2_etmcp.trace_store import TraceStore
    from eval.adapters.tau2_etmcp.et_mcp_agent import ETMCPAgent

    store = TraceStore(task_id="smoke_test")
    store.write(
        "FAILED_PATH",
        agent_id="executor",
        payload={"tool_name": "test_tool", "error_text": "boom"},
    )
    hits = store.query("test_tool boom", limit=3)
    print(f"TraceStore wrote 1, query returned {len(hits)} hit(s).")
    assert len(hits) == 1, "TraceStore query should return the written event."

    print("OK: smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

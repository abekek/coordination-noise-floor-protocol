"""Tests for the minimal LangGraph-friendly wrapper."""

from __future__ import annotations

import pytest

from et_mcp.client.langgraph import EtMcpNode
from et_mcp.client.policy import AgentStep, Policy
from et_mcp.events import FailedPath
from et_mcp.server import build_server


@pytest.fixture
async def node():
    server = build_server()
    await server.lifecycle.init("t1", owner="orchestrator")
    return EtMcpNode(
        server=server, task_id="t1", agent_id="agent_alpha",
        policy=Policy.FAILURE_ONLY,
    )


async def test_record_writes_failed_path(node):
    await node.record(AgentStep(
        outcome="failed",
        failed_path=FailedPath(
            approach="a", reason="r", constraints_hit=[], steps_taken=[],
        ),
    ))
    result = await node.query("what failed?")
    assert len(result["events"]) == 1


async def test_record_skips_reversible_success(node):
    await node.record(AgentStep(outcome="succeeded"))
    result = await node.query("anything")
    assert result["events"] == []


async def test_peer_only_excludes_own_agent_events(node):
    """When peer_only=True, query results exclude events authored by self."""
    # Use the same node to write
    await node.record(AgentStep(
        outcome="failed",
        failed_path=FailedPath(
            approach="self_attempt", reason="r",
            constraints_hit=[], steps_taken=[],
        ),
    ))
    # Create a peer node writing under a different agent_id
    peer = EtMcpNode(
        server=node.server, task_id=node.task_id,
        agent_id="agent_beta", policy=Policy.FAILURE_ONLY,
    )
    await peer.record(AgentStep(
        outcome="failed",
        failed_path=FailedPath(
            approach="peer_attempt", reason="r",
            constraints_hit=[], steps_taken=[],
        ),
    ))
    # Without peer_only: see both events
    all_results = await node.query("anything")
    assert len(all_results["events"]) == 2
    # With peer_only=True: see only peer event (not own)
    peer_results = await node.query("anything", peer_only=True)
    assert len(peer_results["events"]) == 1
    assert peer_results["events"][0]["agent_id"] == "agent_beta"


async def test_peer_only_false_returns_all_events(node):
    """peer_only=False (default) returns all events including own."""
    await node.record(AgentStep(
        outcome="failed",
        failed_path=FailedPath(
            approach="self_attempt", reason="r",
            constraints_hit=[], steps_taken=[],
        ),
    ))
    all_results = await node.query("anything", peer_only=False)
    assert len(all_results["events"]) == 1
    assert all_results["events"][0]["agent_id"] == "agent_alpha"

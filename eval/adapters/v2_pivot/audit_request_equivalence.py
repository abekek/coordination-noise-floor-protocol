"""Offline SHA-256 audit of the trial-0 request-equivalence claim.

The §6.3.0 "Request-equivalence" claim is that at trial~0 of any task,
with an empty TraceStore, the three protocols (no_coord, pull,
intercept) construct an IDENTICAL Anthropic Messages API request
payload before the first call. Specifically:

  * pull's system_augmenter is a no-op when warnings_block() is empty
    (empty TraceStore -> no warnings).
  * intercept's reader hook fires only on tool calls whose key matches
    a prior FAILED_PATH event; with no events recorded, it's a
    pass-through.
  * The tool list is built from env.get_tools() protocol-independently.
  * The user-sim prompt, priming message, system, scenario, and
    sampling params are shared verbatim.

This script verifies that claim by BUILDING the request payload that
each protocol would send for its first agent-side and first
user-simulator-side API call, then SHA-256-hashing the serialized
payload. It does NOT make any network call -- so it costs $0, runs
in seconds, and tests a strictly stronger property than a live run
(a live run would diverge on the very first sampled token even when
the request payload was identical, because the API isn't seeded).

Usage:
    python -m eval.adapters.v2_pivot.audit_request_equivalence \
        --domain retail \
        --task-ids 16 22 32 49 67 \
        --out /tmp/hash_audit.jsonl

The script writes one JSONL line per (task, role, protocol) showing
the sha256 digest, then prints a cross-protocol agreement summary.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Any

_REPO = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.adapters.v2_pivot.conversation import Conversation
from eval.adapters.v2_pivot.orchestrator import (
    AGENT_SYSTEM_TEMPLATE,
    USER_SYSTEM_TEMPLATE,
)
from eval.adapters.v2_pivot.protocols import PROTOCOLS
from eval.adapters.v2_pivot.tools_adapter import all_tools_to_anthropic
from eval.adapters.v2_pivot.trace_store import (
    TraceStore, get_or_create, reset_stores,
)


def _hash_payload(system: str, tools: list[dict], messages: list[dict]) -> str:
    payload = {"system": system, "tools": tools, "messages": messages}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _load_domain(domain: str):
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment, get_tasks
    elif domain == "airline":
        from tau2.domains.airline.environment import get_environment, get_tasks
    else:
        raise ValueError(f"unsupported domain: {domain}")
    return get_environment, get_tasks


def build_first_request_payloads(task, env, proto: str) -> dict[str, dict]:
    """For protocol `proto` at trial 0 (empty store), build the request
    payload that would be sent on:
      (a) the first user-simulator-side API call (the priming `Begin.`)
      (b) the first agent-side API call (after the user-sim's opening msg)

    Returns dict keyed by role -> {system, tools, messages, sha256}.
    Does NOT execute the model; just constructs the kwargs the
    orchestrator would pass to client.messages.create.
    """
    sys_aug, _hook_factory = PROTOCOLS[proto]
    reset_stores()
    store = get_or_create(str(task.id))
    assert len(store) == 0, "store should be empty at trial 0 of first task"

    # ---- agent conversation construction (mirrors run_trial) ----
    base_system = AGENT_SYSTEM_TEMPLATE.format(policy=env.get_policy())
    agent_system = sys_aug(base_system, store)
    anth_tools = all_tools_to_anthropic(env.get_tools())
    agent_conv = Conversation(system=agent_system, tools=anth_tools)

    # ---- user-sim conversation ----
    user_conv = Conversation(
        system=USER_SYSTEM_TEMPLATE.format(scenario=str(task.user_scenario)),
        tools=[],
    )
    user_conv.append_user_text(
        "Begin the conversation by greeting the customer service "
        "representative and stating your reason for contacting them."
    )

    payloads = {}
    # Role 1: first user-sim API call (the priming "Begin." turn).
    payloads["user_sim_open"] = {
        "system": user_conv.system,
        "tools": user_conv.tools,
        "messages": user_conv.messages,
        "sha256": _hash_payload(
            user_conv.system, user_conv.tools, user_conv.messages,
        ),
    }
    # Role 2: the request the agent WOULD see for its first turn.
    # The agent's first message is the user-sim's open-text (which we
    # don't have without sampling). The deterministic part we CAN hash
    # is the agent's pre-message system+tools+messages, i.e. the empty
    # agent conversation. We hash that, plus separately we hash
    # (system, tools, []) and (system, tools) jointly so we can be
    # explicit about what we're claiming is identical.
    payloads["agent_pretext"] = {
        "system": agent_conv.system,
        "tools": agent_conv.tools,
        "messages": agent_conv.messages,
        "sha256": _hash_payload(
            agent_conv.system, agent_conv.tools, agent_conv.messages,
        ),
    }
    # For completeness, also hash the agent system and tools alone --
    # these are what the augmenter / intercept-hook touch.
    payloads["agent_system_only"] = {
        "system": agent_conv.system,
        "tools": [],
        "messages": [],
        "sha256": _hash_payload(agent_conv.system, [], []),
    }
    payloads["agent_tools_only"] = {
        "system": "",
        "tools": agent_conv.tools,
        "messages": [],
        "sha256": _hash_payload("", agent_conv.tools, []),
    }
    return payloads


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="retail")
    p.add_argument("--task-ids", nargs="+", required=True)
    p.add_argument("--protocols", default="no_coord,pull,intercept")
    p.add_argument("--out", default="/tmp/hash_audit.jsonl")
    args = p.parse_args(argv)

    get_environment, get_tasks = _load_domain(args.domain)
    all_tasks = get_tasks(None)
    by_id = {str(t.id): t for t in all_tasks}
    task_ids = list(args.task_ids)
    tasks = [by_id[t] for t in task_ids if t in by_id]
    missing = set(task_ids) - {str(t.id) for t in tasks}
    if missing:
        print(f"warning: missing task ids: {sorted(missing)}", file=sys.stderr)

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]

    out_path = pathlib.Path(args.out)
    records: list[dict] = []
    # task_id -> role -> proto -> sha256
    by_task_role: dict = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )

    with open(out_path, "w") as f_out:
        for task in tasks:
            env = get_environment()
            for proto in protocols:
                payloads = build_first_request_payloads(task, env, proto)
                for role, p in payloads.items():
                    rec = {
                        "task_id": str(task.id),
                        "trial": 0,
                        "protocol": proto,
                        "role": role,
                        "sha256": p["sha256"],
                        "n_messages": len(p["messages"]),
                        "system_len": len(p["system"]),
                        "n_tools": len(p["tools"]),
                    }
                    f_out.write(json.dumps(rec, default=str) + "\n")
                    records.append(rec)
                    by_task_role[str(task.id)][role][proto] = p["sha256"]

    # ---- analyze: cross-protocol agreement per (task, role) ----
    n_total = 0
    n_agree = 0
    disagreements: list[tuple] = []
    by_role_stats: dict = collections.defaultdict(lambda: [0, 0])
    for tid, roles in by_task_role.items():
        for role, by_proto in roles.items():
            n_total += 1
            by_role_stats[role][0] += 1
            distinct = set(by_proto.values())
            if len(distinct) == 1:
                n_agree += 1
                by_role_stats[role][1] += 1
            else:
                disagreements.append((tid, role, dict(by_proto)))

    print(f"wrote {len(records)} records to {out_path}")
    print()
    print(f"Cross-protocol SHA-256 agreement (n_protocols={len(protocols)}):")
    print(f"  overall: {n_agree} / {n_total} = "
          f"{100.0 * n_agree / max(n_total, 1):.1f}%")
    for role, (total, agree) in sorted(by_role_stats.items()):
        print(f"  {role:>20s}: {agree} / {total} = "
              f"{100.0 * agree / max(total, 1):.1f}%")
    if disagreements:
        print()
        print(f"Disagreements ({len(disagreements)}):")
        for tid, role, by_proto in disagreements[:10]:
            print(f"  task={tid} role={role}")
            for proto, h in sorted(by_proto.items()):
                print(f"    {proto:>20s}: {h[:16]}...")
    else:
        print()
        print("All (task, role) cells: byte-identical across protocols.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

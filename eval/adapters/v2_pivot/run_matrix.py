"""3-protocol coordination sweep on tau2 retail tasks via custom harness.

Runs each (task, trial) for each protocol in {no_coord, pull, intercept}
sequentially, sharing a per-task TraceStore across trials within each
protocol cell. Writes per-trial JSONLs + a summary.json at the end.

Usage:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python -m \\
        eval.adapters.v2_pivot.run_matrix \\
        --domain retail \\
        --task-ids 16 22 32 ... \\
        --num-trials 2 \\
        --out-root /tmp/v2pivot_retail_sweep
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

# Repo root on sys.path
_REPO = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.adapters.v2_pivot.orchestrator import (
    run_trial, TrialResult, set_hash_audit_ctx,
)
from eval.adapters.v2_pivot.protocols import PROTOCOLS, PER_TASK_AUGMENTERS
from eval.adapters.v2_pivot.trace_store import TraceStore, reset_stores


def _load_domain(domain: str):
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment, get_tasks
    elif domain == "airline":
        from tau2.domains.airline.environment import get_environment, get_tasks
    else:
        raise ValueError(f"unsupported domain: {domain}")
    return get_environment, get_tasks


def _score_trial(get_environment, env, task) -> tuple[Optional[float], bool]:
    """Direct DB-hash compare — same logic as run_one.py."""
    try:
        from loguru import logger as _logger
        ec = task.evaluation_criteria
        golden_actions = (ec.actions if ec is not None else None) or []
        env_assertions = (ec.env_assertions if ec is not None else None) or []
        if not golden_actions and not env_assertions:
            return 1.0, True
        gold_env = get_environment()
        for action in golden_actions:
            try:
                gold_env.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception as exc:
                _logger.warning(
                    f"golden {action.name}({action.arguments}) errored: {exc}"
                )
        agent_db_match = env.get_db_hash() == gold_env.get_db_hash()
        user_db_match = env.get_user_db_hash() == gold_env.get_user_db_hash()
        db_match = bool(agent_db_match and user_db_match)
        db_reward = 1.0 if db_match else 0.0
        env_assertion_reward = 1.0
        for assertion in env_assertions:
            success = env.run_env_assertion(assertion, raise_assertion_error=False)
            if not success:
                env_assertion_reward = 0.0
        reward = 1.0
        if golden_actions:
            reward *= db_reward
        if env_assertions:
            reward *= env_assertion_reward
        return reward, db_match
    except Exception as exc:
        return None, False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="retail")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--task-ids-file", default=None,
        help="JSON file containing a list of task IDs (overrides --task-ids).")
    p.add_argument("--num-trials", type=int, default=2)
    p.add_argument("--agent-model", default="claude-haiku-4-5")
    p.add_argument("--user-model", default="claude-haiku-4-5")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--protocols", default="no_coord,pull,intercept")
    p.add_argument("--out-root", required=True)
    p.add_argument("--save-transcripts", action="store_true",
        help="Persist full agent message history per trial (large).")
    p.add_argument("--writer", default="all",
        choices=["all", "last_k"],
        help="Writer-side selection policy for FAILED_TRIAL_ACTION events. "
             "'all' records every tool call of a failed trial (default, original). "
             "'last_k' records only the last K tool calls (W1 architectural fix).")
    p.add_argument("--writer-last-k", type=int, default=3,
        help="When --writer=last_k, how many of the trial's last tool calls to record.")
    args = p.parse_args(argv)

    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            task_ids = [str(x) for x in json.load(f)]
    elif args.task_ids:
        task_ids = list(args.task_ids)
    else:
        print("--task-ids or --task-ids-file required", file=sys.stderr)
        return 2

    get_environment, get_tasks = _load_domain(args.domain)
    all_tasks = get_tasks(None)
    by_id = {str(t.id): t for t in all_tasks}
    tasks = [by_id[t] for t in task_ids if t in by_id]
    missing = set(task_ids) - {str(t.id) for t in tasks}
    if missing:
        print(f"warning: missing task ids: {sorted(missing)}", file=sys.stderr)

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for proto in protocols:
        if proto not in PROTOCOLS:
            print(f"unknown protocol: {proto}", file=sys.stderr)
            return 2

    all_records: list[dict] = []
    for proto in protocols:
        sys_aug, hook_factory = PROTOCOLS[proto]
        reset_stores()  # fresh stores per protocol cell
        proto_dir = out_root / proto
        proto_dir.mkdir(parents=True, exist_ok=True)
        trials_path = proto_dir / "trials.jsonl"
        with open(trials_path, "w") as f_trials:
            for task in tasks:
                # Per-task store (cross-trial within this protocol cell)
                from eval.adapters.v2_pivot.trace_store import get_or_create
                store = get_or_create(str(task.id))
                # Oracle-style protocols need the task bound into their
                # augmenter (golden actions live on the task object).
                task_aug = (
                    PER_TASK_AUGMENTERS[proto](task)
                    if proto in PER_TASK_AUGMENTERS
                    else sys_aug
                )
                for trial_idx in range(args.num_trials):
                    env = get_environment()
                    hook = hook_factory(store)
                    t0 = time.time()
                    # Tag hash-audit records with the active protocol so
                    # downstream analysis can group by (task, trial, turn,
                    # role, protocol).
                    set_hash_audit_ctx(protocol=proto)
                    try:
                        result = run_trial(
                            task=task,
                            env=env,
                            agent_model=args.agent_model,
                            user_model=args.user_model,
                            max_assistant_turns=args.max_turns,
                            temperature=args.temperature,
                            intercept_hook=hook,
                            system_augmenter=task_aug,
                            trial=trial_idx,
                            trace_store=store,
                        )
                        reward, db_match = _score_trial(get_environment, env, task)

                        # WRITER-SIDE post-trial signal: if the trial did
                        # not earn reward 1.0, the protocols record some
                        # subset of this trial's tool calls as
                        # FAILED_TRIAL_ACTION events. Future trials' read
                        # paths (pull warning block, intercept hook) hit
                        # against these events.
                        #
                        # --writer=all records every tool call (v1 of this
                        # writer; produces over-attribution).
                        # --writer=last_k records only the last K tool
                        # calls before the trial ended — late-game choices
                        # are more likely to be the immediate cause of
                        # failure than setup calls.
                        if proto != "no_coord" and (reward is None or reward < 1.0):
                            tool_calls_in_order: list[tuple[str, dict]] = []
                            for m in result.agent_messages:
                                if m.get("role") != "assistant":
                                    continue
                                for b in m.get("content", []) or []:
                                    if b.get("type") == "tool_use":
                                        tool_calls_in_order.append((
                                            b.get("name", "?"),
                                            b.get("input", {}) or {},
                                        ))
                            if args.writer == "last_k":
                                to_record = tool_calls_in_order[-args.writer_last_k:]
                            else:  # "all"
                                to_record = tool_calls_in_order
                            for tname, targs in to_record:
                                store.write(
                                    trial=trial_idx,
                                    tool_name=tname,
                                    tool_args=targs,
                                    error_text=(
                                        f"trial {trial_idx} reward={reward} "
                                        f"db_match={db_match}"
                                    ),
                                    event_type="FAILED_TRIAL_ACTION",
                                )
                        rec = {
                            "protocol": proto,
                            "task_id": str(task.id),
                            "trial": trial_idx,
                            "completed": result.completed,
                            "termination_reason": result.termination_reason,
                            "n_assistant_turns": result.n_assistant_turns,
                            "n_tool_calls": result.n_tool_calls,
                            "n_errored_calls": result.n_errored_calls,
                            "n_redundant_calls": result.n_redundant_calls,
                            "duration_s": result.duration_s,
                            "input_tokens_agent": result.input_tokens_agent,
                            "output_tokens_agent": result.output_tokens_agent,
                            "input_tokens_user": result.input_tokens_user,
                            "output_tokens_user": result.output_tokens_user,
                            "reward": reward,
                            "db_match": db_match,
                            "store_events_at_end": len(store),
                        }
                        if args.save_transcripts:
                            rec["messages"] = result.agent_messages
                        f_trials.write(json.dumps(rec, default=str) + "\n")
                        f_trials.flush()
                        all_records.append(rec)
                        print(
                            f"[{proto}] task={task.id} trial={trial_idx} "
                            f"reward={reward} done={result.completed} "
                            f"turns={result.n_assistant_turns} "
                            f"redund={result.n_redundant_calls} "
                            f"dur={result.duration_s:.1f}s "
                            f"store={len(store)}",
                            flush=True,
                        )
                    except Exception as exc:
                        rec = {
                            "protocol": proto,
                            "task_id": str(task.id),
                            "trial": trial_idx,
                            "completed": False,
                            "termination_reason": f"harness_error:{type(exc).__name__}",
                            "error": str(exc)[:500],
                            "n_assistant_turns": 0,
                            "n_tool_calls": 0,
                            "n_errored_calls": 0,
                            "n_redundant_calls": 0,
                            "duration_s": time.time() - t0,
                            "reward": None,
                            "db_match": None,
                        }
                        f_trials.write(json.dumps(rec, default=str) + "\n")
                        f_trials.flush()
                        all_records.append(rec)
                        print(f"[{proto}] task={task.id} trial={trial_idx} ERROR: {exc}",
                              flush=True)

    # Aggregate and write summary
    summary_path = out_root / "summary.json"
    summary = _aggregate(all_records)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary written to {summary_path}")
    print(json.dumps(summary["cells"], indent=2, default=str))
    return 0


def _aggregate(records: list[dict]) -> dict:
    by_proto: dict[str, list[dict]] = {}
    for r in records:
        by_proto.setdefault(r["protocol"], []).append(r)
    cells = {}
    for proto, recs in by_proto.items():
        n = len(recs)
        if n == 0:
            continue
        rewards = [r["reward"] for r in recs if r.get("reward") is not None]
        by_task: dict[str, list[dict]] = {}
        for r in recs:
            by_task.setdefault(r["task_id"], []).append(r)
        for trials_for_task in by_task.values():
            trials_for_task.sort(key=lambda r: r.get("trial", 0))
        max_k = min((len(v) for v in by_task.values()), default=0)
        pass_at_k = {}
        for k in range(1, max_k + 1):
            pass_at_k[k] = sum(
                1 for trials in by_task.values()
                if all(t.get("reward") == 1.0 for t in trials[:k])
            ) / len(by_task)

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else 0.0

        cells[proto] = {
            "n_trials": n,
            "n_tasks": len(by_task),
            "n_completed": sum(1 for r in recs if r.get("completed")),
            "n_harness_errors": sum(
                1 for r in recs
                if (r.get("termination_reason") or "").startswith("harness_error")
            ),
            "success_rate": mean([1.0 if r.get("reward") == 1.0 else 0.0 for r in recs]),
            "pass_at_k": pass_at_k,
            "mean_turns": mean([r.get("n_assistant_turns") for r in recs]),
            "mean_tool_calls": mean([r.get("n_tool_calls") for r in recs]),
            "mean_errored_calls": mean([r.get("n_errored_calls") for r in recs]),
            "mean_redundant_calls": mean([r.get("n_redundant_calls") for r in recs]),
            "mean_input_tokens_agent": mean([r.get("input_tokens_agent") for r in recs]),
            "mean_output_tokens_agent": mean([r.get("output_tokens_agent") for r in recs]),
        }
    return {"cells": cells}


if __name__ == "__main__":
    sys.exit(main())

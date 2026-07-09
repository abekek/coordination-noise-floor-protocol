"""Run one (task × trial) of the custom v2-pivot harness.

Usage (from repo root):
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python -m \\
        eval.adapters.v2_pivot.run_one \\
        --domain retail --task-id 16 \\
        --agent-model claude-haiku-4-5 \\
        --user-model claude-haiku-4-5 \\
        --max-turns 20
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib

# Ensure repo root is importable (tau2 venv doesn't include our adapters by default)
_REPO = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_domain(domain: str):
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment, get_tasks
    elif domain == "airline":
        from tau2.domains.airline.environment import get_environment, get_tasks
    elif domain == "telecom":
        from tau2.domains.telecom.environment import get_environment, get_tasks
    elif domain == "mock":
        from tau2.domains.mock.environment import get_environment, get_tasks
    else:
        raise ValueError(f"unknown domain {domain!r}")
    return get_environment, get_tasks


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="retail")
    p.add_argument("--task-id", required=True)
    p.add_argument("--agent-model", default="claude-haiku-4-5")
    p.add_argument("--user-model", default="claude-haiku-4-5")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args(argv)

    get_environment, get_tasks = _load_domain(args.domain)
    env = get_environment()

    tasks = get_tasks(None)  # all tasks for this domain
    task = next((t for t in tasks if str(t.id) == str(args.task_id)), None)
    if task is None:
        print(f"task id {args.task_id!r} not found in {args.domain}", file=sys.stderr)
        return 2

    from eval.adapters.v2_pivot.orchestrator import run_trial

    result = run_trial(
        task=task,
        env=env,
        agent_model=args.agent_model,
        user_model=args.user_model,
        max_assistant_turns=args.max_turns,
        temperature=args.temperature,
        trial=args.trial,
    )

    # Score directly via DB-hash compare. The env we just ran the trial
    # against is the "predicted" world state. We construct a fresh env
    # and replay the golden action sequence on it; if both end up at the
    # same DB hash, db_reward = 1.
    try:
        from loguru import logger as _logger
        ec = task.evaluation_criteria
        golden_actions = (ec.actions if ec is not None else None) or []
        env_assertions = (ec.env_assertions if ec is not None else None) or []
        if not golden_actions and not env_assertions:
            reward = 1.0
            db_match = True
            env_assertion_reward = 1.0
        else:
            gold_env = get_environment()
            for action in golden_actions:
                try:
                    gold_env.make_tool_call(
                        tool_name=action.name,
                        requestor=action.requestor,
                        **action.arguments,
                    )
                except Exception as exc_g:
                    _logger.warning(
                        f"golden action {action.name}({action.arguments}) errored: {exc_g}"
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

            # Multiplicative reward, mirroring tau2 default (DB × ENV_ASSERTION).
            reward = 1.0
            if golden_actions:
                reward *= db_reward
            if env_assertions:
                reward *= env_assertion_reward
        reward_error = None
    except Exception as exc:
        reward = None
        reward_error = f"{type(exc).__name__}: {exc}"
        db_match = None
        env_assertion_reward = None

    summary = {
        "task_id": result.task_id,
        "trial": result.trial,
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
        "reward_error": reward_error,
        "db_match": db_match if 'db_match' in dir() else None,
    }

    print(json.dumps(summary, indent=2, default=str))

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {**summary, "agent_messages": result.agent_messages},
                f, indent=2, default=str,
            )
        print(f"saved transcript to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

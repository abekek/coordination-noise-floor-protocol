"""CLI for Silo-Bench runner.

Usage:
    python -m benchmarks.silo_bench smoke
        5 tasks × 4 protocols × 1 seed = 20 trials, ~$1-2

    python -m benchmarks.silo_bench sweep --tasks 10 --protocols msg etmcp --seeds 1
        Custom sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.silo_bench.runner import CostBudgetExceeded, list_tasks, run_sweep


def _select_tasks(n_agents: int, limit: int, level: str | None) -> list[str]:
    tasks = list_tasks(n=n_agents)
    if level is not None:
        tasks = [task for task in tasks if task.startswith(f"{level}-")]
    return tasks[:limit]


def smoke(args: argparse.Namespace) -> int:
    tasks = _select_tasks(n_agents=5, limit=5, level=None)
    protocols = ["msg", "broadcast", "sfs", "etmcp"]
    seeds = [42]
    results_dir = Path(args.results_dir)
    print(
        f"Silo-Bench smoke: {len(tasks)} tasks × {len(protocols)} protocols "
        f"× {len(seeds)} seeds = {len(tasks) * len(protocols) * len(seeds)} trials"
    )
    try:
        results = run_sweep(
            tasks=tasks,
            protocols=protocols,
            seeds=seeds,
            results_dir=results_dir,
            max_rounds=args.max_rounds,
            model=args.model,
            max_cost_usd=args.max_cost_usd,
            max_trials=args.max_trials,
            resume=args.resume,
            dry_run=args.dry_run,
            cooldown_seconds=args.cooldown_seconds,
            initial_cooldown_seconds=args.initial_cooldown_seconds,
            zero_token_retries=args.zero_token_retries,
        )
    except CostBudgetExceeded as exc:
        print(f"\nABORTED: {exc}")
        return 2
    total_in = sum(r.total_input_tokens for r in results)
    total_out = sum(r.total_output_tokens for r in results)
    completed = sum(1 for r in results if r.completed)
    print(f"\n{completed}/{len(results)} completed")
    print(f"Total tokens: input={total_in:,} output={total_out:,}")
    print(f"Artifacts: {results_dir}/trials.jsonl")
    return 0


def sweep(args: argparse.Namespace) -> int:
    tasks = _select_tasks(n_agents=args.n_agents, limit=args.tasks, level=args.level)
    seeds = list(range(42, 42 + args.seeds))
    results_dir = Path(args.results_dir)
    print(
        f"Silo-Bench sweep: {len(tasks)} tasks × {len(args.protocols)} protocols "
        f"× {len(seeds)} seeds = {len(tasks) * len(args.protocols) * len(seeds)} trials"
    )
    try:
        results = run_sweep(
            tasks=tasks,
            protocols=args.protocols,
            seeds=seeds,
            results_dir=results_dir,
            max_rounds=args.max_rounds,
            model=args.model,
            max_cost_usd=args.max_cost_usd,
            max_trials=args.max_trials,
            resume=args.resume,
            dry_run=args.dry_run,
            cooldown_seconds=args.cooldown_seconds,
            initial_cooldown_seconds=args.initial_cooldown_seconds,
            zero_token_retries=args.zero_token_retries,
        )
    except CostBudgetExceeded as exc:
        print(f"\nABORTED: {exc}")
        return 2
    completed = sum(1 for r in results if r.completed)
    print(f"\n{completed}/{len(results)} completed; artifacts in {args.results_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="silo_bench",
        description="Run Silo-Bench multi-agent tasks and emit TrialResult JSONL.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke_p = sub.add_parser(
        "smoke",
        help="Run the 20-trial smoke test (5 tasks × 4 protocols × 1 seed)",
    )
    smoke_p.add_argument("--results-dir", default="results/silo_smoke", help="Output directory")
    smoke_p.add_argument("--model", default=None, help="Anthropic model override")
    smoke_p.add_argument("--max-rounds", type=int, default=20, help="Max rounds per trial")
    smoke_p.add_argument("--max-cost-usd", type=float, default=None, help="USD cost ceiling")
    smoke_p.add_argument("--max-trials", type=int, default=None, help="Stop after N new trials")
    smoke_p.add_argument("--resume", action="store_true", help="Skip specs already in trials.jsonl")
    smoke_p.add_argument("--dry-run", action="store_true", help="Print planned trials only")
    smoke_p.add_argument("--cooldown-seconds", type=float, default=0.0, help="Sleep between trials")
    smoke_p.add_argument(
        "--initial-cooldown-seconds",
        type=float,
        default=0.0,
        help="Sleep before the first non-skipped trial",
    )
    smoke_p.add_argument(
        "--zero-token-retries",
        type=int,
        default=0,
        help="Retry zero-token setup/quota failures",
    )

    s = sub.add_parser("sweep", help="Custom sweep across tasks/protocols/seeds")
    s.add_argument("--tasks", type=int, default=30, help="Number of task files to use")
    s.add_argument(
        "--protocols",
        nargs="+",
        choices=["msg", "broadcast", "sfs", "etmcp"],
        default=["msg", "broadcast", "sfs", "etmcp"],
        help="Protocols to evaluate",
    )
    s.add_argument("--seeds", type=int, default=1, help="Number of seeds (starting from 42)")
    s.add_argument("--n-agents", type=int, default=5, help="Agent count filter for task files")
    s.add_argument("--level", choices=["I", "II", "III"], default=None, help="Optional task level")
    s.add_argument("--results-dir", default="results/silo_sweep", help="Output directory")
    s.add_argument("--model", default=None, help="Anthropic model override")
    s.add_argument("--max-rounds", type=int, default=20, help="Max rounds per trial")
    s.add_argument("--max-cost-usd", type=float, default=None, help="USD cost ceiling")
    s.add_argument("--max-trials", type=int, default=None, help="Stop after N new trials")
    s.add_argument("--resume", action="store_true", help="Skip specs already in trials.jsonl")
    s.add_argument("--dry-run", action="store_true", help="Print planned trials only")
    s.add_argument("--cooldown-seconds", type=float, default=0.0, help="Sleep between trials")
    s.add_argument(
        "--initial-cooldown-seconds",
        type=float,
        default=0.0,
        help="Sleep before the first non-skipped trial",
    )
    s.add_argument(
        "--zero-token-retries",
        type=int,
        default=0,
        help="Retry zero-token setup/quota failures",
    )

    args = parser.parse_args(argv)
    if args.cmd == "smoke":
        return smoke(args)
    if args.cmd == "sweep":
        return sweep(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

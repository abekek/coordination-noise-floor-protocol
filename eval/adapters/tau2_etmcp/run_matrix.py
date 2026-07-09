"""Run the 4-protocol coordination factorial against a tau2 domain.

For each of {no_coord, push_scratchpad, message_passing, et_mcp}, runs
the configured task set k times per task with the et_mcp_agent and writes
results to a separate dir per protocol. The cross-trial trace store
(module-level cache in et_mcp_agent.py) is reset between protocols so
each cell starts fresh.

Usage:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python \\
        eval/adapters/tau2_etmcp/run_matrix.py \\
        --domain telecom --num-trials 4 \\
        --agent-llm "anthropic/claude-haiku-4-5" \\
        --user-llm "openai/gpt-4.1" \\
        --out-root /tmp/v2_telecom_sweep \\
        [--protocols et_mcp,no_coord,push_scratchpad,message_passing] \\
        [--task-set-name telecom_small] \\
        [--concurrency 4]

When all four cells finish, runs the analysis automatically and writes
summary.json next to the per-protocol dirs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Iterable


_PROTOCOLS = ["et_mcp", "no_coord", "push_scratchpad", "message_passing"]


def _tau2_cli_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parent / "tau2_etmcp_cli.py"
    )


def _venv_python() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "tau2_bench"
        / "_vendor"
        / ".venv"
        / "bin"
        / "python"
    )


def _run_one(
    protocol: str,
    domain: str,
    agent_llm: str,
    user_llm: str,
    num_trials: int,
    out_dir: pathlib.Path,
    task_set_name: str | None = None,
    task_split_name: str | None = None,
    task_ids: list[str] | None = None,
    concurrency: int = 3,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(_venv_python()),
        str(_tau2_cli_path()),
        "run",
        "--domain", domain,
        "--agent", "et_mcp_agent",
        "--agent-llm", agent_llm,
        "--agent-llm-args", json.dumps({"coord_protocol": protocol}),
        "--user", "user_simulator",
        "--user-llm", user_llm,
        "--num-trials", str(num_trials),
        "--save-to", str(out_dir),
        "--max-concurrency", str(concurrency),
        "--auto-resume",
    ]
    if task_set_name:
        args.extend(["--task-set-name", task_set_name])
    if task_split_name:
        args.extend(["--task-split-name", task_split_name])
    if task_ids:
        args.extend(["--task-ids", *task_ids])
    print(f"[{protocol}] Running: {' '.join(args)}", flush=True)
    start = time.time()
    rc = subprocess.call(args)
    dur = time.time() - start
    print(f"[{protocol}] rc={rc} in {dur:.0f}s", flush=True)
    return rc


def main(argv: Iterable[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--num-trials", type=int, default=4)
    p.add_argument("--agent-llm", default="anthropic/claude-haiku-4-5")
    p.add_argument("--user-llm", default="anthropic/claude-haiku-4-5")
    p.add_argument("--out-root", required=True)
    p.add_argument(
        "--protocols",
        default=",".join(_PROTOCOLS),
        help="Comma-separated list; defaults to all four.",
    )
    p.add_argument("--task-set-name", default=None)
    p.add_argument("--task-split-name", default=None,
        help="When task-set-name is a parent like 'telecom', pick the split (e.g. 'small', 'base').")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--task-ids-file", default=None,
        help="JSON file containing a list of task IDs (alternative to --task-ids when IDs have shell-special chars).")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a protocol if its out dir already has a non-empty results.json.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip running; just aggregate existing per-protocol dirs.",
    )
    args = p.parse_args(argv)

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    task_ids = args.task_ids
    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            task_ids = json.load(f)
        print(f"Loaded {len(task_ids)} task ids from {args.task_ids_file}", flush=True)
    for proto in protocols:
        if proto not in _PROTOCOLS:
            print(f"Unknown protocol: {proto}", file=sys.stderr)
            return 2

    paths: dict[str, str] = {}
    for proto in protocols:
        cell_dir = out_root / proto
        paths[proto] = str(cell_dir)
        if args.summary_only:
            continue
        if args.skip_existing and (cell_dir / "results.json").exists():
            sz = (cell_dir / "results.json").stat().st_size
            if sz > 100:
                print(f"[{proto}] Skipping (results.json exists, {sz} bytes)", flush=True)
                continue
        rc = _run_one(
            protocol=proto,
            domain=args.domain,
            agent_llm=args.agent_llm,
            user_llm=args.user_llm,
            num_trials=args.num_trials,
            out_dir=cell_dir,
            task_set_name=args.task_set_name,
            task_split_name=args.task_split_name,
            task_ids=task_ids,
            concurrency=args.concurrency,
        )
        if rc != 0:
            print(f"[{proto}] non-zero exit; continuing", file=sys.stderr)

    # Run analysis across whatever cells produced results
    print("=== Analysis ===", flush=True)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from eval.adapters.tau2_etmcp.analysis import summarize

    summary_path = out_root / "summary.json"
    summary = summarize(paths, out_path=str(summary_path))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

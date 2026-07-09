"""ET-MCP evaluation harness CLI.

Subcommands:
    smoke          8-trial toy smoke test (real API)
    tp-smoke       8-trial TravelPlanner smoke test
    run            Arbitrary subset run
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from baselines.ca_mcp_style import CaMcpStyleBaseline
from baselines.et_mcp import EtMcpBaseline
from baselines.et_mcp_ablations import (
    EtMcpFailureStrict,
    EtMcpNoQuery,
    EtMcpPushNegative,
    EtMcpRandomQuery,
    EtMcpWriteEverything,
)
from baselines.full_context import FullContextBaseline
from baselines.summarization import SummarizationBaseline
from benchmarks.toy.adapter import ToyBenchmark
from benchmarks.travelplanner.adapter import TravelPlannerBenchmark
from harness.llm import AnthropicClient
from harness.runner import CostBudgetExceeded, TokenBudgetExceeded, TrialRunner, TrialSpec

_ALL_CONDITIONS = [
    "B1_full_context",
    "B2_summarization",
    "B3_ca_mcp_style",
    "ET_MCP_default",
    "ET_MCP_no_query",
    "ET_MCP_random_query",
    "ET_MCP_push_negative",
    "ET_MCP_failure_strict",
    "ET_MCP_write_everything",
]

_SMOKE_TOKEN_CEILING = 100_000
_TP_SMOKE_TOKEN_CEILING = 300_000


def _baseline_factory(condition: str, task_id: str, llm: Any):
    if condition == "B1_full_context":
        return FullContextBaseline()
    if condition == "B2_summarization":
        return SummarizationBaseline(llm=llm)
    if condition == "B3_ca_mcp_style":
        return CaMcpStyleBaseline(task_id=task_id)
    if condition == "ET_MCP_default":
        return EtMcpBaseline(task_id=task_id)
    if condition == "ET_MCP_no_query":
        return EtMcpNoQuery(task_id=task_id)
    if condition == "ET_MCP_random_query":
        return EtMcpRandomQuery(task_id=task_id)
    if condition == "ET_MCP_push_negative":
        return EtMcpPushNegative(task_id=task_id)
    if condition == "ET_MCP_failure_strict":
        return EtMcpFailureStrict(task_id=task_id)
    if condition == "ET_MCP_write_everything":
        return EtMcpWriteEverything(task_id=task_id)
    raise ValueError(f"unknown condition {condition!r}")


async def smoke(model: str = "claude-haiku-4-5") -> int:
    llm = AnthropicClient(model=model)
    results_dir = Path("results/smoke")
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = TrialRunner(
        llm=llm, results_dir=results_dir,
        benchmarks={"toy": ToyBenchmark(), "travelplanner": TravelPlannerBenchmark()},
        baseline_factory=_baseline_factory,
    )
    conditions = ["B1_full_context", "B2_summarization", "B3_ca_mcp_style", "ET_MCP_default"]
    queries = ["q_easy_001", "q_medium_001"]
    specs = [TrialSpec(condition=c, benchmark="toy", query_id=q, seed=42)
             for c in conditions for q in queries]
    print(f"Smoke run: {len(specs)} trials, model={model}, cap={_SMOKE_TOKEN_CEILING} tokens")

    try:
        results = await runner.run_many(specs, max_concurrency=1,
                                        max_total_tokens=_SMOKE_TOKEN_CEILING)
        aborted = False
    except TokenBudgetExceeded as exc:
        print(f"\nABORTED: {exc}")
        results = []
        aborted = True

    total_tokens = sum(r.metrics.get("input_tokens", 0) + r.metrics.get("output_tokens", 0)
                       for r in results)
    print()
    for r in results:
        line = (f"[{r.spec.condition:25s} {r.spec.query_id:13s} seed={r.spec.seed}] "
                f"completed={str(r.completed):5s} "
                f"tokens={r.metrics.get('input_tokens', 0) + r.metrics.get('output_tokens', 0):6d} "
                f"redundant={r.metrics.get('redundant_call_rate', 0.0):.2f} "
                f"trace_events={r.metrics.get('trace_event_count', 0)} "
                f"({r.wall_time_s:5.1f}s)")
        if r.error:
            line += f"  ERROR: {r.error}"
        print(line)
    print(f"\nTotal tokens: {total_tokens:,}")
    if aborted:
        return 2
    by_condition: dict[str, list[Any]] = {}
    for r in results:
        by_condition.setdefault(r.spec.condition, []).append(r)
    print()
    print(f"{'Condition':28s}  completed  mean_input_tokens")
    print("-" * 55)
    for cond, rs in by_condition.items():
        completed = sum(1 for r in rs if r.completed)
        mean_in = sum(r.metrics.get("input_tokens", 0) for r in rs) / len(rs)
        print(f"{cond:28s}  {completed}/{len(rs):d}        {mean_in:8.0f}")
    print(f"\nArtifacts: {results_dir}/trials.jsonl")
    return 0


async def tp_smoke(model: str = "claude-haiku-4-5") -> int:
    llm = AnthropicClient(model=model)
    results_dir = Path("results/tp_smoke")
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = TrialRunner(
        llm=llm, results_dir=results_dir,
        benchmarks={"toy": ToyBenchmark(), "travelplanner": TravelPlannerBenchmark()},
        baseline_factory=_baseline_factory,
    )
    conditions = ["B1_full_context", "B2_summarization", "B3_ca_mcp_style", "ET_MCP_default"]
    bench = TravelPlannerBenchmark()
    easy_queries = bench.load_queries(subset="easy")
    chosen = [easy_queries[0].query_id, easy_queries[1].query_id]
    specs = [TrialSpec(condition=c, benchmark="travelplanner", query_id=q, seed=42)
             for c in conditions for q in chosen]
    print(f"TP smoke run: {len(specs)} trials, model={model}, cap={_TP_SMOKE_TOKEN_CEILING:,} tokens")
    try:
        results = await runner.run_many(specs, max_concurrency=1,
                                        max_total_tokens=_TP_SMOKE_TOKEN_CEILING)
        aborted = False
    except TokenBudgetExceeded as exc:
        print(f"\nABORTED: {exc}")
        results = []
        aborted = True
    total = sum(r.metrics.get("input_tokens", 0) + r.metrics.get("output_tokens", 0)
                for r in results)
    for r in results:
        line = (f"[{r.spec.condition:25s} {r.spec.query_id:12s}] "
                f"completed={str(r.completed):5s} tokens={r.metrics.get('input_tokens',0)+r.metrics.get('output_tokens',0):7d}")
        if r.error:
            line += f"  ERROR: {r.error}"
        print(line)
    print(f"\nTotal tokens: {total:,}")
    if aborted:
        return 2
    return 0


async def run_subset(args: argparse.Namespace) -> int:
    import datetime
    model = getattr(args, "model", "claude-haiku-4-5")
    max_tokens = getattr(args, "max_total_tokens", None)
    max_cost = getattr(args, "max_cost_usd", 200.0)
    llm = AnthropicClient(model=model)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = TrialRunner(
        llm=llm, results_dir=results_dir,
        benchmarks={"toy": ToyBenchmark(), "travelplanner": TravelPlannerBenchmark()},
        baseline_factory=_baseline_factory,
    )
    _benchmark_map = {"toy": ToyBenchmark(), "travelplanner": TravelPlannerBenchmark()}
    benchmark = _benchmark_map[args.benchmark]
    queries = benchmark.load_queries()[: args.queries]
    seeds = [42 + i for i in range(args.seeds)]
    conditions = args.condition if isinstance(args.condition, list) else [args.condition]
    specs = [
        TrialSpec(condition=c, benchmark=args.benchmark, query_id=q.query_id, seed=s)
        for c in conditions for q in queries for s in seeds
    ]
    print(f"Run: {len(specs)} trials | model={model} | benchmark={args.benchmark} | "
          f"conditions={conditions} | max_cost=${max_cost:.0f}")
    try:
        results = await runner.run_many(
            specs, max_concurrency=args.concurrency,
            max_cost_usd=max_cost, max_total_tokens=max_tokens, model=model,
        )
    except (CostBudgetExceeded, TokenBudgetExceeded) as exc:
        print(f"\nABORTED: {exc}")
        return 2
    completed = sum(1 for r in results if r.completed)
    total_tok = sum(r.metrics.get("input_tokens",0)+r.metrics.get("output_tokens",0) for r in results)
    print(f"{completed}/{len(results)} completed | total_tokens={total_tok:,} | artifacts in {results_dir}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke_p = sub.add_parser("smoke")
    smoke_p.add_argument("--model", default="claude-haiku-4-5")

    tp_p = sub.add_parser("tp-smoke")
    tp_p.add_argument("--model", default="claude-haiku-4-5")

    run_p = sub.add_parser("run")
    run_p.add_argument("--benchmark", default="toy", choices=["toy", "travelplanner"])
    run_p.add_argument("--condition", required=True, action="append",
                       choices=_ALL_CONDITIONS,
                       help="Condition to run (repeat flag for multiple conditions)")
    run_p.add_argument("--queries", type=int, default=5)
    run_p.add_argument("--seeds", type=int, default=1)
    run_p.add_argument("--concurrency", type=int, default=4)
    run_p.add_argument("--results-dir", default="results/run")
    run_p.add_argument("--model", default="claude-haiku-4-5")
    run_p.add_argument("--max-total-tokens", type=int, default=None,
                       dest="max_total_tokens",
                       help="Hard token ceiling; aborts run if exceeded")
    run_p.add_argument("--max-cost-usd", type=float, default=200.0,
                       dest="max_cost_usd",
                       help="Hard USD cost ceiling (default: $200)")

    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        return asyncio.run(smoke(model=getattr(args, "model", "claude-haiku-4-5")))
    if args.cmd == "tp-smoke":
        return asyncio.run(tp_smoke(model=getattr(args, "model", "claude-haiku-4-5")))
    if args.cmd == "run":
        return asyncio.run(run_subset(args))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

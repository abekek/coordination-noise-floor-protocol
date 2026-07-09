"""Run the full ablation matrix as a single resumable command.

Usage:
    cd eval
    python experiments/run_matrix.py \
        --benchmark toy \
        --queries 30 \
        --seeds 3 \
        --model claude-haiku-4-5 \
        --max-total-tokens 15000000 \
        --concurrency 2

Writes results to results/lean_<YYYYMMDD_HHMM>/ and a manifest JSON.
"""
from __future__ import annotations
import argparse, asyncio, datetime, json, os, sys
from pathlib import Path

# Allow running from repo root or eval/
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.__main__ import _baseline_factory, _ALL_CONDITIONS
from harness.llm import AnthropicClient
from harness.runner import CostBudgetExceeded, TokenBudgetExceeded, TrialRunner, TrialSpec
from benchmarks.toy.adapter import ToyBenchmark
from benchmarks.travelplanner.adapter import TravelPlannerBenchmark

_BENCHMARK_MAP = {
    "toy": ToyBenchmark,
    "travelplanner": TravelPlannerBenchmark,
}


async def run_matrix(args: argparse.Namespace) -> int:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(f"results/lean_{args.benchmark}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    llm = AnthropicClient(model=args.model)
    benchmarks = {k: cls() for k, cls in _BENCHMARK_MAP.items()}
    runner = TrialRunner(
        llm=llm, results_dir=results_dir,
        benchmarks=benchmarks,
        baseline_factory=_baseline_factory,
    )

    benchmark = benchmarks[args.benchmark]
    all_queries = benchmark.load_queries()
    queries = all_queries[:args.queries]
    seeds = [42 + i for i in range(args.seeds)]
    conditions = args.condition or _ALL_CONDITIONS

    specs = [
        TrialSpec(condition=c, benchmark=args.benchmark, query_id=q.query_id, seed=s)
        for c in conditions for q in queries for s in seeds
    ]

    # Write manifest
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    manifest = {
        "model": args.model,
        "git_sha": git_sha,
        "benchmark": args.benchmark,
        "conditions": conditions,
        "queries": args.queries,
        "seeds": args.seeds,
        "concurrency": args.concurrency,
        "max_cost_usd": args.max_cost_usd,
        "max_total_tokens": args.max_total_tokens,
        "temperature": 0.0,
        "total_specs": len(specs),
        "timestamp": timestamp,
        "results_dir": str(results_dir),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {results_dir}/manifest.json")
    print(f"Running {len(specs)} trials: {len(conditions)} conditions x {len(queries)} queries x {len(seeds)} seeds")
    print(f"Model: {args.model} | Concurrency: {args.concurrency} | Cost cap: ${args.max_cost_usd:.0f}")

    try:
        results = await runner.run_many(
            specs,
            max_concurrency=args.concurrency,
            max_cost_usd=args.max_cost_usd,
            max_total_tokens=args.max_total_tokens,
            model=args.model,
        )
        aborted = False
    except (CostBudgetExceeded, TokenBudgetExceeded) as exc:
        print(f"\nABORTED: {exc}")
        aborted = True
        results = []

    from harness.runner import _trial_cost_usd, _DEFAULT_PRICING
    completed = sum(1 for r in results if r.completed)
    total_tok = sum(r.metrics.get("input_tokens",0)+r.metrics.get("output_tokens",0) for r in results)
    total_cost = sum(_trial_cost_usd(r.metrics, args.model) for r in results)

    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{len(results)} trials")
    print(f"Total tokens: {total_tok:,}   Est. cost: ${total_cost:.4f}")
    print(f"Results: {results_dir}/trials.jsonl")
    if aborted:
        print(f"WARNING: Run was cost-budget aborted -- partial results saved")

    # Per-condition summary
    by_cond: dict[str, list] = {}
    for r in results:
        by_cond.setdefault(r.spec.condition, []).append(r)
    print(f"\n{'Condition':30s}  comp  tokens   redundant  trace_ev")
    print("-"*70)
    for cond, rs in sorted(by_cond.items()):
        comp = sum(1 for r in rs if r.completed)
        tok = sum(r.metrics.get("input_tokens",0)+r.metrics.get("output_tokens",0) for r in rs)
        red = sum(r.metrics.get("redundant_call_rate",0) for r in rs) / len(rs) if rs else 0
        tev = sum(r.metrics.get("trace_event_count",0) for r in rs)
        print(f"{cond:30s}  {comp:2d}/{len(rs):2d}  {tok:7,}   {red:.3f}      {tev:4d}")

    return 2 if aborted else 0


def main():
    p = argparse.ArgumentParser(description="Run full ET-MCP ablation matrix")
    p.add_argument("--benchmark", default="toy", choices=list(_BENCHMARK_MAP.keys()))
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--max-cost-usd", type=float, default=200.0, dest="max_cost_usd",
                   help="Hard USD cost cap (default: $200)")
    p.add_argument("--max-total-tokens", type=int, default=None, dest="max_total_tokens",
                   help="Legacy token cap; ignored if --max-cost-usd is set")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--condition", action="append", default=None,
                   help="Condition to include (repeat for multiple; default: all)")
    return asyncio.run(run_matrix(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired-stats analysis for the v2-pivot sweep results.

Reads the per-cell trials.jsonl files written by run_matrix.py and
produces:
- per-cell aggregates (success rate, pass^1, pass^2)
- paired Wilcoxon signed-rank tests + Cliff's δ on per-task pass^k
- Holm-Bonferroni correction across the (no_coord vs X) pairs per metric
- a LaTeX-ready table fragment for the headline pairwise comparison

Usage:
    eval/benchmarks/tau2_bench/_vendor/.venv/bin/python -m \\
        eval.adapters.v2_pivot.analyze \\
        --root /tmp/v2pivot_n30 \\
        --out /tmp/v2pivot_n30/summary_paired.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from statistics import mean
from typing import Optional


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def per_task_pass_at_k(records: list[dict], k: int) -> dict[str, int]:
    """Map task_id → 1 if all-of-first-k trials had reward=1.0 else 0."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[str(r["task_id"])].append(r)
    out = {}
    for tid, recs in by_task.items():
        recs.sort(key=lambda r: r.get("trial", 0))
        if len(recs) < k:
            continue
        out[tid] = 1 if all((r.get("reward") == 1.0) for r in recs[:k]) else 0
    return out


def _wilcoxon_signed_rank(diffs: list[float]) -> Optional[float]:
    """Two-sided paired Wilcoxon. None if scipy unavailable or insufficient n."""
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) < 5:
        return None
    try:
        from scipy.stats import wilcoxon  # type: ignore
        stat, p = wilcoxon(nonzero, zero_method="wilcox")
        return float(p)
    except ImportError:
        return None


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def _holm(pvals: list[Optional[float]]) -> list[Optional[float]]:
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    indexed.sort(key=lambda x: x[1])
    n = len(indexed)
    out: list[Optional[float]] = list(pvals)
    last = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, max(last, (n - rank) * p))
        last = adj
        out[orig_idx] = adj
    return out


def analyze(root: pathlib.Path, baseline: str = "no_coord") -> dict:
    cells = {}
    raw_records: dict[str, list[dict]] = {}
    for proto_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        proto = proto_dir.name
        recs = load_jsonl(proto_dir / "trials.jsonl")
        if not recs:
            continue
        raw_records[proto] = recs
        # Per-task pass^1/pass^2
        p1 = per_task_pass_at_k(recs, 1)
        p2 = per_task_pass_at_k(recs, 2)
        rewards = [r.get("reward") for r in recs if r.get("reward") is not None]
        tokens_in = [r.get("input_tokens_agent", 0) or 0 for r in recs]
        tokens_out = [r.get("output_tokens_agent", 0) or 0 for r in recs]
        turns = [r.get("n_assistant_turns", 0) or 0 for r in recs]
        tcs = [r.get("n_tool_calls", 0) or 0 for r in recs]
        cells[proto] = {
            "n_trials": len(recs),
            "n_tasks": len({str(r["task_id"]) for r in recs}),
            "success_rate": (sum(1 for r in rewards if r == 1.0) / len(rewards)) if rewards else 0.0,
            "pass_1": (sum(p1.values()) / len(p1)) if p1 else 0.0,
            "pass_2": (sum(p2.values()) / len(p2)) if p2 else 0.0,
            "mean_turns": mean(turns) if turns else 0.0,
            "mean_tool_calls": mean(tcs) if tcs else 0.0,
            "mean_input_tokens_agent": mean(tokens_in) if tokens_in else 0.0,
            "mean_output_tokens_agent": mean(tokens_out) if tokens_out else 0.0,
            "_p1_per_task": p1,
            "_p2_per_task": p2,
        }

    if baseline not in cells:
        return {"cells": cells, "pairwise": []}

    base_p2 = cells[baseline]["_p2_per_task"]
    base_p1 = cells[baseline]["_p1_per_task"]

    pairwise = []
    for proto, cell in cells.items():
        if proto == baseline:
            continue
        common2 = sorted(set(base_p2) & set(cell["_p2_per_task"]))
        a_p2 = [cell["_p2_per_task"][t] for t in common2]
        b_p2 = [base_p2[t] for t in common2]
        diffs2 = [a - b for a, b in zip(a_p2, b_p2)]
        p_p2 = _wilcoxon_signed_rank(diffs2)
        delta_p2 = _cliffs_delta(a_p2, b_p2)

        common1 = sorted(set(base_p1) & set(cell["_p1_per_task"]))
        a_p1 = [cell["_p1_per_task"][t] for t in common1]
        b_p1 = [base_p1[t] for t in common1]
        diffs1 = [a - b for a, b in zip(a_p1, b_p1)]
        p_p1 = _wilcoxon_signed_rank(diffs1)
        delta_p1 = _cliffs_delta(a_p1, b_p1)

        pairwise.append({
            "protocol": proto,
            "vs": baseline,
            "n_paired_tasks": len(common2),
            "pass_2_mean_protocol": (sum(a_p2)/len(a_p2)) if a_p2 else 0.0,
            "pass_2_mean_baseline": (sum(b_p2)/len(b_p2)) if b_p2 else 0.0,
            "pass_2_p": p_p2,
            "pass_2_cliffs_delta": delta_p2,
            "pass_1_mean_protocol": (sum(a_p1)/len(a_p1)) if a_p1 else 0.0,
            "pass_1_mean_baseline": (sum(b_p1)/len(b_p1)) if b_p1 else 0.0,
            "pass_1_p": p_p1,
            "pass_1_cliffs_delta": delta_p1,
        })

    # Holm correction across the (non-baseline) protocols within each metric
    for metric in ("pass_2", "pass_1"):
        pvals = [pw.get(f"{metric}_p") for pw in pairwise]
        adj = _holm(pvals)
        for pw, a in zip(pairwise, adj):
            pw[f"{metric}_p_holm"] = a

    # Strip internal _per_task fields from cells for the JSON dump
    for cell in cells.values():
        cell.pop("_p1_per_task", None)
        cell.pop("_p2_per_task", None)

    return {"cells": cells, "pairwise_vs_baseline": pairwise, "baseline": baseline}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--baseline", default="no_coord")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    root = pathlib.Path(args.root)
    report = analyze(root, baseline=args.baseline)

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

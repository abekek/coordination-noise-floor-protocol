"""Silo-Bench re-exploration / redundancy metric.

Metric: Stale Re-Submission Rounds (RSR)
-----------------------------------------
For each agent in a trial, count the number of rounds in which the agent's
``submission.json`` appears at a round index *strictly greater* than the agent's
own first-submission round (the ``round`` field stored inside the file).
RSR for a trial is the sum over all five agents.

Rationale: once an agent has submitted a final answer, any subsequent round in
which its submission file re-appears is a wasted agent-round -- the agent is
re-emitting an unchanged answer while burning tokens and wall-time.  ET-MCP's
pull-based design lets agents emit selective trace events and terminate
efficiently, whereas push-based protocols (especially P2P/msg) often leave
agents spinning until the 20-round budget is exhausted.

Usage (from eval/ directory):
    uv run python -m analysis.silo_reexploration

Outputs:
    - Markdown summary table to stdout
    - Per-protocol and per-paradigm breakdown
    - Paired Wilcoxon + Cliff's delta vs ET-MCP (Holm-Bonferroni corrected)
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKSPACE_BASE = Path(__file__).parent.parent / "results" / "silo_n5_sweep" / "workspaces"

PROTOCOL_LABELS: dict[str, str] = {
    "etmcp": "ET-MCP",
    "broadcast": "Broadcast",
    "sfs": "Shared FS",
    "msg": "P2P",
}


def compute_trial_rsr(trial_path: Path, n_agents: int = 5) -> int:
    """Return the stale re-submission round count (RSR) for one trial.

    For each agent, counts rounds where a submission file exists at a
    round index > the agent's own first-submission round.
    """
    total_stale = 0
    for agent_id in range(n_agents):
        sub_files = sorted(
            glob.glob(str(trial_path / "rounds" / "*" / f"agent-{agent_id:03d}" / "submission.json"))
        )
        if not sub_files:
            continue
        with open(sub_files[0]) as fh:
            first_data = json.load(fh)
        # The "round" field in submission.json = first submission round
        first_submit_round: int = first_data.get("round", 0)
        stale = sum(
            1 for sf in sub_files
            if int(sf.split("round-")[1].split(os.sep)[0]) > first_submit_round
        )
        total_stale += stale
    return total_stale


def load_all_trials(workspace_base: Path) -> list[dict[str, Any]]:
    """Walk workspaces and compute RSR for every trial."""
    records: list[dict[str, Any]] = []
    for trial_dir in sorted(os.listdir(workspace_base)):
        trial_path = workspace_base / trial_dir
        meta_path = trial_path / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        protocol: str = meta["config"]["protocol"]
        task: str = meta["task"]["case_id"]
        paradigm: str = meta["task"].get("paradigm", "")
        total_rounds: int = meta["execution"]["current_round"]
        all_submitted: bool = meta["execution"].get("all_submitted", False)
        stale_rsr = compute_trial_rsr(trial_path)
        records.append({
            "trial_dir": trial_dir,
            "protocol": protocol,
            "task": task,
            "paradigm": paradigm,
            "total_rounds": total_rounds,
            "all_submitted": all_submitted,
            "stale_rsr": stale_rsr,
        })
    return records


def aggregate_by_protocol(trials: list[dict]) -> dict[str, list[int]]:
    by_proto: dict[str, list[int]] = defaultdict(list)
    for t in trials:
        by_proto[t["protocol"]].append(t["stale_rsr"])
    return dict(by_proto)


def _task_means(trials: list[dict], proto_a: str, proto_b: str) -> tuple[list[float], list[float]]:
    """Return per-task mean RSR paired for two protocols."""
    task_proto: dict[tuple[str, str], list[int]] = defaultdict(list)
    for t in trials:
        task_proto[(t["task"], t["protocol"])].append(t["stale_rsr"])
    tasks = sorted({t["task"] for t in trials})
    a_vals, b_vals = [], []
    for task in tasks:
        a_seeds = task_proto.get((task, proto_a), [])
        b_seeds = task_proto.get((task, proto_b), [])
        if a_seeds and b_seeds:
            a_vals.append(sum(a_seeds) / len(a_seeds))
            b_vals.append(sum(b_seeds) / len(b_seeds))
    return a_vals, b_vals


def run_analysis(workspace_base: Path = WORKSPACE_BASE) -> None:
    """Load all trials, compute RSR, print markdown summary."""
    print(f"Loading trials from {workspace_base} ...")
    trials = load_all_trials(workspace_base)
    if not trials:
        print("No trials found.")
        return

    n_trials = len(trials)
    print(f"Loaded {n_trials} trials.\n")

    # --- Full-corpus summary ---
    by_proto = aggregate_by_protocol(trials)
    print("## RSR Summary — all 360 trials (30 tasks × 4 protocols × 3 seeds)\n")
    print("| Protocol | n | Mean RSR | Median | Std | Min | Max |")
    print("|----------|---|----------|--------|-----|-----|-----|")
    for proto in ["etmcp", "broadcast", "sfs", "msg"]:
        vals = by_proto.get(proto, [])
        if not vals:
            continue
        label = PROTOCOL_LABELS[proto]
        mean_v = sum(vals) / len(vals)
        med_v = statistics.median(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"| {label} | {len(vals)} | {mean_v:.2f} | {med_v:.1f} | {std_v:.2f} | {min(vals)} | {max(vals)} |")

    # --- Per-paradigm breakdown ---
    print("\n## RSR by Paradigm\n")
    print("| Paradigm | ET-MCP | Broadcast | Shared FS | P2P |")
    print("|----------|--------|-----------|-----------|-----|")
    for paradigm in ["Paradigm I", "Paradigm II", "Paradigm III"]:
        row_vals: list[str] = []
        for proto in ["etmcp", "broadcast", "sfs", "msg"]:
            ptlist = [t["stale_rsr"] for t in trials if t["paradigm"] == paradigm and t["protocol"] == proto]
            if ptlist:
                row_vals.append(f"{sum(ptlist)/len(ptlist):.1f}")
            else:
                row_vals.append("–")
        print(f"| {paradigm} | {' | '.join(row_vals)} |")

    # --- Statistical tests (Paradigms II+III only, where mechanism matters) ---
    print("\n## Paired Wilcoxon + Cliff's delta — ET-MCP vs. baselines")
    print("### Paradigms II and III only (complex coordination tasks)\n")

    # Import stats primitives
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from analysis.stats import paired_wilcoxon, cliffs_delta, holm_bonferroni
    except ImportError:
        print("WARNING: scipy not available; skipping significance tests.")
        return

    hard_trials = [t for t in trials if t["paradigm"] in ("Paradigm II", "Paradigm III")]
    baselines = ["msg", "broadcast", "sfs"]
    results: dict[str, dict] = {}
    pvals: list[float] = []

    for b in baselines:
        a_vals, b_vals = _task_means(hard_trials, "etmcp", b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = paired_wilcoxon(a_vals, b_vals)
        delta = cliffs_delta(a_vals, b_vals)
        results[b] = {
            "p": p,
            "delta": delta,
            "etmcp_mean": sum(a_vals) / len(a_vals),
            "base_mean": sum(b_vals) / len(b_vals),
            "n": len(a_vals),
        }
        pvals.append(p)

    rejections = holm_bonferroni(pvals)

    print(f"n = {results['msg']['n']} tasks (Paradigms II and III)  "
          "| paired by task, mean over 3 seeds  "
          "| Holm-Bonferroni corrected (k=3)\n")
    print("| Baseline | ET-MCP mean | Baseline mean | p (Wilcoxon) | Cliff's δ | Significant? |")
    print("|----------|-------------|---------------|--------------|-----------|--------------|")
    for i, b in enumerate(baselines):
        r = results[b]
        label = PROTOCOL_LABELS[b]
        sig = "Yes*" if rejections[i] else "No"
        print(f"| {label} | {r['etmcp_mean']:.1f} | {r['base_mean']:.1f} | "
              f"{r['p']:.4f} | {r['delta']:.3f} | {sig} |")

    print("\nNote: Cliff's δ < 0 means ET-MCP has lower RSR (less redundant re-submission).")
    print("Effect size thresholds: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large.")

    # --- ET-MCP task win rate (Paradigms II+III) ---
    print("\n## Task-level win rate — ET-MCP vs. each baseline (Paradigms II+III)")
    task_proto: dict[tuple[str, str], list[int]] = defaultdict(list)
    for t in hard_trials:
        task_proto[(t["task"], t["protocol"])].append(t["stale_rsr"])
    hard_tasks = sorted({t["task"] for t in hard_trials})
    print()
    for b in baselines:
        wins = ties = losses = 0
        for task in hard_tasks:
            a_seeds = task_proto.get((task, "etmcp"), [])
            b_seeds = task_proto.get((task, b), [])
            if not a_seeds or not b_seeds:
                continue
            a_mean = sum(a_seeds) / len(a_seeds)
            b_mean = sum(b_seeds) / len(b_seeds)
            if a_mean < b_mean:
                wins += 1
            elif a_mean == b_mean:
                ties += 1
            else:
                losses += 1
        label = PROTOCOL_LABELS[b]
        print(f"ET-MCP wins vs {label}: {wins}/{len(hard_tasks)} tasks (tie={ties}, loss={losses})")


if __name__ == "__main__":
    run_analysis()

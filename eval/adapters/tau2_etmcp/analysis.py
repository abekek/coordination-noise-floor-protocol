"""Trial-level analysis for tau2-bench ET-MCP runs.

Reads tau2 results (Results JSON or per-task split-format dir) and produces:

- redundancy metric per trial (count of tool calls matching a prior
  errored call with the same (tool_name, normalized_args) key)
- pass^k per (task, protocol) cell — fraction of tasks where ALL k trials
  succeed (reward == 1)
- per-cell aggregates (success rate, mean tokens, mean tool calls,
  mean redundant calls, mean cost)
- paired Wilcoxon signed-rank on per-trial metrics between two cells,
  paired by (task_id, trial)
- Cliff's delta effect size for the same comparisons

All metric definitions follow the conservative chronological-only rule
(see paper §6.3): the first failing call is never counted as redundant;
only later repeats are. The metric is condition-symmetric — it does not
read the trace store, only the in-transcript tool-call history.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable


# ----------------------------------------------------------------------
# Tau2 results loading
# ----------------------------------------------------------------------


def load_results(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Return a flat list of SimulationRun-as-dict.

    Accepts either:
      - a single JSON file (tau2's monolithic Results format), or
      - a directory containing per-simulation JSON files plus a results.json
        index (tau2's split format).
    """
    p = pathlib.Path(path)
    if p.is_dir():
        sims = []
        # First check for the index file
        idx = p / "results.json"
        if idx.exists():
            with open(idx) as f:
                data = json.load(f)
            for entry in data.get("simulations", []):
                # Could be inline or filename ref
                if isinstance(entry, dict) and "messages" in entry:
                    sims.append(entry)
        # Then scan individual sim files (split-format)
        for f in sorted(p.glob("*.json")):
            if f.name == "results.json":
                continue
            with open(f) as fh:
                try:
                    sim = json.load(fh)
                    if isinstance(sim, dict) and "task_id" in sim:
                        sims.append(sim)
                except json.JSONDecodeError:
                    continue
        return sims
    # Monolithic file
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict) and "simulations" in data:
        return list(data["simulations"])
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "task_id" in data:
        return [data]
    return []


# ----------------------------------------------------------------------
# Per-trial metrics
# ----------------------------------------------------------------------


def _normalize_args(args: dict[str, Any]) -> str:
    """Hashable, order-independent canonical form."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def _iter_tool_call_records(messages: Iterable[dict[str, Any]]) -> Iterable[
    tuple[int, str, str, bool]
]:
    """Yield (turn_index, tool_name, normalized_args, errored) for each
    tool exchange in the message list.

    A tool exchange is an AssistantMessage with tool_calls + the subsequent
    ToolMessage(s). We pair them by tool_call_id when possible, else by
    chronological position.
    """
    pending: dict[str, tuple[int, str, str]] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        # AssistantMessage with tool_calls
        tcs = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if tcs:
            for tc in tcs:
                tc_id = tc.get("id") or f"_pos{i}_{tc.get('name','')}"
                pending[tc_id] = (
                    i,
                    tc.get("name", "unknown"),
                    _normalize_args(tc.get("arguments") or {}),
                )
        # ToolMessage carries error flag
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id") or msg.get("id") or ""
            errored = bool(msg.get("error", False))
            if tc_id in pending:
                turn, name, args_norm = pending.pop(tc_id)
                yield turn, name, args_norm, errored
            else:
                # Tool message with no matching call (shouldn't happen) —
                # fall back to position
                yield i, msg.get("requestor", "unknown"), "", errored


@dataclass
class TrialMetrics:
    """Computed per-trial metrics extracted from a SimulationRun."""

    task_id: str
    trial: int | None
    protocol: str | None  # filled in by the runner using config knowledge
    reward: float
    success: bool
    n_tool_calls: int
    n_errored_calls: int
    n_redundant_calls: int
    redundant_rate: float
    agent_cost: float | None
    user_cost: float | None
    termination_reason: str | None


def compute_trial_metrics(
    sim: dict[str, Any], protocol: str | None = None
) -> TrialMetrics:
    messages = sim.get("messages") or []
    tool_records = list(_iter_tool_call_records(messages))

    first_failure_by_key: dict[tuple[str, str], int] = {}
    redundant = 0
    errored = 0
    for turn, name, args_norm, err in tool_records:
        key = (name, args_norm)
        if err:
            errored += 1
            if key not in first_failure_by_key:
                first_failure_by_key[key] = turn
            # Else: a repeat of an already-failed call that errored again —
            # count as redundant (it was avoidable).
            elif turn > first_failure_by_key[key]:
                redundant += 1
        else:
            # Successful call. If a prior call with the same key errored,
            # then this call wasn't strictly redundant (it succeeded after
            # all). Do not count.
            pass

    total = len(tool_records)
    reward_info = sim.get("reward_info") or {}
    reward_val = reward_info.get("reward", 0.0) if isinstance(reward_info, dict) else 0.0
    return TrialMetrics(
        task_id=sim.get("task_id", "?"),
        trial=sim.get("trial"),
        protocol=protocol,
        reward=float(reward_val),
        success=bool(reward_val == 1.0),
        n_tool_calls=total,
        n_errored_calls=errored,
        n_redundant_calls=redundant,
        redundant_rate=(redundant / total) if total > 0 else 0.0,
        agent_cost=sim.get("agent_cost"),
        user_cost=sim.get("user_cost"),
        termination_reason=sim.get("termination_reason"),
    )


# ----------------------------------------------------------------------
# Cell aggregates and pass^k
# ----------------------------------------------------------------------


@dataclass
class CellAggregate:
    protocol: str
    n_trials: int
    n_tasks: int
    success_rate: float
    pass_at_k: dict[int, float] = field(default_factory=dict)
    mean_tool_calls: float = 0.0
    mean_errored_calls: float = 0.0
    mean_redundant_calls: float = 0.0
    mean_redundant_rate: float = 0.0
    mean_agent_cost: float = 0.0
    mean_user_cost: float = 0.0


def aggregate_cell(metrics: list[TrialMetrics]) -> CellAggregate:
    if not metrics:
        return CellAggregate(protocol="?", n_trials=0, n_tasks=0, success_rate=0.0)
    protocol = metrics[0].protocol or "?"
    tasks_seen = {m.task_id for m in metrics}

    # pass^k: fraction of tasks where the FIRST k trials all succeeded
    by_task: dict[str, list[TrialMetrics]] = defaultdict(list)
    for m in metrics:
        by_task[m.task_id].append(m)
    # Sort each task's trials by trial number for deterministic pass^k
    for t in by_task.values():
        t.sort(key=lambda m: (m.trial if m.trial is not None else 0))
    max_k = min(len(t) for t in by_task.values()) if by_task else 0
    pass_at_k = {}
    for k in range(1, max_k + 1):
        pass_at_k[k] = sum(
            1 for trials in by_task.values()
            if all(t.success for t in trials[:k])
        ) / len(by_task)

    def safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else 0.0

    return CellAggregate(
        protocol=protocol,
        n_trials=len(metrics),
        n_tasks=len(tasks_seen),
        success_rate=mean(1.0 if m.success else 0.0 for m in metrics),
        pass_at_k=pass_at_k,
        mean_tool_calls=safe_mean([m.n_tool_calls for m in metrics]),
        mean_errored_calls=safe_mean([m.n_errored_calls for m in metrics]),
        mean_redundant_calls=safe_mean([m.n_redundant_calls for m in metrics]),
        mean_redundant_rate=safe_mean([m.redundant_rate for m in metrics]),
        mean_agent_cost=safe_mean([m.agent_cost for m in metrics]),
        mean_user_cost=safe_mean([m.user_cost for m in metrics]),
    )


# ----------------------------------------------------------------------
# Paired Wilcoxon signed-rank + Cliff's delta
# ----------------------------------------------------------------------


def _pair_trials(
    a: list[TrialMetrics], b: list[TrialMetrics]
) -> list[tuple[TrialMetrics, TrialMetrics]]:
    """Pair by (task_id, trial)."""
    idx_a = {(m.task_id, m.trial): m for m in a}
    idx_b = {(m.task_id, m.trial): m for m in b}
    keys = sorted(set(idx_a) & set(idx_b))
    return [(idx_a[k], idx_b[k]) for k in keys]


def _wilcoxon_signed_rank(diffs: list[float]) -> tuple[float | None, int]:
    """Two-sided paired Wilcoxon. Returns (p_value, n_nonzero).

    Uses scipy.stats.wilcoxon if available; else returns None.
    """
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) < 5:
        return None, len(nonzero)
    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat, p = wilcoxon(nonzero, zero_method="wilcox")
        return float(p), len(nonzero)
    except ImportError:
        return None, len(nonzero)


def _cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta: P(X>Y) - P(X<Y). Range [-1, 1]. Positive means a > b."""
    if not a or not b:
        return 0.0
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n


def _cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


@dataclass
class PairedComparison:
    metric: str
    n_pairs: int
    p_value: float | None
    cliffs_delta: float
    magnitude: str
    mean_a: float
    mean_b: float
    direction: str  # "a > b" / "a < b" / "tied"


def paired_comparison(
    a: list[TrialMetrics],
    b: list[TrialMetrics],
    metric: str,
) -> PairedComparison:
    """Compare cells a and b on `metric`.

    Supported metrics: 'success', 'redundant_rate', 'redundant_calls',
    'tool_calls', 'agent_cost'.
    """
    extractor = {
        "success": lambda m: 1.0 if m.success else 0.0,
        "redundant_rate": lambda m: m.redundant_rate,
        "redundant_calls": lambda m: float(m.n_redundant_calls),
        "tool_calls": lambda m: float(m.n_tool_calls),
        "agent_cost": lambda m: float(m.agent_cost or 0.0),
    }[metric]

    pairs = _pair_trials(a, b)
    a_vals = [extractor(x) for x, _ in pairs]
    b_vals = [extractor(y) for _, y in pairs]
    diffs = [x - y for x, y in zip(a_vals, b_vals)]
    p, n_nonzero = _wilcoxon_signed_rank(diffs)
    delta = _cliffs_delta(a_vals, b_vals)
    mean_a = mean(a_vals) if a_vals else 0.0
    mean_b = mean(b_vals) if b_vals else 0.0
    if mean_a > mean_b:
        direction = "a > b"
    elif mean_a < mean_b:
        direction = "a < b"
    else:
        direction = "tied"
    return PairedComparison(
        metric=metric,
        n_pairs=n_nonzero,
        p_value=p,
        cliffs_delta=delta,
        magnitude=_cliffs_magnitude(delta),
        mean_a=mean_a,
        mean_b=mean_b,
        direction=direction,
    )


def holm_correct(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni step-down. Preserves None entries."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda x: x[1])
    n = len(indexed)
    out: list[float | None] = list(p_values)
    last = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, max(last, (n - rank) * p))
        last = adj
        out[orig_idx] = adj
    return out


# ----------------------------------------------------------------------
# CLI entry point: summarize a results directory
# ----------------------------------------------------------------------


def summarize(
    results_paths: dict[str, str | pathlib.Path],
    out_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build a per-cell summary from a {protocol: results-path} map and
    emit pairwise comparisons of each non-baseline cell vs. the default
    et_mcp cell.

    Returns a dict suitable for JSON dump.
    """
    cells: dict[str, list[TrialMetrics]] = {}
    for protocol, path in results_paths.items():
        sims = load_results(path)
        cells[protocol] = [compute_trial_metrics(s, protocol=protocol) for s in sims]

    aggregates = {p: aggregate_cell(m) for p, m in cells.items()}

    pairs = []
    baseline_name = "et_mcp"
    if baseline_name in cells:
        baseline = cells[baseline_name]
        for protocol, metrics in cells.items():
            if protocol == baseline_name:
                continue
            for metric in ["success", "redundant_rate", "redundant_calls"]:
                cmp = paired_comparison(baseline, metrics, metric)
                pairs.append({
                    "a": baseline_name,
                    "b": protocol,
                    "metric": metric,
                    "n_pairs": cmp.n_pairs,
                    "p": cmp.p_value,
                    "cliffs_delta": cmp.cliffs_delta,
                    "magnitude": cmp.magnitude,
                    "mean_a": cmp.mean_a,
                    "mean_b": cmp.mean_b,
                    "direction": cmp.direction,
                })

    # Holm-correct within each metric family
    for metric in {p["metric"] for p in pairs}:
        family = [p for p in pairs if p["metric"] == metric]
        p_vals = [p["p"] for p in family]
        adjusted = holm_correct(p_vals)
        for p, adj in zip(family, adjusted):
            p["p_holm"] = adj

    summary = {
        "cells": {
            p: {
                "protocol": agg.protocol,
                "n_trials": agg.n_trials,
                "n_tasks": agg.n_tasks,
                "success_rate": agg.success_rate,
                "pass_at_k": agg.pass_at_k,
                "mean_tool_calls": agg.mean_tool_calls,
                "mean_errored_calls": agg.mean_errored_calls,
                "mean_redundant_calls": agg.mean_redundant_calls,
                "mean_redundant_rate": agg.mean_redundant_rate,
                "mean_agent_cost": agg.mean_agent_cost,
                "mean_user_cost": agg.mean_user_cost,
            }
            for p, agg in aggregates.items()
        },
        "pairwise_vs_et_mcp": pairs,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


def main(argv: list[str]) -> int:
    """Usage: python -m eval.adapters.tau2_etmcp.analysis \\
                  --protocol no_coord:/path/to/no_coord_results \\
                  --protocol push_scratchpad:/path/to/push_results \\
                  ... \\
                  [--out summary.json]
    """
    paths: dict[str, str] = {}
    out_path = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--protocol" and i + 1 < len(argv):
            proto, _, path = argv[i + 1].partition(":")
            paths[proto] = path
            i += 2
        elif a == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]
            i += 2
        else:
            i += 1
    if not paths:
        print("usage: --protocol <name>:<path> [--protocol ...] [--out <path>]")
        return 2
    summary = summarize(paths, out_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

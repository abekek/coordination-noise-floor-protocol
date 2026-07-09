"""LaTeX table renderers for the paper's §6 results section.

All tables use booktabs and acmart-compatible commands. No colors;
no extra packages required beyond what the paper already loads.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis.aggregate import aggregate_trials, per_condition_summary
from analysis.stats import holm_bonferroni, pairwise_comparison


_CONDITION_DISPLAY = {
    "B1_full_context": "B1 Full context",
    "B2_summarization": "B2 Summarization",
    "B3_ca_mcp_style": "B3 CA-MCP-style",
    "ET_MCP_default": "\\textbf{ET-MCP}",
    "ET_MCP_failure_strict": "A1 Failure-strict",
    "ET_MCP_write_everything": "A2 Write-everything",
    "ET_MCP_persistent": "A3 Persistent-store",
}


def _display_condition(cond: str) -> str:
    return _CONDITION_DISPLAY.get(cond, cond.replace("_", " "))


def headline_results_table(
    results_dir: Path | str, *,
    caption: str = "Headline results across conditions, averaged across queries and seeds.",
    label: str = "tab:headline",
) -> str:
    """Render an acmart/booktabs headline results table.

    One row per condition; columns are completion rate, mean tokens,
    redundant call rate, and trace event count.
    """
    summary = per_condition_summary(results_dir)
    if not summary:
        return _empty_table(caption=caption, label=label)

    # Stable order: B1, B2, B3, ET-MCP, then any ablations
    ordering = [
        "B1_full_context", "B2_summarization", "B3_ca_mcp_style",
        "ET_MCP_default", "ET_MCP_failure_strict",
        "ET_MCP_write_everything", "ET_MCP_persistent",
    ]
    conds_in_order = [c for c in ordering if c in summary]
    extras = [c for c in summary if c not in ordering]
    conds_in_order.extend(sorted(extras))

    rows = []
    for cond in conds_in_order:
        s = summary[cond]
        rows.append(" & ".join([
            _display_condition(cond),
            f"{s['completion_rate']:.2f}",
            f"{s['mean_total_tokens']:.0f}",
            f"{s['mean_redundant_call_rate']:.3f}",
            f"{s['mean_trace_events']:.2f}",
        ]) + " \\\\")

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Condition & Completion & Tokens & Redundant & Trace events \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def pairwise_comparison_table(
    results_dir: Path | str, *,
    condition: str,
    baselines: list[str],
    metric: str = "mean_total_tokens",
    caption: str | None = None,
    label: str = "tab:pairwise",
    alpha: float = 0.05,
) -> str:
    """Render an acmart/booktabs pairwise comparison table.

    One row per baseline; columns are p-value (Wilcoxon), Cliff's delta,
    n, and mean change (baseline → condition).
    """
    cells = aggregate_trials(results_dir)
    # Build a {condition: {query_id: cell_metric_value}} map for paired access
    by_cond_query: dict[str, dict[str, float]] = defaultdict(dict)
    for c in cells:
        value = getattr(c, metric)
        by_cond_query[c.condition][c.query_id] = value

    condition_values_by_query = by_cond_query.get(condition, {})
    if not condition_values_by_query:
        return _empty_table(
            caption=caption or f"No trials for {condition}.",
            label=label,
        )

    rows = []
    p_values: list[float] = []
    raw_results: list[dict[str, Any] | None] = []

    for b in baselines:
        baseline_q = by_cond_query.get(b, {})
        shared = sorted(set(condition_values_by_query.keys()) & set(baseline_q.keys()))
        if not shared:
            raw_results.append(None)
            p_values.append(1.0)
            continue
        baseline_values = [baseline_q[q] for q in shared]
        condition_values = [condition_values_by_query[q] for q in shared]
        comp = pairwise_comparison(
            baseline_values=baseline_values,
            condition_values=condition_values,
        )
        raw_results.append(comp)
        p_values.append(comp["p_value"])

    sig_flags = holm_bonferroni(p_values, alpha=alpha)

    for i, b in enumerate(baselines):
        comp = raw_results[i]
        sig = sig_flags[i]
        if comp is None:
            rows.append(" & ".join([
                _display_condition(b), "---", "---", "0", "n/a",
            ]) + " \\\\")
            continue
        sig_mark = "$\\ast$" if sig else ""
        rows.append(" & ".join([
            _display_condition(b),
            f"{comp['p_value']:.4f}{sig_mark}",
            f"{comp['cliffs_delta']:+.3f}",
            f"{comp['n']}",
            f"{comp['baseline_mean']:.0f} $\\to$ {comp['condition_mean']:.0f}",
        ]) + " \\\\")

    if caption is None:
        caption = (
            f"Pairwise comparison of \\texttt{{{condition}}} vs.\\ each baseline "
            f"on metric \\texttt{{{metric}}}. "
            f"$\\ast$ indicates rejected at $\\alpha = {alpha}$ after Holm-Bonferroni."
        )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Baseline & $p$ (Wilcoxon) & Cliff's $\\delta$ & $n$ & mean change \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _empty_table(*, caption: str, label: str) -> str:
    return "\n".join([
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\textit{No results available.}",
        "\\end{table}",
    ])

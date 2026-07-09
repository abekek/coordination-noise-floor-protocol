"""CLI for the analysis module.

Usage:
    python -m analysis report --results-dir results/smoke --output-dir results/smoke/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.aggregate import aggregate_trials, per_condition_summary
from analysis.tables import headline_results_table, pairwise_comparison_table


def report_command(*, results_dir: Path | str,
                   output_dir: Path | str,
                   condition: str = "ET_MCP_default",
                   baselines: list[str] | None = None) -> int:
    if baselines is None:
        baselines = ["B1_full_context", "B2_summarization", "B3_ca_mcp_style"]

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-condition summary (markdown)
    summary = per_condition_summary(results_dir)
    if not summary:
        (output_dir / "summary.md").write_text(
            f"# Report — {results_dir}\n\n_No trials found._\n"
        )
        (output_dir / "headline.tex").write_text("")
        (output_dir / "pairwise_tokens.tex").write_text("")
        return 1

    md_lines = [f"# Report — {results_dir}\n",
                "## Per-condition summary\n",
                "| Condition | n_queries | completion | tokens | redundant | trace_events |",
                "|-----------|-----------|------------|--------|-----------|--------------|"]
    for cond, s in summary.items():
        md_lines.append(
            f"| {cond} | {int(s['n_queries'])} | "
            f"{s['completion_rate']:.2f} | "
            f"{s['mean_total_tokens']:.0f} | "
            f"{s['mean_redundant_call_rate']:.3f} | "
            f"{s['mean_trace_events']:.2f} |"
        )
    md_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(md_lines))

    # 2. Headline LaTeX table
    (output_dir / "headline.tex").write_text(
        headline_results_table(results_dir)
    )

    # 3. Pairwise comparison: ET-MCP vs each baseline on mean_total_tokens
    (output_dir / "pairwise_tokens.tex").write_text(
        pairwise_comparison_table(
            results_dir, condition=condition, baselines=baselines,
            metric="mean_total_tokens", label="tab:pairwise_tokens",
            caption=(f"Pairwise comparison of \\texttt{{{condition}}} vs. each "
                     f"baseline on \\texttt{{mean\\_total\\_tokens}}. "
                     f"$\\ast$ indicates rejected at $\\alpha=0.05$ after Holm-Bonferroni."),
        )
    )

    # 4. Pairwise comparison: ET-MCP vs each baseline on completion_rate
    (output_dir / "pairwise_completion.tex").write_text(
        pairwise_comparison_table(
            results_dir, condition=condition, baselines=baselines,
            metric="completion_rate", label="tab:pairwise_completion",
            caption=(f"Pairwise comparison of \\texttt{{{condition}}} vs. each "
                     f"baseline on \\texttt{{completion\\_rate}}."),
        )
    )

    print(f"Report written to {output_dir}/")
    print(f"  summary.md ({len(summary)} conditions)")
    print(f"  headline.tex")
    print(f"  pairwise_tokens.tex")
    print(f"  pairwise_completion.tex")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="Generate report (summary.md + LaTeX tables)")
    r.add_argument("--results-dir", required=True,
                   help="Path to a results directory containing trials.jsonl")
    r.add_argument("--output-dir", default=None,
                   help="Output directory for report files (default: <results-dir>/report)")
    r.add_argument("--condition", default="ET_MCP_default",
                   help="Condition to compare against baselines")
    r.add_argument("--baselines", nargs="+", default=None,
                   help="Baselines to compare against (default: B1, B2, B3)")
    args = parser.parse_args(argv)

    if args.cmd == "report":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir) / "report"
        return report_command(
            results_dir=args.results_dir, output_dir=output_dir,
            condition=args.condition,
            baselines=args.baselines,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

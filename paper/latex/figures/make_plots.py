"""Generate the matplotlib figures for the τ²-bench retail head-to-head.

Reads trial JSONLs from eval/results/v2pivot_w1_n100/ and writes
publication-quality PDFs into paper/latex/figures/.

Run from project root:
    /Users/alibek/anaconda3/bin/python3 paper/latex/figures/make_plots.py
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict
from math import comb

import matplotlib.pyplot as plt
from matplotlib import rcParams


def wilson_ci(k, n, z=1.96):
    """Two-sided Wilson 95% CI for a proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

REPO = pathlib.Path(__file__).resolve().parents[3]
RESULTS = REPO / "eval" / "results" / "v2pivot_w1_n100"
OUT = pathlib.Path(__file__).resolve().parent

PROTOCOLS = ["no_coord", "pull", "intercept"]
PROTO_LABEL = {"no_coord": "no_coord", "pull": "pull", "intercept": "intercept"}
PROTO_COLOR = {
    "no_coord": "#8b9cb3",  # cool gray
    "pull": "#4c8cb9",      # blue
    "intercept": "#d97a4c", # warm orange (visually distinct from pull)
}


def loads(p):
    return [json.loads(l) for l in open(p)]


def by_task_trial(recs):
    d = defaultdict(dict)
    for r in recs:
        d[r["task_id"]][r["trial"]] = r["reward"]
    return d


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    one = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * one)


def setup_style():
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 9
    rcParams["axes.labelsize"] = 9
    rcParams["axes.titlesize"] = 10
    rcParams["legend.fontsize"] = 8
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 8
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.grid"] = True
    rcParams["axes.axisbelow"] = True
    rcParams["grid.linestyle"] = ":"
    rcParams["grid.alpha"] = 0.5


def fig_trial_split(data):
    """Trial-0 vs trial-1 success rate, by protocol."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = list(range(len(PROTOCOLS)))
    bar_w = 0.36
    t0 = [data[p]["trial_0"] for p in PROTOCOLS]
    t1 = [data[p]["trial_1"] for p in PROTOCOLS]
    b0 = ax.bar([i - bar_w / 2 for i in x], t0, bar_w,
                color="#5e8b7e", edgecolor="black", linewidth=0.5,
                label="trial 0 (empty store)")
    b1 = ax.bar([i + bar_w / 2 for i in x], t1, bar_w,
                color="#a47e63", edgecolor="black", linewidth=0.5,
                label="trial 1 (store filled)")
    for bar in b0 + b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTO_LABEL[p] for p in PROTOCOLS])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 0.82)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_trial_split.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_headline(data):
    """Headline marginal pass^k bars across protocols."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = list(range(len(PROTOCOLS)))
    bar_w = 0.36
    p1 = [data[p]["pass_1"] for p in PROTOCOLS]
    p2 = [data[p]["pass_2"] for p in PROTOCOLS]
    b1 = ax.bar([i - bar_w / 2 for i in x], p1, bar_w,
                color="#4c8cb9", edgecolor="black", linewidth=0.5,
                label=r"pass$^1$ (marginal)")
    b2 = ax.bar([i + bar_w / 2 for i in x], p2, bar_w,
                color="#d97a4c", edgecolor="black", linewidth=0.5,
                label=r"pass$^2$")
    for bar in b1 + b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTO_LABEL[p] for p in PROTOCOLS])
    ax.set_ylabel("Pass$^k$")
    ax.set_ylim(0, 0.82)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_headline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_winloss(pairs):
    """Stacked horizontal bars: wins/losses/ties for each pairwise contrast."""
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    contrasts = list(pairs.keys())
    n = len(contrasts)
    y = list(range(n))
    wins = [pairs[c]["wins"] for c in contrasts]
    losses = [pairs[c]["losses"] for c in contrasts]
    ties = [pairs[c]["ties"] for c in contrasts]
    ax.barh(y, wins, color="#5b9bd5", edgecolor="black", linewidth=0.5,
            label="wins (left)")
    ax.barh(y, losses, left=wins, color="#ed7d31", edgecolor="black",
            linewidth=0.5, label="losses (right)")
    ax.barh(y, ties, left=[w + l for w, l in zip(wins, losses)],
            color="#d9d9d9", edgecolor="black", linewidth=0.5, label="ties")
    for i, c in enumerate(contrasts):
        w, l, t = wins[i], losses[i], ties[i]
        if w > 0:
            ax.text(w / 2, i, f"{w}", ha="center", va="center", fontsize=7.5)
        if l > 0:
            ax.text(w + l / 2, i, f"{l}", ha="center", va="center", fontsize=7.5)
        if t > 5:
            ax.text(w + l + t / 2, i, f"{t}", ha="center", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(contrasts)
    ax.set_xlabel("Tasks (paired, n=100)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_winloss.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_combined(data, pairs, n_tasks=100):
    """Two-panel figure: (a) trial-conditional null with Wilson CIs +
    noise-floor band, (b) paired W/L/T outcomes with no-difference midline.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.05),
                                   gridspec_kw={"width_ratios": [1.0, 1.2]})

    # ----- Left: trial split with Wilson CIs and noise-floor band -----
    x = list(range(len(PROTOCOLS)))
    bar_w = 0.36
    t0 = [data[p]["trial_0"] for p in PROTOCOLS]
    t1 = [data[p]["trial_1"] for p in PROTOCOLS]
    proto_colors = [PROTO_COLOR[p] for p in PROTOCOLS]

    def err(rates, n):
        lows, his = [], []
        for r in rates:
            lo, hi = wilson_ci(int(round(r * n)), n)
            lows.append(max(0.0, r - lo))
            his.append(max(0.0, hi - r))
        return [lows, his]

    # Noise-floor envelope (Haiku retail trial-0 paired gaps, anchored at
    # the t0 mean; half-width 7.5pp = the pooled upper Wilson CI of ~15pp).
    t0_mean = sum(t0) / len(t0)
    ax1.axhspan(t0_mean - 0.075, t0_mean + 0.075, color="red", alpha=0.05, zorder=0)
    ax1.axhline(t0_mean, color="red", linestyle=":", linewidth=0.7,
                alpha=0.5, zorder=0)

    b0 = ax1.bar([i - bar_w / 2 for i in x], t0, bar_w,
                 color=proto_colors, edgecolor="black", linewidth=0.5,
                 yerr=err(t0, n_tasks), capsize=2.5, ecolor="black",
                 error_kw={"linewidth": 0.6})
    b1 = ax1.bar([i + bar_w / 2 for i in x], t1, bar_w,
                 color=proto_colors, edgecolor="black", linewidth=0.5,
                 hatch="//", yerr=err(t1, n_tasks), capsize=2.5,
                 ecolor="black", error_kw={"linewidth": 0.6})
    # Place value labels INSIDE bars (white text) when bar > 0.15, otherwise
    # above. Vertical orientation avoids collisions between adjacent t0/t1
    # bars when values are close (e.g., no_coord t0=t1=0.54).
    for bar in list(b0) + list(b1):
        h = bar.get_height()
        if h >= 0.18:
            ax1.text(bar.get_x() + bar.get_width() / 2, h - 0.015,
                     f"{h:.2f}", ha="center", va="top",
                     fontsize=6.8, color="white", fontweight="bold",
                     rotation=90)
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                     f"{h:.2f}", ha="center", va="bottom",
                     fontsize=6.8, rotation=90)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PROTO_LABEL[p] for p in PROTOCOLS])
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(0, 0.95)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

    # Custom legend: trial-0 vs trial-1 (hatching), not protocol colors.
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor="white", edgecolor="black", label="trial 0 (empty store)"),
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="trial 1 (store filled)"),
        Patch(facecolor="red", alpha=0.18,
              label=r"noise envelope (pooled upper CI $\lesssim$15 pp)"),
    ]
    ax1.legend(handles=leg_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False,
               fontsize=6.8, handlelength=1.4)
    ax1.set_title("(a) Trial-conditional null", fontsize=9, pad=4)

    # ----- Right: paired win/loss/tie with no-difference midline -----
    contrasts = list(pairs.keys())
    y = list(range(len(contrasts)))
    wins = [pairs[c]["wins"] for c in contrasts]
    losses = [pairs[c]["losses"] for c in contrasts]
    ties = [pairs[c]["ties"] for c in contrasts]
    ps = [pairs[c]["p"] for c in contrasts]
    ax2.barh(y, wins, color="#4c8cb9", edgecolor="black", linewidth=0.5,
             label="wins (left side)")
    ax2.barh(y, losses, left=wins, color="#d97a4c", edgecolor="black",
             linewidth=0.5, label="losses (right side)")
    ax2.barh(y, ties, left=[w + l for w, l in zip(wins, losses)],
             color="#d9d9d9", edgecolor="black", linewidth=0.5, label="ties")
    for i, c in enumerate(contrasts):
        w, l, t = wins[i], losses[i], ties[i]
        if w > 0:
            ax2.text(w / 2, i, f"{w}", ha="center", va="center", fontsize=7.3)
        if l > 0:
            ax2.text(w + l / 2, i, f"{l}", ha="center", va="center", fontsize=7.3)
        if t > 5:
            ax2.text(w + l + t / 2, i, f"{t}", ha="center", va="center",
                     fontsize=7.3)
        ax2.text(n_tasks + 1.5, i, f"$p{{=}}{ps[i]:.2f}$",
                 ha="left", va="center", fontsize=7.0, color="black")
    # No-difference reference line: wins == losses at midpoint of w+l span.
    ax2.axvline(n_tasks / 2.0, color="black", linestyle=":", linewidth=0.6,
                alpha=0.45)
    ax2.set_yticks(y)
    ax2.set_yticklabels(contrasts, fontsize=8)
    ax2.set_xlabel(f"Tasks (paired, n={n_tasks})")
    ax2.set_xlim(0, n_tasks + 16)
    ax2.invert_yaxis()
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
               ncol=3, frameon=False, fontsize=7.0)
    ax2.set_title("(b) Paired trial-1 outcomes", fontsize=9, pad=4)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_p2_validation():
    """Bar chart of trial-1 success on coordination-active subset:
    vanilla intercept vs selective intercept at Haiku 4.5 retail n=30
    (paired subset n=19; sign test 2/7/21, p=0.18 underpowered).
    Hatching signals "same intercept family, different variant."
    """
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    labels = ['vanilla\nintercept', 'selective\nintercept (P2)']
    rates = [0.158, 0.316]
    # Wilson 95% CIs for n=19 (subset size).
    lo_v, hi_v = wilson_ci(3, 19)   # 3/19 ≈ 0.158
    lo_s, hi_s = wilson_ci(6, 19)   # 6/19 ≈ 0.316
    errs_low = [rates[0] - lo_v, rates[1] - lo_s]
    errs_hi  = [hi_v - rates[0], hi_s - rates[1]]
    x = list(range(len(labels)))
    intercept_color = PROTO_COLOR["intercept"]
    bars = ax.bar(x, rates,
                  color=[intercept_color, intercept_color],
                  edgecolor='black', linewidth=0.5,
                  hatch=['', '//'],
                  yerr=[errs_low, errs_hi], capsize=4, ecolor='black',
                  error_kw={'linewidth': 0.8})
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight='bold')
    # Annotate the +15.8pp gap directly above the bars.
    y_brk = max(rates) + 0.10
    ax.annotate('', xy=(0, y_brk), xytext=(1, y_brk),
                arrowprops=dict(arrowstyle='-', lw=0.7, color='black'))
    ax.text(0.5, y_brk + 0.012, r"$+15.8$pp", ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Trial-1 success (coord-active)", fontsize=8.5)
    ax.set_ylim(0, 0.75)
    ax.set_title("P2 selective intercept (Haiku 4.5 retail)",
                 fontsize=9, pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_p2_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_noise_floor_by_model(rows):
    """Bar chart of trial-0 paired sign-test gap by model + domain.

    rows: [(group_label, contrast_label, gap_pp, color, n, p), ...]
      e.g., ('Haiku retail', 'pull vs ic', 18.0, PROTO_COLOR['pull'], 100, 0.012)
    Backwards-compatible: a 3-tuple (label, gap, color) is still accepted.

    Renders one bar per (model, domain, contrast) cell, grouped by
    (model, domain). Replicates the noise floor cross-model + cross-domain.
    """
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    # Normalize rows to the full 6-tuple form.
    norm = []
    for r in rows:
        if len(r) == 3:
            norm.append((r[0], '', r[1], r[2], None, None))
        elif len(r) == 6:
            norm.append(r)
        else:
            raise ValueError(f"unexpected row shape: {r}")
    # Group by group_label preserving order.
    groups, seen = [], {}
    for gl, cl, gap, col, n, p in norm:
        if gl not in seen:
            seen[gl] = len(groups)
            groups.append({"label": gl, "items": []})
        groups[seen[gl]]["items"].append((cl, gap, col, n, p))

    # Lay bars out with spacers between groups.
    x_positions, bar_meta, x_cursor = [], [], 0.0
    group_spans = []  # (group_label, x_lo, x_hi)
    for g in groups:
        x_lo = x_cursor
        for cl, gap, col, n, p in g["items"]:
            x_positions.append(x_cursor)
            bar_meta.append((cl, gap, col, n, p))
            x_cursor += 1.0
        group_spans.append((g["label"], x_lo, x_cursor - 1.0))
        x_cursor += 1.6  # spacer between groups (must fit group label)

    gaps = [m[1] for m in bar_meta]
    colors = [m[2] for m in bar_meta]
    bars = ax.bar(x_positions, gaps, width=0.8,
                  color=colors, edgecolor='black', linewidth=0.5)
    # Render zero bars as low hatched "ghost" bars so a measured-zero reads
    # as a deliberate null result rather than missing data.
    for bar, (cl, gap, col, n, p) in zip(bars, bar_meta):
        x_b = bar.get_x() + bar.get_width()/2
        if gap == 0:
            # short hatched ghost bar at baseline (height 0.8pp) so the slot
            # is visually occupied. Same color as the cell, lighter alpha,
            # with diagonal hatch to signal "null measurement".
            ax.bar([x_b], [0.8], width=0.8,
                   color=col, alpha=0.35, edgecolor='black', linewidth=0.5,
                   hatch='///')
        # value label (signed; above positive bars, below negative ones)
        label = f"{gap:+.0f}" if gap == int(gap) else f"{gap:+.1f}"
        if p is not None and p < 0.05:
            label += '*'
        # zero bars get explicit "0 (n.s.)" wording above their ghost bar
        if gap == 0:
            label = '0 pp\n(n.s.)'
            ax.text(x_b, 2.0,
                    label, ha="center", va="bottom", fontsize=6.6,
                    color='gray', linespacing=0.95)
        elif gap > 0:
            ax.text(x_b, gap + 0.55,
                    label, ha="center", va="bottom", fontsize=7.8,
                    fontweight='bold')
        else:
            ax.text(x_b, gap - 0.55,
                    label, ha="center", va="top", fontsize=7.8,
                    fontweight='bold')
        # contrast label in a fixed band below all bars
        if cl:
            ax.text(x_b, -25.0, cl,
                    ha="center", va="top", fontsize=6.2, color='black',
                    rotation=25)

    # Group separators + group labels (above bars).
    for i, (label, x_lo, x_hi) in enumerate(group_spans):
        centre = (x_lo + x_hi) / 2.0
        ax.text(centre, 27.0, label, ha='center', va='bottom',
                fontsize=7.8, fontweight='bold')
        if i < len(group_spans) - 1:
            sep_x = x_hi + 1.3
            ax.axvline(sep_x, color='gray', linestyle='-', linewidth=0.4,
                       alpha=0.4)

    # Pooled envelope over the two Haiku-retail seed groups: shaded band =
    # pooled clean-contrast CI [-2,+12]pp; dotted line = largest pooled
    # upper Wilson CI (~15pp, pull vs intercept). Labels sit to the right
    # of the retail groups, above the Haiku-airline zero bars.
    xlim_lo, xlim_hi = -0.6, x_cursor
    if len(group_spans) >= 2:
        x_lo = group_spans[0][1] - 0.45
        x_hi = group_spans[1][2] + 0.45
        frac = lambda x: (x - xlim_lo) / (xlim_hi - xlim_lo)
        ax.axhspan(-2, 12, xmin=frac(x_lo), xmax=frac(x_hi),
                   color='red', alpha=0.08)
        ax.axhline(15, xmin=frac(x_lo), xmax=frac(x_hi),
                   color='red', linestyle=':', linewidth=0.8)
        ax.text(x_hi + 0.25, 15.6, 'max pooled upper CI ≈15 pp',
                fontsize=6.6, color='red', va='bottom', ha='left',
                alpha=0.9)
        ax.text(x_hi + 0.25, 9.0, 'pooled clean CI [−2,+12] pp',
                fontsize=6.6, color='red', va='bottom', ha='left',
                alpha=0.9)

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks([])
    ax.set_ylabel("Trial-0 signed paired gap (pp)")
    ax.set_ylim(-29, 33)
    ax.set_xlim(xlim_lo, xlim_hi)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_noise_floor.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    nc = loads(RESULTS / "no_coord" / "trials.jsonl")
    pl = loads(RESULTS / "pull" / "trials.jsonl")
    ic = loads(RESULTS / "intercept" / "trials.jsonl")
    nc_d, pl_d, ic_d = by_task_trial(nc), by_task_trial(pl), by_task_trial(ic)
    tasks = sorted(set(nc_d) & set(pl_d) & set(ic_d))

    def t01(d):
        t0 = sum(1 for t in tasks if d[t][0] == 1.0) / len(tasks)
        t1 = sum(1 for t in tasks if d[t][1] == 1.0) / len(tasks)
        return t0, t1

    nc_t0, nc_t1 = t01(nc_d)
    pl_t0, pl_t1 = t01(pl_d)
    ic_t0, ic_t1 = t01(ic_d)

    def p_at(d, k):
        if k == 1:
            return sum(sum(d[t].values()) / 2.0 for t in tasks) / len(tasks)
        return sum(1 for t in tasks if d[t][0] == 1.0 and d[t][1] == 1.0) / len(tasks)

    data = {
        "no_coord": {"trial_0": nc_t0, "trial_1": nc_t1,
                      "pass_1": p_at(nc_d, 1), "pass_2": p_at(nc_d, 2)},
        "pull": {"trial_0": pl_t0, "trial_1": pl_t1,
                  "pass_1": p_at(pl_d, 1), "pass_2": p_at(pl_d, 2)},
        "intercept": {"trial_0": ic_t0, "trial_1": ic_t1,
                       "pass_1": p_at(ic_d, 1), "pass_2": p_at(ic_d, 2)},
    }

    fig_trial_split(data)
    fig_headline(data)

    # Win/loss/tie pairwise on trial 1
    def pair(a, b):
        w = sum(1 for t in tasks if a[t][1] > b[t][1])
        l = sum(1 for t in tasks if a[t][1] < b[t][1])
        tt = sum(1 for t in tasks if a[t][1] == b[t][1])
        return {"wins": w, "losses": l, "ties": tt, "p": sign_p(w, l)}

    pairs = {
        "pull vs no_coord": pair(pl_d, nc_d),
        "intercept vs no_coord": pair(ic_d, nc_d),
        "pull vs intercept": pair(pl_d, ic_d),
    }
    fig_winloss(pairs)
    fig_combined(data, pairs)
    fig_p2_validation()

    # Noise-floor across seed+model+domain+contrast.
    # Rows: (group, contrast, signed_gap_pp, color, n, p_corr).
    # Sign convention: "a − b" = (wins_a − wins_b) / n in pp, from the
    # paired trial-0 sign tests. Verified against raw trials.jsonl
    # (seed-1/seed-2 retail: §6.3.2; airline/Sonnet recomputed 2026-06-09).
    noise_rows = [
        ("Haiku retail\nseed 1 (n=100)", "nc − ic",    10.0, PROTO_COLOR["intercept"], 100, 0.330),
        ("Haiku retail\nseed 1 (n=100)", "pull − nc",   8.0, PROTO_COLOR["pull"],      100, 0.555),
        ("Haiku retail\nseed 1 (n=100)", "pull − ic",  18.0, PROTO_COLOR["no_coord"],  100, 0.012),
        ("Haiku retail\nseed 2 (n=100)", "nc − ic",     0.0, PROTO_COLOR["intercept"], 100, 1.000),
        ("Haiku retail\nseed 2 (n=100)", "pull − nc",  -3.0, PROTO_COLOR["pull"],      100, 1.000),
        ("Haiku retail\nseed 2 (n=100)", "pull − ic",  -3.0, PROTO_COLOR["no_coord"],  100, 1.000),
        ("Haiku airline\n(n=30)",        "nc − ic",     0.0, PROTO_COLOR["intercept"],  30, 1.000),
        ("Haiku airline\n(n=30)",        "pull − nc",   0.0, PROTO_COLOR["pull"],       30, 1.000),
        ("Haiku airline\n(n=30)",        "pull − ic",   0.0, PROTO_COLOR["no_coord"],   30, 1.000),
        ("Sonnet retail\n(n=30)",        "nc − ic",    -3.3, PROTO_COLOR["intercept"],  30, 1.000),
        ("Sonnet retail\n(n=30)",        "pull − nc", -16.7, PROTO_COLOR["pull"],       30, 0.539),
        ("Sonnet retail\n(n=30)",        "pull − ic", -20.0, PROTO_COLOR["no_coord"],   30, 0.328),
    ]
    fig_noise_floor_by_model(noise_rows)

    print("Wrote:", OUT / "fig_combined.pdf")
    print("Wrote:", OUT / "fig_p2_validation.pdf")
    print("Wrote:", OUT / "fig_trial_split.pdf")
    print("Wrote:", OUT / "fig_headline.pdf")
    print("Wrote:", OUT / "fig_winloss.pdf")
    for k, v in data.items():
        print(f"  {k}: t0={v['trial_0']:.3f} t1={v['trial_1']:.3f} "
              f"pass1={v['pass_1']:.3f} pass2={v['pass_2']:.3f}")
    for k, v in pairs.items():
        print(f"  {k}: W/L/T={v['wins']}/{v['losses']}/{v['ties']} p={v['p']:.4f}")


if __name__ == "__main__":
    main()

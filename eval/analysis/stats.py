"""Statistical primitives for pairwise condition comparisons.

- paired_wilcoxon: non-parametric paired test (scipy)
- cliffs_delta: effect size for non-parametric data
- holm_bonferroni: multiple-comparison correction
- pairwise_comparison: convenience that bundles the three for one comparison
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _scipy_stats


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test.

    Returns (p_value, statistic). Tests H0: median of (a - b) == 0.
    Two-sided. Raises ValueError if lengths differ.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0, 0.0
    # All-zeros case: scipy raises, so handle manually.
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    if np.all(diffs == 0):
        return 1.0, 0.0
    result = _scipy_stats.wilcoxon(a, b, zero_method="wilcox",
                                    alternative="two-sided")
    return float(result.pvalue), float(result.statistic)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's δ effect size: #(a > b) - #(a < b) divided by len(a) * len(b).

    Ranges from -1 (all b > a) to +1 (all a > b). Zero means stochastic
    equality. Interpretation thresholds (Romano et al. 2006):
      |δ| < 0.147 negligible; <0.33 small; <0.474 medium; ≥0.474 large.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    greater = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (n_a * n_b)


def holm_bonferroni(pvalues: list[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for multiple comparisons.

    Returns list of bools (same order as input) indicating which
    hypotheses are rejected at family-wise error rate alpha.

    Algorithm: sort pvalues ascending; reject p_(i) if
    p_(i) <= alpha / (m - i + 1) for i = 1..m. Stop at first
    non-rejection. (Step-down sequential procedure.)
    """
    m = len(pvalues)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    out = [False] * m
    for rank, (orig_i, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            out[orig_i] = True
        else:
            # Holm stops on first non-rejection: all remaining stay False
            break
    return out


def pairwise_comparison(
    *, baseline_values: list[float], condition_values: list[float],
) -> dict[str, Any]:
    """Single pairwise comparison (condition vs baseline) on paired data.

    `baseline_values[i]` and `condition_values[i]` must be from the same
    query+seed combination (paired). Returns:
      {"n": N, "p_value": p (Wilcoxon two-sided),
       "cliffs_delta": δ (condition relative to baseline),
       "baseline_mean": m_b, "condition_mean": m_c}
    """
    if len(baseline_values) != len(condition_values):
        raise ValueError("baseline_values and condition_values must be paired")
    p, _stat = paired_wilcoxon(condition_values, baseline_values)
    delta = cliffs_delta(condition_values, baseline_values)
    return {
        "n": len(baseline_values),
        "p_value": p,
        "cliffs_delta": delta,
        "baseline_mean": float(np.mean(baseline_values)) if baseline_values else 0.0,
        "condition_mean": float(np.mean(condition_values)) if condition_values else 0.0,
    }

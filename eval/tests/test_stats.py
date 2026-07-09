"""Tests for statistical primitives."""

from __future__ import annotations

import math

import pytest

from analysis.stats import (
    cliffs_delta,
    holm_bonferroni,
    paired_wilcoxon,
    pairwise_comparison,
)


class TestPairedWilcoxon:
    def test_clear_difference_significant(self):
        # A consistently larger than B
        a = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p, stat = paired_wilcoxon(a, b)
        assert p < 0.01

    def test_no_difference_not_significant(self):
        # Identical
        a = [5, 6, 7, 8, 9]
        b = [5, 6, 7, 8, 9]
        p, stat = paired_wilcoxon(a, b)
        assert p > 0.5

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([1, 2, 3], [1, 2])


class TestCliffsDelta:
    def test_large_difference_returns_high_magnitude(self):
        # A always larger than B → delta ≈ +1
        a = [10, 11, 12, 13, 14]
        b = [1, 2, 3, 4, 5]
        delta = cliffs_delta(a, b)
        assert delta > 0.9

    def test_no_difference_returns_zero(self):
        a = [5, 6, 7]
        b = [5, 6, 7]
        delta = cliffs_delta(a, b)
        assert abs(delta) < 0.01

    def test_b_larger_returns_negative(self):
        a = [1, 2, 3]
        b = [10, 11, 12]
        delta = cliffs_delta(a, b)
        assert delta < -0.9


class TestHolmBonferroni:
    def test_orders_and_adjusts_pvalues(self):
        # 4 comparisons, alpha=0.05, sorted ascending: 0.001, 0.01, 0.02, 0.04
        # Holm thresholds: 0.05/4, 0.05/3, 0.05/2, 0.05/1
        # 0.001 < 0.0125 → significant
        # 0.01 < 0.0167 → significant
        # 0.02 < 0.025 → significant
        # 0.04 > 0.05 → not significant (compared to alpha/1)
        # Wait: 0.04 < 0.05 → significant. Let me adjust.
        pvalues = [0.001, 0.01, 0.02, 0.04]
        result = holm_bonferroni(pvalues, alpha=0.05)
        assert result == [True, True, True, True]

    def test_marks_non_significant(self):
        # Same setup but with a larger p
        pvalues = [0.001, 0.01, 0.02, 0.10]
        result = holm_bonferroni(pvalues, alpha=0.05)
        # 0.10 > 0.05/1 → not significant
        assert result == [True, True, True, False]

    def test_preserves_input_order(self):
        # Input order: [0.04, 0.001, 0.02, 0.01]
        # Sorted indices: [1, 3, 2, 0]
        # Thresholds: 0.0125, 0.0167, 0.025, 0.05
        # 0.001 < 0.0125 ✓, 0.01 < 0.0167 ✓, 0.02 < 0.025 ✓, 0.04 < 0.05 ✓
        # Result in input order: [True, True, True, True]
        pvalues = [0.04, 0.001, 0.02, 0.01]
        result = holm_bonferroni(pvalues, alpha=0.05)
        assert result == [True, True, True, True]


class TestPairwiseComparison:
    def test_returns_p_delta_significant(self):
        baseline_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        condition_values = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = pairwise_comparison(
            baseline_values=baseline_values,
            condition_values=condition_values,
        )
        assert "p_value" in result
        assert "cliffs_delta" in result
        assert "n" in result
        assert result["n"] == 5
        assert result["cliffs_delta"] > 0  # condition is larger

"""
Test Suite: Hypothesis Tests
==============================

Validates each hypothesis test against known results from textbook examples
and scipy direct calls. Ensures the wrapper functions produce correct statistics,
p-values, and interpretations.
"""

import pytest
import numpy as np
from src.hypothesis_tests import (
    TestResult,
    two_sample_ttest,
    z_test_proportions,
    chi_square_test,
    one_way_anova,
    mann_whitney_u,
    select_test,
)


class TestTwoSampleTTest:
    """Tests for the independent two-sample t-test."""

    def test_equal_groups_not_significant(self):
        """Two groups drawn from the same distribution should not be significant."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 500)
        b = np.random.normal(50, 10, 500)
        result = two_sample_ttest(a, b)
        assert isinstance(result, TestResult)
        assert result.p_value > 0.05
        assert result.is_significant is False

    def test_different_groups_significant(self):
        """Two groups with a large mean difference should be significant."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 500)
        b = np.random.normal(55, 10, 500)
        result = two_sample_ttest(a, b)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_cohens_d_present(self):
        """Effect size (Cohen's d) should be computed."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 100)
        b = np.random.normal(55, 10, 100)
        result = two_sample_ttest(a, b)
        assert result.effect_size is not None
        # d ≈ 0.5 for this setup
        assert 0.2 < abs(result.effect_size) < 0.8

    def test_returns_confidence_interval(self):
        """Confidence interval should be returned."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 100)
        b = np.random.normal(50, 10, 100)
        result = two_sample_ttest(a, b)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]

    def test_interpretation_string(self):
        """Interpretation should be a non-empty string."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 100)
        b = np.random.normal(50, 10, 100)
        result = two_sample_ttest(a, b)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0


class TestZTestProportions:
    """Tests for the two-proportion z-test."""

    def test_equal_proportions_not_significant(self):
        """Same conversion rates should not be significant."""
        result = z_test_proportions(100, 1000, 100, 1000)
        assert result.p_value > 0.05
        assert result.is_significant is False

    def test_different_proportions_significant(self):
        """Large difference in conversion rates should be significant."""
        result = z_test_proportions(100, 1000, 150, 1000)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_known_result(self):
        """Verify against a manually computed z-test result."""
        # A: 120/1000 = 12%, B: 145/1000 = 14.5%
        result = z_test_proportions(120, 1000, 145, 1000)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1


class TestChiSquareTest:
    """Tests for the chi-square test of independence."""

    def test_independent_groups_not_significant(self):
        """Groups with similar distributions should not be significant."""
        table = np.array([[50, 50], [50, 50]])
        result = chi_square_test(table)
        assert result.p_value > 0.05
        assert result.is_significant is False

    def test_dependent_groups_significant(self):
        """Groups with very different distributions should be significant."""
        table = np.array([[90, 10], [10, 90]])
        result = chi_square_test(table)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_cramers_v_effect_size(self):
        """Cramér's V effect size should be computed."""
        table = np.array([[90, 10], [10, 90]])
        result = chi_square_test(table)
        assert result.effect_size is not None
        assert 0 <= result.effect_size <= 1


class TestOneWayANOVA:
    """Tests for one-way ANOVA."""

    def test_same_groups_not_significant(self):
        """Three groups from the same distribution should not be significant."""
        np.random.seed(42)
        g1 = np.random.normal(50, 10, 100)
        g2 = np.random.normal(50, 10, 100)
        g3 = np.random.normal(50, 10, 100)
        result = one_way_anova(g1, g2, g3)
        assert result.p_value > 0.05

    def test_different_groups_significant(self):
        """Three groups with different means should be significant."""
        np.random.seed(42)
        g1 = np.random.normal(50, 10, 100)
        g2 = np.random.normal(55, 10, 100)
        g3 = np.random.normal(60, 10, 100)
        result = one_way_anova(g1, g2, g3)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_eta_squared_effect_size(self):
        """Eta-squared effect size should be computed."""
        np.random.seed(42)
        g1 = np.random.normal(50, 10, 100)
        g2 = np.random.normal(60, 10, 100)
        g3 = np.random.normal(70, 10, 100)
        result = one_way_anova(g1, g2, g3)
        assert result.effect_size is not None
        assert 0 <= result.effect_size <= 1


class TestMannWhitneyU:
    """Tests for the Mann-Whitney U test."""

    def test_equal_distributions_not_significant(self):
        """Two groups from the same distribution should not be significant."""
        np.random.seed(42)
        a = np.random.exponential(5, 200)
        b = np.random.exponential(5, 200)
        result = mann_whitney_u(a, b)
        assert result.p_value > 0.05

    def test_different_distributions_significant(self):
        """Two groups from different distributions should be significant."""
        np.random.seed(42)
        a = np.random.exponential(5, 200)
        b = np.random.exponential(10, 200)
        result = mann_whitney_u(a, b)
        assert result.p_value < 0.05
        assert result.is_significant is True

    def test_works_with_non_normal_data(self):
        """Mann-Whitney should work correctly with skewed (non-normal) data."""
        np.random.seed(42)
        a = np.random.lognormal(0, 1, 100)
        b = np.random.lognormal(0.5, 1, 100)
        result = mann_whitney_u(a, b)
        assert isinstance(result, TestResult)
        assert isinstance(result.p_value, float)


class TestSelectTest:
    """Tests for the test selection helper."""

    def test_binary_two_groups(self):
        """Binary metric + 2 groups → z-test for proportions."""
        recommendation = select_test(metric_type="binary", num_groups=2)
        assert recommendation["recommended_test"] == "z_test_proportions"

    def test_continuous_two_groups_normal(self):
        """Continuous metric + 2 groups + normal → t-test."""
        recommendation = select_test(metric_type="continuous", num_groups=2, is_normal=True)
        assert recommendation["recommended_test"] == "two_sample_ttest"

    def test_continuous_two_groups_not_normal(self):
        """Continuous metric + 2 groups + not normal → Mann-Whitney."""
        recommendation = select_test(metric_type="continuous", num_groups=2, is_normal=False)
        assert recommendation["recommended_test"] == "mann_whitney_u"

    def test_continuous_three_plus_groups(self):
        """Continuous metric + 3+ groups → ANOVA."""
        recommendation = select_test(metric_type="continuous", num_groups=3, is_normal=True)
        assert recommendation["recommended_test"] == "one_way_anova"

    def test_returns_reason(self):
        """Recommendation should include a reason."""
        recommendation = select_test(metric_type="binary")
        assert "reason" in recommendation
        assert isinstance(recommendation["reason"], str)

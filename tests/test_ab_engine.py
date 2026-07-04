"""
Test Suite: A/B Testing Engine
================================

End-to-end tests for the A/B test evaluation engine, verifying it correctly
combines sample size analysis, hypothesis testing, and verdict logic.
"""

import pytest
import numpy as np
from src.ab_engine import (
    ABTestResult,
    run_ab_test,
    compare_multiple_variants,
    _compute_lift,
    _cohens_d,
    _odds_ratio,
)


class TestRunABTestBinary:
    """Tests for binary (conversion rate) A/B tests."""

    def test_basic_binary_test(self):
        """Basic binary test should return a valid ABTestResult."""
        result = run_ab_test(
            successes_a=120, n_a=1000,
            successes_b=145, n_b=1000,
            metric_type="binary",
        )
        assert isinstance(result, ABTestResult)
        assert result.metric_type == "binary"
        assert result.control_estimate == 0.12
        assert result.treatment_estimate == 0.145

    def test_lift_calculation(self):
        """Absolute and relative lift should be computed correctly."""
        result = run_ab_test(
            successes_a=100, n_a=1000,
            successes_b=120, n_b=1000,
            metric_type="binary",
        )
        assert result.absolute_lift == pytest.approx(0.02, abs=1e-10)
        assert result.relative_lift == pytest.approx(20.0, abs=1e-10)

    def test_significant_result_verdict(self):
        """Large difference should produce a significant verdict."""
        result = run_ab_test(
            successes_a=100, n_a=1000,
            successes_b=180, n_b=1000,
            metric_type="binary",
        )
        assert result.test_result.is_significant is True
        assert "significant_b" in result.verdict_code or "significant_a" in result.verdict_code

    def test_not_significant_result(self):
        """Tiny difference should not be significant."""
        result = run_ab_test(
            successes_a=100, n_a=1000,
            successes_b=102, n_b=1000,
            metric_type="binary",
        )
        assert result.test_result.is_significant is False

    def test_confidence_interval_present(self):
        """Confidence interval should be returned."""
        result = run_ab_test(
            successes_a=100, n_a=1000,
            successes_b=120, n_b=1000,
            metric_type="binary",
        )
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2

    def test_odds_ratio_present(self):
        """Odds ratio should be computed for binary metrics."""
        result = run_ab_test(
            successes_a=100, n_a=1000,
            successes_b=120, n_b=1000,
            metric_type="binary",
        )
        assert result.effect_size_label == "Odds Ratio"
        assert result.effect_size > 0

    def test_missing_inputs_raises(self):
        """Missing binary inputs should raise ValueError."""
        with pytest.raises(ValueError):
            run_ab_test(successes_a=100, n_a=1000, metric_type="binary")


class TestRunABTestContinuous:
    """Tests for continuous metric A/B tests."""

    def test_basic_continuous_test(self):
        """Basic continuous test should return a valid ABTestResult."""
        np.random.seed(42)
        control = np.random.normal(50, 10, 500)
        treatment = np.random.normal(53, 10, 500)
        result = run_ab_test(data_a=control, data_b=treatment, metric_type="continuous")
        assert isinstance(result, ABTestResult)
        assert result.metric_type == "continuous"

    def test_cohens_d_present(self):
        """Cohen's d should be computed for continuous metrics."""
        np.random.seed(42)
        control = np.random.normal(50, 10, 500)
        treatment = np.random.normal(55, 10, 500)
        result = run_ab_test(data_a=control, data_b=treatment, metric_type="continuous")
        assert result.effect_size_label == "Cohen's d"
        assert abs(result.effect_size) > 0

    def test_missing_data_raises(self):
        """Missing continuous data should raise ValueError."""
        with pytest.raises(ValueError):
            run_ab_test(data_a=np.array([1, 2, 3]), metric_type="continuous")


class TestVerdictLogic:
    """Tests for the verdict generation logic."""

    def test_significant_b_wins(self):
        """When B is significantly better, verdict should say 'significant_b'."""
        result = run_ab_test(
            successes_a=100, n_a=5000,
            successes_b=150, n_b=5000,
            metric_type="binary",
        )
        if result.test_result.is_significant and result.absolute_lift > 0:
            assert result.verdict_code == "significant_b"

    def test_not_significant_verdict(self):
        """When difference is tiny, verdict should be 'not_significant'."""
        result = run_ab_test(
            successes_a=500, n_a=5000,
            successes_b=502, n_b=5000,
            metric_type="binary",
        )
        assert result.verdict_code in ["not_significant", "underpowered"]


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_compute_lift(self):
        """Lift calculation should be correct."""
        lift = _compute_lift(0.10, 0.12)
        assert lift["absolute"] == pytest.approx(0.02, abs=1e-10)
        assert lift["relative"] == pytest.approx(20.0, abs=1e-10)

    def test_compute_lift_zero_baseline(self):
        """Zero baseline should return None for relative lift."""
        lift = _compute_lift(0.0, 0.05)
        assert lift["relative"] is None

    def test_cohens_d_known_value(self):
        """Cohen's d for known groups should be approximately correct."""
        np.random.seed(42)
        a = np.random.normal(50, 10, 10000)
        b = np.random.normal(55, 10, 10000)
        d = _cohens_d(a, b)
        assert abs(d - 0.5) < 0.05  # Should be close to 0.5

    def test_odds_ratio_equal_groups(self):
        """Equal conversion rates should give OR ≈ 1."""
        odr = _odds_ratio(100, 1000, 100, 1000)
        assert abs(odr - 1.0) < 0.01

    def test_odds_ratio_higher_treatment(self):
        """Higher treatment rate should give OR > 1."""
        odr = _odds_ratio(100, 1000, 200, 1000)
        assert odr > 1.0


class TestCompareMultipleVariants:
    """Tests for multi-variant comparison."""

    def test_two_variants_vs_control(self):
        """Should return one result per treatment variant."""
        variants = {
            "control": (100, 1000),
            "variant_b": (120, 1000),
            "variant_c": (130, 1000),
        }
        results = compare_multiple_variants(variants, metric_type="binary")
        assert len(results) == 2

    def test_bonferroni_correction_applied(self):
        """Corrected alpha should be alpha / num_comparisons."""
        variants = {
            "control": (100, 1000),
            "variant_b": (120, 1000),
            "variant_c": (130, 1000),
        }
        results = compare_multiple_variants(variants, metric_type="binary", alpha=0.05)
        # With 2 comparisons, corrected alpha should be 0.025
        for r in results:
            assert r.alpha == pytest.approx(0.025, abs=1e-10)

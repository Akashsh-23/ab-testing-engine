"""
Test Suite: Sample Size Calculators
====================================

Validates sample size calculations against statsmodels reference implementations
and known textbook values. Ensures our calculators are correct, not just functional.
"""

import pytest
import numpy as np
import math
from src.sample_size import (
    sample_size_proportions,
    sample_size_continuous,
    validate_sample_size,
)


class TestSampleSizeProportions:
    """Tests for the two-proportion z-test sample size calculator."""

    def test_basic_calculation(self):
        """Standard case: 10% baseline, 2% MDE, alpha=0.05, power=0.80."""
        n = sample_size_proportions(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0
        # Reference: ~3,623 per group (from statsmodels / online calculators)
        assert 3000 < n < 4500, f"Expected ~3,623 per group, got {n}"

    def test_larger_mde_needs_fewer_samples(self):
        """Larger MDE should require fewer observations."""
        n_small_mde = sample_size_proportions(baseline_rate=0.10, mde=0.01)
        n_large_mde = sample_size_proportions(baseline_rate=0.10, mde=0.05)
        assert n_large_mde < n_small_mde

    def test_higher_power_needs_more_samples(self):
        """Higher power should require more observations."""
        n_80 = sample_size_proportions(baseline_rate=0.10, mde=0.02, power=0.80)
        n_90 = sample_size_proportions(baseline_rate=0.10, mde=0.02, power=0.90)
        assert n_90 > n_80

    def test_lower_alpha_needs_more_samples(self):
        """Lower alpha (stricter) should require more observations."""
        n_05 = sample_size_proportions(baseline_rate=0.10, mde=0.02, alpha=0.05)
        n_01 = sample_size_proportions(baseline_rate=0.10, mde=0.02, alpha=0.01)
        assert n_01 > n_05

    def test_symmetric_baseline(self):
        """Baseline near 50% should require larger samples (max variance)."""
        n_10 = sample_size_proportions(baseline_rate=0.10, mde=0.02)
        n_50 = sample_size_proportions(baseline_rate=0.50, mde=0.02)
        assert n_50 > n_10

    def test_returns_integer(self):
        """Sample size must always be a whole number (rounded up)."""
        n = sample_size_proportions(baseline_rate=0.15, mde=0.03)
        assert isinstance(n, int)

    def test_cross_check_with_statsmodels(self):
        """Cross-validate against statsmodels power analysis."""
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        baseline = 0.10
        mde = 0.02
        effect = proportion_effectsize(baseline, baseline + mde)
        analysis = NormalIndPower()
        reference_n = math.ceil(
            analysis.solve_power(effect_size=effect, alpha=0.05, power=0.80, alternative="two-sided")
        )

        our_n = sample_size_proportions(baseline_rate=baseline, mde=mde, alpha=0.05, power=0.80)

        # Allow ±5% tolerance due to different approximation methods
        assert abs(our_n - reference_n) / reference_n < 0.05, (
            f"Our n={our_n} vs statsmodels n={reference_n}"
        )


class TestSampleSizeContinuous:
    """Tests for the continuous metrics (t-test) sample size calculator."""

    def test_basic_calculation(self):
        """Standard case: mean=50, std=10, MDE=3, alpha=0.05, power=0.80."""
        n = sample_size_continuous(baseline_mean=50, baseline_std=10, mde=3, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0
        # Cohen's d = 3/10 = 0.3 → ~175 per group
        assert 100 < n < 300, f"Expected ~175 per group, got {n}"

    def test_larger_mde_needs_fewer_samples(self):
        """Larger MDE should require fewer observations."""
        n_small = sample_size_continuous(baseline_mean=50, baseline_std=10, mde=1)
        n_large = sample_size_continuous(baseline_mean=50, baseline_std=10, mde=5)
        assert n_large < n_small

    def test_higher_variance_needs_more_samples(self):
        """Higher variance should require more observations."""
        n_low_var = sample_size_continuous(baseline_mean=50, baseline_std=5, mde=3)
        n_high_var = sample_size_continuous(baseline_mean=50, baseline_std=20, mde=3)
        assert n_high_var > n_low_var

    def test_cross_check_with_statsmodels(self):
        """Cross-validate against statsmodels TTestIndPower."""
        from statsmodels.stats.power import TTestIndPower

        baseline_std = 10.0
        mde = 3.0
        cohens_d = mde / baseline_std
        analysis = TTestIndPower()
        reference_n = math.ceil(
            analysis.solve_power(effect_size=cohens_d, alpha=0.05, power=0.80, alternative="two-sided")
        )

        our_n = sample_size_continuous(baseline_mean=50, baseline_std=baseline_std, mde=mde)

        assert abs(our_n - reference_n) / reference_n < 0.05, (
            f"Our n={our_n} vs statsmodels n={reference_n}"
        )


class TestValidateSampleSize:
    """Tests for the sample size validation helper."""

    def test_sufficient_sample(self):
        """When actual >= required, should be sufficient."""
        result = validate_sample_size(actual_n=2000, required_n=1500)
        assert result["is_sufficient"] is True
        assert result["shortfall"] == 0

    def test_insufficient_sample(self):
        """When actual < required, should be insufficient."""
        result = validate_sample_size(actual_n=500, required_n=1500)
        assert result["is_sufficient"] is False
        assert result["shortfall"] == 1000

    def test_exact_match(self):
        """When actual == required, should be sufficient."""
        result = validate_sample_size(actual_n=1500, required_n=1500)
        assert result["is_sufficient"] is True

    def test_returns_dict_with_required_keys(self):
        """Result must contain all expected keys."""
        result = validate_sample_size(actual_n=100, required_n=200)
        assert "is_sufficient" in result
        assert "actual" in result
        assert "required" in result
        assert "shortfall" in result
        assert "message" in result

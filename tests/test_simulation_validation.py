"""
Test Suite: Simulation Validation
===================================

The "unit test for statistics" — validates that the statistical engine
correctly detects true effects at the expected rate (power). This is the
differentiating piece of the project.

Tests:
    1. At the recommended sample size, observed power ≈ target power (~80%)
    2. Under the null hypothesis (no effect), Type I error rate ≈ alpha (~5%)
    3. Power increases monotonically with sample size
    4. Larger effect sizes are easier to detect
"""

import pytest
import numpy as np
from src.simulator import SyntheticExperiment, run_null_validation


class TestPowerValidation:
    """Validate that observed power matches target power at recommended sample size."""

    @pytest.mark.slow
    def test_binary_power_at_recommended_n(self):
        """
        At the sample size recommended by the Phase 1 calculator,
        the test should correctly detect the true effect ~80% of the time.

        This is the core validation: proving the engine is *correct*.
        """
        from src.sample_size import sample_size_proportions

        baseline = 0.10
        effect = 0.03
        alpha = 0.05
        target_power = 0.80

        # Get recommended sample size
        recommended_n = sample_size_proportions(
            baseline_rate=baseline, mde=effect, alpha=alpha, power=target_power
        )

        # Run simulation at recommended N
        sim = SyntheticExperiment(
            baseline_rate=baseline,
            effect_size=effect,
            sample_size=recommended_n,
            metric_type="binary",
            alpha=alpha,
            power=target_power,
            random_seed=42,
        )

        result = sim.run_power_validation(n_simulations=500)
        observed_power = result["observed_power"]

        # Should be within ±8 percentage points of target (Monte Carlo noise)
        assert abs(observed_power - target_power) < 0.08, (
            f"Observed power {observed_power:.3f} too far from target {target_power} "
            f"at n={recommended_n}"
        )

    @pytest.mark.slow
    def test_continuous_power_at_recommended_n(self):
        """Same validation for continuous metrics."""
        from src.sample_size import sample_size_continuous

        baseline_mean = 50.0
        baseline_std = 10.0
        effect = 3.0
        alpha = 0.05
        target_power = 0.80

        recommended_n = sample_size_continuous(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde=effect,
            alpha=alpha,
            power=target_power,
        )

        sim = SyntheticExperiment(
            baseline_rate=baseline_mean,
            effect_size=effect,
            sample_size=recommended_n,
            metric_type="continuous",
            baseline_std=baseline_std,
            alpha=alpha,
            power=target_power,
            random_seed=42,
        )

        result = sim.run_power_validation(n_simulations=500)
        observed_power = result["observed_power"]

        assert abs(observed_power - target_power) < 0.08, (
            f"Observed power {observed_power:.3f} too far from target {target_power} "
            f"at n={recommended_n}"
        )


class TestTypeIErrorRate:
    """Validate that false positive rate ≈ alpha under the null hypothesis."""

    def test_binary_null_calibration(self):
        """Under H0 (no effect), significance rate should ≈ alpha."""
        result = run_null_validation(
            metric_type="binary",
            baseline_rate=0.10,
            sample_size=1000,
            alpha=0.05,
            n_simulations=500,
            random_seed=42,
        )
        assert result["is_calibrated"], (
            f"FPR={result['false_positive_rate']:.3f} vs expected={result['expected_rate']:.3f}"
        )

    def test_continuous_null_calibration(self):
        """Under H0 (no effect), significance rate should ≈ alpha for continuous."""
        sim = SyntheticExperiment(
            baseline_rate=50.0,
            effect_size=0.0,
            sample_size=500,
            metric_type="continuous",
            baseline_std=10.0,
            alpha=0.05,
            random_seed=42,
        )
        result = sim.run_power_validation(n_simulations=500)
        fpr = result["significance_rate"]
        # FPR should be approximately alpha (within ±2.5 percentage points)
        assert abs(fpr - 0.05) < 0.025, f"FPR={fpr:.3f}, expected ≈ 0.05"


class TestPowerMonotonicity:
    """Validate that power increases with sample size."""

    def test_power_increases_with_sample_size(self):
        """Larger samples should yield higher power."""
        powers = []
        for n in [100, 500, 2000]:
            sim = SyntheticExperiment(
                baseline_rate=0.10,
                effect_size=0.02,
                sample_size=n,
                metric_type="binary",
                random_seed=42,
            )
            result = sim.run_power_validation(n_simulations=300)
            powers.append(result["observed_power"])

        # Power should be monotonically non-decreasing
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 0.05, (
                f"Power decreased: {powers[i-1]:.3f} -> {powers[i]:.3f} "
                f"as sample size increased"
            )


class TestEffectSizeImpact:
    """Validate that larger effects are easier to detect."""

    def test_larger_effect_higher_power(self):
        """Larger true effect size should produce higher observed power."""
        powers = []
        for effect in [0.01, 0.03, 0.05]:
            sim = SyntheticExperiment(
                baseline_rate=0.10,
                effect_size=effect,
                sample_size=500,
                metric_type="binary",
                random_seed=42,
            )
            result = sim.run_power_validation(n_simulations=300)
            powers.append(result["observed_power"])

        # Power should increase with effect size
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 0.05, (
                f"Power did not increase with effect size: {powers}"
            )


class TestSyntheticExperiment:
    """Basic tests for the experiment generator."""

    def test_generate_binary_data(self):
        """Binary data should contain only 0s and 1s."""
        sim = SyntheticExperiment(
            baseline_rate=0.15, effect_size=0.03, sample_size=100,
            metric_type="binary", random_seed=42,
        )
        control, treatment = sim.generate_data()
        assert set(np.unique(control)).issubset({0, 1})
        assert set(np.unique(treatment)).issubset({0, 1})
        assert len(control) == 100
        assert len(treatment) == 100

    def test_generate_continuous_data(self):
        """Continuous data should have the right shape."""
        sim = SyntheticExperiment(
            baseline_rate=50.0, effect_size=5.0, sample_size=200,
            metric_type="continuous", baseline_std=10.0, random_seed=42,
        )
        control, treatment = sim.generate_data()
        assert len(control) == 200
        assert len(treatment) == 200

    def test_generate_dataframe(self):
        """DataFrame should have 'group' and 'value' columns."""
        sim = SyntheticExperiment(
            baseline_rate=0.10, effect_size=0.02, sample_size=100,
            metric_type="binary", random_seed=42,
        )
        df = sim.generate_dataframe()
        assert "group" in df.columns
        assert "value" in df.columns
        assert len(df) == 200  # 100 control + 100 treatment

    def test_reproducibility(self):
        """Same seed should produce same data."""
        sim1 = SyntheticExperiment(
            baseline_rate=0.10, effect_size=0.02, sample_size=100,
            metric_type="binary", random_seed=123,
        )
        sim2 = SyntheticExperiment(
            baseline_rate=0.10, effect_size=0.02, sample_size=100,
            metric_type="binary", random_seed=123,
        )
        c1, t1 = sim1.generate_data()
        c2, t2 = sim2.generate_data()
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(t1, t2)

    def test_power_curve_returns_dataframe(self):
        """Power curve should return a DataFrame with expected columns."""
        sim = SyntheticExperiment(
            baseline_rate=0.10, effect_size=0.03, sample_size=500,
            metric_type="binary", random_seed=42,
        )
        df = sim.run_power_curve(
            sample_sizes=[100, 500, 1000],
            n_simulations=50,  # Low for speed
        )
        assert "sample_size" in df.columns
        assert "observed_power" in df.columns
        assert len(df) == 3

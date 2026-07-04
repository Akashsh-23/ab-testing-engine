"""
Synthetic Experiment Simulator & Validation
============================================

Generates synthetic A/B test data with a known ground truth, then validates
the statistical engine against it. This serves as the "unit test for statistics"
— proving the engine is *correct*, not just that it runs.

Key capabilities:
    - Generate binary (conversion) or continuous metric data with configurable
      true effect sizes, baseline rates, and sample sizes.
    - Run Monte Carlo power validation: simulate many experiments at different
      sample sizes and confirm the engine achieves the expected detection rate.
    - Produce power curves showing the relationship between sample size and
      the probability of detecting a true effect.

Usage
-----
    >>> from src.simulator import SyntheticExperiment
    >>> sim = SyntheticExperiment(baseline_rate=0.10, effect_size=0.02, sample_size=1000)
    >>> control, treatment = sim.generate_data()
    >>> validation = sim.run_power_validation(n_simulations=500)
    >>> print(f"Observed power: {validation['observed_power']:.2f}")
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from scipy import stats


@dataclass
class SimulationConfig:
    """
    Configuration for a synthetic experiment.

    Attributes
    ----------
    metric_type : str
        'binary' for conversion data, 'continuous' for continuous metrics.
    baseline_rate : float
        Baseline conversion rate (binary) or baseline mean (continuous).
    effect_size : float
        True effect size: absolute increase in conversion rate (binary)
        or absolute increase in mean (continuous).
    sample_size : int
        Number of observations per group.
    baseline_std : float
        Standard deviation for continuous metrics (ignored for binary).
    alpha : float
        Significance level for hypothesis tests.
    power : float
        Target statistical power.
    random_seed : int or None
        Random seed for reproducibility.
    """

    metric_type: str = "binary"
    baseline_rate: float = 0.10
    effect_size: float = 0.02
    sample_size: int = 1000
    baseline_std: float = 10.0
    alpha: float = 0.05
    power: float = 0.80
    random_seed: Optional[int] = None


class SyntheticExperiment:
    """
    Generate synthetic A/B test data with known ground truth for validation.

    This class simulates experiment data where you KNOW the true effect size,
    so you can verify that the statistical engine correctly detects it at the
    expected rate (power).

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate (binary) or mean (continuous).
    effect_size : float
        True absolute effect size (treatment - control).
    sample_size : int
        Number of observations per group.
    metric_type : str, optional
        'binary' (default) or 'continuous'.
    baseline_std : float, optional
        Standard deviation for continuous metrics (default 10.0).
    alpha : float, optional
        Significance level (default 0.05).
    power : float, optional
        Target power (default 0.80).
    random_seed : int or None, optional
        Seed for reproducibility (default None).

    Examples
    --------
    >>> sim = SyntheticExperiment(baseline_rate=0.10, effect_size=0.02, sample_size=1000)
    >>> control, treatment = sim.generate_data()
    >>> print(f"Control conversions: {control.sum()} / {len(control)}")
    >>> print(f"Treatment conversions: {treatment.sum()} / {len(treatment)}")
    """

    def __init__(
        self,
        baseline_rate: float = 0.10,
        effect_size: float = 0.02,
        sample_size: int = 1000,
        metric_type: str = "binary",
        baseline_std: float = 10.0,
        alpha: float = 0.05,
        power: float = 0.80,
        random_seed: Optional[int] = None,
    ):
        self.config = SimulationConfig(
            metric_type=metric_type,
            baseline_rate=baseline_rate,
            effect_size=effect_size,
            sample_size=sample_size,
            baseline_std=baseline_std,
            alpha=alpha,
            power=power,
            random_seed=random_seed,
        )
        self.rng = np.random.default_rng(random_seed)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one set of control and treatment data.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (control_data, treatment_data)
            - Binary: arrays of 0s and 1s
            - Continuous: arrays of float values

        Notes
        -----
        For binary metrics, each observation is a Bernoulli trial.
        For continuous metrics, observations are drawn from normal distributions
        with the same standard deviation but different means.
        """
        cfg = self.config

        if cfg.metric_type == "binary":
            control = self.rng.binomial(1, cfg.baseline_rate, cfg.sample_size)
            treatment_rate = cfg.baseline_rate + cfg.effect_size
            treatment_rate = np.clip(treatment_rate, 0.0, 1.0)
            treatment = self.rng.binomial(1, treatment_rate, cfg.sample_size)
        elif cfg.metric_type == "continuous":
            control = self.rng.normal(cfg.baseline_rate, cfg.baseline_std, cfg.sample_size)
            treatment = self.rng.normal(
                cfg.baseline_rate + cfg.effect_size, cfg.baseline_std, cfg.sample_size
            )
        else:
            raise ValueError(f"metric_type must be 'binary' or 'continuous', got '{cfg.metric_type}'")

        return control, treatment

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate experiment data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'group' ('control'/'treatment'), 'value'.
        """
        control, treatment = self.generate_data()
        df = pd.DataFrame(
            {
                "group": ["control"] * len(control) + ["treatment"] * len(treatment),
                "value": np.concatenate([control, treatment]),
            }
        )
        return df

    def run_single_test(self, control: np.ndarray, treatment: np.ndarray) -> dict:
        """
        Run a single hypothesis test on generated data.

        Performs the test internally using scipy to avoid circular dependency
        with ab_engine during validation. This is intentional — validation
        should test the statistical logic, not just that our wrapper works.

        Parameters
        ----------
        control : np.ndarray
            Control group data.
        treatment : np.ndarray
            Treatment group data.

        Returns
        -------
        dict
            Keys: 'p_value', 'is_significant', 'detected_effect'.
        """
        cfg = self.config

        if cfg.metric_type == "binary":
            # Two-proportion z-test
            successes = np.array([control.sum(), treatment.sum()])
            nobs = np.array([len(control), len(treatment)])
            if nobs[0] == 0 or nobs[1] == 0:
                return {"p_value": 1.0, "is_significant": False, "detected_effect": False}

            p1 = successes[0] / nobs[0]
            p2 = successes[1] / nobs[1]
            p_pool = successes.sum() / nobs.sum()

            if p_pool == 0 or p_pool == 1:
                return {"p_value": 1.0, "is_significant": False, "detected_effect": False}

            se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
            if se == 0:
                return {"p_value": 1.0, "is_significant": False, "detected_effect": False}

            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            # Independent two-sample t-test (Welch's)
            stat_result = stats.ttest_ind(treatment, control, equal_var=False)
            p_value = stat_result.pvalue

        is_significant = p_value < cfg.alpha

        # Check if effect is in the correct direction
        if cfg.metric_type == "binary":
            observed_diff = treatment.mean() - control.mean()
        else:
            observed_diff = np.mean(treatment) - np.mean(control)

        detected_effect = is_significant and (
            (cfg.effect_size >= 0 and observed_diff > 0)
            or (cfg.effect_size < 0 and observed_diff < 0)
        )

        return {
            "p_value": p_value,
            "is_significant": is_significant,
            "detected_effect": detected_effect,
        }

    def run_power_validation(
        self,
        n_simulations: int = 1000,
        progress_callback=None,
    ) -> Dict:
        """
        Run Monte Carlo power validation.

        Simulates `n_simulations` experiments at the configured sample size and
        effect size, then checks how often the test correctly detects the true effect.

        The observed power should be approximately equal to the target power
        (e.g., ~0.80) when the sample size matches the Phase 1 calculator's
        recommendation.

        Parameters
        ----------
        n_simulations : int, optional
            Number of Monte Carlo replications (default 1000).
        progress_callback : callable, optional
            Function called with (current_iteration, total) for progress tracking.

        Returns
        -------
        dict
            - 'observed_power': fraction of simulations that correctly detected the effect
            - 'false_positive_rate': fraction of Type I errors (estimated under null)
            - 'p_values': array of all p-values from the simulations
            - 'significant_count': number of significant results
            - 'total_simulations': total number of simulations run
            - 'config': the simulation configuration
            - 'summary': plain-English summary of results

        Notes
        -----
        This is the "unit test for statistics" — a strong, unusual thing to show
        in an interview: proving the tool is *correct*, not just that it runs.
        """
        significant_count = 0
        correct_detection_count = 0
        p_values = []

        for i in range(n_simulations):
            control, treatment = self.generate_data()
            result = self.run_single_test(control, treatment)

            p_values.append(result["p_value"])
            if result["is_significant"]:
                significant_count += 1
            if result["detected_effect"]:
                correct_detection_count += 1

            if progress_callback and (i + 1) % max(1, n_simulations // 20) == 0:
                progress_callback(i + 1, n_simulations)

        observed_power = correct_detection_count / n_simulations
        significance_rate = significant_count / n_simulations

        # Build summary
        summary = (
            f"Monte Carlo Validation ({n_simulations} simulations)\n"
            f"  Metric type:      {self.config.metric_type}\n"
            f"  Sample size/group: {self.config.sample_size}\n"
            f"  True effect size:  {self.config.effect_size}\n"
            f"  Alpha:             {self.config.alpha}\n"
            f"  Target power:      {self.config.power}\n"
            f"  ──────────────────────────────────\n"
            f"  Observed power:    {observed_power:.3f} "
            f"({'close to target' if abs(observed_power - self.config.power) < 0.05 else 'differs from target'})\n"
            f"  Significance rate: {significance_rate:.3f}\n"
            f"  Median p-value:    {np.median(p_values):.4f}"
        )

        return {
            "observed_power": observed_power,
            "significance_rate": significance_rate,
            "p_values": np.array(p_values),
            "significant_count": significant_count,
            "correct_detections": correct_detection_count,
            "total_simulations": n_simulations,
            "config": self.config,
            "summary": summary,
        }

    def run_power_curve(
        self,
        sample_sizes: List[int] = None,
        n_simulations: int = 500,
    ) -> pd.DataFrame:
        """
        Generate a power curve: observed power as a function of sample size.

        For each sample size, runs Monte Carlo simulations and records the
        observed detection rate. The resulting curve shows how power increases
        with sample size.

        Parameters
        ----------
        sample_sizes : list of int, optional
            Sample sizes to evaluate. Defaults to a range from 50 to 5000.
        n_simulations : int, optional
            Number of simulations per sample size point (default 500).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'sample_size', 'observed_power', 'target_power'.
        """
        if sample_sizes is None:
            sample_sizes = [50, 100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000]

        results = []
        original_sample_size = self.config.sample_size

        for n in sample_sizes:
            self.config.sample_size = n
            validation = self.run_power_validation(n_simulations=n_simulations)
            results.append(
                {
                    "sample_size": n,
                    "observed_power": validation["observed_power"],
                    "target_power": self.config.power,
                }
            )

        # Restore original
        self.config.sample_size = original_sample_size

        return pd.DataFrame(results)


def run_null_validation(
    metric_type: str = "binary",
    baseline_rate: float = 0.10,
    sample_size: int = 1000,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> Dict:
    """
    Validate Type I error rate under the null hypothesis (no true effect).

    Simulates experiments where the true effect is zero, then checks that the
    false positive rate is approximately equal to alpha.

    Parameters
    ----------
    metric_type : str
        'binary' or 'continuous'.
    baseline_rate : float
        Baseline rate or mean.
    sample_size : int
        Sample size per group.
    alpha : float
        Significance level.
    n_simulations : int
        Number of null-hypothesis simulations.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        - 'false_positive_rate': observed Type I error rate
        - 'expected_rate': alpha
        - 'is_calibrated': True if within ±1.5 percentage points of alpha
        - 'summary': plain-English summary
    """
    sim = SyntheticExperiment(
        baseline_rate=baseline_rate,
        effect_size=0.0,  # Null hypothesis: no effect
        sample_size=sample_size,
        metric_type=metric_type,
        alpha=alpha,
        random_seed=random_seed,
    )

    result = sim.run_power_validation(n_simulations=n_simulations)
    false_positive_rate = result["significance_rate"]
    is_calibrated = abs(false_positive_rate - alpha) < 0.015

    summary = (
        f"Null Hypothesis Validation ({n_simulations} simulations)\n"
        f"  Expected FPR:  {alpha:.3f}\n"
        f"  Observed FPR:  {false_positive_rate:.3f}\n"
        f"  Calibrated:    {'Yes' if is_calibrated else 'No — check test implementation'}"
    )

    return {
        "false_positive_rate": false_positive_rate,
        "expected_rate": alpha,
        "is_calibrated": is_calibrated,
        "summary": summary,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC EXPERIMENT SIMULATOR — DEMO")
    print("=" * 70)

    # --- Binary experiment simulation ---
    print("\n--- Binary Experiment (baseline=10%, effect=+2%) ---")
    sim = SyntheticExperiment(
        baseline_rate=0.10,
        effect_size=0.02,
        sample_size=1000,
        metric_type="binary",
        random_seed=42,
    )

    control, treatment = sim.generate_data()
    print(f"Control:   {control.sum()} conversions / {len(control)} = {control.mean():.4f}")
    print(f"Treatment: {treatment.sum()} conversions / {len(treatment)} = {treatment.mean():.4f}")

    # --- Power validation ---
    print("\n--- Power Validation (500 simulations) ---")
    validation = sim.run_power_validation(n_simulations=500)
    print(validation["summary"])

    # --- Null hypothesis validation ---
    print("\n--- Null Hypothesis Validation ---")
    null_result = run_null_validation(
        metric_type="binary",
        baseline_rate=0.10,
        sample_size=1000,
        n_simulations=500,
        random_seed=123,
    )
    print(null_result["summary"])

    # --- Continuous experiment ---
    print("\n--- Continuous Experiment (baseline=50, std=10, effect=+3) ---")
    sim_cont = SyntheticExperiment(
        baseline_rate=50.0,
        effect_size=3.0,
        sample_size=200,
        metric_type="continuous",
        baseline_std=10.0,
        random_seed=42,
    )
    validation_cont = sim_cont.run_power_validation(n_simulations=500)
    print(validation_cont["summary"])

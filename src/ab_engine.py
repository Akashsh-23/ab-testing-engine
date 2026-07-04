"""
A/B Testing Engine
==================

Core evaluation module that takes control/treatment experiment data and produces
a complete statistical analysis: conversion rates, lift (absolute & relative),
confidence intervals, p-values, effect sizes, and a plain-English verdict.

This module ties together the sample size calculator (Phase 1) and the hypothesis
testing library (Phase 2) into a single, cohesive A/B test evaluation workflow.

Usage
-----
    >>> from src.ab_engine import run_ab_test
    >>> result = run_ab_test(
    ...     successes_a=120, n_a=1000,
    ...     successes_b=145, n_b=1000,
    ...     metric_type='binary'
    ... )
    >>> print(result.verdict)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
from scipy import stats

from src.sample_size import sample_size_proportions, sample_size_continuous, validate_sample_size
from src.hypothesis_tests import (
    two_sample_ttest,
    z_test_proportions,
    chi_square_test,
    mann_whitney_u,
    select_test,
    TestResult,
)


@dataclass
class ABTestResult:
    """
    Complete result object from an A/B test evaluation.

    Contains point estimates, statistical test results, effect sizes,
    power analysis, and a plain-English verdict suitable for presenting
    to non-technical stakeholders.

    Attributes
    ----------
    metric_type : str
        Type of metric analyzed ('binary' for conversions, 'continuous' for means).
    control_estimate : float
        Point estimate for the control group (conversion rate or mean).
    treatment_estimate : float
        Point estimate for the treatment group (conversion rate or mean).
    absolute_lift : float
        Absolute difference: treatment - control.
    relative_lift : float
        Relative lift as a percentage: (treatment - control) / control * 100.
    confidence_interval : tuple
        95% confidence interval on the difference (treatment - control).
    test_result : TestResult
        The full hypothesis test result (statistic, p-value, interpretation).
    effect_size : float
        Standardized effect size (Cohen's d for continuous, risk difference for binary).
    effect_size_label : str
        Label describing the effect size metric used.
    sample_size_check : dict
        Power analysis: whether the observed sample size is sufficient.
    verdict : str
        Plain-English recommendation for stakeholders.
    verdict_code : str
        Machine-readable verdict code: 'significant_b', 'significant_a',
        'not_significant', or 'underpowered'.
    control_n : int
        Sample size for the control group.
    treatment_n : int
        Sample size for the treatment group.
    alpha : float
        Significance level used.
    """

    metric_type: str
    control_estimate: float
    treatment_estimate: float
    absolute_lift: float
    relative_lift: float
    confidence_interval: tuple
    test_result: TestResult
    effect_size: float
    effect_size_label: str
    sample_size_check: dict
    verdict: str
    verdict_code: str
    control_n: int
    treatment_n: int
    alpha: float = 0.05


def _compute_lift(control_est: float, treatment_est: float) -> dict:
    """
    Compute absolute and relative lift between treatment and control.

    Parameters
    ----------
    control_est : float
        Control group point estimate.
    treatment_est : float
        Treatment group point estimate.

    Returns
    -------
    dict
        Keys 'absolute' (treatment - control) and 'relative' (percentage lift).
        Relative lift is None if control_est is zero to avoid division by zero.
    """
    absolute = treatment_est - control_est
    if control_est == 0:
        relative = None
    else:
        relative = (absolute / control_est) * 100
    return {"absolute": absolute, "relative": relative}


def _ci_difference_proportions(
    p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05
) -> tuple:
    """
    Compute confidence interval for the difference of two proportions (p2 - p1).

    Uses the Wald method: (p2 - p1) ± z * sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2).

    Parameters
    ----------
    p1 : float
        Proportion in group 1 (control).
    n1 : int
        Sample size for group 1.
    p2 : float
        Proportion in group 2 (treatment).
    n2 : int
        Sample size for group 2.
    alpha : float, optional
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval.
    """
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return (diff - z_crit * se, diff + z_crit * se)


def _ci_difference_means(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.05
) -> tuple:
    """
    Compute confidence interval for the difference of two means (b - a).

    Uses Welch's approximation for degrees of freedom when variances
    may differ between groups.

    Parameters
    ----------
    a : array-like
        Data for group A (control).
    b : array-like
        Data for group B (treatment).
    alpha : float, optional
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.mean(b) - np.mean(a)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))

    # Welch-Satterthwaite degrees of freedom
    s1_sq, s2_sq = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else min(n1, n2) - 1

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return (diff - t_crit * se, diff + t_crit * se)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d for independent samples.

    Cohen's d = (mean_b - mean_a) / pooled_std

    Interpretation guidelines (Cohen, 1988):
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Parameters
    ----------
    a : array-like
        Data for group A.
    b : array-like
        Data for group B.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(b) - np.mean(a)) / pooled_std


def _odds_ratio(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    """
    Compute odds ratio for two binary outcomes.

    OR = (successes_b / failures_b) / (successes_a / failures_a)

    An OR > 1 indicates the treatment group has higher odds of success.

    Parameters
    ----------
    successes_a : int
        Number of successes in group A.
    n_a : int
        Total sample size for group A.
    successes_b : int
        Number of successes in group B.
    n_b : int
        Total sample size for group B.

    Returns
    -------
    float
        Odds ratio. Returns float('inf') if denominator is zero.
    """
    failures_a = n_a - successes_a
    failures_b = n_b - successes_b

    # Add 0.5 continuity correction to avoid division by zero
    if failures_a == 0 or successes_a == 0 or failures_b == 0 or successes_b == 0:
        odds_a = (successes_a + 0.5) / (failures_a + 0.5)
        odds_b = (successes_b + 0.5) / (failures_b + 0.5)
    else:
        odds_a = successes_a / failures_a
        odds_b = successes_b / failures_b

    return odds_b / odds_a if odds_a != 0 else float("inf")


def _generate_verdict(
    p_value: float,
    alpha: float,
    absolute_lift: float,
    is_sufficient_sample: bool,
) -> tuple:
    """
    Generate a plain-English verdict and machine-readable code.

    Decision logic:
        1. If sample size is insufficient → 'underpowered'
        2. If p_value < alpha and treatment > control → 'significant_b'
        3. If p_value < alpha and control > treatment → 'significant_a'
        4. Otherwise → 'not_significant'

    Parameters
    ----------
    p_value : float
        Observed p-value from the hypothesis test.
    alpha : float
        Significance level.
    absolute_lift : float
        Treatment estimate minus control estimate.
    is_sufficient_sample : bool
        Whether the experiment has adequate sample size per the power analysis.

    Returns
    -------
    tuple
        (verdict_string, verdict_code) where verdict_string is human-readable
        and verdict_code is one of: 'significant_b', 'significant_a',
        'not_significant', 'underpowered'.
    """
    if not is_sufficient_sample:
        return (
            "⚠️ Underpowered — Needs More Data. Your current sample size is below "
            "the minimum required to reliably detect the expected effect size. "
            "Continue collecting data before making a decision.",
            "underpowered",
        )

    if p_value < alpha:
        if absolute_lift > 0:
            return (
                "✅ Significant Winner: Treatment (B). The treatment group shows a "
                f"statistically significant improvement (p={p_value:.4f} < α={alpha}). "
                "Recommendation: Ship it.",
                "significant_b",
            )
        else:
            return (
                "✅ Significant Winner: Control (A). The control group outperforms the "
                f"treatment (p={p_value:.4f} < α={alpha}). "
                "Recommendation: Keep the current version.",
                "significant_a",
            )
    else:
        return (
            f"❌ Not Significant (p={p_value:.4f} ≥ α={alpha}). The observed difference "
            "is not statistically significant. Recommendation: Keep testing or accept "
            "that the difference is too small to detect with current sample size.",
            "not_significant",
        )


def run_ab_test(
    successes_a: int = None,
    n_a: int = None,
    successes_b: int = None,
    n_b: int = None,
    data_a: np.ndarray = None,
    data_b: np.ndarray = None,
    metric_type: str = "binary",
    alpha: float = 0.05,
    mde: float = None,
    power: float = 0.80,
) -> ABTestResult:
    """
    Run a complete A/B test evaluation.

    Supports both binary (conversion) and continuous metrics. For binary metrics,
    provide successes and sample sizes. For continuous metrics, provide raw data arrays.

    The engine automatically:
    1. Selects the appropriate statistical test
    2. Computes point estimates, lift, confidence intervals, and effect size
    3. Runs a power analysis to check if the sample size is sufficient
    4. Produces a plain-English verdict

    Parameters
    ----------
    successes_a : int, optional
        Number of conversions in control group (binary metrics only).
    n_a : int, optional
        Sample size for control group (binary metrics only).
    successes_b : int, optional
        Number of conversions in treatment group (binary metrics only).
    n_b : int, optional
        Sample size for treatment group (binary metrics only).
    data_a : array-like, optional
        Raw data for control group (continuous metrics only).
    data_b : array-like, optional
        Raw data for treatment group (continuous metrics only).
    metric_type : str, optional
        'binary' (default) for conversion rates, 'continuous' for mean comparisons.
    alpha : float, optional
        Significance level (default 0.05).
    mde : float, optional
        Minimum detectable effect for power analysis. If None, uses the observed
        absolute difference as MDE (conservative approach).
    power : float, optional
        Target statistical power for sample size check (default 0.80).

    Returns
    -------
    ABTestResult
        Complete result object with all statistics and a verdict.

    Raises
    ------
    ValueError
        If required parameters are missing for the chosen metric_type.

    Examples
    --------
    Binary (conversion) test:

    >>> result = run_ab_test(
    ...     successes_a=120, n_a=1000,
    ...     successes_b=145, n_b=1000,
    ...     metric_type='binary'
    ... )
    >>> print(f"Lift: {result.relative_lift:.1f}%")
    >>> print(result.verdict)

    Continuous (revenue) test:

    >>> import numpy as np
    >>> np.random.seed(42)
    >>> control = np.random.normal(50, 10, 500)
    >>> treatment = np.random.normal(53, 10, 500)
    >>> result = run_ab_test(data_a=control, data_b=treatment, metric_type='continuous')
    >>> print(result.verdict)
    """

    # ── Validate inputs ─────────────────────────────────────────────────────────
    if metric_type == "binary":
        if any(v is None for v in [successes_a, n_a, successes_b, n_b]):
            raise ValueError(
                "For binary metrics, provide successes_a, n_a, successes_b, n_b."
            )
        control_est = successes_a / n_a
        treatment_est = successes_b / n_b
        control_n = n_a
        treatment_n = n_b
    elif metric_type == "continuous":
        if data_a is None or data_b is None:
            raise ValueError(
                "For continuous metrics, provide data_a and data_b arrays."
            )
        data_a = np.asarray(data_a, dtype=float)
        data_b = np.asarray(data_b, dtype=float)
        control_est = float(np.mean(data_a))
        treatment_est = float(np.mean(data_b))
        control_n = len(data_a)
        treatment_n = len(data_b)
    else:
        raise ValueError(f"metric_type must be 'binary' or 'continuous', got '{metric_type}'")

    # ── Lift ─────────────────────────────────────────────────────────────────────
    lift = _compute_lift(control_est, treatment_est)
    absolute_lift = lift["absolute"]
    relative_lift = lift["relative"] if lift["relative"] is not None else 0.0

    # ── Select and run appropriate hypothesis test ───────────────────────────────
    if metric_type == "binary":
        test_result = z_test_proportions(successes_a, n_a, successes_b, n_b, alpha=alpha)
        ci = _ci_difference_proportions(control_est, control_n, treatment_est, treatment_n, alpha)
        effect = _odds_ratio(successes_a, n_a, successes_b, n_b)
        effect_label = "Odds Ratio"
    else:
        test_result = two_sample_ttest(data_a, data_b, alpha=alpha, equal_var=False)
        ci = _ci_difference_means(data_a, data_b, alpha)
        effect = _cohens_d(data_a, data_b)
        effect_label = "Cohen's d"

    # ── Power analysis / sample size check ───────────────────────────────────────
    observed_mde = abs(absolute_lift) if mde is None else mde

    try:
        if metric_type == "binary":
            if observed_mde > 0:
                required_n = sample_size_proportions(
                    baseline_rate=control_est,
                    mde=observed_mde,
                    alpha=alpha,
                    power=power,
                )
            else:
                # No detectable difference — can't compute required n
                required_n = float("inf")
        else:
            baseline_std = float(np.std(data_a, ddof=1))
            if observed_mde > 0 and baseline_std > 0:
                required_n = sample_size_continuous(
                    baseline_mean=control_est,
                    baseline_std=baseline_std,
                    mde=observed_mde,
                    alpha=alpha,
                    power=power,
                )
            else:
                required_n = float("inf")

        actual_n = min(control_n, treatment_n)
        ss_check = validate_sample_size(actual_n, required_n)
    except Exception:
        ss_check = {
            "is_sufficient": True,
            "actual": min(control_n, treatment_n),
            "required": None,
            "shortfall": 0,
            "message": "Could not compute required sample size; proceeding with available data.",
        }

    # ── Verdict ──────────────────────────────────────────────────────────────────
    verdict, verdict_code = _generate_verdict(
        p_value=test_result.p_value,
        alpha=alpha,
        absolute_lift=absolute_lift,
        is_sufficient_sample=ss_check["is_sufficient"],
    )

    return ABTestResult(
        metric_type=metric_type,
        control_estimate=control_est,
        treatment_estimate=treatment_est,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        confidence_interval=ci,
        test_result=test_result,
        effect_size=effect,
        effect_size_label=effect_label,
        sample_size_check=ss_check,
        verdict=verdict,
        verdict_code=verdict_code,
        control_n=control_n,
        treatment_n=treatment_n,
        alpha=alpha,
    )


def compare_multiple_variants(
    variants: dict,
    control_key: str = "control",
    metric_type: str = "binary",
    alpha: float = 0.05,
) -> List[ABTestResult]:
    """
    Compare multiple treatment variants against a single control.

    Runs a separate A/B test for each treatment vs. the control and applies
    Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    variants : dict
        Dictionary mapping variant names to their data.
        For binary: {'control': (successes, n), 'variant_b': (successes, n), ...}
        For continuous: {'control': np.array, 'variant_b': np.array, ...}
    control_key : str, optional
        Key for the control group in the variants dict (default 'control').
    metric_type : str, optional
        'binary' or 'continuous' (default 'binary').
    alpha : float, optional
        Family-wise significance level (default 0.05). Will be Bonferroni-corrected.

    Returns
    -------
    list of ABTestResult
        One result for each treatment-vs-control comparison.
    """
    treatment_keys = [k for k in variants if k != control_key]
    n_comparisons = len(treatment_keys)
    corrected_alpha = alpha / n_comparisons  # Bonferroni correction

    results = []
    control_data = variants[control_key]

    for key in treatment_keys:
        treatment_data = variants[key]

        if metric_type == "binary":
            result = run_ab_test(
                successes_a=control_data[0],
                n_a=control_data[1],
                successes_b=treatment_data[0],
                n_b=treatment_data[1],
                metric_type="binary",
                alpha=corrected_alpha,
            )
        else:
            result = run_ab_test(
                data_a=control_data,
                data_b=treatment_data,
                metric_type="continuous",
                alpha=corrected_alpha,
            )
        results.append(result)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("A/B TESTING ENGINE — DEMO")
    print("=" * 70)

    # --- Binary test ---
    print("\n--- Binary (Conversion Rate) Test ---")
    result = run_ab_test(
        successes_a=120, n_a=1000,
        successes_b=145, n_b=1000,
        metric_type="binary",
    )
    print(f"Control rate:       {result.control_estimate:.4f}")
    print(f"Treatment rate:     {result.treatment_estimate:.4f}")
    print(f"Absolute lift:      {result.absolute_lift:.4f}")
    print(f"Relative lift:      {result.relative_lift:.1f}%")
    print(f"95% CI:             ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
    print(f"P-value:            {result.test_result.p_value:.4f}")
    print(f"Odds Ratio:         {result.effect_size:.4f}")
    print(f"Sample size check:  {result.sample_size_check['message']}")
    print(f"Verdict:            {result.verdict}")

    # --- Continuous test ---
    print("\n--- Continuous (Revenue) Test ---")
    np.random.seed(42)
    control = np.random.normal(50, 10, 500)
    treatment = np.random.normal(53, 10, 500)
    result = run_ab_test(data_a=control, data_b=treatment, metric_type="continuous")
    print(f"Control mean:       {result.control_estimate:.2f}")
    print(f"Treatment mean:     {result.treatment_estimate:.2f}")
    print(f"Absolute lift:      {result.absolute_lift:.2f}")
    print(f"Relative lift:      {result.relative_lift:.1f}%")
    print(f"95% CI:             ({result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f})")
    print(f"P-value:            {result.test_result.p_value:.4f}")
    print(f"Cohen's d:          {result.effect_size:.4f}")
    print(f"Verdict:            {result.verdict}")

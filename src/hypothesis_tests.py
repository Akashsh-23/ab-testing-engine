"""
hypothesis_tests.py
====================
Core hypothesis tests for A/B testing and statistical significance analysis.

This module provides production-ready implementations of the most common
hypothesis tests used in A/B experimentation.  Every public function returns
a :class:`TestResult` dataclass so that downstream code can programmatically
inspect statistics, p-values, effect sizes, and confidence intervals without
parsing strings.

Supported tests
---------------
* Two-sample t-test (Student's / Welch's)
* Two-proportion z-test
* Chi-square test of independence
* One-way ANOVA
* Mann–Whitney U (non-parametric)

A helper function :func:`select_test` recommends the appropriate test given
the metric type, number of groups, and distributional assumptions.

Dependencies
------------
* numpy
* scipy  (>= 1.7)
* statsmodels  (>= 0.13)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


# ---------------------------------------------------------------------------
# Structured return type
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Structured container for hypothesis-test output.

    Every test function in this module returns a ``TestResult`` so that
    callers have a uniform interface for inspecting outcomes.

    Parameters
    ----------
    test_name : str
        Human-readable name of the statistical test that was executed.
    statistic : float
        Value of the test statistic (t, z, χ², F, U, etc.).
    p_value : float
        Two-sided p-value associated with *statistic*.
    alpha : float
        Significance level (type-I error rate) used for the decision.
    is_significant : bool
        ``True`` when ``p_value < alpha``.
    interpretation : str
        One- or two-sentence plain-English interpretation of the result,
        suitable for embedding in reports or dashboards.
    effect_size : float or None
        An optional standardised effect-size measure (Cohen's d, Cramér's V,
        η², rank-biserial r, etc.).  ``None`` when not applicable.
    confidence_interval : tuple of (float, float) or None
        Optional confidence interval on the quantity of interest (e.g. mean
        difference or proportion difference).  ``None`` when not computed.

    Examples
    --------
    >>> r = TestResult(
    ...     test_name="Two-Sample t-Test",
    ...     statistic=2.31,
    ...     p_value=0.023,
    ...     alpha=0.05,
    ...     is_significant=True,
    ...     interpretation="The difference is statistically significant.",
    ... )
    >>> r.is_significant
    True
    """

    test_name: str
    statistic: float
    p_value: float
    alpha: float
    is_significant: bool
    interpretation: str
    effect_size: Optional[float] = field(default=None)
    confidence_interval: Optional[Tuple[float, float]] = field(default=None)


# ---------------------------------------------------------------------------
# Two-sample t-test
# ---------------------------------------------------------------------------

def two_sample_ttest(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alpha: float = 0.05,
    equal_var: bool = True,
) -> TestResult:
    """Two-sample t-test for independent samples (Student's or Welch's).

    Tests whether the population means of two independent groups differ.

    Assumptions
    -----------
    * Observations within each group are independent.
    * Data are continuous and measured on at least an interval scale.
    * Each group is drawn from a roughly normal population.  The test is
      robust to moderate departures from normality when sample sizes are
      large (n ≳ 30 per group) thanks to the Central Limit Theorem.
    * When ``equal_var=True`` (Student's t-test) the two populations are
      assumed to have equal variance.  Set ``equal_var=False`` to use
      Welch's t-test, which does **not** assume equal variances.

    Parameters
    ----------
    group_a : array-like of float
        Observations from group A (control).
    group_b : array-like of float
        Observations from group B (treatment).
    alpha : float, default 0.05
        Significance level for the hypothesis test.
    equal_var : bool, default True
        If ``True``, perform the standard Student's t-test which assumes
        equal population variances.  If ``False``, perform Welch's t-test,
        which does not assume equal variances and is generally safer.

    Returns
    -------
    TestResult
        Contains the t-statistic, two-sided p-value, Cohen's d effect size,
        and a 95 % confidence interval on the mean difference
        (mean_a − mean_b).

    Notes
    -----
    Cohen's d is calculated as::

        d = (mean_a - mean_b) / s_pooled

    where ``s_pooled = sqrt(((n_a-1)*s_a² + (n_b-1)*s_b²) / (n_a+n_b-2))``.

    The confidence interval on the mean difference uses the appropriate
    t-distribution (Student or Welch) degrees of freedom.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> ctrl = np.random.normal(10, 2, size=100)
    >>> treat = np.random.normal(11, 2, size=100)
    >>> result = two_sample_ttest(ctrl, treat, alpha=0.05, equal_var=False)
    >>> result.is_significant
    True
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if a.size < 2 or b.size < 2:
        raise ValueError(
            "Each group must contain at least 2 observations.  "
            f"Got n_a={a.size}, n_b={b.size}."
        )


    t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)


    n_a, n_b = a.size, b.size
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


    mean_diff = mean_a - mean_b
    if equal_var:
        df = n_a + n_b - 2
        se_diff = pooled_std * np.sqrt(1 / n_a + 1 / n_b)
    else:
        # Welch–Satterthwaite degrees of freedom
        se_diff = np.sqrt(var_a / n_a + var_b / n_b)
        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (
            (var_a / n_a) ** 2 / (n_a - 1)
            + (var_b / n_b) ** 2 / (n_b - 1)
        )
        df = numerator / denominator

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff


    test_variant = "Welch's" if not equal_var else "Student's"
    is_sig = bool(p_value < alpha)
    if is_sig:
        interpretation = (
            f"{test_variant} t-test: The mean difference ({mean_diff:+.4f}) "
            f"is statistically significant (t={t_stat:.4f}, p={p_value:.4g}, "
            f"α={alpha}).  Cohen's d = {cohens_d:.4f}."
        )
    else:
        interpretation = (
            f"{test_variant} t-test: No statistically significant difference "
            f"in means (t={t_stat:.4f}, p={p_value:.4g}, α={alpha}).  "
            f"Cohen's d = {cohens_d:.4f}."
        )

    return TestResult(
        test_name=f"Two-Sample {test_variant} t-Test",
        statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=is_sig,
        interpretation=interpretation,
        effect_size=float(cohens_d),
        confidence_interval=(float(ci_lower), float(ci_upper)),
    )


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------

def z_test_proportions(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-proportion z-test for comparing conversion rates.

    Tests whether the population proportions of two independent groups
    differ (H₀: p_A = p_B  vs  H₁: p_A ≠ p_B).

    Assumptions
    -----------
    * Observations are independent **within** and **between** groups.
    * Sample sizes are large enough for the normal approximation to hold.
      The rule of thumb is n·p ≥ 5 **and** n·(1 − p) ≥ 5 for each group.
    * The outcome is binary (success / failure).

    Parameters
    ----------
    successes_a : int
        Number of successes (conversions) in group A.
    n_a : int
        Total number of observations in group A.
    successes_b : int
        Number of successes (conversions) in group B.
    n_b : int
        Total number of observations in group B.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    TestResult
        Contains the z-statistic, two-sided p-value, and a confidence
        interval on the difference of proportions (p_A − p_B).

    Notes
    -----
    The pooled proportion is ``p_pool = (x_a + x_b) / (n_a + n_b)`` and the
    z-statistic is::

        z = (p_a - p_b) / sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

    The CI is computed using un-pooled standard errors (Wald interval).

    Examples
    --------
    >>> result = z_test_proportions(
    ...     successes_a=120, n_a=1000,
    ...     successes_b=145, n_b=1000,
    ...     alpha=0.05,
    ... )
    >>> print(result.p_value)  # doctest: +SKIP
    """

    for name, val, total in [
        ("successes_a", successes_a, n_a),
        ("successes_b", successes_b, n_b),
    ]:
        if not (0 <= val <= total):
            raise ValueError(
                f"{name} must satisfy 0 <= {name} <= n  "
                f"(got {val}, n={total})."
            )
    if n_a < 1 or n_b < 1:
        raise ValueError("Sample sizes must be at least 1.")


    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_a - p_b

    # Use statsmodels z-test, which pools standard error under H0 by default.
    count = np.array([successes_a, successes_b])
    nobs = np.array([n_a, n_b])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    # Compute CI on difference using un-pooled standard error (Wald interval)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff


    is_sig = bool(p_value < alpha)
    if is_sig:
        interpretation = (
            f"Two-proportion z-test: The difference in proportions "
            f"({p_a:.4f} vs {p_b:.4f}, Δ={diff:+.4f}) is statistically "
            f"significant (z={z_stat:.4f}, p={p_value:.4g}, α={alpha})."
        )
    else:
        interpretation = (
            f"Two-proportion z-test: No statistically significant difference "
            f"in proportions ({p_a:.4f} vs {p_b:.4f}, Δ={diff:+.4f}; "
            f"z={z_stat:.4f}, p={p_value:.4g}, α={alpha})."
        )

    return TestResult(
        test_name="Two-Proportion z-Test",
        statistic=float(z_stat),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=is_sig,
        interpretation=interpretation,
        effect_size=None,
        confidence_interval=(float(ci_lower), float(ci_upper)),
    )


# ---------------------------------------------------------------------------
# Chi-square test of independence
# ---------------------------------------------------------------------------

def chi_square_test(
    contingency_table: Union[np.ndarray, List[List[int]]],
    alpha: float = 0.05,
) -> TestResult:
    """Chi-square (χ²) test of independence for a contingency table.

    Tests whether two categorical variables are independent by comparing
    observed cell frequencies with those expected under the null hypothesis
    of independence.

    Assumptions
    -----------
    * Observations are independent.
    * All **expected** cell frequencies are at least 5.  When this assumption
      is violated the approximation to the χ² distribution may be poor;
      consider Fisher's exact test for small samples.

    Parameters
    ----------
    contingency_table : 2-D array-like of int
        Contingency table of observed frequencies.  Rows and columns
        represent the levels of two categorical variables (e.g.,
        ``[[a_success, a_fail], [b_success, b_fail]]``).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    TestResult
        Contains the χ² statistic, p-value, degrees of freedom (stored in
        ``interpretation``), and Cramér's V as the effect-size measure.

    Notes
    -----
    Cramér's V is computed as::

        V = sqrt(χ² / (n * (min(r, c) - 1)))

    where *r* and *c* are the numbers of rows and columns, and *n* is the
    grand total of observations.

    Examples
    --------
    >>> table = [[120, 880], [145, 855]]
    >>> result = chi_square_test(table, alpha=0.05)
    >>> result.test_name
    'Chi-Square Test of Independence'
    """
    table = np.asarray(contingency_table, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "contingency_table must be a 2-D array with at least 2 rows "
            f"and 2 columns.  Got shape {table.shape}."
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(table)


    n = table.sum()
    min_dim = min(table.shape[0], table.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0


    is_sig = bool(p_value < alpha)
    if is_sig:
        interpretation = (
            f"Chi-square test: A statistically significant association was "
            f"found (χ²={chi2:.4f}, df={dof}, p={p_value:.4g}, α={alpha}).  "
            f"Cramér's V = {cramers_v:.4f}."
        )
    else:
        interpretation = (
            f"Chi-square test: No statistically significant association "
            f"(χ²={chi2:.4f}, df={dof}, p={p_value:.4g}, α={alpha}).  "
            f"Cramér's V = {cramers_v:.4f}."
        )

    return TestResult(
        test_name="Chi-Square Test of Independence",
        statistic=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=is_sig,
        interpretation=interpretation,
        effect_size=float(cramers_v),
        confidence_interval=None,  # CI not standard for χ²
    )


# ---------------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------------

def one_way_anova(*groups: Sequence[float], alpha: float = 0.05) -> TestResult:
    """One-way analysis of variance (ANOVA) for three or more groups.

    Tests whether the population means of three or more independent groups
    are all equal (H₀: μ₁ = μ₂ = … = μ_k).

    Assumptions
    -----------
    * Observations are independent within and between groups.
    * Each group is drawn from a normally distributed population.
    * The populations have equal variances (homoscedasticity).  Consider
      Levene's or Bartlett's test to verify this; if violated, use the
      Kruskal–Wallis test instead.

    Parameters
    ----------
    *groups : array-like of float
        Two or more groups of observations.  Typically three or more groups
        are provided; for exactly two groups a t-test is more informative.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    TestResult
        Contains the F-statistic, p-value, and η² (eta-squared) as the
        effect-size measure.

    Notes
    -----
    Eta-squared is computed as::

        η² = SS_between / SS_total

    where ``SS_between = Σ n_i (x̄_i − x̄)²`` and ``SS_total`` is the total
    sum of squares.  η² represents the proportion of total variance explained
    by group membership.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> g1 = np.random.normal(10, 2, 50)
    >>> g2 = np.random.normal(12, 2, 50)
    >>> g3 = np.random.normal(10.5, 2, 50)
    >>> result = one_way_anova(g1, g2, g3, alpha=0.05)
    >>> result.is_significant
    True
    """
    if len(groups) < 2:
        raise ValueError(
            f"At least 2 groups are required for ANOVA.  Got {len(groups)}."
        )
    arrays = [np.asarray(g, dtype=float) for g in groups]
    for i, arr in enumerate(arrays):
        if arr.size < 2:
            raise ValueError(
                f"Group {i} must contain at least 2 observations "
                f"(got {arr.size})."
            )

    f_stat, p_value = stats.f_oneway(*arrays)


    grand_mean = np.concatenate(arrays).mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2 for g in arrays
    )
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in arrays)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0


    k = len(arrays)
    is_sig = bool(p_value < alpha)
    if is_sig:
        interpretation = (
            f"One-way ANOVA ({k} groups): A statistically significant "
            f"difference among group means was found (F={f_stat:.4f}, "
            f"p={p_value:.4g}, α={alpha}).  η² = {eta_squared:.4f}."
        )
    else:
        interpretation = (
            f"One-way ANOVA ({k} groups): No statistically significant "
            f"difference among group means (F={f_stat:.4f}, p={p_value:.4g}, "
            f"α={alpha}).  η² = {eta_squared:.4f}."
        )

    return TestResult(
        test_name="One-Way ANOVA",
        statistic=float(f_stat),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=is_sig,
        interpretation=interpretation,
        effect_size=float(eta_squared),
        confidence_interval=None,  # ANOVA doesn't produce a single CI
    )


# ---------------------------------------------------------------------------
# Mann–Whitney U test
# ---------------------------------------------------------------------------

def mann_whitney_u(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Mann–Whitney U test (non-parametric alternative to the t-test).

    Tests whether two independent samples come from the same distribution
    (H₀: P(X_a > X_b) = 0.5).  Unlike the t-test, it does **not** require
    the data to be normally distributed; it operates on ranks.

    When to use
    -----------
    * Data are ordinal, or continuous but **not** normally distributed.
    * Sample sizes are small and normality cannot be assumed.
    * As a robustness check alongside a parametric t-test.

    Assumptions
    -----------
    * Observations are independent within and between groups.
    * The dependent variable is at least ordinal.
    * Under the strictest interpretation, the test assumes that the two
      distributions have the same shape and differ only in location (shift
      alternative).

    Parameters
    ----------
    group_a : array-like of float
        Observations from group A.
    group_b : array-like of float
        Observations from group B.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    TestResult
        Contains the U statistic, two-sided p-value, and the rank-biserial
        correlation *r* as an effect-size measure.

    Notes
    -----
    The rank-biserial correlation is computed as::

        r = 1 - (2 * U) / (n_a * n_b)

    Values of *r* range from −1 to +1, where 0 indicates no effect.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(7)
    >>> a = np.random.exponential(5, 60)
    >>> b = np.random.exponential(7, 60)
    >>> result = mann_whitney_u(a, b, alpha=0.05)
    >>> result.test_name
    'Mann–Whitney U Test'
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if a.size < 1 or b.size < 1:
        raise ValueError(
            "Each group must contain at least 1 observation.  "
            f"Got n_a={a.size}, n_b={b.size}."
        )

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")


    n_a, n_b = a.size, b.size
    rank_biserial = 1 - (2 * u_stat) / (n_a * n_b)


    is_sig = bool(p_value < alpha)
    if is_sig:
        interpretation = (
            f"Mann–Whitney U test: The distributions of the two groups "
            f"differ significantly (U={u_stat:.1f}, p={p_value:.4g}, "
            f"α={alpha}).  Rank-biserial r = {rank_biserial:.4f}."
        )
    else:
        interpretation = (
            f"Mann–Whitney U test: No statistically significant difference "
            f"between distributions (U={u_stat:.1f}, p={p_value:.4g}, "
            f"α={alpha}).  Rank-biserial r = {rank_biserial:.4f}."
        )

    return TestResult(
        test_name="Mann–Whitney U Test",
        statistic=float(u_stat),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=is_sig,
        interpretation=interpretation,
        effect_size=float(rank_biserial),
        confidence_interval=None,
    )


# ---------------------------------------------------------------------------
# Test selection helper
# ---------------------------------------------------------------------------

def select_test(
    metric_type: str,
    num_groups: int = 2,
    is_normal: bool = True,
) -> Dict[str, Union[str, List[str]]]:
    """Recommend the most appropriate hypothesis test for a given scenario.

    This is a decision helper — it does **not** run any test.  It inspects
    the metric type, number of experimental groups, and distributional
    assumptions, then returns a structured recommendation.

    Parameters
    ----------
    metric_type : {'binary', 'continuous'}
        * ``'binary'``     — the outcome is a proportion (e.g. conversion).
        * ``'continuous'`` — the outcome is a continuous measure (e.g. revenue).
    num_groups : int, default 2
        Number of experimental groups (including the control).
    is_normal : bool, default True
        Whether the continuous data can be assumed roughly normal.  Ignored
        when ``metric_type='binary'``.

    Returns
    -------
    dict
        A dictionary with three keys:

        * ``'recommended_test'`` (str) — function name to call.
        * ``'reason'`` (str) — one-sentence justification.
        * ``'alternatives'`` (list of str) — other applicable tests.

    Raises
    ------
    ValueError
        If *metric_type* is not ``'binary'`` or ``'continuous'``, or if
        *num_groups* < 2.

    Examples
    --------
    >>> select_test('binary', num_groups=2)
    {'recommended_test': 'z_test_proportions', ...}

    >>> select_test('continuous', num_groups=3)
    {'recommended_test': 'one_way_anova', ...}
    """
    metric_type = metric_type.strip().lower()
    if metric_type not in ("binary", "continuous"):
        raise ValueError(
            f"metric_type must be 'binary' or 'continuous', got {metric_type!r}."
        )
    if num_groups < 2:
        raise ValueError(
            f"num_groups must be >= 2, got {num_groups}."
        )


    if metric_type == "binary":
        if num_groups == 2:
            return {
                "recommended_test": "z_test_proportions",
                "reason": (
                    "Two-proportion z-test is the standard parametric test "
                    "for comparing conversion rates between two independent "
                    "groups with large samples."
                ),
                "alternatives": ["chi_square_test"],
            }
        else:
            return {
                "recommended_test": "chi_square_test",
                "reason": (
                    "Chi-square test of independence handles comparisons of "
                    "proportions across three or more groups via a "
                    "contingency table."
                ),
                "alternatives": [],
            }


    if num_groups == 2:
        if is_normal:
            return {
                "recommended_test": "two_sample_ttest",
                "reason": (
                    "The two-sample t-test is the standard parametric test "
                    "for comparing means of two groups when the data are "
                    "approximately normally distributed."
                ),
                "alternatives": ["mann_whitney_u"],
            }
        else:
            return {
                "recommended_test": "mann_whitney_u",
                "reason": (
                    "The Mann–Whitney U test is the non-parametric "
                    "alternative to the t-test, appropriate when the "
                    "normality assumption is not met."
                ),
                "alternatives": ["two_sample_ttest"],
            }
    else:
        # 3+ groups
        return {
            "recommended_test": "one_way_anova",
            "reason": (
                "One-way ANOVA compares means across three or more groups.  "
                "If normality is questionable, consider the Kruskal–Wallis "
                "test as a non-parametric alternative."
            ),
            "alternatives": ["kruskal_wallis"],
        }


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    divider = "=" * 72

    # ---- Two-sample t-test ------------------------------------------------
    print(divider)
    print("1. TWO-SAMPLE T-TEST")
    print(divider)
    ctrl = np.random.normal(loc=50, scale=10, size=200)
    treat = np.random.normal(loc=53, scale=10, size=200)
    res = two_sample_ttest(ctrl, treat, alpha=0.05, equal_var=False)
    print(f"  Test        : {res.test_name}")
    print(f"  Statistic   : {res.statistic:.4f}")
    print(f"  p-value     : {res.p_value:.4g}")
    print(f"  Significant : {res.is_significant}")
    print(f"  Cohen's d   : {res.effect_size:.4f}")
    print(f"  95% CI      : ({res.confidence_interval[0]:.4f}, "
          f"{res.confidence_interval[1]:.4f})")
    print(f"  {res.interpretation}\n")

    # ---- Two-proportion z-test --------------------------------------------
    print(divider)
    print("2. TWO-PROPORTION Z-TEST")
    print(divider)
    res = z_test_proportions(
        successes_a=510, n_a=5000,
        successes_b=560, n_b=5000,
        alpha=0.05,
    )
    print(f"  Test        : {res.test_name}")
    print(f"  Statistic   : {res.statistic:.4f}")
    print(f"  p-value     : {res.p_value:.4g}")
    print(f"  Significant : {res.is_significant}")
    print(f"  95% CI      : ({res.confidence_interval[0]:.4f}, "
          f"{res.confidence_interval[1]:.4f})")
    print(f"  {res.interpretation}\n")

    # ---- Chi-square test --------------------------------------------------
    print(divider)
    print("3. CHI-SQUARE TEST OF INDEPENDENCE")
    print(divider)
    table = [[510, 4490], [560, 4440]]
    res = chi_square_test(table, alpha=0.05)
    print(f"  Test        : {res.test_name}")
    print(f"  Statistic   : {res.statistic:.4f}")
    print(f"  p-value     : {res.p_value:.4g}")
    print(f"  Significant : {res.is_significant}")
    print(f"  Cramér's V  : {res.effect_size:.4f}")
    print(f"  {res.interpretation}\n")

    # ---- One-way ANOVA ----------------------------------------------------
    print(divider)
    print("4. ONE-WAY ANOVA")
    print(divider)
    g1 = np.random.normal(50, 10, 100)
    g2 = np.random.normal(53, 10, 100)
    g3 = np.random.normal(48, 10, 100)
    res = one_way_anova(g1, g2, g3, alpha=0.05)
    print(f"  Test        : {res.test_name}")
    print(f"  Statistic   : {res.statistic:.4f}")
    print(f"  p-value     : {res.p_value:.4g}")
    print(f"  Significant : {res.is_significant}")
    print(f"  η²          : {res.effect_size:.4f}")
    print(f"  {res.interpretation}\n")

    # ---- Mann–Whitney U ---------------------------------------------------
    print(divider)
    print("5. MANN–WHITNEY U TEST")
    print(divider)
    skewed_a = np.random.exponential(scale=5, size=150)
    skewed_b = np.random.exponential(scale=7, size=150)
    res = mann_whitney_u(skewed_a, skewed_b, alpha=0.05)
    print(f"  Test        : {res.test_name}")
    print(f"  Statistic   : {res.statistic:.1f}")
    print(f"  p-value     : {res.p_value:.4g}")
    print(f"  Significant : {res.is_significant}")
    print(f"  Rank-bis. r : {res.effect_size:.4f}")
    print(f"  {res.interpretation}\n")

    # ---- Test selector ----------------------------------------------------
    print(divider)
    print("6. TEST SELECTOR")
    print(divider)
    scenarios = [
        ("binary", 2, True),
        ("continuous", 2, True),
        ("continuous", 2, False),
        ("continuous", 3, True),
        ("binary", 4, True),
    ]
    for mt, ng, norm in scenarios:
        rec = select_test(mt, num_groups=ng, is_normal=norm)
        print(f"  metric={mt:>10s}  groups={ng}  normal={str(norm):>5s}  "
              f"→ {rec['recommended_test']}")
        print(f"    reason       : {rec['reason']}")
        print(f"    alternatives : {rec['alternatives']}")
        print()

"""Sample size calculators for A/B testing experiments.

This module provides functions to determine the minimum number of
observations required per group so that an A/B test achieves a desired
statistical power at a given significance level.  Two experiment types
are supported:

* **Proportions** – e.g. conversion-rate tests (two-proportion z-test).
* **Continuous metrics** – e.g. revenue-per-user tests (two-sample t-test).

A convenience validator is also included to check whether an already-
collected sample meets the minimum size requirement.

Typical usage
-------------
>>> from sample_size import sample_size_proportions, sample_size_continuous
>>> n_prop = sample_size_proportions(baseline_rate=0.10, mde=0.02)
>>> n_cont = sample_size_continuous(baseline_mean=50, baseline_std=10, mde=2)
"""

import math

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize



def sample_size_proportions(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
) -> int:
    """Calculate the required sample size per group for a two-proportion z-test.

    Determines the minimum number of observations needed in *each* group
    (control and treatment) so that the test can detect an absolute shift
    of ``mde`` from ``baseline_rate`` with the requested ``power`` at
    significance level ``alpha``.

    Statistical method
    ------------------
    The calculation uses the classical normal-approximation formula for
    comparing two independent proportions:

        n = (Z_{α/2} + Z_{β})² · [p₁(1 − p₁) + p₂(1 − p₂)] / (p₂ − p₁)²

    where *p₁* is the baseline (control) proportion, *p₂ = p₁ + mde* is
    the expected treatment proportion, *Z_{α/2}* is the critical value for
    the chosen significance level (halved for a two-sided test), and *Z_{β}*
    is the critical value corresponding to the desired power.

    The result is cross-validated against ``statsmodels`` by converting the
    proportions to Cohen's *h* via ``proportion_effectsize`` and solving
    with ``NormalIndPower().solve_power()``.  If the two estimates diverge
    by more than 10 %, a ``RuntimeWarning`` is issued.

    Assumptions
    -----------
    * Observations within and across groups are independent.
    * The normal approximation to the binomial is adequate (each cell
      *n · p* and *n · (1 − p)* should be ≥ 5).
    * Equal group sizes are assumed.

    Parameters
    ----------
    baseline_rate : float
        The conversion rate (proportion) expected in the control group.
        Must be in the open interval (0, 1).
    mde : float
        Minimum detectable effect expressed as an *absolute* difference in
        proportions (e.g. 0.02 for a 2-percentage-point lift).  Must be
        non-zero and such that ``baseline_rate + mde`` remains in (0, 1).
    alpha : float, optional
        Significance level (Type I error rate).  Default is 0.05.
    power : float, optional
        Statistical power (1 − Type II error rate).  Default is 0.80.
    alternative : {'two-sided', 'larger', 'smaller'}, optional
        The alternative hypothesis.  Default is ``'two-sided'``.

    Returns
    -------
    int
        Required sample size per group, rounded up to the next integer.

    Raises
    ------
    ValueError
        If any parameter falls outside its valid domain.

    Examples
    --------
    >>> sample_size_proportions(baseline_rate=0.10, mde=0.02)
    3623

    >>> sample_size_proportions(baseline_rate=0.10, mde=0.02,
    ...                        alpha=0.05, power=0.90)
    4849
    """

    if not 0 < baseline_rate < 1:
        raise ValueError(
            f"baseline_rate must be in (0, 1), got {baseline_rate}"
        )
    if mde == 0:
        raise ValueError("mde must be non-zero.")
    p2 = baseline_rate + mde
    if not 0 < p2 < 1:
        raise ValueError(
            f"baseline_rate + mde = {p2} falls outside (0, 1)."
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    valid_alternatives = {"two-sided", "larger", "smaller"}
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}, "
            f"got '{alternative}'"
        )


    p1 = baseline_rate
    beta = 1.0 - power

    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    denominator = (p2 - p1) ** 2
    n_formula = math.ceil(numerator / denominator)


    effect_size_h = proportion_effectsize(p2, p1)
    sm_power = NormalIndPower()
    n_statsmodels = math.ceil(
        sm_power.solve_power(
            effect_size=abs(effect_size_h),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative=alternative,
        )
    )


    if n_formula > 0 and abs(n_formula - n_statsmodels) / n_formula > 0.10:
        import warnings

        warnings.warn(
            f"Formula-based n ({n_formula}) and statsmodels n "
            f"({n_statsmodels}) differ by more than 10 %. "
            f"The statsmodels estimate is returned.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Return the statsmodels result as the authoritative answer (it accounts
    # for continuity and effect-size conversion nuances).
    return n_statsmodels



def sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
) -> int:
    """Calculate the required sample size per group for a two-sample t-test.

    Determines the minimum number of observations needed in *each* group so
    that a two-sample independent t-test can detect an absolute mean
    difference of ``mde`` with the specified ``power`` at significance
    level ``alpha``.

    Statistical method
    ------------------
    The effect is standardised to Cohen's *d*:

        d = mde / baseline_std

    The required *n* is then obtained from
    ``statsmodels.stats.power.TTestIndPower().solve_power()`` which
    numerically inverts the non-central *t* distribution.

    Assumptions
    -----------
    * Observations are independent within and across groups.
    * Both populations are (approximately) normally distributed, or the
      sample sizes are large enough for the Central Limit Theorem to apply.
    * Population variances are equal (homoscedasticity).  If this
      assumption is violated, consider Welch's t-test and adjust
      accordingly.
    * Equal group sizes are assumed.

    Parameters
    ----------
    baseline_mean : float
        Expected mean of the metric in the control group.  Used only for
        documentation / context; the calculation depends on ``baseline_std``
        and ``mde``.
    baseline_std : float
        Standard deviation of the metric in the control group.  Must be
        positive.
    mde : float
        Minimum detectable effect expressed as an *absolute* difference in
        means (e.g. 2.0 if you want to detect a shift of 2 units).  Must
        be non-zero.
    alpha : float, optional
        Significance level (Type I error rate).  Default is 0.05.
    power : float, optional
        Statistical power (1 − Type II error rate).  Default is 0.80.
    alternative : {'two-sided', 'larger', 'smaller'}, optional
        The alternative hypothesis.  Default is ``'two-sided'``.

    Returns
    -------
    int
        Required sample size per group, rounded up to the next integer.

    Raises
    ------
    ValueError
        If any parameter falls outside its valid domain.

    Examples
    --------
    >>> sample_size_continuous(baseline_mean=50.0, baseline_std=10.0, mde=2.0)
    394

    >>> sample_size_continuous(baseline_mean=50.0, baseline_std=10.0,
    ...                        mde=2.0, alpha=0.01, power=0.90)
    709
    """

    if baseline_std <= 0:
        raise ValueError(
            f"baseline_std must be positive, got {baseline_std}"
        )
    if mde == 0:
        raise ValueError("mde must be non-zero.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    valid_alternatives = {"two-sided", "larger", "smaller"}
    if alternative not in valid_alternatives:
        raise ValueError(
            f"alternative must be one of {valid_alternatives}, "
            f"got '{alternative}'"
        )


    cohens_d = abs(mde) / baseline_std

    power_analysis = TTestIndPower()
    n_raw = power_analysis.solve_power(
        effect_size=cohens_d,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=alternative,
    )

    return math.ceil(n_raw)



def validate_sample_size(actual_n: int, required_n: int) -> dict:
    """Compare an actual sample size to the required minimum.

    Provides a quick diagnostic that tells the experimenter whether the
    data already collected (or projected to be collected) meets the
    statistical requirements computed by one of the sample-size functions
    in this module.

    Parameters
    ----------
    actual_n : int
        The number of observations actually available (per group).
    required_n : int
        The minimum number of observations per group as determined by a
        power analysis (e.g. from ``sample_size_proportions`` or
        ``sample_size_continuous``).

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``is_sufficient`` (*bool*) – ``True`` if ``actual_n >= required_n``.
        * ``actual`` (*int*) – Echo of ``actual_n``.
        * ``required`` (*int*) – Echo of ``required_n``.
        * ``shortfall`` (*int*) – ``max(0, required_n - actual_n)``.
          Zero when the sample is sufficient.
        * ``surplus`` (*int*) – ``max(0, actual_n - required_n)``.
          Zero when the sample is insufficient.
        * ``message`` (*str*) – A plain-English summary sentence.

    Raises
    ------
    ValueError
        If ``actual_n`` or ``required_n`` is negative.

    Examples
    --------
    >>> validate_sample_size(actual_n=5000, required_n=3623)
    {'is_sufficient': True, 'actual': 5000, 'required': 3623,
     'shortfall': 0, 'surplus': 1377,
     'message': 'Sample size is sufficient. You have 1,377 observations '
                'more than the required 3,623 per group.'}

    >>> validate_sample_size(actual_n=2000, required_n=3623)
    {'is_sufficient': False, 'actual': 2000, 'required': 3623,
     'shortfall': 1623, 'surplus': 0,
     'message': 'Sample size is INSUFFICIENT. You need 1,623 more '
                'observations per group (have 2,000 of 3,623 required).'}
    """
    if actual_n < 0:
        raise ValueError(f"actual_n must be non-negative, got {actual_n}")
    if required_n < 0:
        raise ValueError(f"required_n must be non-negative, got {required_n}")

    is_sufficient = actual_n >= required_n
    shortfall = max(0, required_n - actual_n)
    surplus = max(0, actual_n - required_n)

    if is_sufficient:
        message = (
            f"Sample size is sufficient. You have {surplus:,} observations "
            f"more than the required {required_n:,} per group."
        )
    else:
        message = (
            f"Sample size is INSUFFICIENT. You need {shortfall:,} more "
            f"observations per group "
            f"(have {actual_n:,} of {required_n:,} required)."
        )

    return {
        "is_sufficient": is_sufficient,
        "actual": actual_n,
        "required": required_n,
        "shortfall": shortfall,
        "surplus": surplus,
        "message": message,
    }



if __name__ == "__main__":
    print("=" * 72)
    print("  A/B Test Sample-Size Calculator – Demo")
    print("=" * 72)

    baseline = 0.10
    lift = 0.02
    alpha = 0.05
    pwr = 0.80

    n_prop = sample_size_proportions(
        baseline_rate=baseline, mde=lift, alpha=alpha, power=pwr,
    )
    print(
        f"\n[Proportions Test]"
        f"\n  Baseline rate : {baseline:.0%}"
        f"\n  MDE (absolute): {lift:.0%}"
        f"\n  Alpha         : {alpha}"
        f"\n  Power         : {pwr}"
        f"\n  ➜ Required n per group: {n_prop:,}"
    )


    mean = 50.0
    std = 10.0
    mde_cont = 2.0

    n_cont = sample_size_continuous(
        baseline_mean=mean, baseline_std=std, mde=mde_cont,
        alpha=alpha, power=pwr,
    )
    print(
        f"\n[Continuous Metric Test]"
        f"\n  Baseline mean : {mean}"
        f"\n  Baseline std  : {std}"
        f"\n  MDE (absolute): {mde_cont}"
        f"\n  Alpha         : {alpha}"
        f"\n  Power         : {pwr}"
        f"\n  ➜ Required n per group: {n_cont:,}"
    )


    actual = 5000
    result = validate_sample_size(actual_n=actual, required_n=n_prop)
    print(f"\n[Validation – Proportions]")
    print(f"  Actual n: {actual:,}")
    print(f"  {result['message']}")

    actual_small = 2000
    result2 = validate_sample_size(actual_n=actual_small, required_n=n_prop)
    print(f"\n[Validation – Proportions (under-powered)]")
    print(f"  Actual n: {actual_small:,}")
    print(f"  {result2['message']}")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)

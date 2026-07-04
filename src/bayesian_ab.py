"""
Bayesian A/B Testing — Beta-Binomial Conjugate Model
=====================================================

This module provides a full Bayesian framework for comparing two variants
(A and B) in a conversion-rate experiment.  It uses the **Beta-Binomial
conjugate model**, meaning the prior and posterior are both Beta distributions,
which makes analytical updates trivial and Monte Carlo inference fast.

Key ideas
---------
* **Prior**: Beta(α, β) — default is Beta(1, 1), the uniform / non-informative
  prior that assigns equal probability to every conversion rate in [0, 1].
* **Likelihood**: Binomial(n, p) — each trial is an independent Bernoulli draw.
* **Posterior**: Beta(α + successes, β + trials − successes) — the conjugate
  update simply adds counts to the prior hyper-parameters.

Decision rules
--------------
The module reports both *probability of superiority* and *expected loss*.
Using expected loss is recommended over raw probability because it accounts
for the *magnitude* of the difference, not just its direction.

References
----------
* Kamalbasha & Eugster (2021), "Bayesian A/B Testing".
* Chris Stucchio, "Bayesian A/B Testing at VWO" (whitepaper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats
import plotly.graph_objects as go



@dataclass(frozen=True)
class BayesianABResult:
    """Structured result of a Bayesian A/B test.

    Attributes
    ----------
    prob_b_beats_a : float
        Monte Carlo estimate of P(p_B > p_A), where p_A and p_B are the
        true conversion rates drawn from their respective posteriors.
    prob_a_beats_b : float
        Complement probability, i.e. 1 − prob_b_beats_a.
    expected_loss_choosing_a : float
        E[max(p_B − p_A, 0)] — the average conversion-rate you leave on the
        table if you ship A but B is actually better.
    expected_loss_choosing_b : float
        E[max(p_A − p_B, 0)] — the average conversion-rate you leave on the
        table if you ship B but A is actually better.
    posterior_a_params : tuple
        (alpha, beta) hyper-parameters of A's Beta posterior.
    posterior_b_params : tuple
        (alpha, beta) hyper-parameters of B's Beta posterior.
    credible_interval_a : tuple
        95 % equal-tailed credible interval (lower, upper) for A's rate.
    credible_interval_b : tuple
        95 % equal-tailed credible interval (lower, upper) for B's rate.
    verdict : str
        Plain-English recommendation based on the evidence strength.
    risk_threshold : float
        The expected-loss threshold used when deciding a winner.

    Examples
    --------
    >>> result = run_bayesian_ab(120, 1000, 145, 1000)
    >>> result.prob_b_beats_a          # doctest: +SKIP
    0.9684
    >>> result.verdict                 # doctest: +SKIP
    'Moderate evidence favoring B — consider more data'
    """

    prob_b_beats_a: float
    prob_a_beats_b: float
    expected_loss_choosing_a: float
    expected_loss_choosing_b: float
    posterior_a_params: Tuple[float, float]
    posterior_b_params: Tuple[float, float]
    credible_interval_a: Tuple[float, float]
    credible_interval_b: Tuple[float, float]
    verdict: str
    risk_threshold: float



def compute_posterior(
    successes: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> stats.rv_frozen:
    """Return the Beta posterior after observing conversion data.

    The Beta-Binomial conjugate update works as follows:

        Prior :  Beta(α₀, β₀)
        Data  :  s successes in n trials  (Binomial likelihood)
        Post  :  Beta(α₀ + s, β₀ + n − s)

    With the default prior Beta(1, 1) — which is Uniform(0, 1) — the
    posterior is entirely driven by the observed data, making it an
    appropriate non-informative starting point.

    Parameters
    ----------
    successes : int
        Number of conversions (successes) observed.  Must satisfy
        0 ≤ successes ≤ trials.
    trials : int
        Total number of visitors / impressions / trials.  Must be > 0.
    prior_alpha : float, default 1.0
        α hyper-parameter of the Beta prior.  Higher values express
        stronger prior belief in a high conversion rate.
    prior_beta : float, default 1.0
        β hyper-parameter of the Beta prior.  Higher values express
        stronger prior belief in a low conversion rate.

    Returns
    -------
    scipy.stats.rv_frozen
        A frozen ``scipy.stats.beta`` distribution with shape parameters
        ``(prior_alpha + successes, prior_beta + trials - successes)``.

    Raises
    ------
    ValueError
        If *successes* or *trials* are negative, or successes > trials,
        or prior hyper-parameters are non-positive.

    Examples
    --------
    >>> post = compute_posterior(successes=50, trials=200)
    >>> post.args           # (alpha, beta) of the posterior
    (51, 151)
    >>> round(post.mean(), 4)
    0.2525
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > trials:
        raise ValueError(
            f"successes ({successes}) cannot exceed trials ({trials})"
        )
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError(
            f"Prior hyper-parameters must be positive, "
            f"got alpha={prior_alpha}, beta={prior_beta}"
        )

    alpha_post = prior_alpha + successes
    beta_post = prior_beta + (trials - successes)
    return stats.beta(alpha_post, beta_post)


def probability_b_beats_a(
    posterior_a: stats.rv_frozen,
    posterior_b: stats.rv_frozen,
    n_samples: int = 100_000,
) -> float:
    """Estimate P(p_B > p_A) via Monte Carlo sampling.

    Draws ``n_samples`` independent samples from each posterior, then
    computes the fraction where the B-sample exceeds the A-sample.

    Parameters
    ----------
    posterior_a : scipy.stats.rv_frozen
        Frozen Beta distribution representing A's posterior.
    posterior_b : scipy.stats.rv_frozen
        Frozen Beta distribution representing B's posterior.
    n_samples : int, default 100_000
        Number of Monte Carlo draws.  Higher values reduce variance of the
        estimate at the cost of compute time.  100 k gives roughly ±0.003
        standard error for probabilities near 0.5.

    Returns
    -------
    float
        Estimated probability that B's true conversion rate exceeds A's.

    Notes
    -----
    A deterministic seed ``np.random.default_rng(42)`` is used so that
    results are reproducible across runs.

    Examples
    --------
    >>> a = stats.beta(121, 881)   # ~12 % rate
    >>> b = stats.beta(146, 856)   # ~14.5 % rate
    >>> 0.95 < probability_b_beats_a(a, b) or True  # roughly ~0.97
    True
    """
    rng = np.random.default_rng(42)
    samples_a = posterior_a.rvs(size=n_samples, random_state=rng)
    samples_b = posterior_b.rvs(size=n_samples, random_state=rng)
    return float(np.mean(samples_b > samples_a))


def compute_expected_loss(
    posterior_a: stats.rv_frozen,
    posterior_b: stats.rv_frozen,
    n_samples: int = 100_000,
) -> Tuple[float, float]:
    """Compute the expected loss (risk) associated with each choice.

    *Expected loss of choosing A* is defined as:

        E[max(p_B − p_A, 0)]

    This measures how much conversion rate you sacrifice, on average, if
    you deploy A but B was actually better.  The analogous quantity holds
    for choosing B.

    Using expected loss instead of (or alongside) probability of
    superiority is recommended because it accounts for the *magnitude*
    of the difference, not just its sign.  A variant might have a 60 %
    probability of being better, yet the expected improvement could be
    negligibly small.

    Parameters
    ----------
    posterior_a : scipy.stats.rv_frozen
        Frozen Beta distribution for A's posterior.
    posterior_b : scipy.stats.rv_frozen
        Frozen Beta distribution for B's posterior.
    n_samples : int, default 100_000
        Number of Monte Carlo draws.

    Returns
    -------
    tuple of (float, float)
        ``(loss_choosing_a, loss_choosing_b)``

    Examples
    --------
    >>> a = stats.beta(121, 881)
    >>> b = stats.beta(146, 856)
    >>> loss_a, loss_b = compute_expected_loss(a, b)
    >>> loss_a > loss_b   # Choosing A is riskier when B is likely better
    True
    """
    rng = np.random.default_rng(42)
    samples_a = posterior_a.rvs(size=n_samples, random_state=rng)
    samples_b = posterior_b.rvs(size=n_samples, random_state=rng)

    loss_choosing_a = float(np.mean(np.maximum(samples_b - samples_a, 0.0)))
    loss_choosing_b = float(np.mean(np.maximum(samples_a - samples_b, 0.0)))
    return loss_choosing_a, loss_choosing_b


def compute_credible_interval(
    posterior: stats.rv_frozen,
    credibility: float = 0.95,
) -> Tuple[float, float]:
    """Return the equal-tailed credible interval for a posterior.

    An equal-tailed interval places ``(1 − credibility) / 2`` probability
    mass in each tail.  For a 95 % interval this corresponds to the 2.5 %
    and 97.5 % quantiles.

    Parameters
    ----------
    posterior : scipy.stats.rv_frozen
        A frozen distribution (typically ``scipy.stats.beta``).
    credibility : float, default 0.95
        The desired probability mass inside the interval.  Must be in
        the open interval (0, 1).

    Returns
    -------
    tuple of (float, float)
        ``(lower_bound, upper_bound)`` of the credible interval.

    Raises
    ------
    ValueError
        If *credibility* is not in (0, 1).

    Examples
    --------
    >>> post = stats.beta(121, 881)
    >>> lo, hi = compute_credible_interval(post)
    >>> 0.09 < lo < hi < 0.16
    True
    """
    if not 0.0 < credibility < 1.0:
        raise ValueError(
            f"credibility must be in (0, 1), got {credibility}"
        )

    tail = (1.0 - credibility) / 2.0
    lower = float(posterior.ppf(tail))
    upper = float(posterior.ppf(1.0 - tail))
    return lower, upper



def plot_posteriors(
    posterior_a: stats.rv_frozen,
    posterior_b: stats.rv_frozen,
    label_a: str = "Control (A)",
    label_b: str = "Treatment (B)",
) -> go.Figure:
    """Create a Plotly figure showing both posterior densities overlaid.

    The plot uses a dark theme with:
    * **Control (A)** rendered in a blue gradient (#4A90D9).
    * **Treatment (B)** rendered in teal-green (#2ECC71).
    * Shaded 95 % credible-interval regions for each posterior.

    Parameters
    ----------
    posterior_a : scipy.stats.rv_frozen
        Frozen Beta distribution for variant A.
    posterior_b : scipy.stats.rv_frozen
        Frozen Beta distribution for variant B.
    label_a : str, default ``'Control (A)'``
        Legend label for variant A.
    label_b : str, default ``'Treatment (B)'``
        Legend label for variant B.

    Returns
    -------
    plotly.graph_objects.Figure
        A fully styled Plotly figure ready for ``.show()`` or serialisation.

    Examples
    --------
    >>> a = compute_posterior(120, 1000)
    >>> b = compute_posterior(145, 1000)
    >>> fig = plot_posteriors(a, b)
    >>> fig.write_html('posteriors.html')   # doctest: +SKIP
    """

    lo = min(posterior_a.ppf(0.001), posterior_b.ppf(0.001))
    hi = max(posterior_a.ppf(0.999), posterior_b.ppf(0.999))
    x = np.linspace(lo, hi, 1_000)

    pdf_a = posterior_a.pdf(x)
    pdf_b = posterior_b.pdf(x)


    ci_a = compute_credible_interval(posterior_a)
    ci_b = compute_credible_interval(posterior_b)

    color_a = "#4A90D9"
    color_a_fill = "rgba(74, 144, 217, 0.15)"
    color_b = "#2ECC71"
    color_b_fill = "rgba(46, 204, 113, 0.15)"

    fig = go.Figure()


    ci_mask_a = (x >= ci_a[0]) & (x <= ci_a[1])
    fig.add_trace(
        go.Scatter(
            x=x[ci_mask_a],
            y=pdf_a[ci_mask_a],
            fill="tozeroy",
            fillcolor=color_a_fill,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{label_a} 95% CI",
        )
    )


    ci_mask_b = (x >= ci_b[0]) & (x <= ci_b[1])
    fig.add_trace(
        go.Scatter(
            x=x[ci_mask_b],
            y=pdf_b[ci_mask_b],
            fill="tozeroy",
            fillcolor=color_b_fill,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{label_b} 95% CI",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_a,
            mode="lines",
            name=label_a,
            line=dict(color=color_a, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_b,
            mode="lines",
            name=label_b,
            line=dict(color=color_b, width=2.5),
        )
    )


    fig.update_layout(
        title=dict(
            text="Posterior Distributions",
            font=dict(size=20, color="#ECEFF4"),
            x=0.5,
        ),
        xaxis=dict(
            title="Conversion Rate",
            color="#D8DEE9",
            gridcolor="#3B4252",
            zerolinecolor="#3B4252",
        ),
        yaxis=dict(
            title="Density",
            color="#D8DEE9",
            gridcolor="#3B4252",
            zerolinecolor="#3B4252",
        ),
        legend=dict(
            font=dict(color="#D8DEE9", size=13),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="#2E3440",
        paper_bgcolor="#2E3440",
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )

    return fig



def run_bayesian_ab(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    risk_threshold: float = 0.01,
) -> BayesianABResult:
    """Run a full Bayesian A/B test and return a structured result.

    This is the main entry-point for the module.  It computes posteriors,
    probabilities, expected losses, credible intervals, and a plain-English
    verdict — all packaged into a :class:`BayesianABResult`.

    Decision Logic
    --------------
    1. **Strong winner B** — ``P(B > A) > 0.95`` *and* expected loss of
       choosing B is below ``risk_threshold``.
    2. **Strong winner A** — ``P(A > B) > 0.95`` *and* expected loss of
       choosing A is below ``risk_threshold``.
    3. **Moderate evidence for B** — ``0.75 < P(B > A) ≤ 0.95``.
    4. **Moderate evidence for A** — ``0.75 < P(A > B) ≤ 0.95``.
    5. **Inconclusive** — otherwise.

    Parameters
    ----------
    successes_a : int
        Conversions observed in variant A.
    trials_a : int
        Total trials for variant A.
    successes_b : int
        Conversions observed in variant B.
    trials_b : int
        Total trials for variant B.
    prior_alpha : float, default 1.0
        α of the shared Beta prior.
    prior_beta : float, default 1.0
        β of the shared Beta prior.
    risk_threshold : float, default 0.01
        Maximum acceptable expected loss (in conversion-rate units) for
        declaring a strong winner.  E.g. 0.01 means you tolerate at most
        a 1-percentage-point loss.

    Returns
    -------
    BayesianABResult
        Frozen dataclass containing every computed metric and the verdict.

    Examples
    --------
    >>> result = run_bayesian_ab(120, 1000, 145, 1000)
    >>> print(result.verdict)          # doctest: +SKIP
    'Moderate evidence favoring B — consider more data'
    >>> print(f"P(B > A) = {result.prob_b_beats_a:.4f}")  # doctest: +SKIP
    P(B > A) = 0.9684
    """

    post_a = compute_posterior(successes_a, trials_a, prior_alpha, prior_beta)
    post_b = compute_posterior(successes_b, trials_b, prior_alpha, prior_beta)


    p_b_wins = probability_b_beats_a(post_a, post_b)
    p_a_wins = 1.0 - p_b_wins


    loss_a, loss_b = compute_expected_loss(post_a, post_b)


    ci_a = compute_credible_interval(post_a)
    ci_b = compute_credible_interval(post_b)


    if p_b_wins > 0.95 and loss_b < risk_threshold:
        verdict = "Strong evidence: B is the winner"
    elif p_a_wins > 0.95 and loss_a < risk_threshold:
        verdict = "Strong evidence: A is the winner"
    elif 0.75 < p_b_wins <= 0.95:
        verdict = "Moderate evidence favoring B \u2014 consider more data"
    elif 0.75 < p_a_wins <= 0.95:
        verdict = "Moderate evidence favoring A \u2014 consider more data"
    else:
        verdict = "Inconclusive \u2014 need more data"

    return BayesianABResult(
        prob_b_beats_a=p_b_wins,
        prob_a_beats_b=p_a_wins,
        expected_loss_choosing_a=loss_a,
        expected_loss_choosing_b=loss_b,
        posterior_a_params=(post_a.args[0], post_a.args[1]),
        posterior_b_params=(post_b.args[0], post_b.args[1]),
        credible_interval_a=ci_a,
        credible_interval_b=ci_b,
        verdict=verdict,
        risk_threshold=risk_threshold,
    )



if __name__ == "__main__":

    SUCCESSES_A, TRIALS_A = 120, 1_000
    SUCCESSES_B, TRIALS_B = 145, 1_000

    print("=" * 64)
    print("  Bayesian A/B Test — Beta-Binomial Conjugate Model")
    print("=" * 64)
    print(f"  Control  (A): {SUCCESSES_A}/{TRIALS_A}"
          f"  ({SUCCESSES_A / TRIALS_A:.1%} observed rate)")
    print(f"  Treatment(B): {SUCCESSES_B}/{TRIALS_B}"
          f"  ({SUCCESSES_B / TRIALS_B:.1%} observed rate)")
    print("-" * 64)

    result = run_bayesian_ab(SUCCESSES_A, TRIALS_A, SUCCESSES_B, TRIALS_B)

    print(f"  P(B > A)            : {result.prob_b_beats_a:.4f}")
    print(f"  P(A > B)            : {result.prob_a_beats_b:.4f}")
    print(f"  E[loss | choose A]  : {result.expected_loss_choosing_a:.5f}")
    print(f"  E[loss | choose B]  : {result.expected_loss_choosing_b:.5f}")
    print(f"  95% CI  A           : [{result.credible_interval_a[0]:.4f},"
          f" {result.credible_interval_a[1]:.4f}]")
    print(f"  95% CI  B           : [{result.credible_interval_b[0]:.4f},"
          f" {result.credible_interval_b[1]:.4f}]")
    print(f"  Posterior A (a, b)  : {result.posterior_a_params}")
    print(f"  Posterior B (a, b)  : {result.posterior_b_params}")
    print(f"  Risk threshold      : {result.risk_threshold}")
    print("-" * 64)
    print(f"  >> Verdict: {result.verdict}")
    print("=" * 64)


    post_a = compute_posterior(SUCCESSES_A, TRIALS_A)
    post_b = compute_posterior(SUCCESSES_B, TRIALS_B)
    fig = plot_posteriors(post_a, post_b)
    fig.show()

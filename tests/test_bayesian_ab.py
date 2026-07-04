"""
Test Suite: Bayesian A/B Testing
==================================

Validates the Bayesian A/B testing module including posterior computation,
probability calculations, expected loss, and verdict logic.
"""

import pytest
import numpy as np
from scipy import stats as sp_stats
from src.bayesian_ab import (
    BayesianABResult,
    compute_posterior,
    probability_b_beats_a,
    compute_expected_loss,
    compute_credible_interval,
    run_bayesian_ab,
    plot_posteriors,
)


class TestComputePosterior:
    """Tests for the Beta-Binomial posterior update."""

    def test_uninformative_prior(self):
        """With Beta(1,1) prior and data, posterior should be Beta(1+s, 1+f)."""
        posterior = compute_posterior(successes=50, trials=100, prior_alpha=1, prior_beta=1)
        # Posterior should be Beta(51, 51) → mean ≈ 0.5
        assert abs(posterior.mean() - 0.5) < 0.01

    def test_posterior_mean_matches_formula(self):
        """Posterior mean = (alpha + successes) / (alpha + beta + trials)."""
        posterior = compute_posterior(successes=30, trials=100)
        expected_mean = (1 + 30) / (1 + 1 + 100)
        assert abs(posterior.mean() - expected_mean) < 1e-6

    def test_more_data_narrows_posterior(self):
        """More data should produce a tighter posterior (smaller variance)."""
        post_small = compute_posterior(successes=10, trials=100)
        post_large = compute_posterior(successes=100, trials=1000)
        assert post_large.var() < post_small.var()

    def test_returns_frozen_distribution(self):
        """Should return a frozen scipy.stats distribution."""
        posterior = compute_posterior(successes=50, trials=200)
        # Should be able to call .rvs(), .pdf(), .mean()
        assert hasattr(posterior, "rvs")
        assert hasattr(posterior, "pdf")
        assert hasattr(posterior, "mean")


class TestProbabilityBBeatsA:
    """Tests for P(B > A) Monte Carlo estimation."""

    def test_identical_posteriors_near_half(self):
        """Identical posteriors should give P(B>A) ≈ 0.5."""
        post_a = compute_posterior(100, 1000)
        post_b = compute_posterior(100, 1000)
        prob = probability_b_beats_a(post_a, post_b)
        assert abs(prob - 0.5) < 0.03  # Monte Carlo noise

    def test_clearly_better_b(self):
        """When B has much higher rate, P(B>A) should be high."""
        post_a = compute_posterior(100, 1000)  # 10%
        post_b = compute_posterior(200, 1000)  # 20%
        prob = probability_b_beats_a(post_a, post_b)
        assert prob > 0.99

    def test_clearly_better_a(self):
        """When A has much higher rate, P(B>A) should be low."""
        post_a = compute_posterior(200, 1000)
        post_b = compute_posterior(100, 1000)
        prob = probability_b_beats_a(post_a, post_b)
        assert prob < 0.01

    def test_returns_between_zero_and_one(self):
        """Probability must be in [0, 1]."""
        post_a = compute_posterior(50, 500)
        post_b = compute_posterior(55, 500)
        prob = probability_b_beats_a(post_a, post_b)
        assert 0 <= prob <= 1


class TestExpectedLoss:
    """Tests for expected loss calculations."""

    def test_symmetric_case(self):
        """With identical posteriors, losses should be approximately equal."""
        post_a = compute_posterior(100, 1000)
        post_b = compute_posterior(100, 1000)
        loss_a, loss_b = compute_expected_loss(post_a, post_b)
        assert abs(loss_a - loss_b) < 0.005

    def test_clear_winner_low_loss(self):
        """When B is clearly better, expected loss of choosing B should be tiny."""
        post_a = compute_posterior(100, 1000)
        post_b = compute_posterior(200, 1000)
        loss_a, loss_b = compute_expected_loss(post_a, post_b)
        assert loss_b < 0.001
        assert loss_a > loss_b

    def test_losses_non_negative(self):
        """Expected losses should always be non-negative."""
        post_a = compute_posterior(80, 500)
        post_b = compute_posterior(90, 500)
        loss_a, loss_b = compute_expected_loss(post_a, post_b)
        assert loss_a >= 0
        assert loss_b >= 0


class TestCredibleInterval:
    """Tests for credible interval computation."""

    def test_95_credible_interval(self):
        """95% CI should contain the posterior mean."""
        posterior = compute_posterior(100, 1000)
        ci = compute_credible_interval(posterior, credibility=0.95)
        assert ci[0] < posterior.mean() < ci[1]

    def test_wider_interval_with_less_data(self):
        """Less data should produce a wider credible interval."""
        post_small = compute_posterior(10, 100)
        post_large = compute_posterior(100, 1000)
        ci_small = compute_credible_interval(post_small)
        ci_large = compute_credible_interval(post_large)
        width_small = ci_small[1] - ci_small[0]
        width_large = ci_large[1] - ci_large[0]
        assert width_small > width_large

    def test_returns_tuple_of_two(self):
        """Should return a tuple with (lower, upper)."""
        posterior = compute_posterior(50, 500)
        ci = compute_credible_interval(posterior)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]


class TestRunBayesianAB:
    """Tests for the full Bayesian A/B test orchestrator."""

    def test_returns_result_dataclass(self):
        """Should return a BayesianABResult."""
        result = run_bayesian_ab(
            successes_a=100, trials_a=1000,
            successes_b=120, trials_b=1000,
        )
        assert isinstance(result, BayesianABResult)

    def test_clear_b_winner_verdict(self):
        """When B clearly wins, verdict should indicate B."""
        result = run_bayesian_ab(
            successes_a=100, trials_a=1000,
            successes_b=200, trials_b=1000,
        )
        assert result.prob_b_beats_a > 0.95
        assert "B" in result.verdict or "winner" in result.verdict.lower()

    def test_inconclusive_with_similar_rates(self):
        """Similar rates should produce an inconclusive or moderate verdict."""
        result = run_bayesian_ab(
            successes_a=100, trials_a=1000,
            successes_b=102, trials_b=1000,
        )
        assert result.prob_b_beats_a < 0.95
        assert result.prob_a_beats_b < 0.95

    def test_probabilities_sum_to_one(self):
        """P(B>A) + P(A>B) should equal 1."""
        result = run_bayesian_ab(
            successes_a=50, trials_a=500,
            successes_b=60, trials_b=500,
        )
        assert abs(result.prob_b_beats_a + result.prob_a_beats_b - 1.0) < 0.001


class TestPlotPosteriors:
    """Tests for the posterior distribution plot."""

    def test_returns_plotly_figure(self):
        """Should return a Plotly Figure object."""
        import plotly.graph_objects as go

        post_a = compute_posterior(100, 1000)
        post_b = compute_posterior(120, 1000)
        fig = plot_posteriors(post_a, post_b)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        """Figure should have at least 2 traces (A and B distributions)."""
        post_a = compute_posterior(100, 1000)
        post_b = compute_posterior(120, 1000)
        fig = plot_posteriors(post_a, post_b)
        assert len(fig.data) >= 2

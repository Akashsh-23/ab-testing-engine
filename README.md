# 🧪 A/B Testing & Statistical Significance Engine

A focused, well-scoped statistical experimentation tool that covers **sample size estimation**, **hypothesis testing**, **frequentist A/B test evaluation**, and **Bayesian analysis** — built for depth and defensibility.

> **Design principle:** Every feature in this tool is something you can explain and justify line-by-line. It directly demonstrates: *"Strong understanding of the fundamental concepts of Statistical Modelling & Algorithms — Hypothesis testing, Sample size estimation, A/B testing."*

---

## 🎯 What It Does

Input experiment data (or simulate it) and get back:

- ✅ Whether you have **enough sample size** to detect a meaningful effect
- ✅ Whether the observed difference is **statistically significant** (p-value, CI)
- ✅ **Bayesian probability** that B beats A, with expected loss and credible intervals
- ✅ A **plain-English recommendation**: "Ship it," "Not significant — keep testing," or "Needs more data"

---

## 📊 Supported Statistical Tests

| Test | Use Case | Module |
|------|----------|--------|
| Two-sample t-test | Continuous metrics, 2 groups, normal data | `hypothesis_tests.py` |
| Z-test for proportions | Conversion rates, 2 groups | `hypothesis_tests.py` |
| Chi-square test | Independence in contingency tables | `hypothesis_tests.py` |
| One-way ANOVA | Continuous metrics, 3+ groups | `hypothesis_tests.py` |
| Mann-Whitney U | Non-parametric alternative (skewed data) | `hypothesis_tests.py` |
| Beta-Binomial Bayesian | Bayesian conversion rate comparison | `bayesian_ab.py` |

---

## 🏗️ Project Structure

```
ab-testing-engine/
├── src/
│   ├── __init__.py              # Package init
│   ├── sample_size.py           # Sample size calculators (proportions + continuous)
│   ├── hypothesis_tests.py      # Core statistical tests (t-test, z-test, chi-sq, ANOVA, Mann-Whitney)
│   ├── ab_engine.py             # A/B test evaluation engine (lift, CI, p-value, verdict)
│   ├── bayesian_ab.py           # Bayesian A/B testing (Beta-Binomial, P(B>A), expected loss)
│   └── simulator.py             # Synthetic experiment generator + Monte Carlo validation
├── tests/
│   ├── test_sample_size.py      # Cross-validation against statsmodels
│   ├── test_hypothesis_tests.py # Known-output fixtures for each test
│   ├── test_ab_engine.py        # End-to-end scenario tests
│   ├── test_bayesian_ab.py      # Posterior & probability validation
│   └── test_simulation_validation.py  # The "unit test for statistics"
├── reports/
│   └── figures/                 # Generated visualizations
├── app.py                       # Streamlit dashboard (4 tabs)
├── requirements.txt             # Dependencies
├── .gitignore
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ab-testing-engine

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Quick tests (skip slow Monte Carlo validation)
pytest tests/ -v -m "not slow"

# Specific module
pytest tests/test_sample_size.py -v
```

---

## 📖 End-to-End Walkthrough

### Scenario: E-commerce Checkout Optimization

Your team redesigned the checkout flow and wants to know if it improves conversion rates.

**Step 1 — Sample Size Calculation**

```python
from src.sample_size import sample_size_proportions

n = sample_size_proportions(
    baseline_rate=0.10,    # Current: 10% conversion rate
    mde=0.02,              # Want to detect: 2 percentage point increase
    alpha=0.05,            # 5% significance level
    power=0.80             # 80% power
)
print(f"Need {n:,} users per group ({n*2:,} total)")
# Output: Need ~3,623 users per group (7,246 total)
```

**Step 2 — Run the Experiment (simulated)**

```python
from src.simulator import SyntheticExperiment

sim = SyntheticExperiment(
    baseline_rate=0.10, effect_size=0.02,
    sample_size=4000, metric_type="binary", random_seed=42
)
control, treatment = sim.generate_data()
```

**Step 3 — Frequentist A/B Test**

```python
from src.ab_engine import run_ab_test

result = run_ab_test(
    successes_a=int(control.sum()), n_a=len(control),
    successes_b=int(treatment.sum()), n_b=len(treatment),
    metric_type="binary"
)

print(f"Control rate:   {result.control_estimate:.2%}")
print(f"Treatment rate: {result.treatment_estimate:.2%}")
print(f"Relative lift:  {result.relative_lift:+.1f}%")
print(f"P-value:        {result.test_result.p_value:.4f}")
print(f"Verdict:        {result.verdict}")
```

**Step 4 — Bayesian Analysis**

```python
from src.bayesian_ab import run_bayesian_ab

bayes = run_bayesian_ab(
    successes_a=int(control.sum()), trials_a=len(control),
    successes_b=int(treatment.sum()), trials_b=len(treatment),
)

print(f"P(B > A):       {bayes.prob_b_beats_a:.1%}")
print(f"Expected loss:  {bayes.expected_loss_choosing_b:.5f}")
print(f"Verdict:        {bayes.verdict}")
```

**Step 5 — Validate the Engine**

```python
from src.simulator import SyntheticExperiment

sim = SyntheticExperiment(
    baseline_rate=0.10, effect_size=0.02,
    sample_size=3623,  # Use the recommended N
    metric_type="binary", random_seed=42
)
validation = sim.run_power_validation(n_simulations=1000)
print(validation["summary"])
# Observed power should be ≈ 80% — proving the engine is correct!
```

---

## 🔬 The "Unit Test for Statistics"

The simulation validator (Phase 5) is the differentiating piece of this project. It proves the engine is *correct*, not just that it runs:

1. **Power Validation:** At the recommended sample size, the test detects the true effect ~80% of the time (matching the target power).
2. **Type I Error Calibration:** Under the null hypothesis (no effect), the false positive rate ≈ α (5%).
3. **Monotonicity:** Power increases with sample size — as expected from theory.

This is unusual and impressive to demonstrate in an interview.

---

## 🧠 Statistical Methods

### Frequentist Approach
- Tests the null hypothesis H₀: "no difference between groups"
- Reports p-value: probability of observing this result (or more extreme) if H₀ is true
- Decision: reject H₀ if p < α

### Bayesian Approach
- Uses Beta(1,1) uninformative prior (conjugate to Binomial likelihood)
- Updates with observed data → Beta(α + successes, β + failures) posterior
- Reports P(B > A) via Monte Carlo sampling from posteriors
- Reports expected loss: E[max(other − chosen, 0)]
- Decision: based on probability threshold and risk tolerance

### When They Disagree
The frequentist and Bayesian approaches can disagree on borderline cases. The Bayesian approach provides richer information (probability of winning + expected cost of being wrong) while the frequentist approach provides a simple binary decision with controlled error rates.

---

## 📋 Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.9+ |
| Statistics | SciPy, statsmodels |
| Data | pandas, NumPy |
| Bayesian | scipy.stats (Beta distribution) |
| Visualization | Plotly (interactive), Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Testing | pytest |

---

## 📝 Resume Bullet

> Built a statistical experimentation engine covering sample size estimation, hypothesis testing (t-test, z-test, chi-square, ANOVA, Mann-Whitney), frequentist A/B test evaluation, and Bayesian A/B testing (Beta-Binomial posterior, probability-to-win, expected loss); validated statistical correctness via Monte Carlo simulation, confirming the engine achieves ~80% detection power at the calculated sample size; built an interactive Streamlit dashboard for experiment analysis.

---

## 🗺️ Skill-to-JD Mapping

| JD Requirement | Covered By |
|----------------|------------|
| Hypothesis testing, Sample size estimation, A/B testing | Entire project — direct, word-for-word JD match |
| Strong understanding of Statistical Modelling & Algorithms | All core modules (Phases 1–5) |
| Python, pandas, NumPy | All phases |
| Ability to solve complex business problems | Verdict logic, Bayesian framing |
| Present results to a business audience | Streamlit dashboard, plain-English verdicts |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

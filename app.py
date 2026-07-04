"""
A/B Testing & Statistical Significance Engine — Streamlit Dashboard
=====================================================================

Interactive dashboard for experiment analysis with four tabs:
1. Sample Size Calculator — input assumptions, get required sample size
2. Run A/B Test — full frequentist statistical breakdown + verdict
3. Bayesian View — probability, expected loss, posterior distributions
4. Simulation Validator — Monte Carlo power validation

Launch: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark theme ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font & base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(100, 116, 139, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6));
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        min-width: 180px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Verdict banners */
    .verdict-ship {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.12));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 1.3rem 1.8rem;
        margin: 1.2rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #a7f3d0;
    }
    .verdict-nosig {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.10));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 1.3rem 1.8rem;
        margin: 1.2rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #fde68a;
    }
    .verdict-underpower {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.10));
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 14px;
        padding: 1.3rem 1.8rem;
        margin: 1.2rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #fca5a5;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: 0.2px;
    }

    /* Input styling */
    .stNumberInput > label, .stSelectbox > label, .stSlider > label {
        font-weight: 500;
        color: #cbd5e1;
        font-size: 0.88rem;
    }

    /* Divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95));
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

def render_metric_cards(metrics: list):
    """Render a row of metric cards. Each metric is (label, value, sub_text)."""
    cols = st.columns(len(metrics))
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def render_verdict(verdict_code: str, verdict_text: str):
    """Render a styled verdict banner."""
    if verdict_code in ("significant_b", "significant_a"):
        css_class = "verdict-ship"
    elif verdict_code == "underpowered":
        css_class = "verdict-underpower"
    else:
        css_class = "verdict-nosig"
    st.markdown(f'<div class="{css_class}">{verdict_text}</div>', unsafe_allow_html=True)


def create_plotly_theme():
    """Return a common Plotly layout template for consistent styling."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            zerolinecolor="rgba(100,116,139,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(100,116,139,0.15)",
            zerolinecolor="rgba(100,116,139,0.2)",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )


# ── Main app ─────────────────────────────────────────────────────────────────

def main():
    # Hero header
        st.markdown('<div class="hero-title">A/B Testing Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Statistical significance, sample sizing, and Bayesian analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sample Size Calculator",
        "Run A/B Test",
        "Bayesian View",
        "Simulation Validator",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Sample Size Calculator
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Calculate Required Sample Size")
        st.markdown(
            "Determine how many observations you need **per group** to detect "
            "a meaningful effect with statistical confidence."
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_input, col_result = st.columns([1, 1])

        with col_input:
            metric_type_ss = st.selectbox(
                "Metric Type",
                ["Conversion Rate (Binary)", "Continuous Metric"],
                key="ss_metric_type",
            )

            if metric_type_ss == "Conversion Rate (Binary)":
                baseline_rate = st.number_input(
                    "Baseline Conversion Rate",
                    min_value=0.001, max_value=0.999, value=0.10,
                    step=0.01, format="%.3f",
                    help="Current conversion rate (e.g., 0.10 = 10%)",
                    key="ss_baseline_rate",
                )
                mde = st.number_input(
                    "Minimum Detectable Effect (absolute)",
                    min_value=0.001, max_value=0.50, value=0.02,
                    step=0.005, format="%.3f",
                    help="Smallest effect you want to detect (e.g., 0.02 = 2pp increase)",
                    key="ss_mde_binary",
                )
            else:
                baseline_mean = st.number_input(
                    "Baseline Mean", value=50.0, step=1.0,
                    help="Current mean of your metric",
                    key="ss_baseline_mean",
                )
                baseline_std = st.number_input(
                    "Baseline Std Dev", min_value=0.01, value=10.0, step=0.5,
                    help="Standard deviation of your metric",
                    key="ss_baseline_std",
                )
                mde_cont = st.number_input(
                    "Minimum Detectable Effect (absolute)",
                    min_value=0.01, value=3.0, step=0.5,
                    help="Smallest mean difference you want to detect",
                    key="ss_mde_cont",
                )

            alpha_ss = st.slider(
                "Significance Level (α)", 0.01, 0.10, 0.05, 0.01,
                help="Probability of false positive (Type I error)",
                key="ss_alpha",
            )
            power_ss = st.slider(
                "Statistical Power (1-β)", 0.70, 0.99, 0.80, 0.01,
                help="Probability of detecting a true effect",
                key="ss_power",
            )

            calculate = st.button("Calculate Sample Size", use_container_width=True, type="primary")

        with col_result:
            if calculate:
                from src.sample_size import sample_size_proportions, sample_size_continuous

                try:
                    if metric_type_ss == "Conversion Rate (Binary)":
                        n = sample_size_proportions(
                            baseline_rate=baseline_rate, mde=mde,
                            alpha=alpha_ss, power=power_ss,
                        )
                        treatment_rate = baseline_rate + mde
                        sub_text = f"{baseline_rate:.1%} → {treatment_rate:.1%}"
                    else:
                        n = sample_size_continuous(
                            baseline_mean=baseline_mean, baseline_std=baseline_std,
                            mde=mde_cont, alpha=alpha_ss, power=power_ss,
                        )
                        sub_text = f"Cohen's d = {mde_cont/baseline_std:.2f}"

                    total_n = n * 2

                    render_metric_cards([
                        ("Per Group", f"{n:,}", "observations needed"),
                        ("Total", f"{total_n:,}", "across both groups"),
                        ("Power", f"{power_ss:.0%}", f"α = {alpha_ss}"),
                    ])

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # Power curve visualization
                    st.markdown("#### Power Curve")
                    sample_range = np.linspace(max(20, n * 0.1), n * 2.5, 50).astype(int)
                    from scipy import stats as sp_stats

                    if metric_type_ss == "Conversion Rate (Binary)":
                        from statsmodels.stats.proportion import proportion_effectsize
                        from statsmodels.stats.power import NormalIndPower
                        es = proportion_effectsize(baseline_rate, baseline_rate + mde)
                        analysis = NormalIndPower()
                        powers = [
                            analysis.solve_power(effect_size=es, nobs1=nn, alpha=alpha_ss, alternative="two-sided")
                            for nn in sample_range
                        ]
                    else:
                        from statsmodels.stats.power import TTestIndPower
                        cohens_d = mde_cont / baseline_std
                        analysis = TTestIndPower()
                        powers = [
                            analysis.solve_power(effect_size=cohens_d, nobs1=nn, alpha=alpha_ss, alternative="two-sided")
                            for nn in sample_range
                        ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sample_range, y=powers,
                        mode='lines',
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102,126,234,0.12)',
                        name='Power',
                    ))
                    # Target power line
                    fig.add_hline(y=power_ss, line_dash="dash", line_color="#f093fb",
                                  annotation_text=f"Target: {power_ss:.0%}")
                    # Required N line
                    fig.add_vline(x=n, line_dash="dash", line_color="#10b981",
                                  annotation_text=f"Required N: {n:,}")
                    fig.update_layout(
                        **create_plotly_theme(),
                        title="Power vs. Sample Size per Group",
                        xaxis_title="Sample Size per Group",
                        yaxis_title="Statistical Power",
                        yaxis_range=[0, 1.05],
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.info(f"**Interpretation:** You need at least **{n:,}** observations per group "
                            f"({total_n:,} total) to detect an effect of {sub_text} with "
                            f"{power_ss:.0%} power at α={alpha_ss}.")

                except Exception as e:
                    st.error(f"Error computing sample size: {e}")
            else:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #e2e8f0; margin-top: 0;">👈 Set your parameters</h4>
                    <p style="color: #94a3b8;">
                        Configure your experiment assumptions on the left and click
                        <strong>Calculate</strong> to see the required sample size and power curve.
                    </p>
                    <p style="color: #64748b; font-size: 0.85rem;">
                        <strong>Tips:</strong><br>
                        • Start with your current conversion rate as the baseline<br>
                        • Set MDE to the smallest improvement worth detecting<br>
                        • 80% power and 5% significance are standard defaults
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Run A/B Test
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Run A/B Test Analysis")
        st.markdown(
            "Enter your experiment data to get a full statistical analysis with a clear verdict."
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        data_source = st.radio(
            "Data Source",
            ["Manual Entry", "Simulated Data", "Upload CSV"],
            horizontal=True,
            key="ab_data_source",
        )

        if data_source == "Manual Entry":
            metric_type_ab = st.selectbox(
                "Metric Type",
                ["Binary (Conversions)", "Continuous (Means)"],
                key="ab_metric_type",
            )

            col_a, col_b = st.columns(2)

            if metric_type_ab == "Binary (Conversions)":
                with col_a:
                    st.markdown("**🔵 Control (A)**")
                    n_a = st.number_input("Sample Size (A)", min_value=1, value=1000, step=100, key="ab_n_a")
                    conv_a = st.number_input("Conversions (A)", min_value=0, value=120, step=10, key="ab_conv_a")
                with col_b:
                    st.markdown("**🟢 Treatment (B)**")
                    n_b = st.number_input("Sample Size (B)", min_value=1, value=1000, step=100, key="ab_n_b")
                    conv_b = st.number_input("Conversions (B)", min_value=0, value=145, step=10, key="ab_conv_b")
            else:
                with col_a:
                    st.markdown("**🔵 Control (A)**")
                    mean_a = st.number_input("Mean (A)", value=50.0, step=1.0, key="ab_mean_a")
                    std_a = st.number_input("Std Dev (A)", min_value=0.01, value=10.0, step=0.5, key="ab_std_a")
                    size_a = st.number_input("Sample Size (A)", min_value=10, value=500, step=50, key="ab_size_a")
                with col_b:
                    st.markdown("**🟢 Treatment (B)**")
                    mean_b = st.number_input("Mean (B)", value=53.0, step=1.0, key="ab_mean_b")
                    std_b = st.number_input("Std Dev (B)", min_value=0.01, value=10.0, step=0.5, key="ab_std_b")
                    size_b = st.number_input("Sample Size (B)", min_value=10, value=500, step=50, key="ab_size_b")

        elif data_source == "Simulated Data":
            metric_type_ab = st.selectbox(
                "Metric Type",
                ["Binary (Conversions)", "Continuous (Means)"],
                key="ab_sim_metric_type",
            )
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                if metric_type_ab == "Binary (Conversions)":
                    sim_baseline = st.number_input("Baseline Rate", 0.01, 0.99, 0.10, 0.01, key="ab_sim_baseline")
                    sim_effect = st.number_input("True Effect Size", -0.20, 0.20, 0.025, 0.005,
                                                  format="%.3f", key="ab_sim_effect")
                else:
                    sim_baseline = st.number_input("Baseline Mean", value=50.0, step=1.0, key="ab_sim_mean")
                    sim_std = st.number_input("Std Dev", 0.01, 100.0, 10.0, 0.5, key="ab_sim_std")
                    sim_effect = st.number_input("True Effect", -20.0, 20.0, 3.0, 0.5, key="ab_sim_effect_cont")
            with col_sim2:
                sim_n = st.number_input("Sample Size per Group", 50, 50000, 1000, 100, key="ab_sim_n")
                sim_seed = st.number_input("Random Seed", 0, 99999, 42, key="ab_sim_seed")

        else:  # Upload CSV
            metric_type_ab = "Binary (Conversions)"
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="ab_csv")
            if uploaded_file:
                st.info("Expected columns: 'group' (control/treatment) and 'value' (0/1 or continuous)")

        alpha_ab = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, key="ab_alpha")

        run_test = st.button("Run A/B Test", use_container_width=True, type="primary")

        if run_test:
            from src.ab_engine import run_ab_test

            try:
                # Prepare data based on source
                if data_source == "Manual Entry":
                    if metric_type_ab == "Binary (Conversions)":
                        result = run_ab_test(
                            successes_a=conv_a, n_a=n_a,
                            successes_b=conv_b, n_b=n_b,
                            metric_type="binary", alpha=alpha_ab,
                        )
                    else:
                        np.random.seed(42)
                        data_a = np.random.normal(mean_a, std_a, size_a)
                        data_b = np.random.normal(mean_b, std_b, size_b)
                        result = run_ab_test(
                            data_a=data_a, data_b=data_b,
                            metric_type="continuous", alpha=alpha_ab,
                        )

                elif data_source == "Simulated Data":
                    from src.simulator import SyntheticExperiment
                    sim = SyntheticExperiment(
                        baseline_rate=sim_baseline,
                        effect_size=sim_effect,
                        sample_size=sim_n,
                        metric_type="binary" if metric_type_ab == "Binary (Conversions)" else "continuous",
                        baseline_std=sim_std if metric_type_ab != "Binary (Conversions)" else 1.0,
                        random_seed=sim_seed,
                    )
                    control, treatment = sim.generate_data()
                    if metric_type_ab == "Binary (Conversions)":
                        result = run_ab_test(
                            successes_a=int(control.sum()), n_a=len(control),
                            successes_b=int(treatment.sum()), n_b=len(treatment),
                            metric_type="binary", alpha=alpha_ab,
                        )
                    else:
                        result = run_ab_test(
                            data_a=control, data_b=treatment,
                            metric_type="continuous", alpha=alpha_ab,
                        )

                else:  # CSV Upload
                    if uploaded_file is None:
                        st.warning("Please upload a CSV file first.")
                        st.stop()
                    df = pd.read_csv(uploaded_file)
                    control_data = df[df["group"] == "control"]["value"].values
                    treatment_data = df[df["group"] == "treatment"]["value"].values
                    # Auto-detect metric type
                    unique_vals = np.unique(np.concatenate([control_data, treatment_data]))
                    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        result = run_ab_test(
                            successes_a=int(control_data.sum()), n_a=len(control_data),
                            successes_b=int(treatment_data.sum()), n_b=len(treatment_data),
                            metric_type="binary", alpha=alpha_ab,
                        )
                    else:
                        result = run_ab_test(
                            data_a=control_data, data_b=treatment_data,
                            metric_type="continuous", alpha=alpha_ab,
                        )

                # ── Display results ──────────────────────────────────────────
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # Verdict banner
                render_verdict(result.verdict_code, result.verdict)

                # Key metrics
                if result.metric_type == "binary":
                    render_metric_cards([
                        ("Control Rate", f"{result.control_estimate:.2%}", f"n = {result.control_n:,}"),
                        ("Treatment Rate", f"{result.treatment_estimate:.2%}", f"n = {result.treatment_n:,}"),
                        ("Relative Lift", f"{result.relative_lift:+.1f}%", f"Absolute: {result.absolute_lift:+.4f}"),
                        ("P-Value", f"{result.test_result.p_value:.4f}", f"α = {alpha_ab}"),
                    ])
                else:
                    render_metric_cards([
                        ("Control Mean", f"{result.control_estimate:.2f}", f"n = {result.control_n:,}"),
                        ("Treatment Mean", f"{result.treatment_estimate:.2f}", f"n = {result.treatment_n:,}"),
                        ("Relative Lift", f"{result.relative_lift:+.1f}%", f"Absolute: {result.absolute_lift:+.2f}"),
                        ("P-Value", f"{result.test_result.p_value:.4f}", f"α = {alpha_ab}"),
                    ])

                # Detailed stats
                col_detail1, col_detail2 = st.columns(2)

                with col_detail1:
                    st.markdown("#### Statistical Details")
                    details = {
                        "Test Used": result.test_result.test_name,
                        "Test Statistic": f"{result.test_result.statistic:.4f}",
                        "P-Value": f"{result.test_result.p_value:.6f}",
                        "Significant": "Yes" if result.test_result.is_significant else "No",
                        f"{result.effect_size_label}": f"{result.effect_size:.4f}",
                        "95% CI (difference)": f"({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})",
                    }
                    st.table(pd.DataFrame(details.items(), columns=["Metric", "Value"]))

                with col_detail2:
                    st.markdown("#### Power Analysis")
                    ss = result.sample_size_check
                    power_details = {
                        "Sufficient Sample": "Yes" if ss["is_sufficient"] else "No",
                        "Actual N (min group)": f"{ss['actual']:,}",
                        "Required N": f"{ss['required']:,}" if ss['required'] is not None else "N/A",
                        "Shortfall": f"{ss['shortfall']:,}" if ss['shortfall'] > 0 else "0",
                    }
                    st.table(pd.DataFrame(power_details.items(), columns=["Metric", "Value"]))
                    st.caption(ss["message"])

                # Confidence Interval visualization
                st.markdown("#### Confidence Interval on Difference")
                ci_low, ci_high = result.confidence_interval
                ci_mid = result.absolute_lift

                fig_ci = go.Figure()
                fig_ci.add_trace(go.Scatter(
                    x=[ci_low, ci_high], y=[0, 0],
                    mode='lines', line=dict(color='#667eea', width=6),
                    name='95% CI',
                ))
                fig_ci.add_trace(go.Scatter(
                    x=[ci_mid], y=[0],
                    mode='markers',
                    marker=dict(size=14, color='#f093fb', symbol='diamond'),
                    name='Point Estimate',
                ))
                fig_ci.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                 annotation_text="No Effect")
                fig_ci.update_layout(
                    **create_plotly_theme(),
                    title="95% Confidence Interval on Treatment − Control",
                    xaxis_title="Difference",
                    yaxis_visible=False,
                    height=200,
                    showlegend=True,
                )
                st.plotly_chart(fig_ci, use_container_width=True)

                st.markdown(f"*{result.test_result.interpretation}*")

            except Exception as e:
                st.error(f"Error running A/B test: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Bayesian View
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Bayesian A/B Test Analysis")
        st.markdown(
            "Get the **probability that B beats A**, expected loss, and posterior distributions "
            "using the Beta-Binomial conjugate model."
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_bay_input, col_bay_result = st.columns([1, 1.5])

        with col_bay_input:
            st.markdown("**🔵 Control (A)**")
            bay_conv_a = st.number_input("Conversions (A)", 0, 100000, 120, 10, key="bay_conv_a")
            bay_n_a = st.number_input("Sample Size (A)", 1, 1000000, 1000, 100, key="bay_n_a")

            st.markdown("**🟢 Treatment (B)**")
            bay_conv_b = st.number_input("Conversions (B)", 0, 100000, 145, 10, key="bay_conv_b")
            bay_n_b = st.number_input("Sample Size (B)", 1, 1000000, 1000, 100, key="bay_n_b")

            st.markdown("**Prior Settings**")
            prior_alpha = st.number_input("Prior α (successes)", 0.1, 100.0, 1.0, 0.5, key="bay_prior_a")
            prior_beta = st.number_input("Prior β (failures)", 0.1, 100.0, 1.0, 0.5, key="bay_prior_b")
            risk_threshold = st.number_input("Risk Threshold", 0.001, 0.10, 0.01, 0.005,
                                              format="%.3f", key="bay_risk")

            run_bayesian = st.button("Run Bayesian Analysis", use_container_width=True, type="primary")

        with col_bay_result:
            if run_bayesian:
                from src.bayesian_ab import run_bayesian_ab, compute_posterior, plot_posteriors

                try:
                    bayes_result = run_bayesian_ab(
                        successes_a=bay_conv_a, trials_a=bay_n_a,
                        successes_b=bay_conv_b, trials_b=bay_n_b,
                        prior_alpha=prior_alpha, prior_beta=prior_beta,
                        risk_threshold=risk_threshold,
                    )

                    # Verdict
                    if bayes_result.prob_b_beats_a > 0.5:
                        verdict_css = "verdict-ship" if bayes_result.prob_b_beats_a > 0.95 else "verdict-nosig"
                    else:
                        verdict_css = "verdict-ship" if bayes_result.prob_a_beats_b > 0.95 else "verdict-nosig"
                    st.markdown(
                        f'<div class="{verdict_css}">{bayes_result.verdict}</div>',
                        unsafe_allow_html=True,
                    )

                    # Key probabilities
                    render_metric_cards([
                        ("P(B > A)", f"{bayes_result.prob_b_beats_a:.1%}", "Probability B wins"),
                        ("P(A > B)", f"{bayes_result.prob_a_beats_b:.1%}", "Probability A wins"),
                    ])

                    render_metric_cards([
                        ("E[Loss|A]", f"{bayes_result.expected_loss_choosing_a:.5f}",
                         "Expected loss if choosing A"),
                        ("E[Loss|B]", f"{bayes_result.expected_loss_choosing_b:.5f}",
                         "Expected loss if choosing B"),
                    ])

                    # Credible intervals
                    st.markdown("#### 95% Credible Intervals")
                    ci_table = pd.DataFrame({
                        "Variant": ["Control (A)", "Treatment (B)"],
                        "Lower": [f"{bayes_result.credible_interval_a[0]:.4f}",
                                  f"{bayes_result.credible_interval_b[0]:.4f}"],
                        "Upper": [f"{bayes_result.credible_interval_a[1]:.4f}",
                                  f"{bayes_result.credible_interval_b[1]:.4f}"],
                        "Posterior (α, β)": [
                            f"Beta({bayes_result.posterior_a_params[0]:.0f}, {bayes_result.posterior_a_params[1]:.0f})",
                            f"Beta({bayes_result.posterior_b_params[0]:.0f}, {bayes_result.posterior_b_params[1]:.0f})",
                        ],
                    })
                    st.table(ci_table)

                    # Posterior plot
                    st.markdown("#### Posterior Distributions")
                    post_a = compute_posterior(bay_conv_a, bay_n_a, prior_alpha, prior_beta)
                    post_b = compute_posterior(bay_conv_b, bay_n_b, prior_alpha, prior_beta)
                    fig = plot_posteriors(post_a, post_b)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)

                    # Frequentist comparison
                    st.markdown("#### Frequentist vs. Bayesian Comparison")
                    from src.ab_engine import run_ab_test
                    freq_result = run_ab_test(
                        successes_a=bay_conv_a, n_a=bay_n_a,
                        successes_b=bay_conv_b, n_b=bay_n_b,
                        metric_type="binary",
                    )

                    comparison = pd.DataFrame({
                        "Approach": ["Frequentist", "Bayesian"],
                        "Conclusion": [
                            "Significant" if freq_result.test_result.is_significant else "Not Significant",
                            f"P(B>A) = {bayes_result.prob_b_beats_a:.1%}",
                        ],
                        "Key Metric": [
                            f"p-value = {freq_result.test_result.p_value:.4f}",
                            f"E[Loss|B] = {bayes_result.expected_loss_choosing_b:.5f}",
                        ],
                        "Verdict": [freq_result.verdict_code, bayes_result.verdict[:50]],
                    })
                    st.table(comparison)

                    agree = (
                        (freq_result.test_result.is_significant and bayes_result.prob_b_beats_a > 0.95)
                        or (not freq_result.test_result.is_significant and bayes_result.prob_b_beats_a <= 0.95)
                    )
                    if agree:
                        st.success("**Frequentist and Bayesian approaches agree** on this experiment.")
                    else:
                        st.warning(
                            "**Approaches disagree.** This can happen when the effect is borderline. "
                            "The Bayesian approach provides a richer picture by quantifying the "
                            "probability of each variant being better and the expected cost of a wrong decision."
                        )

                except Exception as e:
                    st.error(f"Error in Bayesian analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #e2e8f0; margin-top: 0;">Bayesian A/B Testing</h4>
                    <p style="color: #94a3b8;">
                        Unlike frequentist testing, Bayesian analysis tells you the
                        <strong>probability that B is better than A</strong> and the
                        <strong>expected cost</strong> of making the wrong decision.
                    </p>
                    <p style="color: #64748b; font-size: 0.85rem;">
                        <strong>How it works:</strong><br>
                        • Uses Beta(1,1) uninformative prior (default)<br>
                        • Updates with observed conversions → Beta posterior<br>
                        • Monte Carlo sampling to estimate P(B > A)<br>
                        • Expected loss = E[max(other − chosen, 0)]
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Simulation Validator
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### Monte Carlo Simulation Validator")
        st.markdown(
            "Prove the engine is **correct** by simulating experiments with a known true effect "
            "and verifying that detection rates match the expected power."
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_sim_input, col_sim_result = st.columns([1, 1.5])

        with col_sim_input:
            sim_metric = st.selectbox("Metric Type", ["Binary", "Continuous"], key="sim_metric")
            sim_val_baseline = st.number_input(
                "Baseline Rate" if sim_metric == "Binary" else "Baseline Mean",
                value=0.10 if sim_metric == "Binary" else 50.0,
                step=0.01 if sim_metric == "Binary" else 1.0,
                key="sim_val_baseline",
            )
            if sim_metric == "Continuous":
                sim_val_std = st.number_input("Std Dev", 0.01, 100.0, 10.0, 0.5, key="sim_val_std")
            sim_val_effect = st.number_input(
                "True Effect Size",
                value=0.02 if sim_metric == "Binary" else 3.0,
                step=0.005 if sim_metric == "Binary" else 0.5,
                format="%.3f" if sim_metric == "Binary" else "%.1f",
                key="sim_val_effect",
            )
            sim_val_n = st.number_input("Sample Size per Group", 50, 50000, 1000, 100, key="sim_val_n")
            sim_val_nsim = st.number_input("Number of Simulations", 100, 5000, 500, 100, key="sim_val_nsim")
            sim_val_alpha = st.slider("Alpha", 0.01, 0.10, 0.05, 0.01, key="sim_val_alpha")
            sim_val_power = st.slider("Target Power", 0.70, 0.99, 0.80, 0.01, key="sim_val_power")

            run_simulation = st.button("Run Simulation", use_container_width=True, type="primary")

        with col_sim_result:
            if run_simulation:
                from src.simulator import SyntheticExperiment, run_null_validation

                try:
                    with st.spinner("Running Monte Carlo simulations..."):
                        sim = SyntheticExperiment(
                            baseline_rate=sim_val_baseline,
                            effect_size=sim_val_effect,
                            sample_size=sim_val_n,
                            metric_type="binary" if sim_metric == "Binary" else "continuous",
                            baseline_std=sim_val_std if sim_metric == "Continuous" else 1.0,
                            alpha=sim_val_alpha,
                            power=sim_val_power,
                            random_seed=42,
                        )

                        result = sim.run_power_validation(n_simulations=sim_val_nsim)

                    # Key results
                    power_diff = abs(result["observed_power"] - sim_val_power)
                    power_ok = power_diff < 0.08

                    render_metric_cards([
                        ("Observed Power", f"{result['observed_power']:.1%}",
                         "Close to target" if power_ok else "Differs from target"),
                        ("Target Power", f"{sim_val_power:.0%}", f"α = {sim_val_alpha}"),
                        ("Detections", f"{result['correct_detections']}/{result['total_simulations']}",
                         "Correct effect detections"),
                    ])

                    if power_ok:
                        st.success(
                            f"**Validation passed!** Observed power ({result['observed_power']:.1%}) "
                            f"is within ±8pp of the target ({sim_val_power:.0%}). "
                            "The statistical engine is correctly calibrated."
                        )
                    else:
                        st.warning(
                            f"Observed power ({result['observed_power']:.1%}) differs from "
                            f"target ({sim_val_power:.0%}). This may indicate the sample size "
                            "doesn't match the recommended value for this effect size."
                        )

                    # P-value distribution
                    st.markdown("#### P-Value Distribution")
                    fig_pval = go.Figure()
                    fig_pval.add_trace(go.Histogram(
                        x=result["p_values"],
                        nbinsx=50,
                        marker_color='rgba(102,126,234,0.6)',
                        marker_line=dict(color='#667eea', width=1),
                        name='P-values',
                    ))
                    fig_pval.add_vline(
                        x=sim_val_alpha, line_dash="dash", line_color="#ef4444",
                        annotation_text=f"α = {sim_val_alpha}",
                    )
                    fig_pval.update_layout(
                        **create_plotly_theme(),
                        title="Distribution of P-Values Across Simulations",
                        xaxis_title="P-Value",
                        yaxis_title="Count",
                        height=350,
                    )
                    st.plotly_chart(fig_pval, use_container_width=True)

                    # Null hypothesis validation
                    st.markdown("#### Type I Error Calibration (Null Hypothesis)")
                    with st.spinner("Running null validation..."):
                        null_result = run_null_validation(
                            metric_type="binary" if sim_metric == "Binary" else "continuous",
                            baseline_rate=sim_val_baseline,
                            sample_size=sim_val_n,
                            alpha=sim_val_alpha,
                            n_simulations=min(sim_val_nsim, 500),
                            random_seed=123,
                        )

                    render_metric_cards([
                        ("False Positive Rate", f"{null_result['false_positive_rate']:.1%}",
                         f"Expected: {sim_val_alpha:.0%}"),
                        ("Calibrated", "Yes" if null_result["is_calibrated"] else "No",
                         "FPR ≈ α"),
                    ])

                    if null_result["is_calibrated"]:
                        st.success("**Type I error is properly controlled.** False positive rate ≈ α.")
                    else:
                        st.warning("False positive rate deviates from α. Check test implementation.")

                    # Summary
                    st.markdown("#### Full Simulation Summary")
                    st.code(result["summary"])

                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #e2e8f0; margin-top: 0;">Statistical Model Validation</h4>
                    <p style="color: #94a3b8;">
                        This validator verifies the mathematical correctness of the
                        statistical calculations under controlled parameters:
                    </p>
                    <ul style="color: #94a3b8; margin-left: 1rem;">
                        <li>Observed statistical power matches theoretical expectations (~80% target power)</li>
                        <li>Type I error control matches alpha level (~5% false positive rate under the null hypothesis)</li>
                        <li>Monotonicity of power as sample size scales</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color:#475569; font-size:0.82rem;">'
        'A/B Testing & Statistical Significance Engine • '
        'Built with Python, SciPy, Streamlit & Plotly'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

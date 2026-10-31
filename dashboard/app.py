"""
Streamlit Web Application: Machine Learning-Based Risk-Return Profiling of Indian Mutual Funds
Run with: streamlit run dashboard/app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Indian Mutual Funds ML Risk-Return Profiler",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ml_results", "clustered_funds_complete.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "ml_results", "kmeans_evaluation_metrics.csv")
PROFILES_PATH = os.path.join(BASE_DIR, "data", "ml_results", "cluster_profiles_summary.csv")
PCA_VAR_PATH = os.path.join(BASE_DIR, "data", "ml_results", "pca_explained_variance.csv")
PCA_LOAD_PATH = os.path.join(BASE_DIR, "data", "ml_results", "pca_loadings.csv")

@st.cache_data
def load_all_data():
    df = pd.read_csv(DATA_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    profiles_df = pd.read_csv(PROFILES_PATH)
    pca_var_df = pd.read_csv(PCA_VAR_PATH)
    pca_load_df = pd.read_csv(PCA_LOAD_PATH, index_col=0)
    return df, eval_df, profiles_df, pca_var_df, pca_load_df

try:
    df, eval_df, profiles_df, pca_var_df, pca_load_df = load_all_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}. Please ensure `run_ml_analysis.py` has been executed.")
    st.stop()

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: bold; color: #1B365D; margin-bottom: 5px; }
    .sub-header { font-size: 15px; color: #555555; margin-bottom: 25px; }
    .metric-card { background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #1B365D; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "Executive Dashboard",
        "Interactive Cluster Map (PCA & Risk-Return)",
        "Cluster Profiles & Radar Analytics",
        "Fund Explorer & Peer Comparison",
        "Algorithm Evaluation (K=2-8 & Benchmarks)",
        "Methodology & Data Dictionary"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Details:**
- **Universe:** 132 Indian Equity Schemes
- **Time Horizon:** 5 Years (2021-2026)
- **Benchmark:** NIFTY 500 TRI
- **Risk-Free Rate:** 6.50% p.a.
""")

# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================
if page == "Executive Dashboard":
    st.markdown('<div class="main-header">Machine Learning-Based Risk–Return Profiling of Indian Mutual Funds</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised machine learning taxonomy classifying 132 Indian equity funds across 12 SEBI categories.</div>', unsafe_allow_html=True)

    # KPI Top Bar
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Equity Funds", f"{len(df)}", "100% Pre-2021 Inception")
    kpi2.metric("Evaluation Period", "5.0 Years", "1,232 Trading Days")
    kpi3.metric("Optimal Clusters (K)", "4 Clusters", "K-Means Silhouette Peak")
    kpi4.metric("PCA Variance (2D)", "88.96%", "PC1: 62.8%, PC2: 26.1%")
    kpi5.metric("Universe Median CAGR", f"{df['cagr_5y_pct'].median():.2f}%", f"Range: {df['cagr_5y_pct'].min():.1f}% to {df['cagr_5y_pct'].max():.1f}%")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Interactive Risk–Return Efficient Frontier")
        fig_scatter = px.scatter(
            df,
            x="annualized_volatility_pct",
            y="cagr_5y_pct",
            color="cluster_profile_label",
            size="beta_nifty500",
            hover_name="scheme_name",
            hover_data={
                "category": True,
                "fund_house": True,
                "sharpe_ratio": ":.2f",
                "max_drawdown_pct": ":.2f%",
                "beta_nifty500": ":.2f"
            },
            labels={
                "annualized_volatility_pct": "Annualized Volatility (%)",
                "cagr_5y_pct": "5-Year CAGR (%)",
                "cluster_profile_label": "Cluster Profile",
                "beta_nifty500": "Beta"
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=480
        )
        fig_scatter.add_hline(y=6.5, line_dash="dash", line_color="gray", annotation_text="Rf = 6.50%")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("Cluster Distribution & Share")
        cluster_counts = df["cluster_profile_label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster Profile", "Count"]
        fig_pie = px.pie(
            cluster_counts,
            values="Count",
            names="Cluster Profile",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.45,
            height=480
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Summary of Discovered Financial Profiles")
    profile_cards = [
        ("Core Quality Compounders", "18 Funds (13.6%)", "Mean CAGR: 11.57% | Vol: 12.47% | Sharpe: 0.41 | Beta: 0.81", "Dominant: Large Cap, Large & Mid Cap. Superior downside preservation with mildest drawdowns (-16.3%)."),
        ("Aggressive High-Growth", "43 Funds (32.6%)", "Mean CAGR: 13.45% | Vol: 16.40% | Sharpe: 0.43 | Beta: 1.01", "Dominant: Mid Cap, Small Cap. Highest capital growth compensated by elevated volatility and deep drawdowns (-23.3%)."),
        ("Moderate Balanced Growth", "63 Funds (47.7%)", "Mean CAGR: 9.44% | Vol: 14.05% | Sharpe: 0.21 | Beta: 0.93", "Dominant: Flexi Cap, ELSS, Focused. Mainstream equity cohort tracking market benchmarks closely."),
        ("Defensive Capital Preservers", "8 Funds (6.1%)", "Mean CAGR: 4.42% | Vol: 14.22% | Sharpe: -0.16 | Beta: 0.79", "Dominant: Sub-par Value & Sectoral laggards. Lagged the 6.50% risk-free rate with severe historical drawdowns (-27.1%).")
    ]
    cols = st.columns(4)
    for idx, (title, count_str, stats_str, desc_str) in enumerate(profile_cards):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <b>{title}</b><br>
                <small style="color:#1B365D;font-weight:bold;">{count_str}</small><br>
                <hr style="margin:8px 0;">
                <span style="font-size:12px;">{stats_str}</span><br>
                <small style="color:#555;">{desc_str}</small>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE 2: INTERACTIVE CLUSTER MAP (PCA & RISK-RETURN)
# =============================================================================
elif page == "Interactive Cluster Map (PCA & Risk-Return)":
    st.markdown('<div class="main-header">Principal Component Analysis & 2D Manifold Space</div>', unsafe_allow_html=True)
    st.write("PCA reduces 7 correlated financial dimensions into 2 orthogonal axes capturing **88.96% of cumulative variance**.")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Filter Visualization")
        cat_filter = st.multiselect("Filter by SEBI Category:", options=sorted(df["category"].unique()), default=sorted(df["category"].unique()))
        cluster_filter = st.multiselect("Filter by Cluster:", options=sorted(df["cluster_profile_label"].unique()), default=sorted(df["cluster_profile_label"].unique()))
        
    filtered_df = df[(df["category"].isin(cat_filter)) & (df["cluster_profile_label"].isin(cluster_filter))]

    with col1:
        tab_pca, tab_rr, tab_sharpe = st.tabs(["PCA Latent Space (PC1 vs PC2)", "Volatility vs CAGR Space", "Sharpe vs Sortino Ratio"])
        
        with tab_pca:
            fig_pca = px.scatter(
                filtered_df,
                x="PC1",
                y="PC2",
                color="cluster_profile_label",
                hover_name="scheme_name",
                hover_data={"category": True, "cagr_5y_pct": ":.2f%", "sharpe_ratio": ":.2f"},
                labels={"PC1": "PC1: Market Risk & Volatility Factor (62.8%)", "PC2": "PC2: Efficiency & Alpha Factor (26.1%)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=550
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with tab_rr:
            fig_rr = px.scatter(
                filtered_df,
                x="annualized_volatility_pct",
                y="cagr_5y_pct",
                color="cluster_profile_label",
                hover_name="scheme_name",
                hover_data={"category": True, "sharpe_ratio": ":.2f", "max_drawdown_pct": ":.2f%"},
                labels={"annualized_volatility_pct": "Annualized Volatility (%)", "cagr_5y_pct": "5Y CAGR (%)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=550
            )
            st.plotly_chart(fig_rr, use_container_width=True)

        with tab_sharpe:
            fig_ss = px.scatter(
                filtered_df,
                x="sharpe_ratio",
                y="sortino_ratio",
                color="cluster_profile_label",
                hover_name="scheme_name",
                hover_data={"category": True, "downside_deviation_rf_pct": ":.2f%"},
                labels={"sharpe_ratio": "Sharpe Ratio (Total Risk)", "sortino_ratio": "Sortino Ratio (Downside Risk)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=550
            )
            fig_ss.add_shape(type="line", x0=-0.5, y0=-0.5, x1=1.0, y1=1.5, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_ss, use_container_width=True)

    st.subheader("PCA Component Loadings (Eigenvector Weights)")
    st.dataframe(pca_load_df[["PC1", "PC2", "PC3"]].style.background_gradient(cmap="coolwarm", axis=0), use_container_width=True)

# =============================================================================
# PAGE 3: CLUSTER PROFILES & RADAR ANALYTICS
# =============================================================================
elif page == "Cluster Profiles & Radar Analytics":
    st.markdown('<div class="main-header">Cluster Risk–Return Signatures & Radar Profiling</div>', unsafe_allow_html=True)
    
    # Radar Chart
    st.subheader("Multi-Dimensional Radar Chart Comparison")
    radar_features = ["cagr_5y_pct", "annualized_volatility_pct", "downside_deviation_rf_pct", "max_drawdown_pct", "daily_cvar_95_pct", "sharpe_ratio", "beta_nifty500"]
    clean_feature_names = ["5Y CAGR", "Volatility", "Downside Dev", "Max Drawdown", "CVaR (95%)", "Sharpe Ratio", "Market Beta"]

    # Compute standardized means for radar
    std_df = df.copy()
    for feat in radar_features:
        mu = std_df[feat].mean()
        sig = std_df[feat].std()
        std_df[feat] = (std_df[feat] - mu) / sig

    radar_means = std_df.groupby("cluster_profile_label")[radar_features].mean()

    fig_radar = go.Figure()
    for profile in radar_means.index:
        vals = radar_means.loc[profile].tolist()
        vals += vals[:1]  # close
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=clean_feature_names + [clean_feature_names[0]],
            fill='toself',
            name=profile
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2.5, 2.5])),
        showlegend=True,
        height=550
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Comprehensive Cluster Metrics Comparison Table")
    summary_display_cols = [
        "cluster", "profile_label", "fund_count", "pct_of_total",
        "cagr_5y_pct_mean", "annualized_volatility_pct_mean",
        "sharpe_ratio_mean", "sortino_ratio_mean",
        "beta_nifty500_mean", "max_drawdown_pct_mean"
    ]
    st.dataframe(profiles_df[summary_display_cols], use_container_width=True)

    st.subheader("Category Breakdown per Cluster")
    cat_ct = pd.crosstab(df["category"], df["cluster_profile_label"])
    fig_bar = px.bar(
        cat_ct,
        barmode="stack",
        labels={"value": "Number of Funds", "category": "SEBI Fund Category"},
        height=450,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# PAGE 4: FUND EXPLORER & PEER COMPARISON
# =============================================================================
elif page == "Fund Explorer & Peer Comparison":
    st.markdown('<div class="main-header">Individual Fund Deep-Dive & Peer Benchmarking</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        sel_cat = st.selectbox("Select SEBI Category:", ["All Categories"] + sorted(df["category"].unique().tolist()))
        filtered_funds = df if sel_cat == "All Categories" else df[df["category"] == sel_cat]
        sel_fund_name = st.selectbox("Select Scheme:", filtered_funds["scheme_name"].sort_values().tolist())
        fund_row = df[df["scheme_name"] == sel_fund_name].iloc[0]

    with col2:
        st.markdown(f"### {fund_row['scheme_name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Assigned Cluster", f"{fund_row['cluster_profile_label']}")
        c2.metric("5-Year CAGR", f"{fund_row['cagr_5y_pct']:.2f}%")
        c3.metric("Annualized Volatility", f"{fund_row['annualized_volatility_pct']:.2f}%")
        c4.metric("Sharpe Ratio", f"{fund_row['sharpe_ratio']:.2f}")

    st.markdown("---")
    st.subheader("Comparison: Fund vs. Cluster Average vs. Universe Average")
    
    cluster_means = df[df["kmeans_cluster"] == fund_row["kmeans_cluster"]].mean(numeric_only=True)
    universe_means = df.mean(numeric_only=True)

    comp_metrics = [
        ("5-Year CAGR (%)", "cagr_5y_pct"),
        ("Annualized Volatility (%)", "annualized_volatility_pct"),
        ("Downside Deviation (%)", "downside_deviation_rf_pct"),
        ("Maximum Drawdown (%)", "max_drawdown_pct"),
        ("Daily CVaR 95% (%)", "daily_cvar_95_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Market Beta (vs Nifty 500)", "beta_nifty500"),
        ("Tracking Error (%)", "tracking_error_nifty500_pct")
    ]

    comp_rows = []
    for label, col_key in comp_metrics:
        fund_v = fund_row[col_key]
        clust_v = cluster_means[col_key]
        univ_v = universe_means[col_key]
        comp_rows.append({
            "Metric": label,
            "This Fund": round(fund_v, 2),
            "Cluster Average": round(clust_v, 2),
            "Universe Average": round(univ_v, 2),
            "Diff vs Cluster": round(fund_v - clust_v, 2)
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    st.subheader(f"Peer Schemes in Cluster: {fund_row['cluster_profile_label']}")
    peers_df = df[df["kmeans_cluster"] == fund_row["kmeans_cluster"]][
        ["scheme_code", "scheme_name", "category", "fund_house", "cagr_5y_pct", "annualized_volatility_pct", "sharpe_ratio", "beta_nifty500"]
    ].sort_values(by="sharpe_ratio", ascending=False).reset_index(drop=True)
    st.dataframe(peers_df, use_container_width=True)

# =============================================================================
# PAGE 5: ALGORITHM EVALUATION (K=2-8 & BENCHMARKS)
# =============================================================================
elif page == "Algorithm Evaluation (K=2-8 & Benchmarks)":
    st.markdown('<div class="main-header">Statistical Clustering Evaluation & Model Justification</div>', unsafe_allow_html=True)
    st.write("Evaluation of K-Means across K = 2 to 8 using four independent validation metrics.")

    st.dataframe(eval_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = px.line(eval_df, x="k", y="inertia", markers=True, title="Elbow Curve: Within-Cluster Sum of Squares (Inertia)", labels={"inertia": "Inertia (WCSS)", "k": "Clusters (K)"})
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K = 4")
        st.plotly_chart(fig_elbow, use_container_width=True)

        fig_db = px.line(eval_df, x="k", y="davies_bouldin_index", markers=True, title="Davies–Bouldin Index (Lower is Better)", labels={"davies_bouldin_index": "Davies-Bouldin Index", "k": "Clusters (K)"})
        fig_db.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K = 4")
        st.plotly_chart(fig_db, use_container_width=True)

    with c2:
        fig_sil = px.line(eval_df, x="k", y="silhouette_score", markers=True, title="Silhouette Score (Higher is Better)", labels={"silhouette_score": "Silhouette Score", "k": "Clusters (K)"})
        fig_sil.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K = 4")
        st.plotly_chart(fig_sil, use_container_width=True)

        fig_ch = px.line(eval_df, x="k", y="calinski_harabasz_score", markers=True, title="Calinski–Harabasz Variance Ratio (Higher is Better)", labels={"calinski_harabasz_score": "Variance Ratio", "k": "Clusters (K)"})
        fig_ch.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="K = 4")
        st.plotly_chart(fig_ch, use_container_width=True)

    st.markdown("### Comparison Across Algorithms")
    comp_algos = pd.DataFrame([
        {"Algorithm": "K-Means (K=4)", "Silhouette Score": 0.356, "Davies-Bouldin Index": 0.964, "Calinski-Harabasz": 76.6, "Outlier Handling": "Assigns all funds to clusters", "Strengths": "Clean spherical separation, highly interpretable centroids"},
        {"Algorithm": "Hierarchical (Ward, n=4)", "Silhouette Score": 0.264, "Davies-Bouldin Index": 1.101, "Calinski-Harabasz": 64.2, "Outlier Handling": "Hierarchical tree structure", "Strengths": "Reveals deterministic linkage taxonomy, no random initialization"},
        {"Algorithm": "DBSCAN (eps=1.45)", "Silhouette Score": 0.312, "Davies-Bouldin Index": 1.050, "Calinski-Harabasz": 52.4, "Outlier Handling": "Identifies 13 noise/outlier funds (-1)", "Strengths": "Density-based, isolates idiosyncratic funds without forcing into clusters"}
    ])
    st.dataframe(comp_algos, use_container_width=True)

# =============================================================================
# PAGE 6: METHODOLOGY & DATA DICTIONARY
# =============================================================================
elif page == "Methodology & Data Dictionary":
    st.markdown('<div class="main-header">Methodology, Formulas & Variable Specifications</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Mathematical Formulations
    1. **5-Year CAGR (Compound Annual Growth Rate):**
       $$\\text{CAGR} = \\left(\\frac{\\text{NAV}_{\\text{end}}}{\\text{NAV}_{\\text{start}}}\\right)^{\\frac{365.25}{\\Delta \\text{days}}} - 1$$
    2. **Annualized Volatility (Total Risk):**
       $$\\sigma_{\\text{ann}} = \\text{std}(r_{\\text{daily}}) \\times \\sqrt{250}$$
    3. **Downside Deviation (Harmful Risk below $R_f = 6.50\\%$):**
       $$\\sigma_{\\text{down}} = \\sqrt{\\frac{1}{M} \\sum_{t=1}^M \\min\\left(r_t - \\frac{R_f}{250}, 0\\right)^2} \\times \\sqrt{250}$$
    4. **Maximum Drawdown (MDD):**
       $$\\text{MDD} = \\min_{t} \\left( \\frac{\\text{NAV}_t - \\max_{s \\le t} \\text{NAV}_s}{\\max_{s \\le t} \\text{NAV}_s} \\right)$$
    5. **Historical Value at Risk (VaR 95%):**
       $$\\text{VaR}_{95} = -Q_{0.05}(r_{\\text{daily}})$$
    6. **Conditional Value at Risk (CVaR 95% / Expected Shortfall):**
       $$\\text{CVaR}_{95} = -E[r_t \\mid r_t \\le -\\text{VaR}_{95}]$$
    7. **Sharpe Ratio:**
       $$\\text{Sharpe} = \\frac{\\text{CAGR} - R_f}{\\sigma_{\\text{ann}}}$$
    8. **Sortino Ratio:**
       $$\\text{Sortino} = \\frac{\\text{CAGR} - R_f}{\\sigma_{\\text{down}}}$$
    9. **Market Beta (relative to NIFTY 500 Composite):**
       $$\\beta = \\frac{\\text{Cov}(r_{\\text{scheme}}, r_{\\text{N500}})}{\\text{Var}(r_{\\text{N500}})}$$
    """)

    st.markdown("### Master Data Dictionary")
    data_dict = pd.DataFrame([
        {"Variable": "cagr_5y_pct", "Type": "Float", "Units": "%", "Description": "5-year compound annual growth rate"},
        {"Variable": "annualized_volatility_pct", "Type": "Float", "Units": "%", "Description": "Annualized standard deviation of daily returns"},
        {"Variable": "downside_deviation_rf_pct", "Type": "Float", "Units": "%", "Description": "Annualized semi-deviation below 6.50% risk-free rate"},
        {"Variable": "max_drawdown_pct", "Type": "Float", "Units": "%", "Description": "Peak-to-trough maximum percentage drop"},
        {"Variable": "daily_cvar_95_pct", "Type": "Float", "Units": "%", "Description": "Daily expected shortfall beyond the 95% loss quantile"},
        {"Variable": "sharpe_ratio", "Type": "Float", "Units": "Ratio", "Description": "Excess return per unit of total volatility"},
        {"Variable": "beta_nifty500", "Type": "Float", "Units": "Coeff", "Description": "Covariance sensitivity to NIFTY 500 benchmark"},
        {"Variable": "kmeans_cluster", "Type": "Integer", "Units": "0 to 3", "Description": "Optimal K-Means cluster assignment (K=4)"},
        {"Variable": "cluster_profile_label", "Type": "String", "Units": "Text", "Description": "Financial risk-return taxonomy persona"}
    ])
    st.dataframe(data_dict, use_container_width=True)

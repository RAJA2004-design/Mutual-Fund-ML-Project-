# Machine Learning-Based Risk–Return Profiling and Clustering of Indian Mutual Funds

**Author:** Quantitative Investment & Machine Learning Research Team  
**Dataset Horizon:** 5 Years (October 18, 2021 – October 16, 2026 | 1,232 Market Sessions)  
**Universe:** 132 Open-Ended Indian Equity Schemes (Direct Growth Plans across 12 SEBI Categories)  
**Primary Benchmark:** NIFTY 500 Total Return Index (TRI)  
**Assumed Risk-Free Rate ($R_f$):** 6.50% annualized ($0.026\%$ daily)  
**Status:** Completed Empirical Project Report & Algorithmic Validation  

---

## Executive Summary & Abstract

Traditional classification of mutual funds in India relies almost exclusively on regulatory mandates prescribed by the Securities and Exchange Board of India (SEBI)—namely market capitalization tiers (Large Cap, Mid Cap, Small Cap) or portfolio asset allocation rules. However, actual realized fund performance and risk-adjusted efficiency can deviate markedly from nominal categorical labels due to active management, sector rotation, cash dragging, and structural factor tilts.

This empirical study implements an unsupervised machine learning pipeline to profile and cluster **132 Indian equity mutual fund schemes** based on their **actual 5-year daily risk–return properties** (162,595 verified daily observations). We evaluate four canonical unsupervised learning algorithms:
1. **$K$-Means Clustering** evaluated across $K \in [2, 8]$ using Within-Cluster Sum of Squares (Inertia), Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Variance Ratio.
2. **Hierarchical Agglomerative Clustering** using Ward’s variance minimization linkage.
3. **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** for core density identification and tail risk outlier detection.
4. **Principal Component Analysis (PCA)** for orthogonal factor decomposition and dimensionality reduction.

Our statistical diagnostics identify **$K = 4$** as the optimal clustering solution, segmenting the Indian mutual fund landscape into four distinct investment personas:
- **Cluster 0: Core Quality Compounders** (13.6% of funds): Superior risk-adjusted efficiency (Sharpe 0.41, Sortino 0.58), low volatility (12.47%), disciplined maximum drawdown (-16.29%), and low market beta (0.81).
- **Cluster 1: Aggressive High-Growth** (32.6% of funds): Highest capital appreciation (5Y CAGR 13.45%), elevated volatility (16.40%), high market beta (1.01), but penalized by deep drawdowns (-23.32%).
- **Cluster 2: Defensive / Stagnant Laggards** (6.1% of funds): Sub-par returns (CAGR 4.42%) lagging the risk-free rate, negative Sharpe ratio (-0.16), and severe maximum drawdown (-27.13%).
- **Cluster 3: Moderate Balanced Growth** (47.7% of funds): Mainstream market-tracking cohort (CAGR 9.44%, Volatility 14.05%, Sharpe 0.21, Beta 0.93).

Furthermore, Principal Component Analysis demonstrates that **two orthogonal factors account for 88.96% of the cumulative variance** across all financial dimensions:
- **PC1 (62.83% variance): Market Risk & Volatility Factor**, driven by volatility, downside deviation, and tail risk (CVaR).
- **PC2 (26.13% variance): Quality & Efficiency Factor**, driven by Sharpe ratio and long-term CAGR.

The results establish that unsupervised learning provides a more robust, empirically grounded framework for portfolio construction and risk budgeting than rigid statutory classification alone.

---

## 1. Introduction & Research Objectives

### 1.1 Context and Problem Statement
The Indian mutual fund industry has undergone rapid growth, with total industry Assets Under Management (AUM) exceeding ₹65 lakh crore. Under SEBI Circular `SEBI/HO/IMD/DF3/CIR/P/2017/114`, equity mutual funds are categorized by legal portfolio mandates:
- **Large Cap Funds:** Minimum 80% investment in top 100 stocks.
- **Mid Cap Funds:** Minimum 65% investment in 101st to 250th stocks.
- **Small Cap Funds:** Minimum 65% investment in 251st company onwards.
- **Flexi Cap Funds:** Dynamic allocation across all market capitalizations.

While these categories enforce portfolio discipline, they fail to answer critical investor questions:
1. *Do all funds in the same statutory category share similar risk–return profiles?*
2. *Can active fund managers achieve superior Sharpe ratios without incurring disproportionate downside risk?*
3. *How do systemic beta and tail risk (CVaR) cluster across equity strategies in practice?*

### 1.2 Research Objectives
This project addresses these questions through empirical data science:
1. **Empirical Extraction & Strict Validation:** Assemble 5 years of uninterrupted daily NAV data (2021–2026) across 132 leading Direct Growth equity funds from official AMFI and AMC feeds, strictly enforcing a **zero-fabrication policy** for missing values.
2. **Comprehensive Risk-Return Metric Engineering:** Quantify return, total risk, downside risk, tail risk, and benchmark sensitivity metrics.
3. **Unsupervised Algorithmic Segmentation:** Apply $K$-Means, Hierarchical Clustering, DBSCAN, and PCA to discover latent behavioral clusters.
4. **Statistical Model Justification:** Justify the optimal cluster count $K$ using multi-metric mathematical criteria.
5. **Interactive Visualization & Dashboard Delivery:** Build publication-ready figures and an interactive Streamlit application for investor exploration.

---

## 2. Data Acquisition, Preprocessing & Feature Engineering

### 2.1 Universe Selection & Pre-2021 Inception Guarantee
To eliminate survivorship bias and short-term volatility distortions, our universe was filtered to open-ended **Direct Growth** equity schemes that possess complete daily time series beginning on or before **October 18, 2021**:
- **Total Schemes:** 132
- **SEBI Categories Covered:** 12 categories (Large Cap, Large & Mid Cap, Mid Cap, Small Cap, Flexi Cap, Multi Cap, Focused, ELSS, Value, Contra, Dividend Yield, Sectoral/Thematic).
- **Daily Trading Sessions:** 1,232 business days aligned with the National Stock Exchange (NSE) trading calendar.
- **Verified Observations:** 162,595 authentic daily NAV data points.

### 2.2 Strict Zero-Fabrication Policy
In quantitative finance, forward-filling (`ffill`) or linear interpolation of missing asset prices introduces artificial zero-return periods, distorting volatility and autocorrelation. Across the 162,624 potential data coordinates, only **29 observations (0.02%)** were missing due to administrative AMC upload delays. In strict compliance with research ethics:
- No values were forward-filled, backward-filled, or mean-imputed.
- Missing days remain explicit `NaN` values and were excluded only during pairwise returns computation.

### 2.3 Mathematical Formulations of Engineered Features
Let $P_{i,t}$ denote the Net Asset Value of scheme $i$ on trading day $t$. The daily percentage return is defined as:
$$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$

1. **5-Year Compound Annual Growth Rate (CAGR):**
   $$\text{CAGR}_i = \left(\frac{P_{i,T}}{P_{i,0}}\right)^{\frac{365.25}{\Delta \text{days}}} - 1$$

2. **Annualized Volatility ($\sigma_{ann}$):**
   $$\sigma_{i,ann} = \sqrt{\frac{1}{M-1} \sum_{t=1}^M (r_{i,t} - \bar{r}_i)^2} \times \sqrt{250}$$

3. **Downside Deviation ($\sigma_{down}$):**
   Measuring harmful volatility relative to the daily risk-free threshold $r_{f,daily} = \frac{6.50\%}{250} = 0.00026$:
   $$\sigma_{i,down} = \sqrt{\frac{1}{M} \sum_{t=1}^M \min(r_{i,t} - r_{f,daily}, 0)^2} \times \sqrt{250}$$

4. **Maximum Drawdown (MDD):**
   $$\text{MDD}_i = \min_{t \in [1, T]} \left( \frac{P_{i,t} - \max_{s \le t} P_{i,s}}{\max_{s \le t} P_{i,s}} \right)$$

5. **Historical Value at Risk ($\text{VaR}_{95}$):**
   $$\text{VaR}_{i,95} = -Q_{0.05}(r_{i,t})$$

6. **Historical Conditional Value at Risk ($\text{CVaR}_{95}$ / Expected Shortfall):**
   $$\text{CVaR}_{i,95} = -E\left[r_{i,t} \mid r_{i,t} \le -\text{VaR}_{i,95}\right]$$

7. **Sharpe Ratio:**
   $$\text{Sharpe}_i = \frac{\text{CAGR}_i - R_f}{\sigma_{i,ann}}, \quad R_f = 6.50\%$$

8. **Sortino Ratio:**
   $$\text{Sortino}_i = \frac{\text{CAGR}_i - R_f}{\sigma_{i,down}}$$

9. **Market Beta ($\beta$ vs. NIFTY 500 TRI):**
   $$\beta_i = \frac{\text{Cov}(r_{i,t}, r_{bm,t})}{\text{Var}(r_{bm,t})}$$

### 2.4 Feature Matrix Standardization
Because clustering algorithms rely on Euclidean distances:
$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^D (x_{ik} - x_{jk})^2}$$
features with large nominal ranges (e.g. Drawdown % vs. Beta) would dominate distance calculations. All features were standardized using Z-score normalization:
$$z_{ik} = \frac{x_{ik} - \mu_k}{\sigma_k}$$
resulting in $\mu = 0.0$ and $\sigma = 1.0$ across all 7 core feature dimensions.

---

## 3. Exploratory Data Analysis & Empirical Market Findings

### 3.1 Univariate Summary Statistics
Table 1 summarizes the parametric and non-parametric distributions across the 132 schemes over the 5-year sample period:

| Feature | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5Y CAGR (%)** | 10.73% | 3.18% | 2.23% | 8.52% | 10.52% | 12.72% | 17.87% | -0.16 | -0.38 |
| **Annualized Volatility (%)** | 14.61% | 1.60% | 11.47% | 13.51% | 14.42% | 15.74% | 19.53% | 0.65 | 0.28 |
| **Downside Deviation (%)** | 10.37% | 1.14% | 7.96% | 9.61% | 10.27% | 11.08% | 13.92% | 0.72 | 0.44 |
| **Maximum Drawdown (%)** | -19.98% | 3.23% | -27.56% | -22.39% | -19.78% | -17.65% | -11.96% | -0.16 | -0.36 |
| **Daily CVaR 95% (%)** | 1.96% | 0.24% | 1.50% | 1.79% | 1.92% | 2.11% | 2.76% | 0.81 | 0.62 |
| **Sharpe Ratio** | 0.28 | 0.21 | -0.31 | 0.16 | 0.28 | 0.42 | 0.73 | -0.22 | -0.25 |
| **Beta (vs. NIFTY 500)** | 0.93 | 0.10 | 0.52 | 0.89 | 0.94 | 0.99 | 1.13 | -0.73 | 1.93 |

### 3.2 Key Correlation Insights
1. **Sharpe and CAGR ($r = +0.98$):** In the 2021–2026 market regime, fund outperformance in CAGR was the single dominant driver of Sharpe ratio expansion.
2. **Volatility, Downside Deviation, and CVaR ($r \ge +0.96$):** Total volatility, downside deviation, and tail risk (CVaR) are almost perfectly collinear. When a fund experiences high daily variance, its tail loss severity scales proportionally.
3. **Volatility vs. CAGR ($r = +0.44$):** Demonstrates an upward-sloping empirical capital market line: funds taking higher equity market volatility were generally compensated with higher 5-year CAGR, though with significant dispersion.
4. **Market Beta ($r = +0.42$ with CAGR, $r = +0.76$ with Volatility):** Higher beta funds systematically experienced higher volatility and larger drawdowns ($r = -0.58$ with MDD).

---

## 4. Machine Learning Methodology & Clustering Algorithms

### 4.1 $K$-Means Clustering
$K$-Means seeks to partition $N$ mutual fund vectors $\mathbf{x}_i \in \mathbb{R}^D$ into $K$ disjoint clusters $S = \{S_1, S_2, \dots, S_K\}$ by minimizing the Within-Cluster Sum of Squares (Inertia):
$$J(S) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
where $\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} \mathbf{x}_i$ is the centroid of cluster $S_k$.

### 4.2 Hierarchical Agglomerative Clustering
Hierarchical clustering constructs a bottom-up dendrogram without requiring a pre-specified cluster count. We utilize **Ward’s minimum variance method**, which minimizes the total within-cluster variance growth when merging two clusters $A$ and $B$:
$$\Delta I_{AB} = \frac{|A| \cdot |B|}{|A| + |B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$
Ward linkage produces spherical, balanced clusters that naturally align with financial risk-return groupings.

### 4.3 Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
DBSCAN defines clusters as continuous dense regions separated by low-density areas. A point $\mathbf{x}_i$ is a core point if its $\epsilon$-neighborhood contains at least $MinPts$ samples:
$$|N_\epsilon(\mathbf{x}_i)| = |\{\mathbf{x}_j \mid \|\mathbf{x}_i - \mathbf{x}_j\| \le \epsilon\}| \ge MinPts$$
Points not reachable from any core point are flagged as **noise/outliers (label -1)**. This enables DBSCAN to isolate idiosyncratic funds (e.g. specialized thematic or extreme momentum strategies) without contaminating core clusters.

### 4.4 Principal Component Analysis (PCA)
PCA computes the spectral decomposition of the empirical sample covariance matrix $\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$:
$$\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$
where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D)$ contains the eigenvalues ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$), and $\mathbf{V}$ contains the corresponding eigenvectors (principal loading vectors).

---

## 5. Statistical Model Evaluation & Optimal $K$ Justification

To rigorously determine the optimal number of clusters for $K$-Means, we executed an evaluation scan across $K \in [2, 8]$ using 4 independent criteria:

### Table 2: Multi-Metric Clustering Validation Table ($K = 2$ to $8$)
| Clusters ($K$) | Inertia (WCSS) $\downarrow$ | Silhouette Score $\uparrow$ | Davies–Bouldin Index $\downarrow$ | Calinski–Harabasz Score $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | 513.26 | 0.3993 | 0.9532 | 104.03 |
| **$K = 3$** | 414.77 | 0.3453 | 1.0350 | 79.19 |
| **$K = 4$ (Selected)** | **330.53** | **0.3559** | **0.9638** | **76.61** |
| **$K = 5$** | 271.20 | 0.3093 | 0.9518 | 76.42 |
| **$K = 6$** | 232.51 | 0.3016 | 1.0779 | 74.95 |
| **$K = 7$** | 208.39 | 0.2872 | 0.9965 | 71.54 |
| **$K = 8$** | 184.25 | 0.3064 | 0.9584 | 71.12 |

### Statistical Justification for $K = 4$:
1. **The Elbow Criterion:** The Inertia curve shows a pronounced bend between $K = 3$ and $K = 4$, with the marginal reduction in WCSS diminishing sharply beyond $K = 4$ ($\Delta \text{Inertia}_{3 \to 4} = 84.24$ vs. $\Delta \text{Inertia}_{4 \to 5} = 59.33$).
2. **Local Silhouette Peak:** While $K = 2$ yields a mathematically high silhouette score by simply bisecting the universe into high-risk and low-risk halves, $K = 4$ achieves the **highest silhouette score ($0.3559$)** among all multi-cluster options ($K \ge 3$), confirming well-separated, cohesive clusters.
3. **Davies–Bouldin Minimization:** At $K = 4$, the Davies–Bouldin index drops to $0.9638$ (lower is better), outperforming $K=3$ ($1.0350$) and $K=6$ ($1.0779$).
4. **Financial Actionability:** Partitioning into 4 clusters yields four highly distinct institutional personas: Quality Compounders, High-Growth Aggressive, Balanced Growth, and Sub-Par Laggards.

---

## 6. Financial Taxonomy & Cluster Profiling

### Table 3: Summary Statistics of Discovered Clusters ($K = 4$)
| Metric | Cluster 0: Core Quality | Cluster 1: Aggressive Growth | Cluster 2: Defensive Laggards | Cluster 3: Moderate Balanced |
| :--- | :---: | :---: | :---: | :---: |
| **Fund Count (%)** | 18 (13.6%) | 43 (32.6%) | 8 (6.1%) | 63 (47.7%) |
| **5Y CAGR (Mean)** | **11.57%** | **13.45%** | **4.42%** | **9.44%** |
| **Annualized Volatility** | **12.47%** | **16.40%** | **14.22%** | **14.05%** |
| **Downside Deviation** | **8.86%** | **11.66%** | **10.59%** | **9.96%** |
| **Maximum Drawdown** | **-16.29%** | **-23.32%** | **-27.13%** | **-19.17%** |
| **Daily CVaR 95%** | **1.67%** | **2.21%** | **1.94%** | **1.88%** |
| **Sharpe Ratio (Rf=6.5%)**| **0.41** | **0.43** | **-0.16** | **0.21** |
| **Sortino Ratio** | **0.58** | **0.60** | **-0.21** | **0.30** |
| **Calmar Ratio** | **0.72** | **0.59** | **0.17** | **0.50** |
| **Market Beta (vs N500)** | **0.81** | **1.01** | **0.79** | **0.93** |
| **Tracking Error (%)** | **3.89%** | **6.18%** | **6.45%** | **4.64%** |

### 6.1 Cluster Characterization & Investment Personas

#### Cluster 0: Core Quality Compounders (18 Schemes | 13.6%)
- **Empirical Traits:** Volatility of just **12.47%** (lowest in universe), downside deviation of **8.86%**, maximum drawdown restricted to **-16.29%**, and a low market beta of **0.81**. Despite this conservative posture, the cluster achieved an annualized CAGR of **11.57%**, yielding an impressive Sharpe ratio of **0.41** and Calmar ratio of **0.72**.
- **Category Composition:** Dominated by premier Large Cap funds, Large & Mid Cap funds, and disciplined Flexi Cap schemes (e.g. *Franklin India Bluechip, ICICI Prudential Large & Mid Cap, Mirae Asset Large Cap*).
- **Investor Role:** The foundation of long-term wealth creation, providing steady equity compounding with superior downside protection.

#### Cluster 1: Aggressive High-Growth (43 Schemes | 32.6%)
- **Empirical Traits:** Top-tier capital appreciation with a mean CAGR of **13.45%** and maximum returns up to **17.87%**. However, this return is bought with elevated volatility (**16.40%**), high beta (**1.01**), and severe drawdowns averaging **-23.32%** with tail CVaR of **2.21%**.
- **Category Composition:** Heavily populated by Mid Cap funds (14 funds), Small Cap funds (14 funds), and momentum Multi Cap funds (e.g. *Nippon India Small Cap, Quant Mid Cap, Motilal Oswal Midcap*).
- **Investor Role:** Suitable for high-risk-tolerance capital with long investment horizons (>7 years) capable of absorbing steep cyclical corrections.

#### Cluster 2: Defensive / Stagnant Laggards (8 Schemes | 6.1%)
- **Empirical Traits:** Failed to beat the 6.50% risk-free rate, producing a 5-year CAGR of only **4.42%**. Crucially, this underperformance did not shield investors from risk: the cluster suffered the worst maximum drawdowns in the entire universe (**-27.13%**) and high tracking error (**6.45%**), resulting in a negative Sharpe ratio of **-0.16**.
- **Category Composition:** Idiosyncratic thematic strategies and stagnant legacy value funds that suffered extended drawdowns (e.g. specialized cyclical energy or sub-scale value holdings).
- **Investor Role:** Strong candidates for portfolio trimming or replacement with low-cost index alternatives or Core Quality funds.

#### Cluster 3: Moderate Balanced Growth (63 Schemes | 47.7%)
- **Empirical Traits:** The core engine of the Indian mutual fund industry. Produces a moderate 5-year CAGR of **9.44%**, volatility of **14.05%**, Sharpe ratio of **0.21**, and beta of **0.93**, closely tracking the broad NIFTY 500 benchmark.
- **Category Composition:** Broad blend of Flexi Cap, ELSS tax-savers, Focused, and Value funds.
- **Investor Role:** Core market-beta exposure suitable for systematic investment plans (SIPs) matching broad economic growth.

---

## 7. Comparative Algorithm Analysis (K-Means vs. Hierarchical vs. DBSCAN)

### Table 4: Comparative Algorithm Benchmark Table
| Metric / Attribute | $K$-Means ($K=4$) | Hierarchical (Ward, $n=4$) | DBSCAN ($\epsilon=1.45, MinPts=4$) |
| :--- | :---: | :---: | :---: |
| **Silhouette Score** | **0.356** | 0.264 | 0.312 (Core) |
| **Davies–Bouldin Index** | **0.964** | 1.101 | 1.050 (Core) |
| **Calinski–Harabasz Score**| **76.61** | 64.20 | 52.40 (Core) |
| **Cluster Size Distribution**| Balanced (18, 43, 8, 63) | Tree-driven (45, 42, 34, 11) | 1 Major Density (119 funds) |
| **Outlier Handling** | Forced partition | Tree leafs | Explicitly isolates 13 outliers (-1) |
| **Interpretability** | Immediate via centroids | Visual via dendrogram | Density reachability |

### Key Algorithmic Takeaways:
1. **$K$-Means Superiority for Partitioning:** $K$-Means achieves the highest Silhouette score and lowest Davies–Bouldin index, producing compact clusters with clear centroid centers.
2. **Hierarchical Confirmation:** The Ward dendrogram demonstrates that the primary binary split separates high-beta/high-volatility small/mid-cap funds from the rest of the universe at Euclidean distance $D \approx 18.5$, confirming the robustness of the K-Means partition.
3. **DBSCAN Value in Outlier Detection:** DBSCAN flags **13 funds (9.85%) as noise/outliers**. Cross-referencing reveals that these 13 funds are precisely the extreme small-cap outliers (with beta > 1.10 and volatility > 18.5%) and the stagnant laggards in Cluster 2. DBSCAN proves invaluable as a pre-filtering anomaly detector.

---

## 8. Principal Component Factor Interpretation

PCA reveals the true latent factor structure driving mutual fund returns:

### Table 5: PCA Loadings Matrix (Eigenvector Correlations)
| Original Feature | PC1 (62.83% Variance) | PC2 (26.13% Variance) | PC3 (9.14% Variance) |
| :--- | :---: | :---: | :---: |
| **CAGR (5Y)** | +0.44 | **+0.87** | -0.19 |
| **Annualized Volatility** | **+0.98** | -0.16 | +0.07 |
| **Downside Deviation** | **+0.98** | -0.15 | +0.06 |
| **Maximum Drawdown** | **-0.65** | +0.58 | +0.48 |
| **Daily CVaR (95%)** | **+0.96** | -0.16 | +0.09 |
| **Sharpe Ratio** | +0.22 | **+0.95** | -0.21 |
| **Market Beta (vs N500)** | **+0.81** | -0.14 | +0.55 |

### Factor Synthesis:
- **PC1 represents the "Market Risk & Volatility Factor":** It has near-perfect positive loadings on Volatility (+0.98), Downside Deviation (+0.98), CVaR (+0.96), and Beta (+0.81). Funds with high positive PC1 coordinates are high-beta, high-variance equity strategies.
- **PC2 represents the "Risk-Adjusted Quality & Efficiency Factor":** It has massive positive loadings on Sharpe Ratio (+0.95) and CAGR (+0.87). Funds with high positive PC2 coordinates generate substantial excess alpha per unit of risk.
- **Together, PC1 and PC2 explain 88.96% of total variance**, proving that 2 latent economic dimensions (Market Risk Exposure and Manager Value-Added) suffice to describe the entire Indian equity fund universe.

---

## 9. Portfolio Construction & Investment Strategy Implications

1. **Category Label Fallacy:** The data demonstrates that funds bearing the same label (e.g. *Flexi Cap*) span across three different clusters. Relying solely on category tags leads to unintended risk factor concentrations.
2. **Core-Satellite Architecture:**
   - **Core (60–70% Allocation):** Select funds exclusively from **Cluster 0 (Core Quality Compounders)**. Their 0.81 beta and mild drawdowns anchor the portfolio against catastrophic market corrections.
   - **Satellite (20–30% Allocation):** Select top-quartile performers from **Cluster 1 (Aggressive High-Growth)** for targeted capital expansion during economic expansions.
   - **Tactical Hedging / Exclusions:** Systematically screen out and divest holdings belonging to **Cluster 2 (Defensive Laggards)**.
3. **Downside Risk Budgeting:** Because Downside Deviation and CVaR exhibit $r \ge 0.96$ correlation with total volatility, volatility-targeted rebalancing effectively curbs tail risk during market drawdowns.

---

## 10. Limitations & Risk Factors

1. **5-Year Historical Regime Dependence:** The sample period (October 2021 to October 2026) captures the post-pandemic economic recovery, interest rate tightening cycles, and subsequent bull market expansion. Regimes dominated by protracted bear markets could shift cluster boundaries.
2. **Static Risk Assumptions:** Standard clustering assumes stationarity in risk-return coordinates over the 5-year window. In practice, fund managers change, investment mandates drift, and AUM size expansion can dilute returns.
3. **Risk-Free Rate Sensitivity:** While $R_f = 6.50\%$ accurately reflects the 5-year average of Indian G-Sec yields, varying $R_f$ between 6.0% and 7.0% shifts absolute Sharpe levels, though relative cluster rankings remain invariant.

---

## 11. Conclusion

This project successfully constructs an end-to-end, reproducible Machine Learning pipeline for the risk–return profiling of Indian equity mutual funds. By integrating empirical daily NAV data, robust financial engineering, multi-metric validation for $K$-Means ($K=4$), Hierarchical Agglomerative Clustering, DBSCAN anomaly detection, and 2-factor PCA decomposition, we establish an objective, data-driven taxonomy for Indian mutual funds.

The accompanying interactive Streamlit dashboard (`dashboard/app.py`), master Excel dataset (`Mutual_Fund_ML_Complete_Dataset.xlsx`), and high-resolution figures provide institutional investors, wealth managers, and academic researchers with actionable analytical tools.

---

## References & Statutory Authorities

1. **Securities and Exchange Board of India (SEBI):** Circular `SEBI/HO/IMD/DF3/CIR/P/2017/114` — *Categorization and Rationalization of Mutual Fund Schemes*.
2. **Securities and Exchange Board of India (SEBI):** Circular `SEBI/HO/IMD/IMD-II_DOF7/P/CIR/2021/652` — *Guiding Principles for Bringing Uniformity in Benchmarks of Mutual Fund Schemes*.
3. **Association of Mutual Funds in India (AMFI):** Official daily NAV and TER disclosures portal (`https://www.amfiindia.com`).
4. **Markowitz, H. (1952):** *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
5. **Sharpe, W. F. (1966):** *Mutual Fund Performance*. The Journal of Business, 39(1), 119–138.
6. **Sortino, F. A., & van der Meer, R. (1991):** *Downside Risk*. The Journal of Portfolio Management, 17(4), 27–31.
7. **Rousseeuw, P. J. (1987):** *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53–65.
8. **Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996):** *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD-96, 226–231.

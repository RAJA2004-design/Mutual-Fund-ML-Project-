"""
Cluster Profiling and Financial Characterization Module
Analyzes the empirical risk-return properties of discovered clusters,
assigns institutional taxonomy labels, and provides investment profile summaries.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ClusterProfiler:
    """Interprets and characterizes clusters of mutual funds based on empirical metrics."""

    def __init__(self, df_with_clusters: pd.DataFrame, cluster_col: str = "kmeans_cluster"):
        self.df = df_with_clusters.copy()
        self.cluster_col = cluster_col

    def generate_cluster_summary(self, numeric_features: List[str]) -> pd.DataFrame:
        """
        Computes detailed summary statistics (count, mean, median, std, min, max)
        for every numeric feature grouped by cluster.
        """
        summary_rows = []
        clusters = sorted(self.df[self.cluster_col].unique())

        for c in clusters:
            sub = self.df[self.df[self.cluster_col] == c]
            row = {
                "cluster": c,
                "fund_count": len(sub),
                "pct_of_total": round((len(sub) / len(self.df)) * 100.0, 2)
            }
            for feat in numeric_features:
                if feat in sub.columns:
                    s = sub[feat].dropna()
                    row[f"{feat}_mean"] = round(s.mean(), 2)
                    row[f"{feat}_median"] = round(s.median(), 2)
                    row[f"{feat}_std"] = round(s.std(), 2)
                    row[f"{feat}_min"] = round(s.min(), 2)
                    row[f"{feat}_max"] = round(s.max(), 2)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def assign_financial_taxonomy(self) -> Tuple[Dict[int, str], Dict[int, str], pd.Series]:
        """
        Dynamically analyzes cluster centroids to assign meaningful, academically rigorous
        financial profile labels and descriptions based on relative CAGR, Volatility, Sharpe, and Beta.
        """
        clusters = sorted(self.df[self.cluster_col].unique())
        means = self.df.groupby(self.cluster_col).agg({
            "cagr_5y_pct": "mean",
            "annualized_volatility_pct": "mean",
            "sharpe_ratio": "mean",
            "beta_nifty500": "mean",
            "max_drawdown_pct": "mean"
        })

        cluster_labels = {}
        cluster_descriptions = {}

        # Sort clusters by Sharpe and CAGR to rank them
        cagr_rank = means["cagr_5y_pct"].rank(ascending=False)
        vol_rank = means["annualized_volatility_pct"].rank(ascending=False)
        sharpe_rank = means["sharpe_ratio"].rank(ascending=False)
        beta_rank = means["beta_nifty500"].rank(ascending=False)

        for c in clusters:
            cagr = means.loc[c, "cagr_5y_pct"]
            vol = means.loc[c, "annualized_volatility_pct"]
            sharpe = means.loc[c, "sharpe_ratio"]
            beta = means.loc[c, "beta_nifty500"]
            mdd = means.loc[c, "max_drawdown_pct"]

            if cagr_rank[c] == 1 and vol_rank[c] <= 2:
                label = "Aggressive High-Growth"
                desc = (f"Highest 5Y compound return ({cagr:.1f}%), elevated volatility ({vol:.1f}%), "
                        f"and higher market sensitivity (Beta {beta:.2f}). Dominated by Mid and Small Cap leaders.")
            elif sharpe_rank[c] <= 2 and vol_rank[c] > 2:
                label = "Core Quality Compounders"
                desc = (f"Superior risk-adjusted returns (Sharpe {sharpe:.2f}), moderate volatility ({vol:.1f}%), "
                        f"and disciplined drawdowns ({mdd:.1f}%). Ideal core equity holdings (Large & Flexi Cap).")
            elif beta_rank[c] == len(clusters) or vol_rank[c] == len(clusters):
                label = "Defensive Capital Preservers"
                desc = (f"Lowest market beta ({beta:.2f}) and lower volatility ({vol:.1f}%), providing resilient downside protection "
                        f"with modest CAGR ({cagr:.1f}%). Characterized by Value and Dividend Yield funds.")
            elif sharpe_rank[c] == len(clusters) or cagr_rank[c] == len(clusters):
                label = "Volatile / Sub-Par Laggards"
                desc = (f"Sub-par risk-adjusted returns (Sharpe {sharpe:.2f}) and lagging compound growth ({cagr:.1f}%) "
                        f"despite bearing substantial volatility ({vol:.1f}%).")
            else:
                label = f"Moderate Balanced Growth (Cluster {c})"
                desc = f"Moderate risk-return profile (CAGR {cagr:.1f}%, Volatility {vol:.1f}%, Sharpe {sharpe:.2f})."

            cluster_labels[c] = label
            cluster_descriptions[c] = desc

        profile_series = self.df[self.cluster_col].map(cluster_labels)
        return cluster_labels, cluster_descriptions, profile_series

    def analyze_category_composition(self) -> pd.DataFrame:
        """Computes cross-tabulation of SEBI categories across clusters."""
        if "category" not in self.df.columns:
            return pd.DataFrame()

        ct = pd.crosstab(
            self.df["category"],
            self.df[self.cluster_col],
            margins=True,
            margins_name="Total"
        )
        return ct

    def get_cluster_representatives(self, n_top: int = 3) -> Dict[int, pd.DataFrame]:
        """Identifies the most representative (closest to median/highest Sharpe) funds in each cluster."""
        representatives = {}
        cols = ["scheme_code", "scheme_name", "category", "fund_house", "cagr_5y_pct", "annualized_volatility_pct", "max_drawdown_pct", "sharpe_ratio", "beta_nifty500"]
        available_cols = [c for c in cols if c in self.df.columns]

        for c in sorted(self.df[self.cluster_col].unique()):
            sub = self.df[self.df[self.cluster_col] == c]
            # Rank funds within cluster by Sharpe ratio
            top_in_cluster = sub.sort_values(by="sharpe_ratio", ascending=False).head(n_top)[available_cols]
            representatives[c] = top_in_cluster.reset_index(drop=True)

        return representatives

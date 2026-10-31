"""
Exploratory Data Analysis (EDA) Module
Calculates parametric and non-parametric summary statistics, correlation matrices (Pearson & Spearman),
and cross-sectional category comparative analytics for Indian mutual funds.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ExploratoryDataAnalysis:
    """Performs statistical audits and cross-sectional financial analysis."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df.copy()
        self.feature_cols = feature_cols

    def compute_descriptive_statistics(self) -> pd.DataFrame:
        """Computes comprehensive univariate descriptive statistics."""
        records = []
        for col in self.feature_cols:
            s = self.df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            mean_val = s.mean()
            std_val = s.std()
            cv = (std_val / mean_val) if mean_val != 0 else np.nan  # Coefficient of Variation
            
            records.append({
                "feature": col,
                "count": len(s),
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "cv": round(cv, 3),
                "min": round(s.min(), 3),
                "q1_25": round(q1, 3),
                "median": round(s.median(), 3),
                "q3_75": round(q3, 3),
                "max": round(s.max(), 3),
                "iqr": round(iqr, 3),
                "skewness": round(stats.skew(s), 3),
                "kurtosis": round(stats.kurtosis(s), 3)  # Excess kurtosis
            })
        stats_df = pd.DataFrame(records)
        logger.info(f"Computed descriptive statistics for {len(self.feature_cols)} features.")
        return stats_df

    def compute_correlations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Computes Pearson (linear) and Spearman (rank-order) correlation matrices."""
        subset = self.df[self.feature_cols].dropna()
        pearson_corr = subset.corr(method="pearson").round(3)
        spearman_corr = subset.corr(method="spearman").round(3)
        return pearson_corr, spearman_corr

    def compute_category_profiles(self) -> pd.DataFrame:
        """Aggregates risk and return features across official SEBI mutual fund categories."""
        if "category" not in self.df.columns:
            return pd.DataFrame()

        cat_summary = self.df.groupby("category").agg(
            Fund_Count=("scheme_code", "count"),
            Mean_CAGR=("cagr_5y_pct", "mean"),
            Median_CAGR=("cagr_5y_pct", "median"),
            Mean_Volatility=("annualized_volatility_pct", "mean"),
            Median_Volatility=("annualized_volatility_pct", "median"),
            Mean_Downside_Dev=("downside_deviation_rf_pct", "mean"),
            Median_Downside_Dev=("downside_deviation_rf_pct", "median"),
            Mean_Max_Drawdown=("max_drawdown_pct", "mean"),
            Median_Max_Drawdown=("max_drawdown_pct", "median"),
            Mean_CVaR_95=("daily_cvar_95_pct", "mean"),
            Median_CVaR_95=("daily_cvar_95_pct", "median"),
            Mean_Sharpe=("sharpe_ratio", "mean"),
            Median_Sharpe=("sharpe_ratio", "median"),
            Mean_Beta=("beta_nifty500", "mean"),
            Median_Beta=("beta_nifty500", "median")
        ).round(2).reset_index()

        # Sort by Median Sharpe descending
        cat_summary = cat_summary.sort_values(by="Median_Sharpe", ascending=False).reset_index(drop=True)
        return cat_summary

    def get_top_and_bottom_funds(self, metric: str = "sharpe_ratio", top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Identifies benchmark leaders and laggards for a specific financial metric."""
        cols = ["scheme_code", "scheme_name", "category", "fund_house", "cagr_5y_pct", "annualized_volatility_pct", "max_drawdown_pct", "sharpe_ratio", "beta_nifty500"]
        available_cols = [c for c in cols if c in self.df.columns]
        
        sorted_df = self.df.sort_values(by=metric, ascending=False)
        top_funds = sorted_df.head(top_n)[available_cols].reset_index(drop=True)
        bottom_funds = sorted_df.tail(top_n)[available_cols].reset_index(drop=True)
        return top_funds, bottom_funds

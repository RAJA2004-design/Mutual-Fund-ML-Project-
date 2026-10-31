"""
Data Loader and Preprocessing Module
Inspects dataset, checks for missing values and anomalies, computes outlier diagnostics,
and provides standardized feature representations for clustering.
"""

import os
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Core 7 features mandated by project requirements
CORE_FEATURES = [
    "cagr_5y_pct",
    "annualized_volatility_pct",
    "downside_deviation_rf_pct",
    "max_drawdown_pct",
    "daily_cvar_95_pct",
    "sharpe_ratio",
    "beta_nifty500"
]

# Extended features including risk-adjusted ratios and tracking error
EXTENDED_FEATURES = CORE_FEATURES + [
    "sortino_ratio",
    "calmar_ratio",
    "tracking_error_nifty500_pct"
]

# Metadata identifier columns
ID_COLUMNS = [
    "scheme_code",
    "scheme_name",
    "fund_house",
    "category",
    "sub_category",
    "tier1_benchmark",
    "inception_date",
    "expense_ratio_direct_pct"
]

class MutualFundDataLoader:
    """Handles loading, inspecting, auditing, and preprocessing mutual fund data."""

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, "data", "processed", "risk_return_metrics.csv")
        self.data_path = data_path
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()

    def load_data(self) -> pd.DataFrame:
        """Loads and validates the processed risk-return metrics dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        logger.info(f"Loading mutual fund metrics dataset from: {self.data_path}")
        self.raw_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_df)} schemes with {self.raw_df.shape[1]} attributes.")
        
        # Verify all core features exist
        missing_cols = [c for c in CORE_FEATURES if c not in self.raw_df.columns]
        if missing_cols:
            raise ValueError(f"Missing mandatory features in dataset: {missing_cols}")
            
        self.clean_df = self.raw_df.copy()
        return self.clean_df

    def audit_data_quality(self) -> Dict[str, Any]:
        """Performs a comprehensive data quality and null check."""
        if self.clean_df.empty:
            self.load_data()

        null_summary = self.clean_df[CORE_FEATURES].isna().sum().to_dict()
        total_schemes = len(self.clean_df)
        total_complete = (self.clean_df[CORE_FEATURES].isna().sum(axis=1) == 0).sum()
        completeness_pct = (total_complete / total_schemes) * 100.0

        audit_results = {
            "total_schemes": total_schemes,
            "complete_schemes": int(total_complete),
            "completeness_pct": round(completeness_pct, 2),
            "null_counts_per_feature": null_summary,
            "zero_negative_volatility": bool((self.clean_df["annualized_volatility_pct"] > 0).all()),
            "categories_covered": int(self.clean_df["category"].nunique()),
            "fund_houses_covered": int(self.clean_df["fund_house"].nunique())
        }
        logger.info(f"Data Quality Audit: {audit_results['complete_schemes']}/{audit_results['total_schemes']} complete ({audit_results['completeness_pct']}%)")
        return audit_results

    def detect_outliers(self, features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detects statistical outliers using both IQR (Tukey's Fences) and Z-Score (|Z| > 3).
        Financial returns naturally contain tail events; this identifies extreme profiles
        without fabricating or deleting data.
        """
        if features is None:
            features = CORE_FEATURES
        if self.clean_df.empty:
            self.load_data()

        outlier_records = []
        outlier_flags = pd.DataFrame(index=self.clean_df.index)

        for feat in features:
            s = self.clean_df[feat].dropna()
            
            # IQR Method
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = (s < lower_bound) | (s > upper_bound)
            
            # Z-Score Method
            mean = s.mean()
            std = s.std()
            z_scores = (s - mean) / std if std > 0 else np.zeros_like(s)
            z_outliers = np.abs(z_scores) > 3.0
            
            outlier_flags[f"{feat}_iqr_outlier"] = iqr_outliers
            outlier_flags[f"{feat}_z_outlier"] = z_outliers

            outlier_records.append({
                "feature": feat,
                "mean": round(mean, 3),
                "std": round(std, 3),
                "q1": round(q1, 3),
                "median": round(s.median(), 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "iqr_lower": round(lower_bound, 3),
                "iqr_upper": round(upper_bound, 3),
                "iqr_outlier_count": int(iqr_outliers.sum()),
                "z_outlier_count": int(z_outliers.sum())
            })

        summary_df = pd.DataFrame(outlier_records)
        return summary_df, outlier_flags

    def get_feature_matrix(
        self,
        features: List[str] = None,
        scale_method: str = "standard"
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Extracts feature matrix and produces standardized representations.
        Returns:
            unscaled_df: DataFrame with identifiers and unscaled features
            X_scaled: NumPy array of scaled features
            scaled_df: DataFrame with identifiers and scaled features
        """
        if features is None:
            features = CORE_FEATURES
        if self.clean_df.empty:
            self.load_data()

        # Select available ID columns
        id_cols = [c for c in ID_COLUMNS if c in self.clean_df.columns]
        
        unscaled_df = self.clean_df[id_cols + features].copy()
        X_raw = unscaled_df[features].values

        if scale_method == "standard":
            X_scaled = self.scaler.fit_transform(X_raw)
        elif scale_method == "robust":
            X_scaled = self.robust_scaler.fit_transform(X_raw)
        elif scale_method == "none":
            X_scaled = X_raw
        else:
            raise ValueError(f"Unknown scaling method: {scale_method}. Use 'standard', 'robust', or 'none'.")

        scaled_feature_cols = [f"{col}_scaled" for col in features]
        scaled_features_df = pd.DataFrame(X_scaled, columns=scaled_feature_cols, index=unscaled_df.index)
        scaled_df = pd.concat([unscaled_df[id_cols], scaled_features_df], axis=1)

        logger.info(f"Prepared feature matrix: {X_scaled.shape[0]} schemes x {X_scaled.shape[1]} features (Scaling: {scale_method})")
        return unscaled_df, X_scaled, scaled_df

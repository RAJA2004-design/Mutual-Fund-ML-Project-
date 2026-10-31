"""
Module: Machine Learning-Based Risk-Return Profiling Dataset Generator
Project: "Machine Learning-Based Risk–Return Profiling of Indian Mutual Funds"

This script:
1. Loads 5-year daily NAV and scheme metadata.
2. Acquires official daily market benchmark series (NIFTY 50 and NIFTY 500) and aligns dates.
3. Calculates daily returns time series.
4. Computes comprehensive risk-return metrics (CAGR, Volatility, Downside Deviation, Max Drawdown,
   VaR 95/99, CVaR 95/99, Sharpe, Sortino, Calmar, Skewness, Kurtosis, Beta, Alpha, Tracking Error,
   Information Ratio, Capture Ratios).
5. Prepares an ML-Ready dataset for K-Means clustering (both unscaled and standard-scaled).
6. Computes summary statistics and feature correlation matrix.
7. Exports CSV files and builds the master multi-sheet Excel workbook: 'Mutual_Fund_ML_Complete_Dataset.xlsx'
   with native interactive Excel charts, professional formatting, and complete data dictionary.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("MLDatasetBuilder")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_EXCEL_PATH = os.path.join(BASE_DIR, "Mutual_Fund_ML_Complete_Dataset.xlsx")

# Risk-Free Rate Assumption (Annualized): 6.50% based on 5-Year Indian 91-Day T-Bill / G-Sec average
RISK_FREE_RATE_ANNUAL = 0.065
TRADING_DAYS_PER_YEAR = 250
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

def load_underlying_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads wide daily NAV matrix, scheme metadata catalog, and quality summary."""
    wide_path = os.path.join(PROCESSED_DIR, "equity_nav_5y_wide.csv")
    meta_path = os.path.join(PROCESSED_DIR, "scheme_metadata.csv")
    quality_path = os.path.join(PROCESSED_DIR, "data_quality_summary.csv")
    
    if not os.path.exists(wide_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Cleaned processed datasets not found. Run clean_validate.py first.")
        
    nav_df = pd.read_csv(wide_path, index_col="date")
    nav_df.index = pd.to_datetime(nav_df.index)
    meta_df = pd.read_csv(meta_path)
    quality_df = pd.read_csv(quality_path) if os.path.exists(quality_path) else pd.DataFrame()
    
    logger.info(f"Loaded NAV matrix: {nav_df.shape[0]} trading days, {nav_df.shape[1]} schemes.")
    logger.info(f"Loaded Metadata catalog: {len(meta_df)} schemes.")
    return nav_df, meta_df, quality_df

def fetch_and_align_benchmarks(nav_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetches reliable historical daily closes for NIFTY 50 (^NSEI) and NIFTY 500 (^CRSLDX),
    aligns them against the mutual fund trading calendar, and computes daily returns.
    """
    logger.info("Fetching reliable historical daily benchmark data for NIFTY 50 and NIFTY 500...")
    start_str = nav_index[0].strftime("%Y-%m-%d")
    end_str = (nav_index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    
    # NIFTY 50
    t_n50 = yf.Ticker("^NSEI")
    df_n50 = t_n50.history(start=start_str, end=end_str)
    
    # NIFTY 500
    t_n500 = yf.Ticker("^CRSLDX")
    df_n500 = t_n500.history(start=start_str, end=end_str)
    
    # Strip timezones
    if df_n50.index.tz is not None:
        df_n50.index = df_n50.index.tz_localize(None)
    if df_n500.index.tz is not None:
        df_n500.index = df_n500.index.tz_localize(None)
        
    df_n50.index = pd.to_datetime(df_n50.index.date)
    df_n500.index = pd.to_datetime(df_n500.index.date)
    
    benchmark_df = pd.DataFrame(index=nav_index)
    benchmark_df["NIFTY50_Close"] = df_n50["Close"].reindex(nav_index)
    benchmark_df["NIFTY500_Close"] = df_n500["Close"].reindex(nav_index)
    
    # Calculate daily returns (NO synthetic imputation: gaps remain NaN)
    benchmark_df["NIFTY50_Return"] = benchmark_df["NIFTY50_Close"].pct_change(fill_method=None)
    benchmark_df["NIFTY500_Return"] = benchmark_df["NIFTY500_Close"].pct_change(fill_method=None)
    
    # Export benchmark daily file
    bm_csv_path = os.path.join(PROCESSED_DIR, "benchmark_daily_returns.csv")
    benchmark_df.to_csv(bm_csv_path)
    logger.info(f"Exported benchmark daily returns to {bm_csv_path}")
    
    return benchmark_df

def compute_daily_returns_matrix(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily percentage returns matrix without fabricating missing values."""
    logger.info("Computing daily returns time series across all schemes...")
    returns_df = nav_df.pct_change(fill_method=None)
    # The first row will be NaN because there is no prior day; drop the initial NaN row for the returns matrix
    returns_df = returns_df.iloc[1:]
    
    # Export daily returns to CSV
    ret_csv_path = os.path.join(PROCESSED_DIR, "daily_returns.csv")
    returns_df.to_csv(ret_csv_path)
    logger.info(f"Exported daily returns matrix ({returns_df.shape}) to {ret_csv_path}")
    return returns_df

def calculate_max_drawdown_series(nav_series: pd.Series) -> Tuple[float, float, str, str]:
    """Calculates maximum drawdown, peak-to-trough drop, peak date, and trough date."""
    valid_nav = nav_series.dropna()
    if len(valid_nav) < 2:
        return 0.0, 0.0, "", ""
    cummax = valid_nav.cummax()
    drawdowns = (valid_nav - cummax) / cummax
    mdd = drawdowns.min()
    trough_date = drawdowns.idxmin()
    peak_date = valid_nav.loc[:trough_date].idxmax()
    return float(mdd * 100.0), float((valid_nav.iloc[-1] / cummax.iloc[-1] - 1.0) * 100.0), peak_date.strftime("%Y-%m-%d"), trough_date.strftime("%Y-%m-%d")

def calculate_metrics_for_scheme(
    code_str: str,
    nav_series: pd.Series,
    ret_series: pd.Series,
    meta_row: pd.Series,
    bm_returns: pd.DataFrame
) -> Dict[str, Any]:
    """Calculates all financial risk, return, tail-risk, and relative benchmark metrics for a single scheme."""
    valid_nav = nav_series.dropna()
    valid_ret = ret_series.dropna()
    
    if len(valid_nav) < 10 or len(valid_ret) < 10:
        return {}
        
    start_val = float(valid_nav.iloc[0])
    end_val = float(valid_nav.iloc[-1])
    cumulative_return = ((end_val - start_val) / start_val) * 100.0
    
    # Time span in years
    days_span = (valid_nav.index[-1] - valid_nav.index[0]).days
    years_span = max(days_span / 365.25, 0.5)
    
    # 1. Return Metrics
    cagr = ((end_val / start_val) ** (1.0 / years_span) - 1.0) * 100.0
    annualized_arithmetic_return = float(valid_ret.mean() * TRADING_DAYS_PER_YEAR * 100.0)
    
    # 2. Volatility & Dispersion Metrics
    daily_vol = float(valid_ret.std())
    annualized_volatility = float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    
    # Downside Deviation (relative to risk-free rate)
    downside_diff = valid_ret - RISK_FREE_RATE_DAILY
    downside_negative = np.minimum(downside_diff, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside_negative ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    
    # Downside Deviation (relative to 0 return)
    downside_zero = np.minimum(valid_ret, 0.0)
    downside_deviation_zero = float(np.sqrt(np.mean(downside_zero ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    
    # 3. Maximum Drawdown
    mdd_pct, current_dd_pct, peak_dt, trough_dt = calculate_max_drawdown_series(valid_nav)
    
    # 4. Tail Risk Metrics (VaR & CVaR)
    # Historical Value at Risk (VaR): loss quantile (positive convention)
    daily_var_95 = float(-np.percentile(valid_ret, 5.0) * 100.0)
    daily_var_99 = float(-np.percentile(valid_ret, 1.0) * 100.0)
    ann_var_95 = float(daily_var_95 * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    # Conditional VaR (Expected Shortfall): average of losses beyond VaR threshold
    tail_losses_95 = valid_ret[valid_ret <= -daily_var_95 / 100.0]
    daily_cvar_95 = float(-tail_losses_95.mean() * 100.0) if len(tail_losses_95) > 0 else daily_var_95
    
    tail_losses_99 = valid_ret[valid_ret <= -daily_var_99 / 100.0]
    daily_cvar_99 = float(-tail_losses_99.mean() * 100.0) if len(tail_losses_99) > 0 else daily_var_99
    ann_cvar_95 = float(daily_cvar_95 * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    # 5. Risk-Adjusted Return Ratios
    sharpe_ratio = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) / annualized_volatility) if annualized_volatility > 0 else 0.0
    sortino_ratio = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) / downside_deviation) if downside_deviation > 0 else 0.0
    calmar_ratio = float(cagr / abs(mdd_pct)) if abs(mdd_pct) > 0 else 0.0
    
    # 6. Distribution Moments & Trading Stats
    return_skewness = float(stats.skew(valid_ret))
    return_kurtosis = float(stats.kurtosis(valid_ret))  # Excess kurtosis
    positive_days_pct = float((valid_ret > 0).sum() / len(valid_ret) * 100.0)
    best_day_pct = float(valid_ret.max() * 100.0)
    worst_day_pct = float(valid_ret.min() * 100.0)
    
    # 7. Benchmark Relative Metrics (Calculated against reliable NIFTY 50 and NIFTY 500)
    # NIFTY 50
    aligned_50 = pd.concat([valid_ret, bm_returns["NIFTY50_Return"]], axis=1, join="inner").dropna()
    aligned_50.columns = ["scheme", "benchmark"]
    if len(aligned_50) >= 50:
        cov_matrix_50 = np.cov(aligned_50["scheme"], aligned_50["benchmark"])
        var_bm_50 = cov_matrix_50[1, 1]
        beta_50 = float(cov_matrix_50[0, 1] / var_bm_50) if var_bm_50 > 0 else 1.0
        
        # Benchmark CAGR
        bm_ret_series_50 = aligned_50["benchmark"]
        bm_cagr_50 = float(((1.0 + bm_ret_series_50).prod() ** (TRADING_DAYS_PER_YEAR / len(bm_ret_series_50)) - 1.0) * 100.0)
        alpha_jensen_50 = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) - beta_50 * (bm_cagr_50 - (RISK_FREE_RATE_ANNUAL * 100.0)))
        
        diff_50 = aligned_50["scheme"] - aligned_50["benchmark"]
        tracking_error_50 = float(diff_50.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
        information_ratio_50 = float((cagr - bm_cagr_50) / tracking_error_50) if tracking_error_50 > 0 else 0.0
        treynor_ratio_50 = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) / beta_50) if beta_50 != 0 else 0.0
        
        # Capture ratios
        up_days_50 = aligned_50[aligned_50["benchmark"] > 0]
        down_days_50 = aligned_50[aligned_50["benchmark"] < 0]
        up_capture_50 = float((up_days_50["scheme"].mean() / up_days_50["benchmark"].mean()) * 100.0) if not up_days_50.empty else 100.0
        down_capture_50 = float((down_days_50["scheme"].mean() / down_days_50["benchmark"].mean()) * 100.0) if not down_days_50.empty else 100.0
    else:
        beta_50 = alpha_jensen_50 = tracking_error_50 = information_ratio_50 = treynor_ratio_50 = up_capture_50 = down_capture_50 = None

    # NIFTY 500
    aligned_500 = pd.concat([valid_ret, bm_returns["NIFTY500_Return"]], axis=1, join="inner").dropna()
    aligned_500.columns = ["scheme", "benchmark"]
    if len(aligned_500) >= 50:
        cov_matrix_500 = np.cov(aligned_500["scheme"], aligned_500["benchmark"])
        var_bm_500 = cov_matrix_500[1, 1]
        beta_500 = float(cov_matrix_500[0, 1] / var_bm_500) if var_bm_500 > 0 else 1.0
        
        bm_ret_series_500 = aligned_500["benchmark"]
        bm_cagr_500 = float(((1.0 + bm_ret_series_500).prod() ** (TRADING_DAYS_PER_YEAR / len(bm_ret_series_500)) - 1.0) * 100.0)
        alpha_jensen_500 = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) - beta_500 * (bm_cagr_500 - (RISK_FREE_RATE_ANNUAL * 100.0)))
        
        diff_500 = aligned_500["scheme"] - aligned_500["benchmark"]
        tracking_error_500 = float(diff_500.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
        information_ratio_500 = float((cagr - bm_cagr_500) / tracking_error_500) if tracking_error_500 > 0 else 0.0
        treynor_ratio_500 = float((cagr - (RISK_FREE_RATE_ANNUAL * 100.0)) / beta_500) if beta_500 != 0 else 0.0
        
        up_days_500 = aligned_500[aligned_500["benchmark"] > 0]
        down_days_500 = aligned_500[aligned_500["benchmark"] < 0]
        up_capture_500 = float((up_days_500["scheme"].mean() / up_days_500["benchmark"].mean()) * 100.0) if not up_days_500.empty else 100.0
        down_capture_500 = float((down_days_500["scheme"].mean() / down_days_500["benchmark"].mean()) * 100.0) if not down_days_500.empty else 100.0
    else:
        beta_500 = alpha_jensen_500 = tracking_error_500 = information_ratio_500 = treynor_ratio_500 = up_capture_500 = down_capture_500 = None

    scheme_code_int = int(code_str)
    
    return {
        "scheme_code": scheme_code_int,
        "scheme_name": meta_row.get("scheme_name", ""),
        "fund_house": meta_row.get("fund_house", ""),
        "category": meta_row.get("category", ""),
        "sub_category": meta_row.get("sub_category", ""),
        "tier1_benchmark": meta_row.get("benchmark", "NIFTY 500 TRI"),
        "inception_date": meta_row.get("inception_date", ""),
        "expense_ratio_direct_pct": meta_row.get("expense_ratio_direct_pct", np.nan),
        "expense_ratio_regular_pct": meta_row.get("expense_ratio_regular_pct", np.nan),
        "start_date": valid_nav.index[0].strftime("%Y-%m-%d"),
        "end_date": valid_nav.index[-1].strftime("%Y-%m-%d"),
        "start_nav": round(start_val, 4),
        "end_nav": round(end_val, 4),
        "min_nav": round(float(valid_nav.min()), 4),
        "max_nav": round(float(valid_nav.max()), 4),
        "cumulative_return_pct": round(cumulative_return, 2),
        "cagr_5y_pct": round(cagr, 2),
        "annualized_arithmetic_return_pct": round(annualized_arithmetic_return, 2),
        "annualized_volatility_pct": round(annualized_volatility, 2),
        "downside_deviation_rf_pct": round(downside_deviation, 2),
        "downside_deviation_zero_pct": round(downside_deviation_zero, 2),
        "max_drawdown_pct": round(mdd_pct, 2),
        "current_drawdown_pct": round(current_dd_pct, 2),
        "mdd_peak_date": peak_dt,
        "mdd_trough_date": trough_dt,
        "daily_var_95_pct": round(daily_var_95, 2),
        "daily_var_99_pct": round(daily_var_99, 2),
        "annualized_var_95_pct": round(ann_var_95, 2),
        "daily_cvar_95_pct": round(daily_cvar_95, 2),
        "daily_cvar_99_pct": round(daily_cvar_99, 2),
        "annualized_cvar_95_pct": round(ann_cvar_95, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "sortino_ratio": round(sortino_ratio, 3),
        "calmar_ratio": round(calmar_ratio, 3),
        "skewness": round(return_skewness, 3),
        "kurtosis": round(return_kurtosis, 3),
        "win_rate_pct": round(positive_days_pct, 2),
        "best_day_pct": round(best_day_pct, 2),
        "worst_day_pct": round(worst_day_pct, 2),
        # Benchmark NIFTY 50
        "beta_nifty50": round(beta_50, 3) if beta_50 is not None else np.nan,
        "jensen_alpha_nifty50_pct": round(alpha_jensen_50, 2) if alpha_jensen_50 is not None else np.nan,
        "treynor_ratio_nifty50": round(treynor_ratio_50, 2) if treynor_ratio_50 is not None else np.nan,
        "tracking_error_nifty50_pct": round(tracking_error_50, 2) if tracking_error_50 is not None else np.nan,
        "information_ratio_nifty50": round(information_ratio_50, 3) if information_ratio_50 is not None else np.nan,
        "up_capture_nifty50_pct": round(up_capture_50, 2) if up_capture_50 is not None else np.nan,
        "down_capture_nifty50_pct": round(down_capture_50, 2) if down_capture_50 is not None else np.nan,
        # Benchmark NIFTY 500
        "beta_nifty500": round(beta_500, 3) if beta_500 is not None else np.nan,
        "jensen_alpha_nifty500_pct": round(alpha_jensen_500, 2) if alpha_jensen_500 is not None else np.nan,
        "treynor_ratio_nifty500": round(treynor_ratio_500, 2) if treynor_ratio_500 is not None else np.nan,
        "tracking_error_nifty500_pct": round(tracking_error_500, 2) if tracking_error_500 is not None else np.nan,
        "information_ratio_nifty500": round(information_ratio_500, 3) if information_ratio_500 is not None else np.nan,
        "up_capture_nifty500_pct": round(up_capture_500, 2) if up_capture_500 is not None else np.nan,
        "down_capture_nifty500_pct": round(down_capture_500, 2) if down_capture_500 is not None else np.nan
    }

def construct_ml_feature_set(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the specialized ML-ready dataset for K-Means clustering.
    Returns:
    1. Unscaled ML Features table with scheme descriptors.
    2. Standard-scaled (Z-score normalized) ML Features table ready for clustering algorithms.
    """
    logger.info("Constructing ML-Ready feature matrix for K-Means clustering...")
    id_cols = ["scheme_code", "scheme_name", "category", "fund_house", "tier1_benchmark"]
    
    # Selected 14 key numerical features capturing return, risk, tail risk, risk-adjusted performance, moments, and market sensitivity
    numeric_feature_cols = [
        "cagr_5y_pct",
        "annualized_volatility_pct",
        "downside_deviation_rf_pct",
        "max_drawdown_pct",
        "daily_var_95_pct",
        "daily_cvar_95_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "skewness",
        "kurtosis",
        "beta_nifty500",
        "tracking_error_nifty500_pct",
        "win_rate_pct"
    ]
    
    # 1. Unscaled Feature DataFrame
    ml_unscaled_df = metrics_df[id_cols + numeric_feature_cols].copy()
    
    # 2. Standard-Scaled Feature DataFrame (Z-Score: (x - mu) / sigma)
    ml_scaled_df = ml_unscaled_df.copy()
    for col in numeric_feature_cols:
        col_vals = ml_scaled_df[col].dropna()
        mu = col_vals.mean()
        sigma = col_vals.std()
        if sigma > 0:
            ml_scaled_df[col] = (ml_scaled_df[col] - mu) / sigma
            ml_scaled_df[col] = ml_scaled_df[col].round(4)
            
    # Export unscaled ML dataset to CSV
    ml_csv_path = os.path.join(PROCESSED_DIR, "ml_features_kmeans.csv")
    ml_unscaled_df.to_csv(ml_csv_path, index=False)
    logger.info(f"Exported ML-ready feature matrix to {ml_csv_path}")
    
    # Export scaled ML dataset to CSV
    ml_scaled_csv_path = os.path.join(PROCESSED_DIR, "ml_features_kmeans_standardized.csv")
    ml_scaled_df.to_csv(ml_scaled_csv_path, index=False)
    logger.info(f"Exported standardized ML feature matrix to {ml_scaled_csv_path}")
    
    return ml_unscaled_df, ml_scaled_df

def create_summary_statistics_tables(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates overall descriptive statistics and cross-sectional category summaries."""
    numeric_cols = [
        "cagr_5y_pct", "annualized_volatility_pct", "downside_deviation_rf_pct",
        "max_drawdown_pct", "daily_var_95_pct", "daily_cvar_95_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "beta_nifty500", "tracking_error_nifty500_pct", "skewness", "kurtosis"
    ]
    
    # Overall summary stats
    desc = metrics_df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skewness"] = metrics_df[numeric_cols].apply(stats.skew)
    desc["kurtosis"] = metrics_df[numeric_cols].apply(stats.kurtosis)
    desc = desc.round(3)
    desc = desc.reset_index().rename(columns={"index": "feature"})
    
    # Category summary
    cat_summary = metrics_df.groupby("category").agg(
        Scheme_Count=("scheme_code", "count"),
        Median_CAGR_pct=("cagr_5y_pct", "median"),
        Mean_CAGR_pct=("cagr_5y_pct", "mean"),
        Median_Volatility_pct=("annualized_volatility_pct", "median"),
        Median_Downside_Dev_pct=("downside_deviation_rf_pct", "median"),
        Median_Max_Drawdown_pct=("max_drawdown_pct", "median"),
        Median_Sharpe_Ratio=("sharpe_ratio", "median"),
        Median_Sortino_Ratio=("sortino_ratio", "median"),
        Median_Calmar_Ratio=("calmar_ratio", "median"),
        Median_Beta_Nifty500=("beta_nifty500", "median"),
        Median_Tracking_Error_pct=("tracking_error_nifty500_pct", "median")
    ).round(2).reset_index()
    
    return desc, cat_summary

def get_data_dictionary_content() -> pd.DataFrame:
    """Returns the comprehensive Data Dictionary as a structured DataFrame."""
    dict_records = [
        {
            "Variable_Name": "scheme_code",
            "Category": "Identification",
            "Data_Type": "Integer",
            "Formula / Source": "AMFI Official Scheme Code",
            "Units / Scale": "6-digit ID",
            "Description": "Unique numerical scheme identifier assigned by the Association of Mutual Funds in India."
        },
        {
            "Variable_Name": "scheme_name",
            "Category": "Identification",
            "Data_Type": "String",
            "Formula / Source": "AMFI Master Record",
            "Units / Scale": "Text",
            "Description": "Full official registered name of the mutual fund scheme."
        },
        {
            "Variable_Name": "fund_house",
            "Category": "Identification",
            "Data_Type": "String",
            "Formula / Source": "AMFI / AMC Disclosure",
            "Units / Scale": "Text",
            "Description": "Name of the Asset Management Company (AMC) managing the scheme."
        },
        {
            "Variable_Name": "category",
            "Category": "Classification",
            "Data_Type": "String",
            "Formula / Source": "SEBI Categorisation Circular 2017",
            "Units / Scale": "Text (12 Categories)",
            "Description": "Official SEBI investment category (e.g. Large Cap, Mid Cap, Small Cap, Flexi Cap, ELSS)."
        },
        {
            "Variable_Name": "tier1_benchmark",
            "Category": "Benchmark",
            "Data_Type": "String",
            "Formula / Source": "SEBI Two-Tier Benchmark Circular 2021",
            "Units / Scale": "Text",
            "Description": "Official Tier-1 benchmark index mandated for performance comparison."
        },
        {
            "Variable_Name": "start_nav / end_nav",
            "Category": "Price Levels",
            "Data_Type": "Float",
            "Formula / Source": "AMFI NAV Historical Series",
            "Units / Scale": "INR (Rs)",
            "Description": "Net Asset Value per unit at beginning (2021-10-18) and end (2026-10-16) of the 5-year evaluation period."
        },
        {
            "Variable_Name": "cumulative_return_pct",
            "Category": "Return Metric",
            "Data_Type": "Float",
            "Formula / Source": "(NAV_end - NAV_start) / NAV_start * 100",
            "Units / Scale": "Percentage (%)",
            "Description": "Total percentage capital appreciation over the complete 5-year period."
        },
        {
            "Variable_Name": "cagr_5y_pct",
            "Category": "Return Metric",
            "Data_Type": "Float",
            "Formula / Source": "((NAV_end / NAV_start) ^ (365.25 / delta_days) - 1) * 100",
            "Units / Scale": "Annualized %",
            "Description": "Compound Annual Growth Rate reflecting the smooth annual rate of return over 5 years."
        },
        {
            "Variable_Name": "annualized_volatility_pct",
            "Category": "Risk Metric",
            "Data_Type": "Float",
            "Formula / Source": "std(r_daily) * sqrt(250) * 100",
            "Units / Scale": "Annualized %",
            "Description": "Standard deviation of daily returns annualized assuming 250 trading sessions per year."
        },
        {
            "Variable_Name": "downside_deviation_rf_pct",
            "Category": "Risk Metric",
            "Data_Type": "Float",
            "Formula / Source": "sqrt(mean(min(r_daily - Rf_daily, 0)^2)) * sqrt(250) * 100",
            "Units / Scale": "Annualized %",
            "Description": "Semi-deviation focusing exclusively on returns below the risk-free rate threshold (Rf = 6.50% p.a.)."
        },
        {
            "Variable_Name": "max_drawdown_pct",
            "Category": "Risk Metric",
            "Data_Type": "Float",
            "Formula / Source": "min_t((NAV_t - cummax(NAV)) / cummax(NAV)) * 100",
            "Units / Scale": "Percentage (%) (negative)",
            "Description": "Maximum peak-to-trough decline experienced by the fund over the 5-year sample window."
        },
        {
            "Variable_Name": "daily_var_95_pct",
            "Category": "Tail Risk",
            "Data_Type": "Float",
            "Formula / Source": "-percentile(r_daily, 5.0) * 100",
            "Units / Scale": "Percentage (%)",
            "Description": "Historical Value at Risk at 95% confidence level: maximum expected daily loss on 95 out of 100 days."
        },
        {
            "Variable_Name": "daily_cvar_95_pct",
            "Category": "Tail Risk",
            "Data_Type": "Float",
            "Formula / Source": "-E[r_daily | r_daily <= -VaR_95] * 100",
            "Units / Scale": "Percentage (%)",
            "Description": "Conditional VaR (Expected Shortfall): average magnitude of loss when daily loss exceeds the 95% VaR threshold."
        },
        {
            "Variable_Name": "sharpe_ratio",
            "Category": "Risk-Adjusted Return",
            "Data_Type": "Float",
            "Formula / Source": "(CAGR - Rf_annual) / Annualized_Volatility",
            "Units / Scale": "Ratio (dimensionless)",
            "Description": "Excess return per unit of total risk (volatility) above the risk-free rate (Rf = 6.50%)."
        },
        {
            "Variable_Name": "sortino_ratio",
            "Category": "Risk-Adjusted Return",
            "Data_Type": "Float",
            "Formula / Source": "(CAGR - Rf_annual) / Downside_Deviation",
            "Units / Scale": "Ratio (dimensionless)",
            "Description": "Excess return per unit of harmful downside risk, penalizing only negative volatility."
        },
        {
            "Variable_Name": "calmar_ratio",
            "Category": "Risk-Adjusted Return",
            "Data_Type": "Float",
            "Formula / Source": "CAGR / abs(Max_Drawdown)",
            "Units / Scale": "Ratio (dimensionless)",
            "Description": "Annual compound return relative to the worst historical maximum drawdown."
        },
        {
            "Variable_Name": "skewness",
            "Category": "Distribution Moment",
            "Data_Type": "Float",
            "Formula / Source": "Fisher-Pearson sample skewness of r_daily",
            "Units / Scale": "Dimensionless",
            "Description": "Asymmetry of daily return distribution (negative skewness indicates frequent small gains and occasional large drops)."
        },
        {
            "Variable_Name": "kurtosis",
            "Category": "Distribution Moment",
            "Data_Type": "Float",
            "Formula / Source": "Fisher sample excess kurtosis of r_daily",
            "Units / Scale": "Dimensionless (Normal = 0)",
            "Description": "Fat-tailedness of returns; values > 0 indicate leptokurtic distribution with extreme tail event risk."
        },
        {
            "Variable_Name": "beta_nifty50 / beta_nifty500",
            "Category": "Systemic Risk",
            "Data_Type": "Float",
            "Formula / Source": "cov(r_scheme, r_benchmark) / var(r_benchmark)",
            "Units / Scale": "Coefficient",
            "Description": "Sensitivity of the scheme's returns to broad market movements in NIFTY 50 and NIFTY 500."
        },
        {
            "Variable_Name": "jensen_alpha_nifty500_pct",
            "Category": "Relative Performance",
            "Data_Type": "Float",
            "Formula / Source": "(CAGR_s - Rf) - Beta * (CAGR_bm - Rf)",
            "Units / Scale": "Annualized %",
            "Description": "Manager value-added: excess return generated beyond what is predicted by the CAPM model."
        },
        {
            "Variable_Name": "tracking_error_nifty500_pct",
            "Category": "Relative Risk",
            "Data_Type": "Float",
            "Formula / Source": "std(r_scheme - r_benchmark) * sqrt(250) * 100",
            "Units / Scale": "Annualized %",
            "Description": "Annualized standard deviation of excess returns relative to the benchmark index."
        },
        {
            "Variable_Name": "information_ratio_nifty500",
            "Category": "Relative Performance",
            "Data_Type": "Float",
            "Formula / Source": "(CAGR_scheme - CAGR_benchmark) / Tracking_Error",
            "Units / Scale": "Ratio",
            "Description": "Consistency of alpha generation per unit of active benchmark tracking risk."
        },
        {
            "Variable_Name": "up_capture / down_capture",
            "Category": "Market Timing",
            "Data_Type": "Float",
            "Formula / Source": "mean(r_scheme | bm > 0) / mean(bm | bm > 0) * 100",
            "Units / Scale": "Percentage (%)",
            "Description": "Percentage of benchmark gains captured in bull sessions vs. percentage of benchmark losses captured in bear sessions."
        },
        {
            "Variable_Name": "expense_ratio_direct_pct",
            "Category": "Cost",
            "Data_Type": "Float",
            "Formula / Source": "AMFI Official TER Disclosure",
            "Units / Scale": "Annualized %",
            "Description": "Total Expense Ratio (TER) charged by the AMC for Direct Growth plans."
        }
    ]
    return pd.DataFrame(dict_records)

def build_excel_workbook(
    excel_path: str,
    data_dict_df: pd.DataFrame,
    ml_unscaled_df: pd.DataFrame,
    ml_scaled_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_stats_df: pd.DataFrame,
    cat_summary_df: pd.DataFrame,
    corr_matrix_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    daily_returns_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    meta_df: pd.DataFrame
):
    """
    Builds the professional master Excel workbook 'Mutual_Fund_ML_Complete_Dataset.xlsx'
    with 10 formatted sheets and native Excel interactive charts using xlsxwriter.
    """
    logger.info(f"Assembling multi-sheet master Excel workbook at {excel_path}...")
    
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Define Professional Color Palette & Formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1B365D',  # Deep Navy
            'font_color': '#FFFFFF',
            'border': 1,
            'font_name': 'Segoe UI',
            'font_size': 10
        })
        
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#1B365D',
            'font_name': 'Segoe UI'
        })
        
        subtitle_format = workbook.add_format({
            'italic': True,
            'font_size': 10,
            'font_color': '#555555',
            'font_name': 'Segoe UI'
        })
        
        cell_format = workbook.add_format({
            'font_name': 'Segoe UI',
            'font_size': 9,
            'border': 1,
            'border_color': '#E0E0E0'
        })
        
        num_format_2dec = workbook.add_format({
            'font_name': 'Segoe UI',
            'font_size': 9,
            'num_format': '0.00',
            'border': 1,
            'border_color': '#E0E0E0'
        })
        
        num_format_3dec = workbook.add_format({
            'font_name': 'Segoe UI',
            'font_size': 9,
            'num_format': '0.000',
            'border': 1,
            'border_color': '#E0E0E0'
        })
        
        # Sheet 1: Readme & Data Dictionary
        logger.info("Writing Sheet 1: Readme_Data_Dictionary...")
        dict_sheet_name = "Readme_Data_Dictionary"
        data_dict_df.to_excel(writer, sheet_name=dict_sheet_name, index=False, startrow=3)
        ws_dict = writer.sheets[dict_sheet_name]
        ws_dict.write("A1", "Project: Machine Learning-Based Risk–Return Profiling of Indian Mutual Funds", title_format)
        ws_dict.write("A2", "Master Data Dictionary, Mathematical Formulations, and Variable Specifications (Rf = 6.50% p.a., N = 250 trading days/year)", subtitle_format)
        for col_idx, col_name in enumerate(data_dict_df.columns):
            ws_dict.write(3, col_idx, col_name, header_format)
            ws_dict.set_column(col_idx, col_idx, max(len(col_name) + 3, 22))

        # Sheet 2: ML_Ready_Features
        logger.info("Writing Sheet 2: ML_Ready_Features...")
        ml_sheet_name = "ML_Ready_Features"
        ml_unscaled_df.to_excel(writer, sheet_name=ml_sheet_name, index=False, startrow=2)
        ws_ml = writer.sheets[ml_sheet_name]
        ws_ml.write("A1", "Machine Learning-Ready Feature Matrix for K-Means Clustering (Raw Unscaled)", title_format)
        ws_ml.freeze_panes(3, 3)
        for col_idx, col_name in enumerate(ml_unscaled_df.columns):
            ws_ml.write(2, col_idx, col_name, header_format)
            ws_ml.set_column(col_idx, col_idx, max(len(col_name) + 3, 15))

        # Sheet 3: ML_Features_Standardized
        logger.info("Writing Sheet 3: ML_Features_Standardized...")
        std_sheet_name = "ML_Features_Standardized"
        ml_scaled_df.to_excel(writer, sheet_name=std_sheet_name, index=False, startrow=2)
        ws_std = writer.sheets[std_sheet_name]
        ws_std.write("A1", "Standardized Feature Matrix for K-Means Clustering (Z-score normalized: mean=0, std=1)", title_format)
        ws_std.freeze_panes(3, 3)
        for col_idx, col_name in enumerate(ml_scaled_df.columns):
            ws_std.write(2, col_idx, col_name, header_format)
            ws_std.set_column(col_idx, col_idx, max(len(col_name) + 3, 15))

        # Sheet 4: Risk_Return_Metrics
        logger.info("Writing Sheet 4: Risk_Return_Metrics...")
        metrics_sheet_name = "Risk_Return_Metrics"
        metrics_df.to_excel(writer, sheet_name=metrics_sheet_name, index=False, startrow=2)
        ws_metrics = writer.sheets[metrics_sheet_name]
        ws_metrics.write("A1", "Complete Risk-Return Performance Metrics (132 Schemes, 5-Year Evaluation Window)", title_format)
        ws_metrics.freeze_panes(3, 3)
        for col_idx, col_name in enumerate(metrics_df.columns):
            ws_metrics.write(2, col_idx, col_name, header_format)
            ws_metrics.set_column(col_idx, col_idx, max(len(col_name) + 3, 14))

        # Sheet 5: Summary_Statistics
        logger.info("Writing Sheet 5: Summary_Statistics...")
        stats_sheet_name = "Summary_Statistics"
        summary_stats_df.to_excel(writer, sheet_name=stats_sheet_name, index=False, startrow=2)
        ws_stats = writer.sheets[stats_sheet_name]
        ws_stats.write("A1", "Descriptive Statistics Across All 132 Equity Schemes", title_format)
        for col_idx, col_name in enumerate(summary_stats_df.columns):
            ws_stats.write(2, col_idx, col_name, header_format)
            ws_stats.set_column(col_idx, col_idx, 16)
            
        start_cat_row = len(summary_stats_df) + 6
        ws_stats.write(start_cat_row - 2, 0, "Cross-Sectional Category Aggregates (12 SEBI Categories)", title_format)
        cat_summary_df.to_excel(writer, sheet_name=stats_sheet_name, index=False, startrow=start_cat_row)
        for col_idx, col_name in enumerate(cat_summary_df.columns):
            ws_stats.write(start_cat_row, col_idx, col_name, header_format)

        # Sheet 6: Correlation_Matrix
        logger.info("Writing Sheet 6: Correlation_Matrix...")
        corr_sheet_name = "Correlation_Matrix"
        corr_matrix_df.to_excel(writer, sheet_name=corr_sheet_name, startrow=2)
        ws_corr = writer.sheets[corr_sheet_name]
        ws_corr.write("A1", "Pearson Correlation Matrix of Core Risk-Return Features", title_format)
        ws_corr.set_column(0, len(corr_matrix_df.columns), 18)

        # Sheet 7: Data_Quality_Report
        logger.info("Writing Sheet 7: Data_Quality_Report...")
        qual_sheet_name = "Data_Quality_Report"
        quality_df.to_excel(writer, sheet_name=qual_sheet_name, index=False, startrow=2)
        ws_qual = writer.sheets[qual_sheet_name]
        ws_qual.write("A1", "Data Quality & Completeness Audit (Zero-Fabrication Policy Verified)", title_format)
        for col_idx, col_name in enumerate(quality_df.columns):
            ws_qual.write(2, col_idx, col_name, header_format)
            ws_qual.set_column(col_idx, col_idx, 16)

        # Sheet 8: Daily_Returns
        logger.info("Writing Sheet 8: Daily_Returns...")
        ret_sheet_name = "Daily_Returns"
        daily_returns_df.to_excel(writer, sheet_name=ret_sheet_name, startrow=2)
        ws_ret = writer.sheets[ret_sheet_name]
        ws_ret.write("A1", "Daily Percentage Returns Time Series (1,231 Market Sessions x 132 Schemes)", title_format)
        ws_ret.freeze_panes(3, 1)
        ws_ret.set_column(0, 0, 14)

        # Sheet 9: Cleaned_Daily_NAV
        logger.info("Writing Sheet 9: Cleaned_Daily_NAV...")
        nav_sheet_name = "Cleaned_Daily_NAV"
        nav_df.to_excel(writer, sheet_name=nav_sheet_name, startrow=2)
        ws_nav = writer.sheets[nav_sheet_name]
        ws_nav.write("A1", "Cleaned Wide Daily NAV Series (1,232 Trading Days x 132 Schemes)", title_format)
        ws_nav.freeze_panes(3, 1)
        ws_nav.set_column(0, 0, 14)

        # Sheet 10: Scheme_Metadata
        logger.info("Writing Sheet 10: Scheme_Metadata...")
        meta_sheet_name = "Scheme_Metadata"
        meta_df.to_excel(writer, sheet_name=meta_sheet_name, index=False, startrow=2)
        ws_meta = writer.sheets[meta_sheet_name]
        ws_meta.write("A1", "Scheme Metadata & Regulatory Attributes", title_format)
        ws_meta.freeze_panes(3, 3)
        for col_idx, col_name in enumerate(meta_df.columns):
            ws_meta.write(2, col_idx, col_name, header_format)
            ws_meta.set_column(col_idx, col_idx, 16)

        # Sheet 11: Charts_Visualizations
        logger.info("Writing Sheet 11: Charts_Visualizations and building native Excel charts...")
        charts_sheet_name = "Charts_Visualizations"
        workbook.add_worksheet(charts_sheet_name)
        ws_charts = workbook.get_worksheet_by_name(charts_sheet_name)
        ws_charts.write("A1", "Project Visualizations & Empirical Risk–Return Profiles", title_format)
        ws_charts.write("A2", "Interactive native Excel charts generated directly from the underlying 5-year mutual fund dataset", subtitle_format)

        # CHART 1: Risk vs. Return Scatter Chart (Volatility % vs. CAGR %)
        chart_scatter1 = workbook.add_chart({'type': 'scatter'})
        # Data is in Risk_Return_Metrics:
        # Col P is cagr_5y_pct (index 15), Col R is annualized_volatility_pct (index 17)
        # Row 3 to 134 (132 schemes)
        chart_scatter1.add_series({
            'name': '132 Equity Schemes',
            'categories': f"='{metrics_sheet_name}'!$R$4:$R$135",  # X: Volatility
            'values': f"='{metrics_sheet_name}'!$P$4:$P$135",      # Y: CAGR
            'marker': {
                'type': 'circle',
                'size': 6,
                'border': {'color': '#1B365D'},
                'fill': {'color': '#2E6B9E'}
            }
        })
        chart_scatter1.set_title({'name': 'Risk vs. Return: Annualized Volatility vs. 5Y CAGR'})
        chart_scatter1.set_x_axis({'name': 'Annualized Volatility (%)', 'grid': {'visible': True}})
        chart_scatter1.set_y_axis({'name': '5-Year CAGR (%)', 'grid': {'visible': True}})
        chart_scatter1.set_size({'width': 720, 'height': 420})
        ws_charts.insert_chart('A4', chart_scatter1)

        # CHART 2: Sharpe Ratio vs. Sortino Ratio Scatter Chart
        chart_scatter2 = workbook.add_chart({'type': 'scatter'})
        # Sharpe is Col AE (index 30), Sortino is Col AF (index 31)
        chart_scatter2.add_series({
            'name': 'Schemes',
            'categories': f"='{metrics_sheet_name}'!$AE$4:$AE$135",  # X: Sharpe
            'values': f"='{metrics_sheet_name}'!$AF$4:$AF$135",      # Y: Sortino
            'marker': {
                'type': 'diamond',
                'size': 6,
                'border': {'color': '#006400'},
                'fill': {'color': '#2E8B57'}
            }
        })
        chart_scatter2.set_title({'name': 'Risk-Adjusted Efficiency: Sharpe Ratio vs. Sortino Ratio'})
        chart_scatter2.set_x_axis({'name': 'Sharpe Ratio (Total Risk)', 'grid': {'visible': True}})
        chart_scatter2.set_y_axis({'name': 'Sortino Ratio (Downside Risk)', 'grid': {'visible': True}})
        chart_scatter2.set_size({'width': 720, 'height': 420})
        ws_charts.insert_chart('M4', chart_scatter2)

        # CHART 3: Category Performance Column Chart (Median CAGR by Category)
        # Located in Summary_Statistics at start_cat_row
        cat_rows = len(cat_summary_df)
        chart_bar1 = workbook.add_chart({'type': 'column'})
        chart_bar1.add_series({
            'name': 'Median 5Y CAGR (%)',
            'categories': f"='{stats_sheet_name}'!$A${start_cat_row + 2}:$A${start_cat_row + 1 + cat_rows}",
            'values': f"='{stats_sheet_name}'!$B${start_cat_row + 2}:$B${start_cat_row + 1 + cat_rows}",
            'fill': {'color': '#1B365D'},
            'data_labels': {'value': True}
        })
        chart_bar1.set_title({'name': 'Median 5-Year CAGR by SEBI Equity Category'})
        chart_bar1.set_y_axis({'name': 'Median CAGR (%)'})
        chart_bar1.set_size({'width': 720, 'height': 420})
        ws_charts.insert_chart('A26', chart_bar1)

        # CHART 4: Category Drawdown Column Chart (Median Max Drawdown by Category)
        chart_bar2 = workbook.add_chart({'type': 'column'})
        chart_bar2.add_series({
            'name': 'Median Max Drawdown (%)',
            'categories': f"='{stats_sheet_name}'!$A${start_cat_row + 2}:$A${start_cat_row + 1 + cat_rows}",
            'values': f"='{stats_sheet_name}'!$F${start_cat_row + 2}:$F${start_cat_row + 1 + cat_rows}",
            'fill': {'color': '#B22222'},  # Dark Crimson
            'data_labels': {'value': True}
        })
        chart_bar2.set_title({'name': 'Maximum Drawdown Severity by SEBI Category'})
        chart_bar2.set_y_axis({'name': 'Median Max Drawdown (%)'})
        chart_bar2.set_size({'width': 720, 'height': 420})
        ws_charts.insert_chart('M26', chart_bar2)

    logger.info(f"Master workbook successfully built and saved to {excel_path}")

def run_ml_dataset_generation():
    """Main pipeline execution for ML Dataset generation."""
    logger.info("=================================================================")
    logger.info("BUILDING COMPLETE ML DATASET & EXCEL WORKBOOK")
    logger.info("=================================================================")
    
    # 1. Load underlying cleaned 5-year data
    nav_df, meta_df, quality_df = load_underlying_data()
    
    # 2. Acquire & align market benchmarks
    bm_returns = fetch_and_align_benchmarks(nav_df.index)
    
    # 3. Calculate daily returns matrix
    daily_returns_df = compute_daily_returns_matrix(nav_df)
    
    # 4. Compute all risk-return metrics per scheme
    logger.info("Calculating comprehensive risk-return metrics for all 132 schemes...")
    metrics_list = []
    meta_indexed = meta_df.set_index("scheme_code")
    
    for code_str in nav_df.columns:
        nav_series = nav_df[code_str]
        ret_series = daily_returns_df[code_str] if code_str in daily_returns_df.columns else pd.Series(dtype=float)
        code_int = int(code_str)
        meta_row = meta_indexed.loc[code_int] if code_int in meta_indexed.index else pd.Series(dtype=object)
        
        m = calculate_metrics_for_scheme(code_str, nav_series, ret_series, meta_row, bm_returns)
        if m:
            metrics_list.append(m)
            
    metrics_df = pd.DataFrame(metrics_list)
    logger.info(f"Computed metrics table: {metrics_df.shape[0]} schemes, {metrics_df.shape[1]} attributes.")
    
    # Export full metrics to CSV
    metrics_csv_path = os.path.join(PROCESSED_DIR, "risk_return_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Exported calculated metrics to {metrics_csv_path}")
    
    # 5. Build ML-Ready dataset for K-Means clustering (unscaled & standard-scaled)
    ml_unscaled_df, ml_scaled_df = construct_ml_feature_set(metrics_df)
    
    # 6. Generate Summary Statistics & Correlation Matrix
    summary_stats_df, cat_summary_df = create_summary_statistics_tables(metrics_df)
    
    # Feature correlation matrix
    numeric_feature_cols = [
        "cagr_5y_pct", "annualized_volatility_pct", "downside_deviation_rf_pct",
        "max_drawdown_pct", "daily_var_95_pct", "daily_cvar_95_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "beta_nifty500", "tracking_error_nifty500_pct", "skewness", "kurtosis"
    ]
    corr_matrix_df = metrics_df[numeric_feature_cols].corr().round(3)
    corr_csv_path = os.path.join(PROCESSED_DIR, "features_correlation_matrix.csv")
    corr_matrix_df.to_csv(corr_csv_path)
    logger.info(f"Exported correlation matrix to {corr_csv_path}")
    
    # 7. Data Dictionary
    data_dict_df = get_data_dictionary_content()
    
    # 8. Build Master Excel Workbook
    build_excel_workbook(
        excel_path=OUTPUT_EXCEL_PATH,
        data_dict_df=data_dict_df,
        ml_unscaled_df=ml_unscaled_df,
        ml_scaled_df=ml_scaled_df,
        metrics_df=metrics_df,
        summary_stats_df=summary_stats_df,
        cat_summary_df=cat_summary_df,
        corr_matrix_df=corr_matrix_df,
        quality_df=quality_df,
        daily_returns_df=daily_returns_df,
        nav_df=nav_df,
        meta_df=meta_df
    )
    
    logger.info("=================================================================")
    logger.info(f"ALL ML DATASETS AND EXCEL WORKBOOK GENERATED SUCCESSFULLY!")
    logger.info(f"Master Workbook: {OUTPUT_EXCEL_PATH}")
    logger.info("=================================================================")

if __name__ == "__main__":
    run_ml_dataset_generation()

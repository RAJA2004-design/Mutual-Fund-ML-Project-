"""
Module 2: Data Cleaning, Validation, and Dataset Serialization
Cleans raw JSON historical NAV feeds, strictly enforces the non-fabrication rule for missing values,
detects outliers and anomalies, performs financial audit checks, and exports CSV & Excel datasets.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_SCHEMES_DIR = os.path.join(RAW_DATA_DIR, "schemes_raw_json")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 5-Year Analysis Window
START_DATE = "2021-10-18"
END_DATE = "2026-10-16"

def parse_date_str(d_str: str) -> str:
    """Standardizes date strings (DD-MM-YYYY or YYYY-MM-DD) to ISO format YYYY-MM-DD."""
    d_str = d_str.strip()
    try:
        if "-" in d_str:
            parts = d_str.split("-")
            if len(parts[0]) == 4:
                return d_str  # Already YYYY-MM-DD
            elif len(parts[2]) == 4:
                return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return d_str
    except Exception:
        return d_str

def load_and_clean_scheme_history(scheme_code: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw JSON for a scheme, cleans dates and NAV, removes duplicates,
    filters to the 5-year window, and validates numeric integrity.
    """
    file_path = os.path.join(RAW_SCHEMES_DIR, f"scheme_{scheme_code}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found for scheme {scheme_code}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    meta = data.get("meta", {})
    raw_records = data.get("data", [])
    
    if not raw_records:
        return pd.DataFrame(), meta
        
    parsed_rows = []
    for r in raw_records:
        raw_dt = r.get("date", "")
        raw_nav = r.get("nav", "")
        iso_dt = parse_date_str(raw_dt)
        try:
            val = float(raw_nav)
            parsed_rows.append({"date": iso_dt, "nav": val})
        except (ValueError, TypeError):
            continue
            
    df = pd.DataFrame(parsed_rows)
    if df.empty:
        return df, meta
        
    # Deduplicate dates (keep first verified record)
    df = df.drop_duplicates(subset=["date"], keep="first")
    
    # Sort chronologically ascending
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter to 5-year lookback window
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    df_5y = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    df_5y.reset_index(drop=True, inplace=True)
    
    # Validation checks
    assert (df_5y["nav"] > 0).all(), f"Error: Non-positive NAV encountered in scheme {scheme_code}!"
    
    return df_5y, meta

def execute_cleaning_and_validation():
    """Main cleaning, alignment, validation, and serialization pipeline."""
    manifest_path = os.path.join(RAW_DATA_DIR, "schemes_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run fetch_data.py first.")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        schemes_manifest = json.load(f)
        
    logger.info(f"Loaded {len(schemes_manifest)} schemes from manifest.")
    
    # Containers
    cleaned_series_dict = {}
    scheme_data_quality_records = []
    updated_metadata_list = []
    all_anomalies = []
    
    # 1. First pass: Clean individual scheme records and discover market trading days
    date_occurrence_counter = {}
    
    for s_info in schemes_manifest:
        code = s_info["scheme_code"]
        name = s_info["scheme_name"]
        df_5y, meta = load_and_clean_scheme_history(code)
        
        if df_5y.empty:
            logger.warning(f"Scheme {code} has no records in 5-year window!")
            continue
            
        cleaned_series_dict[code] = df_5y
        
        for dt in df_5y["date"]:
            date_occurrence_counter[dt] = date_occurrence_counter.get(dt, 0) + 1

    # 2. Establish Standard Indian Equity Mutual Fund Trading Calendar
    # A day is an official market trading day if >= 50% of active equity schemes reported NAV
    total_schemes = len(cleaned_series_dict)
    threshold = total_schemes * 0.50
    official_trading_days = sorted([dt for dt, count in date_occurrence_counter.items() if count >= threshold])
    trading_calendar_index = pd.DatetimeIndex(official_trading_days)
    
    logger.info(f"Identified {len(trading_calendar_index)} standard mutual fund trading days between {START_DATE} and {END_DATE}.")
    
    # 3. Second pass: Align against trading calendar, identify gaps without fabrication, compute audit metrics
    wide_dict = {}
    long_nav_rows = []
    
    for s_info in schemes_manifest:
        code = s_info["scheme_code"]
        name = s_info["scheme_name"]
        cat = s_info["category"]
        
        if code not in cleaned_series_dict:
            continue
            
        df_scheme = cleaned_series_dict[code]
        df_scheme_indexed = df_scheme.set_index("date")
        
        # Reindex to trading calendar:
        # STRICT NON-FABRICATION: Gaps become NaN. No ffill/bfill/interpolation!
        aligned_series = df_scheme_indexed["nav"].reindex(trading_calendar_index)
        wide_dict[str(code)] = aligned_series
        
        # Populate Long Format rows
        for dt, nav_val in aligned_series.items():
            long_nav_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "scheme_code": code,
                "scheme_name": name,
                "category": cat,
                "nav": nav_val if pd.notna(nav_val) else None
            })
            
        # Quality Metrics Calculation
        total_expected = len(trading_calendar_index)
        present_count = int(aligned_series.count())
        missing_count = int(aligned_series.isna().sum())
        completeness_pct = round((present_count / total_expected) * 100, 2)
        
        # Non-null values for financial calculations
        valid_navs = aligned_series.dropna()
        if len(valid_navs) >= 2:
            start_val = float(valid_navs.iloc[0])
            end_val = float(valid_navs.iloc[-1])
            min_val = float(valid_navs.min())
            max_val = float(valid_navs.max())
            
            # 5-Year CAGR: (End/Start)^(1/5) - 1
            years = (valid_navs.index[-1] - valid_navs.index[0]).days / 365.25
            cagr = ((end_val / start_val) ** (1.0 / max(years, 1.0)) - 1.0) * 100.0 if years > 0 and start_val > 0 else 0.0
            
            # Daily returns & annualized volatility
            daily_returns = valid_navs.pct_change().dropna()
            ann_volatility = float(daily_returns.std() * np.sqrt(250) * 100.0) if len(daily_returns) > 1 else 0.0
            
            # Outlier / Anomaly Detection: Single-day absolute jump > 15%
            extreme_moves = daily_returns[daily_returns.abs() > 0.15]
            if not extreme_moves.empty:
                for dt_move, ret_val in extreme_moves.items():
                    all_anomalies.append({
                        "scheme_code": code,
                        "scheme_name": name,
                        "date": dt_move.strftime("%Y-%m-%d"),
                        "daily_return_pct": round(ret_val * 100, 2),
                        "nav": valid_navs.loc[dt_move]
                    })
        else:
            start_val = end_val = min_val = max_val = cagr = ann_volatility = None

        earliest_5y_dt = valid_navs.index[0].strftime("%Y-%m-%d") if not valid_navs.empty else None
        latest_5y_dt = valid_navs.index[-1].strftime("%Y-%m-%d") if not valid_navs.empty else None
        
        # Scheme metadata enrichment
        meta_row = dict(s_info)
        meta_row.update({
            "start_date_5y": earliest_5y_dt,
            "end_date_5y": latest_5y_dt,
            "expected_trading_days": total_expected,
            "actual_trading_days_present": present_count,
            "missing_trading_days": missing_count,
            "completeness_pct": completeness_pct,
            "start_nav_5y": start_val,
            "end_nav_5y": end_val,
            "min_nav_5y": min_val,
            "max_nav_5y": max_val,
            "cagr_5y_pct": round(cagr, 2) if cagr is not None else None,
            "annualized_volatility_pct": round(ann_volatility, 2) if ann_volatility is not None else None
        })
        updated_metadata_list.append(meta_row)
        
        # Quality Summary record
        scheme_data_quality_records.append({
            "scheme_code": code,
            "scheme_name": name,
            "category": cat,
            "fund_house": s_info.get("fund_house"),
            "expected_days": total_expected,
            "present_days": present_count,
            "missing_days": missing_count,
            "completeness_pct": completeness_pct,
            "anomalies_detected": len(extreme_moves) if len(valid_navs) >= 2 else 0
        })

    # Convert to DataFrames
    wide_nav_df = pd.DataFrame(wide_dict, index=trading_calendar_index)
    wide_nav_df.index.name = "date"
    metadata_df = pd.DataFrame(updated_metadata_list)
    quality_df = pd.DataFrame(scheme_data_quality_records)
    long_nav_df = pd.DataFrame(long_nav_rows)
    
    # 4. Serialize to CSV files
    logger.info("Serializing cleaned datasets to CSV...")
    wide_csv_path = os.path.join(PROCESSED_DATA_DIR, "equity_nav_5y_wide.csv")
    wide_nav_df.to_csv(wide_csv_path)
    logger.info(f"Exported Wide NAV Matrix: {wide_csv_path} (Shape: {wide_nav_df.shape})")
    
    long_csv_path = os.path.join(PROCESSED_DATA_DIR, "equity_nav_5y_long.csv")
    long_nav_df.to_csv(long_csv_path, index=False)
    logger.info(f"Exported Long NAV Table: {long_csv_path} (Shape: {long_nav_df.shape})")
    
    meta_csv_path = os.path.join(PROCESSED_DATA_DIR, "scheme_metadata.csv")
    metadata_df.to_csv(meta_csv_path, index=False)
    logger.info(f"Exported Scheme Metadata: {meta_csv_path} (Shape: {metadata_df.shape})")
    
    quality_csv_path = os.path.join(PROCESSED_DATA_DIR, "data_quality_summary.csv")
    quality_df.to_csv(quality_csv_path, index=False)
    logger.info(f"Exported Data Quality Summary: {quality_csv_path} (Shape: {quality_df.shape})")
    
    # 5. Serialize to Excel Workbooks
    logger.info("Serializing to Excel workbooks (.xlsx)...")
    
    # Combined Multi-Sheet Workbook for NAV Data
    excel_nav_path = os.path.join(PROCESSED_DATA_DIR, "equity_nav_5y.xlsx")
    with pd.ExcelWriter(excel_nav_path, engine="openpyxl") as writer:
        # Sheet 1: Metadata Catalog
        metadata_df.to_excel(writer, sheet_name="Scheme_Catalog", index=False)
        # Sheet 2: Wide Daily NAV Matrix
        wide_nav_df.to_excel(writer, sheet_name="Daily_NAV_Matrix")
        # Sheet 3: Quality Summary
        quality_df.to_excel(writer, sheet_name="Data_Quality_Audit", index=False)
    logger.info(f"Exported Multi-Sheet Excel Workbook: {excel_nav_path}")
    
    # Dedicated Metadata Excel Workbook
    excel_meta_path = os.path.join(PROCESSED_DATA_DIR, "scheme_metadata.xlsx")
    metadata_df.to_excel(excel_meta_path, index=False, engine="openpyxl")
    logger.info(f"Exported Scheme Metadata Excel: {excel_meta_path}")
    
    # Save anomalies audit file
    anomalies_path = os.path.join(PROCESSED_DATA_DIR, "detected_anomalies.json")
    with open(anomalies_path, "w", encoding="utf-8") as f:
        json.dump(all_anomalies, f, indent=2)
    logger.info(f"Preserved {len(all_anomalies)} detected market anomalies to {anomalies_path}")
    
    return {
        "wide_nav_shape": wide_nav_df.shape,
        "long_nav_shape": long_nav_df.shape,
        "total_schemes": len(metadata_df),
        "total_trading_days": len(trading_calendar_index),
        "anomalies_count": len(all_anomalies)
    }

if __name__ == "__main__":
    execute_cleaning_and_validation()

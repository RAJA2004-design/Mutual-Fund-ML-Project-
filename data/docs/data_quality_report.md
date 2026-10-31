# Data Quality & Integrity Report: 5-Year Indian Equity Mutual Fund NAV Dataset

**Date of Audit:** October 17, 2026  
**Temporal Window:** 2021-10-18 to 2026-10-16 (5 Years Lookback)  
**Total Target Schemes:** 132 Indian Equity Schemes (Direct Growth Plans)  
**Total Identified Trading Calendar Days:** 1232 days  
**Overall Dataset Completeness:** 99.98%  
**Policy on Missing Values:** **Strict Zero-Fabrication (No Imputation, No Forward-Fill)**

---

## 1. Executive Summary

This report provides an end-to-end data quality audit for the curated historical dataset of **132 Indian open-ended equity mutual fund schemes**. The dataset encompasses daily historical Net Asset Value (NAV) records spanning five calendar years (2021-10-18 to 2026-10-16) sourced directly from official Association of Mutual Funds in India (AMFI) portals and verified daily feeds.

### Key Quality Metrics at a Glance
| Metric | Value | Audit Status |
| :--- | :--- | :--- |
| **Total Schemes Audited** | 132 schemes | Verified (Target 100–150 achieved) |
| **SEBI Equity Categories Covered** | 12 distinct categories | Full market-cap and sector coverage |
| **Trading Days in 5Y Window** | 1232 days | Aligned with NSE/BSE business calendar |
| **Total NAV Points Recorded** | 162,595 observations | 100% verified authentic points |
| **Total Missing Data Points** | 29 (0.02%) | Unimputed / preserved as `NaN` |
| **Fully Complete Schemes (>=99.5%)** | 132 schemes (100.0%) | Excellent continuity |
| **High Continuity Schemes (95.0% - 99.5%)** | 0 schemes (0.0%) | High reliability |
| **Schemes Below 95% Continuity** | 0 schemes | Tracked & documented |
| **Non-Positive NAVs ($NAV \le 0$)** | 0 occurrences | Zero errors |
| **Extreme Outlier Returns ($|R_t| > 15\%$)** | 0 events | Audited (no split artifact errors) |

---

## 2. Policy Enforcement: Non-Fabrication of Missing Values

> [!IMPORTANT]
> **Zero-Fabrication Compliance**
> In strict adherence to quantitative data integrity and user guidelines, **no missing values were synthesized, interpolated, forward-filled (`ffill`), or backward-filled (`bfill`)**. 
>
> Any missing trading day for a scheme remains an explicit `NaN` in the wide matrix and is omitted from long records, ensuring that future machine learning models or statistical estimators are never biased by artificial constant prices or synthetic return trajectories.

### Analysis of Missing Values
- **Trading Gaps:** Missing values reflect genuine non-reporting or delayed reporting dates on specific regional holidays or administrative disclosure lags by respective AMCs.
- **Calendar Alignment:** All schemes are mapped against the unified empirical market trading calendar (1232 trading days), allowing precise multi-asset time-series synchronization.

---

## 3. Coverage by Equity Category

The dataset balances 12 distinct SEBI equity mutual fund categories, providing broad diversification across large-cap, mid-cap, small-cap, thematic, and value/contra investment styles.

| Equity Category | Scheme Count | Avg Completeness (%) | Median 5Y CAGR (%) | Min 5Y CAGR (%) | Max 5Y CAGR (%) | Median Volatility (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contra Fund** | 3 | 99.95% | 12.30% | 11.16% | 13.03% | 14.36% |
| **Dividend Yield Fund** | 4 | 99.96% | 10.60% | 9.13% | 14.05% | 12.94% |
| **ELSS** | 14 | 100.00% | 9.91% | 4.26% | 15.87% | 13.79% |
| **Flexi Cap Fund** | 14 | 99.98% | 8.93% | 3.19% | 14.46% | 13.71% |
| **Focused Fund** | 12 | 99.99% | 10.20% | 2.23% | 15.50% | 14.29% |
| **Large & Mid Cap Fund** | 12 | 99.97% | 10.89% | 7.04% | 17.38% | 14.63% |
| **Large Cap Fund** | 14 | 99.99% | 7.80% | 4.47% | 11.27% | 13.41% |
| **Mid Cap Fund** | 14 | 99.99% | 13.74% | 9.36% | 16.57% | 15.50% |
| **Multi Cap Fund** | 7 | 99.99% | 12.51% | 9.30% | 14.51% | 15.27% |
| **Sectoral/ Thematic** | 14 | 99.96% | 9.87% | 2.81% | 16.53% | 15.79% |
| **Small Cap Fund** | 14 | 99.99% | 13.88% | 6.52% | 17.87% | 16.02% |
| **Value Fund** | 10 | 99.97% | 10.80% | 6.22% | 12.21% | 13.69% |

---

## 4. Top Asset Management Companies (AMCs) Represented

The dataset features leading Indian fund houses with high institutional governance, long-standing historical track records, and large asset bases:

| Asset Management Company (AMC) | Schemes Included |
| :--- | :---: |
| Franklin Templeton Mutual Fund | 13 |
| ICICI Prudential Mutual Fund | 12 |
| HDFC Mutual Fund | 9 |
| SBI Mutual Fund | 8 |
| Bandhan Mutual Fund | 7 |
| DSP Mutual Fund | 7 |
| Edelweiss Mutual Fund | 7 |
| Nippon India Mutual Fund | 7 |
| Tata Mutual Fund | 7 |
| quant Mutual Fund | 7 |
| Axis Mutual Fund | 7 |
| Kotak Mahindra Mutual Fund | 6 |

---

## 5. SEBI Tier-1 Benchmark Mapping

As per SEBI circular `SEBI/HO/IMD/IMD-II_DOF7/P/CIR/2021/652`, each equity mutual fund scheme is benchmarked against an official Tier-1 index reflecting its universe:

| Benchmark Index | Number of Schemes |
| :--- | :---: |
| NIFTY 500 TRI | 57 |
| NIFTY 100 TRI | 14 |
| NIFTY Midcap 150 TRI | 14 |
| NIFTY Smallcap 250 TRI | 14 |
| NIFTY LargeMidcap 250 TRI | 12 |
| NIFTY 500 Multicap 50:25:25 TRI | 7 |
| NIFTY Dividend Opportunities 50 TRI | 4 |
| NIFTY IT TRI | 4 |
| NIFTY Bank TRI / NIFTY Financial Services TRI | 2 |
| BSE TeCK TRI / NIFTY IT TRI | 1 |
| NIFTY Infrastructure TRI | 1 |
| NIFTY India Consumption TRI | 1 |
| NIFTY Services Sector TRI | 1 |

---

## 6. Financial Audit & Statistical Summary

### Descriptive Statistics across All 132 Schemes
- **5-Year Compound Annual Growth Rate (CAGR):**
  - **Mean:** 10.73%
  - **Median:** 10.52%
  - **Standard Deviation:** 3.18%
  - **Interquartile Range (IQR):** 8.52% (25th percentile) to 12.72% (75th percentile)
  - **Min CAGR:** 2.23%
  - **Max CAGR:** 17.87%

- **Annualized Volatility (Standard Deviation of Daily Returns $\times \sqrt{250}$):**
  - **Mean:** 14.62%
  - **Median:** 14.42%
  - **Range:** 11.47% to 19.53%

### Outlier & Anomaly Audit Findings
- **Single-Day Return Anomalies ($|R_t| > 15\%$):** A total of **0** occurrences were flagged during the scan.
- All non-zero NAV values were strictly verified positive ($NAV > 0$).
- All dates are verified monotonically ascending with zero duplicate date collisions.

---

## 7. Artifacts & Generated Files Index

The generated datasets are stored in `d:\MF_ML_P\data\processed\` with dual CSV and Excel formats:

1. **`equity_nav_5y_wide.csv`**: Wide daily NAV matrix (1232 rows $\times$ 132 scheme columns).
2. **`equity_nav_5y_long.csv`**: Normalized tabular format with columns `date`, `scheme_code`, `scheme_name`, `category`, `nav`.
3. **`equity_nav_5y.xlsx`**: Comprehensive Excel workbook containing sheets:
   - `Daily_NAV_Matrix`: Aligned wide daily time series.
   - `Scheme_Catalog`: Full metadata catalog for all 132 schemes.
   - `Data_Quality_Audit`: Per-scheme quality and completeness metrics.
4. **`scheme_metadata.csv` & `scheme_metadata.xlsx`**: Detailed metadata table covering Scheme Code, Scheme Name, AMC, Category, Benchmark, Inception Date, TER (Direct & Regular), and 5-Year summary statistics.
5. **`data_quality_summary.csv`**: Per-scheme audit table detailing expected vs actual observations and completeness percentages.
6. **`detected_anomalies.json`**: Machine-readable audit log of any flagged daily return spikes.

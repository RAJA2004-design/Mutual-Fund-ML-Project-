# Official Data Sources Documentation: Indian Mutual Fund Historical NAV & Metadata

This document catalogs all official authorities, statutory portals, application programming interfaces (APIs), and regulatory frameworks utilized to construct the 5-year Indian equity mutual fund dataset.

---

## 1. Association of Mutual Funds in India (AMFI)

AMFI is the non-profit apex regulatory and trade association of all SEBI-registered Asset Management Companies (AMCs) in India. By statutory mandate under the SEBI (Mutual Funds) Regulations, 1996, AMFI hosts and publishes official daily Net Asset Value (NAV) feeds, portfolio disclosures, and expense ratios.

### Official Feeds & Endpoints Utilized:
1. **Master Daily NAV Text Feed:**
   - **URL:** `https://www.amfiindia.com/spages/NAVAll.txt`
   - **Update Frequency:** Daily (updated on every trading day post-market close, typically between 21:00 and 23:00 IST).
   - **Data Schema:** Semicolon-delimited format containing:
     - `Scheme Code` (AMFI 6-digit numeric identifier)
     - `ISIN Div Payout / ISIN Growth`
     - `ISIN Div Reinvestment`
     - `Scheme Name`
     - `Plan` (Direct Plan / Regular Plan)
     - `Option` (Growth Option / IDCW Option)
     - `Net Asset Value`
     - `Date` (DD-Mmm-YYYY)
   - **Local Raw Preservation:** Archived intact in `data/raw/nav_all_amfi.txt`.

2. **Official AMFI Historical NAV Report Portal:**
   - **URL:** `https://www.amfiindia.com/net-asset-value/nav-history`
   - **Backend Query Endpoint:** `https://www.amfiindia.com/api/nav-history`
   - **Parameters:** `query_type=historical_period&from_date=YYYY-MM-DD&to_date=YYYY-MM-DD&sd_id=<scheme_code>`
   - **Coverage:** Historical daily NAV archives from 2006 to present.

3. **Official AMFI Total Expense Ratio (TER) Disclosures:**
   - **URL:** `https://www.amfiindia.com/ter-of-mf-schemes`
   - **Backend API Endpoint:** `https://www.amfiindia.com/api/populate-te-rdata-revised`
   - **Regulatory Reference:** Regulation 66(7), 66(9), and 66(10) of SEBI (Mutual Funds) Regulations.
   - **Data Schema:**
     - Direct Plan TER (`D_TER`) & Base Expense Ratio (`D_BER`)
     - Regular Plan TER (`R_TER`) & Base Expense Ratio (`R_BER`)
     - Brokerage and Transaction costs
     - Disclosure Date (`TER_Date`)
   - **Local Raw Preservation:** Archived intact in `data/raw/amfi_ter_disclosures.json`.

---

## 2. Open AMFI Mutual Fund API (`mfapi.in`)

`mfapi.in` is an open-access public API that mirrors and archives AMFI's official daily feeds, providing programmatic access to the entire time-series history of AMFI scheme codes.

- **Base Endpoint:** `https://api.mfapi.in/mf/{scheme_code}`
- **Fidelity & Parity:** Verified 100% identical in NAV values and dates against official AMFI text dumps and AMC daily uploads.
- **Data Schema:**
  - `meta`: Fund house, scheme type, scheme category, scheme code, scheme name, ISIN.
  - `data`: List of historical records `{"date": "DD-MM-YYYY", "nav": "<float>"}` dating back to scheme inception.
- **Local Raw Preservation:** Each scheme's untouched response is preserved as `data/raw/schemes_raw_json/scheme_<scheme_code>.json`.

---

## 3. Securities and Exchange Board of India (SEBI) Statutory Framework

All categorization, scheme definitions, and benchmark assignments are governed by binding SEBI circulars:

1. **Categorization and Rationalization of Mutual Fund Schemes:**
   - **Circular:** `SEBI/HO/IMD/DF3/CIR/P/2017/114` (dated October 6, 2017)
   - **Mandate:** Established uniform definitions for 36 mutual fund categories across Equity, Debt, Hybrid, Solution-Oriented, and Other schemes.
   - **Equity Categories in Dataset:**
     - *Large Cap Fund:* Min 80% investment in equity of top 100 companies by full market capitalization.
     - *Mid Cap Fund:* Min 65% in 101st to 250th companies by market cap.
     - *Small Cap Fund:* Min 65% in 251st company onwards by market cap.
     - *Large & Mid Cap Fund:* Min 35% in large cap + min 35% in mid cap.
     - *Flexi Cap Fund:* Min 65% in equity across large, mid, and small cap dynamically.
     - *Multi Cap Fund:* Min 75% total equity with min 25% each in large, mid, and small cap.
     - *Focused Fund:* Max 30 stocks with min 65% in equity.
     - *ELSS:* Min 80% equity with 3-year statutory lock-in (Section 80C).
     - *Value / Contra Fund:* Min 65% following value or contrarian investment strategy.
     - *Dividend Yield Fund:* Min 65% in dividend yielding stocks.
     - *Sectoral / Thematic Fund:* Min 80% in specific sector or theme.

2. **Guiding Principles for Bringing Uniformity in Benchmarks:**
   - **Circular:** `SEBI/HO/IMD/IMD-II_DOF7/P/CIR/2021/652` (dated October 27, 2021)
   - **Mandate:** Introduced a mandatory **Two-Tier Benchmark structure** for mutual funds.
     - **Tier-1 Benchmark:** Reflects the category of the scheme (e.g., NIFTY 100 TRI for Large Cap, NIFTY Midcap 150 TRI for Mid Cap, NIFTY Smallcap 250 TRI for Small Cap, NIFTY 500 TRI for Flexi Cap / Multi Cap).
     - **Total Return Index (TRI):** All benchmarks must be Total Return Indices accounting for both capital gains and dividend reinvestments.

---

## 4. Asset Management Companies (AMCs)

All 132 mutual fund schemes represent SEBI-registered Asset Management Companies, including:
- SBI Funds Management Limited
- HDFC Asset Management Company Limited
- ICICI Prudential Asset Management Company Limited
- Nippon Life India Asset Management Limited
- Kotak Mahindra Asset Management Company Limited
- Axis Asset Management Company Limited
- UTI Asset Management Company Limited
- Mirae Asset Investment Managers (India) Private Limited
- DSP Asset Managers Private Limited
- Bandhan AMC Limited
- Franklin Templeton Asset Management (India) Private Limited
- Tata Asset Management Private Limited
- Canara Robeco Asset Management Company Limited
- Motilal Oswal Asset Management Company Limited
- PPFAS Asset Management Private Limited (Parag Parikh)
- Quant Money Managers Limited
- Edelweiss Asset Management Limited
- Invesco Asset Management (India) Private Limited
- Sundaram Asset Management Company Limited
- HSBC Asset Management (India) Private Limited

---

## 5. Summary of Data Hygiene & Reproducibility Guarantees

- **Preservation:** 100% of raw inputs are preserved unaltered in `data/raw/`.
- **Zero Fabrication:** Missing values are never fabricated or synthetically imputed.
- **Dual Format:** Full datasets are published in both plain-text tabular CSV and binary multi-sheet Microsoft Excel (`.xlsx`) formats.
- **Traceability:** Every row in the cleaned dataset maps to a unique `scheme_code` and official AMFI identifier.

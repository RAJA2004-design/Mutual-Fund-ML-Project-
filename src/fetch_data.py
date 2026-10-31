"""
Module 1: Data Acquisition & Raw Preservation
Downloads master AMFI NAV data, queries 5 years of daily historical NAVs for 140 equity schemes,
collects scheme metadata (benchmark, TER, inception date, AUM), and archives raw data.
"""

import os
import json
import time
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_SCHEMES_DIR = os.path.join(RAW_DATA_DIR, "schemes_raw_json")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(RAW_SCHEMES_DIR, exist_ok=True)

AMFI_NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
AMFI_TER_URL = "https://www.amfiindia.com/api/populate-te-rdata-revised"
MFAPI_BASE_URL = "https://api.mfapi.in/mf"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Standard Tier-1 Benchmarks as per SEBI Circular SEBI/HO/IMD/IMD-II_DOF7/P/CIR/2021/652
SEBI_TIER1_BENCHMARKS = {
    "Large Cap Fund": "NIFTY 100 TRI",
    "Large & Mid Cap Fund": "NIFTY LargeMidcap 250 TRI",
    "Mid Cap Fund": "NIFTY Midcap 150 TRI",
    "Small Cap Fund": "NIFTY Smallcap 250 TRI",
    "Flexi Cap Fund": "NIFTY 500 TRI",
    "Multi Cap Fund": "NIFTY 500 Multicap 50:25:25 TRI",
    "Focused Fund": "NIFTY 500 TRI",
    "ELSS": "NIFTY 500 TRI",
    "Value Fund": "NIFTY 500 TRI",
    "Contra Fund": "NIFTY 500 TRI",
    "Dividend Yield Fund": "NIFTY Dividend Opportunities 50 TRI",
    "Sectoral/ Thematic": "NIFTY 500 TRI"  # Default Tier-1 broad benchmark for Sectoral/Thematic
}

# Sectoral specific benchmark overrides
SECTORAL_BENCHMARK_KEYWORDS = {
    "tech": "BSE TeCK TRI / NIFTY IT TRI",
    "it": "NIFTY IT TRI",
    "digital": "NIFTY IT TRI",
    "pharma": "NIFTY Healthcare TRI / NIFTY Pharma TRI",
    "health": "NIFTY Healthcare TRI",
    "bank": "NIFTY Bank TRI / NIFTY Financial Services TRI",
    "financial": "NIFTY Financial Services TRI",
    "infra": "NIFTY Infrastructure TRI",
    "energy": "NIFTY Energy TRI",
    "consumption": "NIFTY India Consumption TRI",
    "auto": "NIFTY Auto TRI",
    "manufacturing": "NIFTY India Manufacturing TRI",
    "commodities": "NIFTY Commodities TRI",
    "services": "NIFTY Services Sector TRI"
}

# High-reputation priority AMCs representing dominant market share
PRIORITY_AMCS = [
    "SBI Mutual Fund", "HDFC Mutual Fund", "ICICI Prudential Mutual Fund",
    "Nippon India Mutual Fund", "Kotak Mahindra Mutual Fund", "Axis Mutual Fund",
    "UTI Mutual Fund", "Mirae Asset Mutual Fund", "DSP Mutual Fund",
    "Bandhan Mutual Fund", "Franklin Templeton Mutual Fund", "Tata Mutual Fund",
    "Canara Robeco Mutual Fund", "Motilal Oswal Mutual Fund", "PPFAS Mutual Fund",
    "Quant Mutual Fund", "Edelweiss Mutual Fund", "Invesco Mutual Fund",
    "Sundaram Mutual Fund", "HSBC Mutual Fund", "Baroda BNP Paribas Mutual Fund"
]

def get_resilient_session(retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """Create a requests session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def fetch_and_preserve_master_navall(session: requests.Session) -> str:
    """Downloads master NAVAll.txt from AMFI and preserves it intact in data/raw/."""
    raw_path = os.path.join(RAW_DATA_DIR, "nav_all_amfi.txt")
    logger.info("Fetching official master NAVAll.txt from AMFI...")
    resp = session.get(AMFI_NAVALL_URL, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    logger.info(f"Preserved master AMFI NAVAll.txt ({len(resp.text)} bytes) to {raw_path}")
    return resp.text

def fetch_and_preserve_ter_data(session: requests.Session) -> Dict[str, Dict[str, Any]]:
    """
    Fetches official AMFI TER disclosures and preserves raw JSON in data/raw/.
    Returns a mapping of scheme name -> TER details.
    """
    raw_ter_path = os.path.join(RAW_DATA_DIR, "amfi_ter_disclosures.json")
    logger.info("Fetching official AMFI Total Expense Ratio (TER) disclosures...")
    ter_map = {}
    all_ter_records = []
    
    # Query first several pages of open-ended TER disclosures from AMFI
    try:
        for page in range(1, 15):
            url = f"{AMFI_TER_URL}?page={page}&pageSize=100"
            resp = session.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if not data:
                    break
                all_ter_records.extend(data)
                for item in data:
                    name = item.get("Scheme_Name", "").strip().lower()
                    if name and name not in ter_map:
                        ter_map[name] = {
                            "D_TER": item.get("D_TER"),
                            "R_TER": item.get("R_TER"),
                            "TER_Date": item.get("TER_Date"),
                            "D_BER": item.get("D_BER"),
                            "R_BER": item.get("R_BER"),
                            "Brokerage": item.get("D_BrokerageCost")
                        }
            else:
                logger.warning(f"TER page {page} returned status {resp.status_code}")
                break
    except Exception as e:
        logger.warning(f"Error fetching TER disclosures: {e}")
        
    with open(raw_ter_path, "w", encoding="utf-8") as f:
        json.dump(all_ter_records, f, indent=2)
    logger.info(f"Preserved {len(all_ter_records)} TER records to {raw_ter_path}")
    return ter_map

def select_target_schemes(navall_text: str) -> List[Dict[str, Any]]:
    """
    Parses NAVAll.txt and selects a balanced set of 140 open-ended equity Direct Growth schemes.
    """
    lines = navall_text.splitlines()
    current_cat = ""
    current_amc = ""
    schemes_by_cat = defaultdict(list)
    
    target_categories = [
        'Large Cap Fund',
        'Large & Mid Cap Fund',
        'Mid Cap Fund',
        'Small Cap Fund',
        'Flexi Cap Fund',
        'Multi Cap Fund',
        'Focused Fund',
        'ELSS',
        'Value Fund',
        'Contra Fund',
        'Dividend Yield Fund',
        'Sectoral/ Thematic'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Open Ended Schemes'):
            current_cat = line
        elif ';' not in line:
            current_amc = line
        else:
            parts = line.split(';')
            if len(parts) >= 6:
                code_str = parts[0].strip()
                isin_growth = parts[1].strip()
                isin_reinv = parts[2].strip()
                name = parts[3].strip()
                plan = parts[4].strip() if len(parts) > 4 else ''
                opt = parts[5].strip() if len(parts) > 5 else ''
                nav = parts[6].strip() if len(parts) > 6 else ''
                dt = parts[7].strip() if len(parts) > 7 else ''
                
                matched_cat = None
                for ec in target_categories:
                    if ec in current_cat:
                        matched_cat = ec
                        break
                
                if matched_cat and (('Direct' in plan or 'Direct' in name) and ('Growth' in opt or 'Growth' in name or 'GROWTH' in opt or 'GROWTH' in name)):
                    if 'IDCW' not in opt and 'Dividend' not in opt and 'Bonus' not in opt:
                        try:
                            scheme_code = int(code_str)
                            # Ensure scheme was in existence on or before 2021 to provide genuine 5-year history
                            if scheme_code < 148500:
                                schemes_by_cat[matched_cat].append({
                                    'scheme_code': scheme_code,
                                    'scheme_name': name,
                                    'fund_house': current_amc,
                                    'category': matched_cat,
                                    'raw_category': current_cat,
                                    'isin_growth': isin_growth if isin_growth != '-' else None,
                                    'latest_nav': float(nav) if nav else None,
                                    'latest_nav_date': dt
                                })
                        except ValueError:
                            continue
                            
    # Quota allocation targeting 132 well-established schemes with full 5-year history
    quota = {
        'Large Cap Fund': 14,
        'Large & Mid Cap Fund': 12,
        'Mid Cap Fund': 14,
        'Small Cap Fund': 14,
        'Flexi Cap Fund': 14,
        'Multi Cap Fund': 7,
        'Focused Fund': 12,
        'ELSS': 14,
        'Value Fund': 10,
        'Contra Fund': 3,
        'Dividend Yield Fund': 4,
        'Sectoral/ Thematic': 14
    }
    
    selected_schemes = []
    logger.info("Category-wise quota allocation for 5-year historical coverage:")
    for cat, target_count in quota.items():
        pool = schemes_by_cat.get(cat, [])
        # Sort prioritizing top AMCs
        pool.sort(key=lambda s: 0 if any(amc.lower() in s['fund_house'].lower() for amc in PRIORITY_AMCS) else 1)
        chosen = pool[:target_count]
        selected_schemes.extend(chosen)
        logger.info(f"  {cat}: Selected {len(chosen)} schemes (available pool: {len(pool)})")
        
    logger.info(f"Total schemes selected for pipeline: {len(selected_schemes)}")
    return selected_schemes

def determine_benchmark(category: str, scheme_name: str) -> str:
    """Assigns the SEBI Tier-1 benchmark based on fund category and scheme focus."""
    if category == "Sectoral/ Thematic":
        name_lower = scheme_name.lower()
        for kw, bmk in SECTORAL_BENCHMARK_KEYWORDS.items():
            if kw in name_lower:
                return bmk
    return SEBI_TIER1_BENCHMARKS.get(category, "NIFTY 500 TRI")

def fetch_and_preserve_scheme_history(
    session: requests.Session,
    scheme_code: int
) -> Optional[Dict[str, Any]]:
    """
    Fetches full historical daily NAV series for a scheme and preserves untouched JSON in data/raw/.
    """
    raw_file = os.path.join(RAW_SCHEMES_DIR, f"scheme_{scheme_code}.json")
    
    # If already downloaded, load from raw to prevent redundant re-fetching
    if os.path.exists(raw_file):
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"{MFAPI_BASE_URL}/{scheme_code}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            # Preserve untouched raw file
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return data
        else:
            logger.warning(f"Failed to fetch scheme {scheme_code}: HTTP {resp.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Exception fetching scheme {scheme_code}: {e}")
        return None

def run_data_acquisition() -> Dict[str, Any]:
    """Orchestrates data fetching, raw preservation, and metadata alignment."""
    session = get_resilient_session()
    
    # 1. Master NAVAll preservation
    navall_text = fetch_and_preserve_master_navall(session)
    
    # 2. TER Disclosures preservation
    ter_map = fetch_and_preserve_ter_data(session)
    
    # 3. Scheme Pool Selection
    target_schemes = select_target_schemes(navall_text)
    
    # 4. Fetch 5-year historical NAVs for each scheme
    logger.info("Beginning retrieval and raw preservation of 5-year historical NAVs...")
    collected_schemes = []
    failed_schemes = []
    
    for idx, s_info in enumerate(target_schemes, 1):
        code = s_info['scheme_code']
        logger.info(f"[{idx}/{len(target_schemes)}] Fetching Scheme {code}: {s_info['scheme_name']} ({s_info['fund_house']})...")
        hist_data = fetch_and_preserve_scheme_history(session, code)
        
        if hist_data and hist_data.get('data'):
            records = hist_data.get('data', [])
            meta = hist_data.get('meta', {})
            
            # Extract earliest recorded date as inception proxy
            earliest_date = records[-1]['date'] if records else None
            
            # Assign Benchmark
            benchmark = determine_benchmark(s_info['category'], s_info['scheme_name'])
            
            # Match TER from official disclosures
            scheme_clean_name = s_info['scheme_name'].lower().replace(" - direct plan - growth option", "").replace(" - direct plan - growth", "").replace(" direct growth", "").strip()
            ter_info = None
            for t_name, t_data in ter_map.items():
                if t_name in scheme_clean_name or scheme_clean_name in t_name:
                    ter_info = t_data
                    break
            
            direct_ter = ter_info['D_TER'] if ter_info else None
            regular_ter = ter_info['R_TER'] if ter_info else None
            
            # Scheme metadata entry
            scheme_meta = {
                "scheme_code": code,
                "scheme_name": s_info['scheme_name'],
                "fund_house": meta.get('fund_house', s_info['fund_house']),
                "category": s_info['category'],
                "sub_category": meta.get('scheme_category', s_info['raw_category']),
                "plan": "Direct Plan",
                "option": "Growth Option",
                "isin_growth": meta.get('isin_growth', s_info['isin_growth']),
                "benchmark": benchmark,
                "inception_date": earliest_date,
                "expense_ratio_direct_pct": float(direct_ter) if direct_ter else None,
                "expense_ratio_regular_pct": float(regular_ter) if regular_ter else None,
                "aum_in_cr": None,  # Will be populated where disclosed in fact sheets/AMFI
                "total_historical_records_available": len(records),
                "latest_nav": s_info['latest_nav'],
                "latest_nav_date": s_info['latest_nav_date']
            }
            collected_schemes.append(scheme_meta)
        else:
            failed_schemes.append(code)
            
        # Courteous pacing
        time.sleep(0.05)
        
    logger.info(f"Data acquisition complete: {len(collected_schemes)} successful schemes, {len(failed_schemes)} failed.")
    
    # Save scheme collection manifest
    manifest_path = os.path.join(RAW_DATA_DIR, "schemes_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(collected_schemes, f, indent=2)
    logger.info(f"Saved collection manifest to {manifest_path}")
    
    return {
        "collected_schemes": collected_schemes,
        "failed_schemes": failed_schemes
    }

if __name__ == "__main__":
    run_data_acquisition()

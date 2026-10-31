"""
Main Pipeline Runner for Indian Equity Mutual Funds Data Collection & Validation
Executes data acquisition, raw preservation, cleaning, validation, and documentation generation.
"""

import sys
import os
import time
import logging

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from fetch_data import run_data_acquisition
from clean_validate import execute_cleaning_and_validation
from generate_report import run_reporting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("PipelineRunner")

def main():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("STARTING INDIAN EQUITY MUTUAL FUNDS DATA COLLECTION PIPELINE")
    logger.info("=================================================================")
    
    # Step 1: Data Acquisition & Raw Preservation
    logger.info("\n--- STEP 1: FETCHING MASTER DATA & HISTORICAL NAV SERIES ---")
    acq_results = run_data_acquisition()
    logger.info(f"Step 1 Complete: {len(acq_results['collected_schemes'])} schemes collected.")
    
    # Step 2: Data Cleaning, Non-Fabrication Enforcement, & Dataset Serialization
    logger.info("\n--- STEP 2: DATA CLEANING, VALIDATION & DATASET SERIALIZATION ---")
    val_results = execute_cleaning_and_validation()
    logger.info(f"Step 2 Complete: Wide matrix shape {val_results['wide_nav_shape']}, Long table shape {val_results['long_nav_shape']}.")
    
    # Step 3: Data Quality Report & Documentation Generation
    logger.info("\n--- STEP 3: GENERATING DATA QUALITY REPORT & DOCUMENTATION ---")
    run_reporting()
    logger.info("Step 3 Complete: Quality report and data sources documentation generated.")
    
    elapsed = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()

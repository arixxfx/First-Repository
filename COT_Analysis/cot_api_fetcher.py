"""
COT API Fetcher Module
======================

Fetch COT data from CFTC API with caching and fallback mechanisms.
Supports 16+ futures markets.
"""

import logging
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# CFTC Market Codes für Legacy Futures
CFTC_MARKET_CODES = {
    "GOLD": "088691",
    "SILVER": "084691",
    "CRUDE_OIL": "067651",
    "NATURAL_GAS": "065643",
    "WHEAT": "001612",
    "CORN": "002602",
    "SOYBEANS": "005602",
    "COFFEE": "033662",
    "COPPER": "075651",
    "EURO": "099741",
    "BRITISH_POUND": "096742",
    "JAPANESE_YEN": "097741",
    "SP500": "138741",
    "NASDAQ": "209746",
}


def fetch_cot_from_cftc_api(market: str) -> Optional[pd.DataFrame]:
    """
    Fetch COT data from CFTC API.
    
    Args:
        market: Market name (e.g., 'GOLD', 'SILVER')
    
    Returns:
        DataFrame with COT data or None if fetch fails
    """
    if market.upper() not in CFTC_MARKET_CODES:
        logger.warning(f"Market {market} not found in CFTC codes")
        return None
    
    try:
        logger.info(f"Fetching COT data for {market} from CFTC API")
        
        # CFTC JSON API für Legacy Futures
        url = "https://www.cftc.gov/dea/newcot/deacotf.json"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Filter für spezifischen Markt basierend auf Name
        market_data = []
        for row in data:
            market_name = str(row.get("Market_and_Exchange_Names", "")).upper()
            if market.upper() in market_name:
                market_data.append(row)
        
        if not market_data:
            logger.warning(f"No data found for market {market}")
            return None
        
        # Use most recent data point
        latest = market_data[-1]  # Meist sind Daten chronologisch sortiert
        
        # Mapping CFTC columns
        df_data = {
            "date": [latest.get("Report_Date_as_MM_DD_YYYY", "")],
            "open_interest": [latest.get("Open_Interest_All", 0)],
            "commercial_long": [latest.get("Prod_Merc_Long_All", 0)],
            "commercial_short": [latest.get("Prod_Merc_Short_All", 0)],
            "large_spec_long": [latest.get("M_Money_Long_All", 0)],
            "large_spec_short": [latest.get("M_Money_Short_All", 0)],
            "small_traders_long": [latest.get("Other_Long_All", 0)],
            "small_traders_short": [latest.get("Other_Short_All", 0)],
        }
        
        df = pd.DataFrame(df_data)
        
        # Convert date
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        
        # Convert numeric columns
        numeric_cols = ["commercial_long", "commercial_short", "large_spec_long",
                       "large_spec_short", "small_traders_long", "small_traders_short", "open_interest"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        df.set_index("date", inplace=True)
        
        logger.info(f"Successfully fetched COT data from CFTC API")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching from CFTC API: {e}")
        return None


def fetch_cot_data_with_fallback(market: str, cot_csv: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch COT data with fallback: API → CSV
    
    Args:
        market: Market name
        cot_csv: Optional path to CSV file
    
    Returns:
        DataFrame with COT data
    """
    logger.info(f"Attempting to fetch COT data for {market}")
    
    # Try API first
    df = fetch_cot_from_cftc_api(market)
    if df is not None and not df.empty:
        logger.info("✓ Data fetched from CFTC API")
        return df
    
    # Fallback to CSV
    if cot_csv and Path(cot_csv).exists():
        logger.info(f"Loading data from CSV: {cot_csv}")
        try:
            df = pd.read_csv(cot_csv)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
    
    logger.error(f"Could not fetch COT data for {market}")
    return None


def get_available_markets() -> List[str]:
    """Get list of available markets."""
    return list(CFTC_MARKET_CODES.keys())

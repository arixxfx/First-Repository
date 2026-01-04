#!/usr/bin/env python3
"""
COT (Commitments of Traders) Automated Analysis
================================================

A comprehensive, production-ready script for analyzing CFTC COT data combined with price charts.

Features:
  - Fetch COT data from CFTC (CSV/local file) and price data from yfinance
  - Compute net positions (commercial, large specs, small traders)
  - Calculate weekly deltas and extremes
  - Generate COT Index (0-100 normalization)
  - Detect extreme positions and signals
  - Compare COT signals with price action
  - Generate charts (PNG) and CSV reports
  - Weekly actionable trading signals (BULLISH/BEARISH/NEUTRAL)

Dependencies:
  pip install pandas numpy matplotlib yfinance requests

Usage:
  python COT_automated_analyses.py \\
    --market GOLD \\
    --price-symbol GC=F \\
    --cot-csv cot_data.csv \\
    --start 2020-01-01 \\
    --end 2025-11-16 \\
    --output-dir ./cot_reports

Author: Automated Trading Analysis
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cot_analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_cot_data(
    market: str, 
    cot_csv: str, 
    source: str = "cftc"
) -> pd.DataFrame:
    """
    Fetch COT data from file or URL.
    
    Args:
        market: Market name (e.g. 'GOLD', 'SILVER')
        cot_csv: Path to CSV file or URL
        source: Data source identifier (default 'cftc')
    
    Returns:
        DataFrame with COT data, datetime index
    
    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If required columns missing
    """
    logger.info(f"Fetching COT data for {market} from {cot_csv}")
    
    # Try to load from file or URL
    try:
        if cot_csv.startswith("http"):
            df = pd.read_csv(cot_csv)
        else:
            df = pd.read_csv(cot_csv)
    except FileNotFoundError as e:
        logger.error(f"COT CSV file not found: {cot_csv}")
        raise
    except Exception as e:
        logger.error(f"Error reading COT CSV: {e}")
        raise
    
    # Validate required columns
    required_cols = {
        "date", "commercial_long", "commercial_short",
        "large_spec_long", "large_spec_short",
        "small_traders_long", "small_traders_short", "open_interest"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert date to datetime and set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    
    # Ensure weekly frequency (fill missing weeks)
    df = df.asfreq("W-FRI", method="ffill")
    
    logger.info(f"Loaded {len(df)} weeks of COT data")
    return df


def fetch_price_data(
    symbol: str, 
    start: str, 
    end: str
) -> pd.Series:
    """
    Fetch price data from yfinance and resample to weekly.
    
    Args:
        symbol: Ticker symbol (e.g. 'GC=F' for gold futures)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        Series with weekly closing prices
    """
    logger.info(f"Fetching price data for {symbol} from {start} to {end}")
    
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError(f"No price data found for {symbol}")
        
        # Resample to weekly (Friday close)
        weekly = data["Close"].resample("W-FRI").last()
        logger.info(f"Loaded {len(weekly)} weeks of price data")
        return weekly
    
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        raise


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_cot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate net positions and weekly deltas.
    
    Args:
        df: COT DataFrame
    
    Returns:
        DataFrame with added columns: net positions and deltas
    """
    logger.info("Preprocessing COT data: computing net positions and deltas")
    
    # Calculate net positions
    df["commercial_net"] = df["commercial_long"] - df["commercial_short"]
    df["large_spec_net"] = df["large_spec_long"] - df["large_spec_short"]
    df["small_traders_net"] = df["small_traders_long"] - df["small_traders_short"]
    
    # Calculate deltas (week-over-week change)
    df["commercial_net_delta"] = df["commercial_net"].diff()
    df["large_spec_net_delta"] = df["large_spec_net"].diff()
    df["small_traders_net_delta"] = df["small_traders_net"].diff()
    df["open_interest_delta"] = df["open_interest"].diff()
    df["open_interest_delta_pct"] = df["open_interest"].pct_change() * 100
    
    logger.info("Net positions and deltas computed")
    return df


# ============================================================================
# COT INDEX & EXTREMES
# ============================================================================

def compute_cot_index(
    series: pd.Series, 
    window: int = 104
) -> pd.Series:
    """
    Normalize series to 0-100 COT Index based on rolling min/max.
    
    Args:
        series: Series to normalize (e.g. commercial_net)
        window: Rolling window in weeks (default 104 = 2 years)
    
    Returns:
        Series with COT Index (0-100)
    """
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()
    
    # Avoid division by zero
    denom = rolling_max - rolling_min
    denom = denom.replace(0, 1)
    
    cot_index = ((series - rolling_min) / denom) * 100
    cot_index = cot_index.clip(0, 100)
    
    return cot_index


def detect_extremes(
    df: pd.DataFrame,
    window: int = 104,
    upper_pct: float = 80,
    lower_pct: float = 20,
    use_quantiles: bool = False,
) -> pd.DataFrame:
    """
    Detect extreme COT positions (very long or very short).
    
    Args:
        df: COT DataFrame with net positions
        window: Rolling window in weeks
        upper_pct: Upper threshold for "extreme long" (default 80)
        lower_pct: Lower threshold for "extreme short" (default 20)
        use_quantiles: If True, use 90th/10th percentile instead of fixed pct
    
    Returns:
        DataFrame with added columns: cot_index_* and extreme_* flags
    """
    logger.info("Detecting extreme positions")
    
    # Compute COT Indices
    df["commercial_cot_index"] = compute_cot_index(df["commercial_net"], window)
    df["large_spec_cot_index"] = compute_cot_index(df["large_spec_net"], window)
    df["small_traders_cot_index"] = compute_cot_index(df["small_traders_net"], window)
    
    # Set thresholds (quantile or fixed)
    if use_quantiles:
        upper_thresh = df["commercial_cot_index"].quantile(0.90)
        lower_thresh = df["commercial_cot_index"].quantile(0.10)
        logger.info(f"Using quantile thresholds: {lower_thresh:.1f} / {upper_thresh:.1f}")
    else:
        upper_thresh = upper_pct
        lower_thresh = lower_pct
    
    # Detect commercial extremes
    df["commercial_extreme_long"] = df["commercial_cot_index"] > upper_thresh
    df["commercial_extreme_short"] = df["commercial_cot_index"] < lower_thresh
    
    # Detect large spec extremes (note: contrarian signal)
    df["large_spec_extreme_long"] = df["large_spec_cot_index"] > upper_thresh
    df["large_spec_extreme_short"] = df["large_spec_cot_index"] < lower_thresh
    
    # Detect small traders extremes
    df["small_traders_extreme_long"] = df["small_traders_cot_index"] > upper_thresh
    df["small_traders_extreme_short"] = df["small_traders_cot_index"] < lower_thresh
    
    logger.info("Extreme positions detected")
    return df


# ============================================================================
# DELTA ANALYSIS
# ============================================================================

def analyze_deltas(
    df: pd.DataFrame,
    delta_threshold_pct: float = 0.10,
    oi_threshold_pct: float = 5.0,
) -> List[str]:
    """
    Analyze significant position changes and open interest movements.
    
    Args:
        df: COT DataFrame with deltas
        delta_threshold_pct: Threshold as % of rolling mean (default 10%)
        oi_threshold_pct: Open interest change threshold in % (default 5%)
    
    Returns:
        List of signal messages
    """
    signals = []
    logger.info("Analyzing position deltas")
    
    # Latest row
    if len(df) == 0:
        return signals
    
    latest = df.iloc[-1]
    
    # Check commercial net delta
    if not pd.isna(latest["commercial_net_delta"]):
        avg_abs_delta = abs(df["commercial_net_delta"]).rolling(window=20).mean().iloc[-1]
        if avg_abs_delta > 0 and abs(latest["commercial_net_delta"]) > delta_threshold_pct * avg_abs_delta:
            direction = "INCREASED" if latest["commercial_net_delta"] > 0 else "DECREASED"
            signals.append(f"Commercial net position {direction} significantly (delta: {latest['commercial_net_delta']:.0f})")
    
    # Check large spec net delta
    if not pd.isna(latest["large_spec_net_delta"]):
        avg_abs_delta = abs(df["large_spec_net_delta"]).rolling(window=20).mean().iloc[-1]
        if avg_abs_delta > 0 and abs(latest["large_spec_net_delta"]) > delta_threshold_pct * avg_abs_delta:
            direction = "INCREASED" if latest["large_spec_net_delta"] > 0 else "DECREASED"
            signals.append(f"Large spec net position {direction} significantly (delta: {latest['large_spec_net_delta']:.0f})")
    
    # Check open interest change
    if not pd.isna(latest["open_interest_delta_pct"]):
        if latest["open_interest_delta_pct"] > oi_threshold_pct:
            signals.append(f"Open Interest INCREASED {latest['open_interest_delta_pct']:.1f}% (possible new participation)")
        elif latest["open_interest_delta_pct"] < -oi_threshold_pct:
            signals.append(f"Open Interest DECREASED {abs(latest['open_interest_delta_pct']):.1f}% (possible exit)")
    
    logger.info(f"Found {len(signals)} delta signals")
    return signals


# ============================================================================
# PRICE ANALYSIS
# ============================================================================

def compare_with_price(
    df: pd.DataFrame,
    price_series: pd.Series,
    lookback_weeks: int = 4,
) -> Dict[str, any]:
    """
    Compare COT signals with recent price action.
    
    Args:
        df: COT DataFrame with extreme flags
        price_series: Weekly price series
        lookback_weeks: Weeks to look back for price trends (default 4)
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("Comparing COT signals with price action")
    
    result = {
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "reasons": [],
    }
    
    if len(df) == 0 or len(price_series) == 0:
        return result
    
    # Align indices
    common_idx = df.index.intersection(price_series.index)
    if len(common_idx) == 0:
        logger.warning("No overlapping dates between COT and price data")
        return result
    
    df_aligned = df.loc[common_idx]
    price_aligned = price_series.loc[common_idx]
    
    latest = df_aligned.iloc[-1]
    price_latest = price_aligned.iloc[-1]
    
    # Calculate recent price change
    if len(price_aligned) >= lookback_weeks:
        price_lookback = price_aligned.iloc[-lookback_weeks]
        price_change_pct = ((price_latest - price_lookback) / price_lookback) * 100
    else:
        price_change_pct = 0
    
    # Calculate moving averages (10 and 40 weeks approximating 50 and 200 day MA)
    ma_10 = price_aligned.rolling(window=10).mean().iloc[-1]
    ma_40 = price_aligned.rolling(window=40).mean().iloc[-1]
    
    # Rule: Commercials extreme long + Price down = Bullish (bottom forming)
    if latest.get("commercial_extreme_long", False) and price_change_pct < -2:
        result["signal"] = "BULLISH"
        result["confidence"] += 0.4
        result["reasons"].append(
            f"Commercials EXTREME LONG (COT: {latest.get('commercial_cot_index', 0):.1f}) "
            f"+ Price down {price_change_pct:.1f}% (4w) = Possible bottom forming"
        )
    
    # Rule: Commercials extreme short + Price up = Bearish (top forming)
    if latest.get("commercial_extreme_short", False) and price_change_pct > 2:
        result["signal"] = "BEARISH"
        result["confidence"] += 0.4
        result["reasons"].append(
            f"Commercials EXTREME SHORT (COT: {latest.get('commercial_cot_index', 0):.1f}) "
            f"+ Price up {price_change_pct:.1f}% (4w) = Possible top forming"
        )
    
    # Rule: Large specs (contrarian) extreme long + price up = Bearish signal
    if latest.get("large_spec_extreme_long", False) and price_change_pct > 2:
        result["signal"] = "BEARISH"
        result["confidence"] += 0.3
        result["reasons"].append(
            f"Large Specs EXTREME LONG (COT: {latest.get('large_spec_cot_index', 0):.1f}) "
            f"+ Price rising (contrarian signal)"
        )
    
    # Price near moving averages
    price_above_ma10 = price_latest > ma_10 * 1.02
    price_below_ma10 = price_latest < ma_10 * 0.98
    price_above_ma40 = price_latest > ma_40 * 1.02
    price_below_ma40 = price_latest < ma_40 * 0.98
    
    if price_below_ma10 and price_below_ma40:
        result["reasons"].append(f"Price below both 10-week ({ma_10:.2f}) and 40-week MA ({ma_40:.2f}) = downtrend")
    
    if price_above_ma10 and price_above_ma40:
        result["reasons"].append(f"Price above both 10-week ({ma_10:.2f}) and 40-week MA ({ma_40:.2f}) = uptrend")
    
    logger.info(f"Price comparison complete: {result['signal']}")
    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cot_and_price(
    df: pd.DataFrame,
    price_series: pd.Series,
    output_file: str,
    market: str = "Market",
) -> None:
    """
    Generate multi-panel chart with price, COT index, and commercials net.
    
    Args:
        df: COT DataFrame with indices and extremes
        price_series: Weekly price series
        output_file: Output PNG filename
        market: Market name for title
    """
    logger.info(f"Generating chart: {output_file}")
    
    # Align indices
    common_idx = df.index.intersection(price_series.index)
    df_aligned = df.loc[common_idx].copy()
    price_aligned = price_series.loc[common_idx].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"COT Analysis: {market}", fontsize=14, fontweight="bold")
    
    # Panel 1: Price with commercial net as secondary axis
    ax1 = axes[0]
    ax1_2 = ax1.twinx()
    
    ax1.plot(price_aligned.index, price_aligned.values, "b-", linewidth=2, label="Price (weekly close)")
    ax1.set_ylabel("Price", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    
    # Commercials net on secondary axis
    colors_comm = ["g" if x > 0 else "r" for x in df_aligned["commercial_net"].values]
    ax1_2.bar(df_aligned.index, df_aligned["commercial_net"].values, alpha=0.3, color=colors_comm, label="Commercial Net")
    ax1_2.set_ylabel("Commercial Net Position", color="gray")
    ax1_2.tick_params(axis="y", labelcolor="gray")
    
    # Mark extremes
    extreme_long_idx = df_aligned[df_aligned.get("commercial_extreme_long", False)].index
    extreme_short_idx = df_aligned[df_aligned.get("commercial_extreme_short", False)].index
    if len(extreme_long_idx) > 0:
        ax1.scatter(extreme_long_idx, price_aligned.loc[extreme_long_idx], marker="^", color="green", s=100, zorder=5, label="Comm. Extreme Long")
    if len(extreme_short_idx) > 0:
        ax1.scatter(extreme_short_idx, price_aligned.loc[extreme_short_idx], marker="v", color="red", s=100, zorder=5, label="Comm. Extreme Short")
    
    ax1.legend(loc="upper left")
    ax1.set_title("Weekly Price vs Commercial Net Position")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: COT Indices
    ax2 = axes[1]
    ax2.plot(df_aligned.index, df_aligned["commercial_cot_index"].values, "b-", linewidth=2, label="Commercial COT Index")
    ax2.plot(df_aligned.index, df_aligned["large_spec_cot_index"].values, "r-", linewidth=2, label="Large Spec COT Index")
    ax2.axhline(80, color="orange", linestyle="--", alpha=0.5, label="Extreme Long (80)")
    ax2.axhline(20, color="orange", linestyle="--", alpha=0.5, label="Extreme Short (20)")
    ax2.fill_between(df_aligned.index, 80, 100, alpha=0.1, color="green")
    ax2.fill_between(df_aligned.index, 0, 20, alpha=0.1, color="red")
    ax2.set_ylabel("COT Index (0-100)")
    ax2.set_ylim([0, 100])
    ax2.legend(loc="best")
    ax2.set_title("COT Indices (Commercial & Large Specs)")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Open Interest
    ax3 = axes[2]
    ax3.bar(df_aligned.index, df_aligned["open_interest"].values, alpha=0.6, color="steelblue", label="Open Interest")
    ax3.set_ylabel("Open Interest")
    ax3.set_xlabel("Date")
    ax3.legend(loc="best")
    ax3.set_title("Open Interest")
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Chart saved: {output_file}")
    plt.close()


def generate_pdf_report(
    df: pd.DataFrame,
    price_series: pd.Series,
    output_file: str,
    market: str = "Market",
) -> None:
    """
    Generate PDF report with charts.
    
    Args:
        df: COT DataFrame
        price_series: Weekly price series
        output_file: Output PDF filename
        market: Market name
    """
    logger.info(f"Generating PDF report: {output_file}")
    
    common_idx = df.index.intersection(price_series.index)
    df_aligned = df.loc[common_idx].copy()
    price_aligned = price_series.loc[common_idx].copy()
    
    with PdfPages(output_file) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        
        summary_text = f"""
COT ANALYSIS REPORT - {market.upper()}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Latest Values (as of {df_aligned.index[-1].strftime('%Y-%m-%d')}):
  Commercial Net Position: {df_aligned['commercial_net'].iloc[-1]:,.0f}
  Commercial COT Index: {df_aligned['commercial_cot_index'].iloc[-1]:.1f}
  Large Spec Net Position: {df_aligned['large_spec_net'].iloc[-1]:,.0f}
  Large Spec COT Index: {df_aligned['large_spec_cot_index'].iloc[-1]:.1f}
  Open Interest: {df_aligned['open_interest'].iloc[-1]:,.0f}
  
Recent Price Data:
  Latest Close: {price_aligned.iloc[-1]:.2f}
  4-Week Change: {((price_aligned.iloc[-1] - price_aligned.iloc[-5]) / price_aligned.iloc[-5] * 100):.2f}% if len(price_aligned) >= 5 else N/A
  10-Week MA: {price_aligned.rolling(10).mean().iloc[-1]:.2f}
  40-Week MA: {price_aligned.rolling(40).mean().iloc[-1]:.2f}

Data Range: {df_aligned.index[0].strftime('%Y-%m-%d')} to {df_aligned.index[-1].strftime('%Y-%m-%d')}
Total Weeks: {len(df_aligned)}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        
        # Page 2: Chart
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        fig.suptitle(f"COT Analysis: {market}", fontsize=14, fontweight="bold")
        
        ax1 = axes[0]
        ax1_2 = ax1.twinx()
        ax1.plot(price_aligned.index, price_aligned.values, "b-", linewidth=2)
        ax1_2.bar(df_aligned.index, df_aligned["commercial_net"].values, alpha=0.3)
        ax1.set_ylabel("Price")
        ax1_2.set_ylabel("Commercial Net")
        ax1.set_title("Price vs Commercial Net")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(df_aligned.index, df_aligned["commercial_cot_index"].values, "b-", linewidth=2, label="Commercial")
        ax2.plot(df_aligned.index, df_aligned["large_spec_cot_index"].values, "r-", linewidth=2, label="Large Specs")
        ax2.axhline(80, color="orange", linestyle="--", alpha=0.5)
        ax2.axhline(20, color="orange", linestyle="--", alpha=0.5)
        ax2.set_ylabel("COT Index")
        ax2.set_ylim([0, 100])
        ax2.legend(loc="best")
        ax2.set_title("COT Indices")
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        ax3.bar(df_aligned.index, df_aligned["open_interest"].values, alpha=0.6)
        ax3.set_ylabel("Open Interest")
        ax3.set_xlabel("Date")
        ax3.set_title("Open Interest")
        ax3.grid(True, alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
    
    logger.info(f"PDF report saved: {output_file}")


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    price_series: pd.Series,
    delta_signals: List[str],
    price_signals: Dict,
    output_dir: str,
    market: str = "Market",
    price_symbol: str = "SYMBOL",
) -> None:
    """
    Generate CSV, PNG charts, and console report with trading signal.
    
    Args:
        df: Processed COT DataFrame
        price_series: Weekly price series
        delta_signals: List of delta signal messages
        price_signals: Dict with price comparison results
        output_dir: Output directory for reports
        market: Market name
        price_symbol: Price ticker symbol
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating reports to {output_dir}")
    
    # Export CSV
    csv_file = output_path / f"cot_{market}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_file)
    logger.info(f"CSV exported: {csv_file}")
    
    # Generate PNG chart
    png_file = output_path / f"cot_{market}_{datetime.now().strftime('%Y%m%d')}.png"
    plot_cot_and_price(df, price_series, str(png_file), market=market)
    
    # Generate PDF report
    pdf_file = output_path / f"cot_{market}_{datetime.now().strftime('%Y%m%d')}.pdf"
    generate_pdf_report(df, price_series, str(pdf_file), market=market)
    
    # Console signal report
    latest_row = df.iloc[-1]
    signal = price_signals.get("signal", "NEUTRAL")
    confidence = price_signals.get("confidence", 0.0)
    reasons = price_signals.get("reasons", [])
    
    print("\n" + "=" * 80)
    print(f"COT ANALYSIS REPORT - {market.upper()} ({price_symbol})")
    print("=" * 80)
    print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Date: {latest_row.name.strftime('%Y-%m-%d')}")
    print()
    
    print("CURRENT POSITIONS:")
    print(f"  Commercial Net: {latest_row.get('commercial_net', 0):,.0f}")
    print(f"  Commercial COT Index: {latest_row.get('commercial_cot_index', 0):.1f}")
    print(f"  Large Spec Net: {latest_row.get('large_spec_net', 0):,.0f}")
    print(f"  Large Spec COT Index: {latest_row.get('large_spec_cot_index', 0):.1f}")
    print(f"  Small Traders Net: {latest_row.get('small_traders_net', 0):,.0f}")
    print(f"  Open Interest: {latest_row.get('open_interest', 0):,.0f}")
    print()
    
    print("POSITION CHANGES (week-over-week):")
    print(f"  Commercial Net Delta: {latest_row.get('commercial_net_delta', 0):+,.0f}")
    print(f"  Large Spec Net Delta: {latest_row.get('large_spec_net_delta', 0):+,.0f}")
    print(f"  Open Interest Change: {latest_row.get('open_interest_delta_pct', 0):+.2f}%")
    print()
    
    print("EXTREMES:")
    if latest_row.get("commercial_extreme_long", False):
        print("  ⬆️  COMMERCIAL EXTREME LONG")
    if latest_row.get("commercial_extreme_short", False):
        print("  ⬇️  COMMERCIAL EXTREME SHORT")
    if latest_row.get("large_spec_extreme_long", False):
        print("  ⬆️  LARGE SPEC EXTREME LONG (contrarian)")
    if latest_row.get("large_spec_extreme_short", False):
        print("  ⬇️  LARGE SPEC EXTREME SHORT (contrarian)")
    print()
    
    if delta_signals:
        print("DELTA SIGNALS:")
        for sig in delta_signals:
            print(f"  • {sig}")
        print()
    
    print("=" * 80)
    print(f"TRADING SIGNAL: {signal} (Confidence: {confidence:.0%})")
    print("=" * 80)
    for reason in reasons:
        print(f"  • {reason}")
    print()
    
    print(f"Files generated:")
    print(f"  CSV:  {csv_file}")
    print(f"  PNG:  {png_file}")
    print(f"  PDF:  {pdf_file}")
    print()


# ============================================================================
# MAIN ANALYSIS FLOW
# ============================================================================

def run_analysis(
    market: str,
    price_symbol: str,
    cot_csv: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    window: int = 104,
    upper_pct: float = 80,
    lower_pct: float = 20,
    delta_threshold_pct: float = 0.10,
    use_quantiles: bool = False,
) -> None:
    """
    Execute complete COT analysis pipeline.
    
    Args:
        market: Market name (e.g. 'GOLD')
        price_symbol: Price ticker (e.g. 'GC=F')
        cot_csv: Path to COT CSV file
        start_date: Analysis start date
        end_date: Analysis end date
        output_dir: Output directory for reports
        window: Rolling window for COT Index
        upper_pct: Upper extreme threshold
        lower_pct: Lower extreme threshold
        delta_threshold_pct: Delta signal threshold
        use_quantiles: Use quantile-based thresholds
    """
    logger.info(f"Starting COT analysis for {market} ({price_symbol})")
    
    try:
        # Data fetching
        cot_df = fetch_cot_data(market, cot_csv)
        price_df = fetch_price_data(price_symbol, start_date, end_date)
        
        # Preprocessing
        cot_df = preprocess_cot(cot_df)
        
        # COT Index and extremes
        cot_df = detect_extremes(cot_df, window=window, upper_pct=upper_pct, 
                                 lower_pct=lower_pct, use_quantiles=use_quantiles)
        
        # Analysis
        delta_signals = analyze_deltas(cot_df, delta_threshold_pct=delta_threshold_pct)
        price_comparison = compare_with_price(cot_df, price_df, lookback_weeks=4)
        
        # Reporting
        generate_report(
            cot_df, price_df,
            delta_signals, price_comparison,
            output_dir,
            market=market,
            price_symbol=price_symbol
        )
        
        logger.info("Analysis complete")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


# ============================================================================
# SMOKE TESTS
# ============================================================================

def run_smoke_tests() -> None:
    """Run basic smoke tests with synthetic data."""
    logger.info("Running smoke tests...")
    
    # Create synthetic COT data
    dates = pd.date_range(start="2020-01-01", periods=200, freq="W-FRI")
    np.random.seed(42)
    
    synthetic_cot = pd.DataFrame({
        "commercial_long": np.random.randint(100000, 500000, 200),
        "commercial_short": np.random.randint(100000, 500000, 200),
        "large_spec_long": np.random.randint(50000, 300000, 200),
        "large_spec_short": np.random.randint(50000, 300000, 200),
        "small_traders_long": np.random.randint(20000, 100000, 200),
        "small_traders_short": np.random.randint(20000, 100000, 200),
        "open_interest": np.random.randint(1000000, 5000000, 200),
        "date": dates,
    })
    
    # Save synthetic CSV
    synthetic_file = "/tmp/cot_synthetic.csv"
    synthetic_cot.to_csv(synthetic_file, index=False)
    logger.info(f"Synthetic COT saved to {synthetic_file}")
    
    # Test fetch and preprocess
    try:
        cot = fetch_cot_data("TEST", synthetic_file)
        cot = preprocess_cot(cot)
        logger.info("✓ Preprocess test passed")
    except Exception as e:
        logger.error(f"✗ Preprocess test failed: {e}")
        return
    
    # Test detect extremes
    try:
        cot = detect_extremes(cot, window=52)
        logger.info("✓ Detect extremes test passed")
    except Exception as e:
        logger.error(f"✗ Detect extremes test failed: {e}")
        return
    
    # Test analyze deltas
    try:
        signals = analyze_deltas(cot)
        logger.info(f"✓ Analyze deltas test passed (found {len(signals)} signals)")
    except Exception as e:
        logger.error(f"✗ Analyze deltas test failed: {e}")
        return
    
    # Create synthetic price data
    synthetic_price = pd.Series(
        np.cumsum(np.random.randn(200)) + 1000,
        index=dates,
        name="price"
    )
    
    # Test compare with price
    try:
        comparison = compare_with_price(cot, synthetic_price)
        logger.info(f"✓ Compare with price test passed (signal: {comparison['signal']})")
    except Exception as e:
        logger.error(f"✗ Compare with price test failed: {e}")
        return
    
    logger.info("All smoke tests passed!")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="COT (Commitments of Traders) Automated Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python COT_automated_analyses.py \\
    --market GOLD \\
    --price-symbol GC=F \\
    --cot-csv /path/to/cot_data.csv \\
    --start 2020-01-01 \\
    --end 2025-11-16 \\
    --output-dir ./reports
  
  python COT_automated_analyses.py \\
    --market SILVER \\
    --price-symbol SI=F \\
    --cot-csv /path/to/cot_silver.csv \\
    --window 52 \\
    --upper-pct 75 \\
    --lower-pct 25 \\
    --test
        """
    )
    
    parser.add_argument("--market", type=str, default="GOLD", help="Market name (e.g. GOLD, SILVER)")
    parser.add_argument("--price-symbol", type=str, default="GC=F", help="Price ticker symbol (e.g. GC=F)")
    parser.add_argument("--cot-csv", type=str, required=True, help="Path to COT CSV file")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="./cot_reports", help="Output directory for reports")
    parser.add_argument("--window", type=int, default=104, help="COT Index rolling window (weeks, default 104)")
    parser.add_argument("--upper-pct", type=float, default=80, help="Upper extreme threshold (default 80)")
    parser.add_argument("--lower-pct", type=float, default=20, help="Lower extreme threshold (default 20)")
    parser.add_argument("--delta-threshold-pct", type=float, default=0.10, help="Delta threshold % (default 0.10)")
    parser.add_argument("--use-quantiles", action="store_true", help="Use quantile-based thresholds (90/10 percentile)")
    parser.add_argument("--test", action="store_true", help="Run smoke tests and exit")
    
    args = parser.parse_args()
    
    if args.test:
        run_smoke_tests()
        return
    
    run_analysis(
        market=args.market,
        price_symbol=args.price_symbol,
        cot_csv=args.cot_csv,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        window=args.window,
        upper_pct=args.upper_pct,
        lower_pct=args.lower_pct,
        delta_threshold_pct=args.delta_threshold_pct,
        use_quantiles=args.use_quantiles,
    )


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE
# ============================================================================

"""
USAGE INSTRUCTIONS
==================

1. Install dependencies:
   pip install pandas numpy matplotlib yfinance requests

2. Prepare your COT data:
   - Expected columns: date, commercial_long, commercial_short,
     large_spec_long, large_spec_short, small_traders_long, small_traders_short, open_interest
   - Format: CSV with header row

3. Run analysis:
   python COT_automated_analyses.py \
     --market GOLD \
     --price-symbol GC=F \
     --cot-csv /path/to/cot_gold.csv \
     --start 2020-01-01 \
     --end 2025-11-16 \
     --output-dir ./reports

4. Run tests with synthetic data:
   python COT_automated_analyses.py --test

5. Customize thresholds:
   python COT_automated_analyses.py \
     --market GOLD \
     --price-symbol GC=F \
     --cot-csv cot_data.csv \
     --window 52 \
     --upper-pct 75 \
     --lower-pct 25 \
     --use-quantiles

6. Output files:
   - cot_MARKET_YYYYMMDD.csv       (processed data)
   - cot_MARKET_YYYYMMDD.png       (charts)
   - cot_MARKET_YYYYMMDD.pdf       (PDF report)
   - cot_analysis.log              (execution log)

7. Scheduling (cron, weekly):
   0 17 * * FRI python /path/to/COT_automated_analyses.py \
     --market GOLD --price-symbol GC=F --cot-csv /path/to/cot.csv \
     --output-dir /path/to/reports
"""

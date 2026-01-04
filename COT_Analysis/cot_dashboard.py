#!/usr/bin/env python3
"""
COT Analysis Dashboard
======================

Interactive Streamlit dashboard for real-time COT analysis.
Select any futures market and see live COT signals.

Requirements:
  pip install streamlit pandas numpy matplotlib yfinance requests

Usage:
  streamlit run cot_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cot_api_fetcher import fetch_cot_data_with_fallback, get_available_markets
except ImportError:
    st.error("Error: cot_api_fetcher.py not found in same directory")
    st.stop()

import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Market to Price Symbol Mapping
MARKET_TO_SYMBOL = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE_OIL": "CL=F",
    "NATURAL_GAS": "NG=F",
    "WHEAT": "ZWH=F",
    "CORN": "ZCZ=F",
    "SOYBEANS": "ZSX=F",
    "COFFEE": "KCZ=F",
    "COPPER": "HGZ=F",
    "EURO": "EURUSD=X",
    "BRITISH_POUND": "GBPUSD=X",
    "JAPANESE_YEN": "JPYUSD=X",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def preprocess_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate net positions and deltas."""
    df["commercial_net"] = df["commercial_long"] - df["commercial_short"]
    df["large_spec_net"] = df["large_spec_long"] - df["large_spec_short"]
    df["small_traders_net"] = df["small_traders_long"] - df["small_traders_short"]
    
    df["commercial_net_delta"] = df["commercial_net"].diff()
    df["large_spec_net_delta"] = df["large_spec_net"].diff()
    df["open_interest_delta_pct"] = df["open_interest"].pct_change() * 100
    
    return df


def compute_cot_index(series: pd.Series, window: int = 104) -> pd.Series:
    """Compute COT Index (0-100)."""
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()
    denom = rolling_max - rolling_min
    denom = denom.replace(0, 1)
    cot_index = ((series - rolling_min) / denom) * 100
    return cot_index.clip(0, 100)


def detect_extremes(df: pd.DataFrame, window: int = 104, upper_pct: float = 80, lower_pct: float = 20) -> pd.DataFrame:
    """Detect extreme positions."""
    df["commercial_cot_index"] = compute_cot_index(df["commercial_net"], window)
    df["large_spec_cot_index"] = compute_cot_index(df["large_spec_net"], window)
    
    df["commercial_extreme_long"] = df["commercial_cot_index"] > upper_pct
    df["commercial_extreme_short"] = df["commercial_cot_index"] < lower_pct
    df["large_spec_extreme_long"] = df["large_spec_cot_index"] > upper_pct
    df["large_spec_extreme_short"] = df["large_spec_cot_index"] < lower_pct
    
    return df


def fetch_price_data(symbol: str, days: int = 365) -> pd.Series:
    """Fetch weekly price data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.Series()
        weekly = data["Close"].resample("W-FRI").last()
        return weekly
    except Exception as e:
        logger.error(f"Error fetching price: {e}")
        return pd.Series()


def compare_with_price(df: pd.DataFrame, price_series: pd.Series) -> Dict:
    """Compare COT signals with price action."""
    result = {"signal": "NEUTRAL", "confidence": 0.0, "reasons": []}
    
    if len(df) == 0 or len(price_series) == 0:
        return result
    
    common_idx = df.index.intersection(price_series.index)
    if len(common_idx) == 0:
        return result
    
    df_aligned = df.loc[common_idx]
    price_aligned = price_series.loc[common_idx]
    
    latest = df_aligned.iloc[-1]
    price_latest = price_aligned.iloc[-1]
    
    # Price change
    if len(price_aligned) >= 4:
        price_lookback = price_aligned.iloc[-4]
        price_change_pct = ((price_latest - price_lookback) / price_lookback) * 100
    else:
        price_change_pct = 0
    
    # Rules
    if latest.get("commercial_extreme_long", False) and price_change_pct < -2:
        result["signal"] = "BULLISH"
        result["confidence"] = 0.7
        result["reasons"].append(
            f"Commercials EXTREME LONG (COT: {latest.get('commercial_cot_index', 0):.1f}) + "
            f"Price down {price_change_pct:.1f}% → Possible bottom"
        )
    
    if latest.get("commercial_extreme_short", False) and price_change_pct > 2:
        result["signal"] = "BEARISH"
        result["confidence"] = 0.7
        result["reasons"].append(
            f"Commercials EXTREME SHORT (COT: {latest.get('commercial_cot_index', 0):.1f}) + "
            f"Price up {price_change_pct:.1f}% → Possible top"
        )
    
    if latest.get("large_spec_extreme_long", False) and price_change_pct > 2:
        result["signal"] = "BEARISH"
        result["confidence"] = 0.5
        result["reasons"].append("Large Specs EXTREME LONG (contrarian signal)")
    
    return result


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="COT Analysis Dashboard", layout="wide")
    
    st.title("📊 COT Analysis Dashboard")
    st.markdown("Real-time Commitments of Traders analysis with CFTC API data")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    available_markets = get_available_markets()
    selected_market = st.sidebar.selectbox(
        "📈 Select Market",
        available_markets,
        index=0 if available_markets else None
    )
    
    if not selected_market:
        st.error("No markets available")
        return
    
    window = st.sidebar.slider("COT Index Window (weeks)", min_value=26, max_value=156, value=104, step=4)
    upper_pct = st.sidebar.slider("Upper Extreme Threshold (%)", min_value=50, max_value=95, value=80)
    lower_pct = st.sidebar.slider("Lower Extreme Threshold (%)", min_value=5, max_value=50, value=20)
    lookback_days = st.sidebar.slider("Price Lookback (days)", min_value=30, max_value=730, value=365, step=30)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Data fetches from CFTC API (weekly updates)")
    
    # Fetch data
    with st.spinner(f"Fetching data for {selected_market}..."):
        try:
            cot_df = fetch_cot_data_with_fallback(selected_market)
            
            if cot_df is None or cot_df.empty:
                st.error(f"❌ No COT data found for {selected_market}")
                return
            
            # Preprocess
            cot_df = preprocess_cot(cot_df)
            cot_df = detect_extremes(cot_df, window=window, upper_pct=upper_pct, lower_pct=lower_pct)
            
            # Fetch price data
            price_symbol = MARKET_TO_SYMBOL.get(selected_market, selected_market)
            price_df = fetch_price_data(price_symbol, days=lookback_days)
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            logger.error(f"Error: {e}", exc_info=True)
            return
    
    # Metrics
    if not cot_df.empty:
        latest = cot_df.iloc[-1]
        latest_date = cot_df.index[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Commercial Net",
                f"{latest.get('commercial_net', 0):,.0f}",
                f"{latest.get('commercial_net_delta', 0):+,.0f}"
            )
        
        with col2:
            cot_idx = latest.get("commercial_cot_index", 0)
            st.metric(
                "Commercial COT Index",
                f"{cot_idx:.1f}",
                "🔴 Short" if latest.get("commercial_extreme_short", False) else "🟢 Long" if latest.get("commercial_extreme_long", False) else "⚪ Neutral"
            )
        
        with col3:
            st.metric(
                "Open Interest",
                f"{latest.get('open_interest', 0):,.0f}",
                f"{latest.get('open_interest_delta_pct', 0):+.2f}%"
            )
        
        with col4:
            st.metric(
                "Data Date",
                latest_date.strftime("%Y-%m-%d"),
                "(CFTC Weekly)"
            )
        
        st.markdown("---")
        
        # Trading Signal
        if not price_df.empty:
            price_signal = compare_with_price(cot_df, price_df)
            
            signal_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}[price_signal["signal"]]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{signal_emoji} Trading Signal: {price_signal['signal']}")
                if price_signal["reasons"]:
                    for reason in price_signal["reasons"]:
                        st.write(f"• {reason}")
            
            with col2:
                st.metric("Confidence", f"{price_signal['confidence']:.0%}")
        
        st.markdown("---")
        
        # Charts
        st.subheader("📈 Analysis Charts")
        
        # Align indices
        if not price_df.empty:
            common_idx = cot_df.index.intersection(price_df.index)
            if len(common_idx) > 0:
                cot_aligned = cot_df.loc[common_idx]
                price_aligned = price_df.loc[common_idx]
                
                # Chart 1: Price + Commercial Net
                fig1, ax1 = plt.subplots(figsize=(14, 6))
                ax1_twin = ax1.twinx()
                
                ax1.plot(price_aligned.index, price_aligned.values, "b-", linewidth=2.5, label="Price")
                ax1.set_ylabel("Price", color="b", fontsize=11, fontweight="bold")
                ax1.tick_params(axis="y", labelcolor="b")
                
                colors = ["g" if x > 0 else "r" for x in cot_aligned["commercial_net"].values]
                ax1_twin.bar(cot_aligned.index, cot_aligned["commercial_net"].values, alpha=0.3, color=colors)
                ax1_twin.set_ylabel("Commercial Net Position", color="gray", fontsize=11, fontweight="bold")
                ax1_twin.tick_params(axis="y", labelcolor="gray")
                
                # Mark extremes
                extreme_long = cot_aligned[cot_aligned.get("commercial_extreme_long", False)]
                extreme_short = cot_aligned[cot_aligned.get("commercial_extreme_short", False)]
                
                if len(extreme_long) > 0:
                    ax1.scatter(extreme_long.index, price_aligned.loc[extreme_long.index], 
                               marker="^", color="green", s=150, zorder=5, label="Extreme Long", edgecolors="darkgreen", linewidth=2)
                if len(extreme_short) > 0:
                    ax1.scatter(extreme_short.index, price_aligned.loc[extreme_short.index], 
                               marker="v", color="red", s=150, zorder=5, label="Extreme Short", edgecolors="darkred", linewidth=2)
                
                ax1.set_title(f"{selected_market} - Weekly Price + Commercial Net Position", fontsize=12, fontweight="bold")
                ax1.legend(loc="upper left", fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                st.pyplot(fig1)
            else:
                st.warning("⚠️ No overlapping dates between COT and price data")
        
        # Chart 2: COT Indices
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        
        ax2.plot(cot_df.index, cot_df["commercial_cot_index"].values, 
                 "b-", linewidth=2.5, label="Commercial COT Index")
        ax2.plot(cot_df.index, cot_df["large_spec_cot_index"].values, 
                 "r-", linewidth=2.5, label="Large Spec COT Index")
        
        ax2.axhline(upper_pct, color="orange", linestyle="--", alpha=0.6, linewidth=2, label=f"Extreme Long ({upper_pct}%)")
        ax2.axhline(lower_pct, color="orange", linestyle="--", alpha=0.6, linewidth=2, label=f"Extreme Short ({lower_pct}%)")
        ax2.fill_between(cot_df.index, upper_pct, 100, alpha=0.1, color="green")
        ax2.fill_between(cot_df.index, 0, lower_pct, alpha=0.1, color="red")
        
        ax2.set_ylabel("COT Index (0-100)", fontsize=11, fontweight="bold")
        ax2.set_ylim([0, 100])
        ax2.set_title(f"{selected_market} - COT Indices", fontsize=12, fontweight="bold")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Chart 3: Open Interest
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        
        ax3.bar(cot_df.index, cot_df["open_interest"].values, alpha=0.6, color="steelblue", edgecolor="darkblue", linewidth=0.5)
        ax3.set_title(f"{selected_market} - Open Interest", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Open Interest", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")
        
        st.pyplot(fig3)
        
        st.markdown("---")
        
        # Data Table
        st.subheader("📋 Recent Data (Last 10 Weeks)")
        
        display_cols = ["commercial_net", "commercial_cot_index", "large_spec_net", 
                       "large_spec_cot_index", "open_interest", "open_interest_delta_pct"]
        display_df = cot_df[display_cols].tail(10).copy()
        display_df.columns = ["Comm. Net", "Comm. COT", "LS Net", "LS COT", "OI", "OI Δ%"]
        display_df = display_df.round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.caption(
            f"**Last Update:** {latest_date.strftime('%Y-%m-%d')} | "
            f"**Data Source:** CFTC API | **Symbol:** {price_symbol}"
        )


if __name__ == "__main__":
    main()

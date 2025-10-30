# metrics/sentiment.py
"""
Derivatives Market Sentiment Metrics for MAPS Framework.
Includes Funding Rate, Open Interest, Long/Short Ratio, OI Change, Funding Skew,
Volume Imbalance, Liquidation Heat, OI Delta Divergence, and Volatility Skew.

Author: ysf-bot-framework
Version: 2025.1
Updated: 2025-10-28

sentiment	Pandas	Orta ölçekli veri	Async
"""

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


# ==========================================================
# === Funding Rate =========================================
# ==========================================================

def Funding_Rate(funding_rates):
    """
    Computes the rolling mean funding rate.
    Indicates sentiment bias (positive = long bias, negative = short bias).
    """
    funding_rates = np.asarray(funding_rates)
    return pd.Series(funding_rates).rolling(8).mean().to_numpy()


# ==========================================================
# === Open Interest ========================================
# ==========================================================

def Open_Interest(oi_series):
    """
    Raw Open Interest series (total notional open positions).
    Often normalized by market cap or volume for analysis.
    """
    return np.asarray(oi_series)


# ==========================================================
# === Long / Short Ratio ===================================
# ==========================================================

def Long_Short_Ratio(long_positions, short_positions):
    """
    Ratio of long to short positions.
    > 1 → bullish sentiment, < 1 → bearish sentiment.
    """
    long_positions = np.asarray(long_positions)
    short_positions = np.asarray(short_positions)
    return long_positions / (short_positions + 1e-9)


# ==========================================================
# === OI Change Rate =======================================
# ==========================================================

def OI_Change_Rate(oi_series, window: int = 4):
    """
    Percentage change in Open Interest across rolling window.
    """
    oi_series = pd.Series(oi_series)
    return oi_series.pct_change(window).to_numpy()


# ==========================================================
# === Funding Rate Skew ====================================
# ==========================================================

def Funding_Rate_Skew(funding_long, funding_short):
    """
    Measures funding asymmetry between long and short positions.
    Positive skew = over-leveraged longs.
    """
    funding_long = np.asarray(funding_long)
    funding_short = np.asarray(funding_short)
    return funding_long - funding_short


# ==========================================================
# === Volume Imbalance =====================================
# ==========================================================

def Volume_Imbalance(buy_volume, sell_volume):
    """
    Measures buy vs. sell volume imbalance.
    > 0 = buyers dominating, < 0 = sellers dominating.
    """
    buy_volume = np.asarray(buy_volume)
    sell_volume = np.asarray(sell_volume)
    return (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)


# ==========================================================
# === Liquidation Heat =====================================
# ==========================================================

def Liquidation_Heat(long_liquidations, short_liquidations, window: int = 24):
    """
    Measures intensity of forced liquidations (normalized).
    High heat = elevated liquidation pressure.
    """
    long_liquidations = pd.Series(long_liquidations)
    short_liquidations = pd.Series(short_liquidations)
    heat = (long_liquidations + short_liquidations).rolling(window).mean()
    return heat.to_numpy()


# ==========================================================
# === OI Delta Divergence ==================================
# ==========================================================

def OI_Delta_Divergence(oi_change, price_change):
    """
    Measures divergence between Open Interest delta and price movement.
    Divergence → unsustainable positioning.
    """
    oi_change = np.asarray(oi_change)
    price_change = np.asarray(price_change)
    return oi_change * np.sign(price_change)


# ==========================================================
# === Volatility Skew ======================================
# ==========================================================

def Volatility_Skew(call_iv, put_iv):
    """
    Measures options market bias: difference between call and put implied volatility.
    > 0 → call dominance, < 0 → put dominance.
    """
    call_iv = np.asarray(call_iv)
    put_iv = np.asarray(put_iv)
    return call_iv - put_iv


# ==========================================================
# === Composite Derivative Sentiment Score ================
# ==========================================================

def Deriv_Sentiment_Score(funding, oi, ls_ratio, oi_change, skew, vol_imbalance):
    """
    Composite sentiment score.
    Aggregates multiple metrics to produce sentiment index.
    """
    score = (
        0.2 * funding +
        0.15 * ls_ratio +
        0.15 * oi_change +
        0.1 * skew +
        0.1 * vol_imbalance +
        0.3 * np.log(oi + 1e-9)
    )
    return score


# Tüm metrik fonksiyonları için template
def metric_template(series, *args, **kwargs):
    # 1. Input validation
    if series is None or len(series) == 0:
        return np.nan
    
    # 2. NaN check
    if isinstance(series, (pd.Series, pd.DataFrame)):
        if series.isna().all():
            return np.nan
        # Forward fill then backward fill
        series = series.ffill().bfill()
    
    # 3. Length check
    if len(series) < kwargs.get('min_periods', 2):
        return np.nan
    
    # 4. Try-except wrapper
    try:
        # Actual calculation
        result = calculate_metric(series, *args, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        return np.nan
        
        
# ==========================================================
# === Export ===============================================
# ==========================================================

__all__ = [
    "Funding_Rate",
    "Open_Interest",
    "Long_Short_Ratio",
    "OI_Change_Rate",
    "Funding_Rate_Skew",
    "Volume_Imbalance",
    "Liquidation_Heat",
    "OI_Delta_Divergence",
    "Volatility_Skew",
    "Deriv_Sentiment_Score"
]

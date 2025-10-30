# metrics/microstructure.py
"""
Market Microstructure Metrics for MAPS framework.
Includes Order Flow Imbalance (OFI), Cumulative Volume Delta (CVD),
Microprice Deviation, Market Impact, Depth Elasticity, and
Taker Dominance calculations.

Author: ysf-bot-framework
Version: 2025.1
Updated: 2025-10-28

microstructure	Numpy	Düşük seviye array işlemleri	Sync (sıralı)
"""

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


# ==========================================================
# === Order Flow Imbalance (OFI) ===========================
# ==========================================================

def OFI(bid_price, bid_size, ask_price, ask_size):
    """
    Computes Order Flow Imbalance (OFI) per tick.
    Measures aggressiveness of limit order changes.
    Reference: Cont, Stoikov & Talreja (2014)
    """
    bid_price = np.asarray(bid_price)
    ask_price = np.asarray(ask_price)
    bid_size = np.asarray(bid_size)
    ask_size = np.asarray(ask_size)

    d_bid = np.diff(bid_size, prepend=bid_size[0])
    d_ask = np.diff(ask_size, prepend=ask_size[0])

    ofi = np.where(bid_price >= bid_price[0], d_bid, 0) - np.where(ask_price <= ask_price[0], d_ask, 0)
    return ofi


# ==========================================================
# === Cumulative Volume Delta (CVD) ========================
# ==========================================================

def CVD(buy_volume, sell_volume):
    """
    Cumulative volume delta across ticks or bars.
    CVD = Σ(Buy - Sell)
    """
    buy_volume = np.asarray(buy_volume)
    sell_volume = np.asarray(sell_volume)
    return np.cumsum(buy_volume - sell_volume)


# ==========================================================
# === Microprice Deviation =================================
# ==========================================================

def Microprice_Deviation(best_bid, best_ask, bid_size, ask_size):
    """
    Computes microprice = (P_ask * V_bid + P_bid * V_ask) / (V_bid + V_ask)
    Returns deviation from midprice.
    """
    best_bid = np.asarray(best_bid)
    best_ask = np.asarray(best_ask)
    bid_size = np.asarray(bid_size)
    ask_size = np.asarray(ask_size)

    midprice = (best_bid + best_ask) / 2
    microprice = (best_ask * bid_size + best_bid * ask_size) / (bid_size + ask_size + 1e-9)
    return microprice - midprice


# ==========================================================
# === Market Impact ========================================
# ==========================================================

def Market_Impact(trade_volume, price_series, window: int = 20):
    """
    Rolling market impact measure:
    Impact = corr(|ΔP|, Volume)
    """
    price_chg = np.abs(np.diff(price_series, prepend=price_series[0]))
    if isinstance(price_series, pd.Series):
        return pd.Series(price_chg).rolling(window).corr(pd.Series(trade_volume))
    else:
        corr_list = []
        for i in range(len(trade_volume)):
            if i < window:
                corr_list.append(np.nan)
            else:
                corr = np.corrcoef(price_chg[i-window:i], trade_volume[i-window:i])[0, 1]
                corr_list.append(corr)
        return np.array(corr_list)


# ==========================================================
# === Depth Elasticity =====================================
# ==========================================================

def Depth_Elasticity(depth_price, depth_volume, window: int = 10):
    """
    Elasticity of order book depth:
    E = %ΔVolume / %ΔPrice
    Measures how much liquidity changes per unit price move.
    """
    depth_price = np.asarray(depth_price)
    depth_volume = np.asarray(depth_volume)
    pct_p = np.diff(depth_price, prepend=depth_price[0]) / (depth_price + 1e-9)
    pct_v = np.diff(depth_volume, prepend=depth_volume[0]) / (depth_volume + 1e-9)
    elasticity = pct_v / (pct_p + 1e-9)
    if len(elasticity) > window:
        elasticity[:window] = np.nan
    return elasticity


# ==========================================================
# === Taker Dominance Ratio ================================
# ==========================================================

def Taker_Dominance_Ratio(taker_buy_volume, taker_sell_volume):
    """
    Measures aggressive trade dominance:
    Ratio > 1 → buyers dominating; < 1 → sellers dominating.
    """
    taker_buy_volume = np.asarray(taker_buy_volume)
    taker_sell_volume = np.asarray(taker_sell_volume)
    return taker_buy_volume / (taker_sell_volume + 1e-9)


# ==========================================================
# === Liquidity Density ====================================
# ==========================================================

def Liquidity_Density(depth_volume, tick_range: int = 10):
    """
    Measures liquidity per price tick.
    """
    depth_volume = np.asarray(depth_volume)
    return np.mean(depth_volume[-tick_range:]) / tick_range



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
# === Utility Export =======================================
# ==========================================================

__all__ = [
    "OFI",
    "CVD",
    "Microprice_Deviation",
    "Market_Impact",
    "Depth_Elasticity",
    "Taker_Dominance_Ratio",
    "Liquidity_Density"
]

# 📁 metrics/microstructure.py
"""
version : 1.2.0
Market Microstructure Metrics (Standardized NumPy Implementation)
MAPS Framework - Low-Level Order Flow Analysis

Includes:
- OFI
- CVD
- Microprice_Deviation
- Market_Impact
- Depth_Elasticity
- Taker_Dominance_Ratio
- Liquidity_Density

Author: ysf-bot-framework
Version: 2025.2
Updated: 2025-10-31
"""

import numpy as np
from typing import Union, Dict, Any, Optional
import asyncio

# Import standardization utilities
from .standard import metric_standard, MetricStandard


# ==========================================================
# === Order Flow Imbalance (OFI) ===========================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=2, fillna=True)
async def ofi(data: Dict[str, np.ndarray]) -> np.ndarray:
    # Input validation and conversion
    for key in ['bid_price', 'bid_size', 'ask_price', 'ask_size']:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")
        data[key] = _safe_clean(data[key])
    """
    Order Flow Imbalance (Cont, Stoikov & Talreja, 2014)
    OFI = ΔBid_Size (if bid_price up) - ΔAsk_Size (if ask_price down)
    
    Args:
        data: Dictionary containing:
            - bid_price: np.ndarray
            - bid_size: np.ndarray  
            - ask_price: np.ndarray
            - ask_size: np.ndarray
    
    Returns:
        OFI values as numpy array
    """
    bid_price = data['bid_price'].astype(float)
    ask_price = data['ask_price'].astype(float)
    bid_size = data['bid_size'].astype(float)
    ask_size = data['ask_size'].astype(float)

    # Input validation
    if len(bid_price) != len(ask_price) != len(bid_size) != len(ask_size):
        raise ValueError("All input arrays must have same length")

    # Calculate differences with NaN protection
    d_bid = np.diff(bid_size, prepend=bid_size[0])
    d_ask = np.diff(ask_size, prepend=ask_size[0])

    # Detect price movements with roll protection
    bid_up = np.full_like(bid_price, False, dtype=bool)
    ask_down = np.full_like(ask_price, False, dtype=bool)
    
    if len(bid_price) > 1:
        bid_up[1:] = bid_price[1:] >= bid_price[:-1]
        ask_down[1:] = ask_price[1:] <= ask_price[:-1]

    # Calculate OFI with zero division protection
    ofi = np.where(bid_up, d_bid, 0.0) - np.where(ask_down, d_ask, 0.0)
    
    # Replace extreme values with NaN
    ofi = np.where(np.abs(ofi) > 1e10, np.nan, ofi)
    
    return ofi


# ==========================================================
# === Cumulative Volume Delta (CVD) ========================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=1, fillna=True)
async def cvd(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Cumulative Volume Delta.
    CVD = cumulative sum of (buy_volume - sell_volume)
    
    Args:
        data: Dictionary containing:
            - buy_volume: np.ndarray
            - sell_volume: np.ndarray
    
    Returns:
        CVD values as numpy array
    """
    buy_volume = data['buy_volume'].astype(float)
    sell_volume = data['sell_volume'].astype(float)

    if len(buy_volume) != len(sell_volume):
        raise ValueError("Buy and sell volume arrays must have same length")

    delta = buy_volume - sell_volume
    
    # Handle NaN values in input
    delta = np.nan_to_num(delta, nan=0.0)
    
    cvd = np.cumsum(delta)
    
    # Protect against overflow
    cvd = np.where(np.abs(cvd) > 1e15, np.nan, cvd)
    
    return cvd


# ==========================================================
# === Microprice Deviation =================================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=1, fillna=True)
async def microprice_deviation(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Microprice deviation from midprice.
    Microprice = (Ask * BidVol + Bid * AskVol) / (BidVol + AskVol)
    Deviation = Microprice - Midprice
    
    Args:
        data: Dictionary containing:
            - best_bid: np.ndarray
            - best_ask: np.ndarray  
            - bid_size: np.ndarray
            - ask_size: np.ndarray
    
    Returns:
        Microprice deviation values as numpy array
    """
    best_bid = data['best_bid'].astype(float)
    best_ask = data['best_ask'].astype(float)
    bid_size = data['bid_size'].astype(float)
    ask_size = data['ask_size'].astype(float)

    # Input validation
    if len(best_bid) != len(best_ask) != len(bid_size) != len(ask_size):
        raise ValueError("All input arrays must have same length")

    # Calculate midprice and microprice with NaN protection
    mid = (best_bid + best_ask) / 2.0
    
    # Avoid division by zero
    total_size = bid_size + ask_size
    safe_total = np.where(total_size == 0, 1.0, total_size)
    
    micro = (best_ask * bid_size + best_bid * ask_size) / safe_total
    
    deviation = micro - mid
    
    # Clean extreme values
    deviation = np.where(np.abs(deviation) > 1e10, np.nan, deviation)
    
    return deviation


# ==========================================================
# === Market Impact ========================================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=20, fillna=True)
async def market_impact(data: Dict[str, np.ndarray], window: int = 20) -> np.ndarray:
    """
    Rolling correlation between |ΔP| and trade volume.
    Impact = Corr(|ΔP|, Volume)
    
    Args:
        data: Dictionary containing:
            - trade_volume: np.ndarray
            - price_series: np.ndarray
        window: Rolling window size
    
    Returns:
        Market impact values as numpy array
    """
    trade_volume = data['trade_volume'].astype(float)
    price_series = data['price_series'].astype(float)

    if len(trade_volume) != len(price_series):
        raise ValueError("Trade volume and price series must have same length")

    n = len(price_series)
    price_change = np.abs(np.diff(price_series, prepend=price_series[0]))
    
    impact = np.full(n, np.nan, dtype=float)

    for i in range(window, n):
        pv = trade_volume[i - window:i]
        pr = price_change[i - window:i]
        
        # Remove NaN values from window
        mask = ~(np.isnan(pv) | np.isnan(pr))
        pv_clean = pv[mask]
        pr_clean = pr[mask]
        
        if len(pv_clean) < 2 or np.std(pv_clean) == 0 or np.std(pr_clean) == 0:
            impact[i] = np.nan
        else:
            corr_matrix = np.corrcoef(pv_clean, pr_clean)
            impact[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else np.nan

    return impact


# ==========================================================
# === Depth Elasticity =====================================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=10, fillna=True)
async def depth_elasticity(data: Dict[str, np.ndarray], window: int = 10) -> np.ndarray:
    """
    Elasticity of order book depth:
    E = %ΔVolume / %ΔPrice
    
    Args:
        data: Dictionary containing:
            - depth_price: np.ndarray
            - depth_volume: np.ndarray
        window: Smoothing window
    
    Returns:
        Depth elasticity values as numpy array
    """
    depth_price = data['depth_price'].astype(float)
    depth_volume = data['depth_volume'].astype(float)

    if len(depth_price) != len(depth_volume):
        raise ValueError("Depth price and volume arrays must have same length")

    # Calculate percentage changes with NaN protection
    pct_price = np.diff(depth_price, prepend=depth_price[0]) / (depth_price + 1e-12)
    pct_volume = np.diff(depth_volume, prepend=depth_volume[0]) / (depth_volume + 1e-12)
    
    # Calculate elasticity with zero division protection
    elasticity = np.divide(pct_volume, pct_price, 
                          out=np.full_like(pct_volume, np.nan), 
                          where=np.abs(pct_price) > 1e-12)

    # Apply rolling mean for stability
    if window > 1 and len(elasticity) >= window:
        elasticity_smooth = np.full_like(elasticity, np.nan)
        for i in range(window - 1, len(elasticity)):
            window_data = elasticity[i - window + 1: i + 1]
            window_clean = window_data[~np.isnan(window_data)]
            if len(window_clean) > 0:
                elasticity_smooth[i] = np.mean(window_clean)
        elasticity = elasticity_smooth

    return elasticity


# ==========================================================
# === Taker Dominance Ratio ================================
# ==========================================================

@metric_standard(input_type="numpy", output_type="numpy", min_periods=1, fillna=True)
async def taker_dominance_ratio(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Aggressive taker dominance ratio.
    > 1 → buyer dominance, < 1 → seller dominance
    
    Args:
        data: Dictionary containing:
            - taker_buy_volume: np.ndarray
            - taker_sell_volume: np.ndarray
    
    Returns:
        Taker dominance ratio values as numpy array
    """
    taker_buy_volume = data['taker_buy_volume'].astype(float)
    taker_sell_volume = data['taker_sell_volume'].astype(float)

    if len(taker_buy_volume) != len(taker_sell_volume):
        raise ValueError("Taker buy and sell volume arrays must have same length")

    # Calculate ratio with zero division protection
    ratio = np.divide(taker_buy_volume, taker_sell_volume,
                     out=np.full_like(taker_buy_volume, np.nan),
                     where=taker_sell_volume > 1e-12)

    # Clean extreme values
    ratio = np.where(ratio > 1e6, np.nan, ratio)
    ratio = np.where(ratio < -1e6, np.nan, ratio)
    
    return ratio


# ==========================================================
# === Liquidity Density ====================================
# ==========================================================

@metric_standard(input_type="numpy", output_type="native", min_periods=10, fillna=True)
async def liquidity_density(data: Dict[str, np.ndarray], tick_range: int = 10) -> float:
    """
    Average liquidity per price tick.
    
    Args:
        data: Dictionary containing:
            - depth_volume: np.ndarray
        tick_range: Number of ticks to average
    
    Returns:
        Liquidity density as float
    """
    depth_volume = data['depth_volume'].astype(float)

    if len(depth_volume) < tick_range:
        return np.nan

    # Use last tick_range values
    recent_volume = depth_volume[-tick_range:]
    
    # Remove NaN values
    recent_volume = recent_volume[~np.isnan(recent_volume)]
    
    if len(recent_volume) == 0:
        return np.nan

    density = float(np.mean(recent_volume) / tick_range)
    
    return density if not np.isinf(density) else np.nan


# ==========================================================
# === Sync Wrappers for Compatibility ======================
# ==========================================================

def ofi_sync(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Synchronous wrapper for OFI"""
    return asyncio.run(OFI(data))

def cvd_sync(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Synchronous wrapper for CVD"""
    return asyncio.run(CVD(data))

def microprice_deviation_sync(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Synchronous wrapper for Microprice_Deviation"""
    return asyncio.run(Microprice_Deviation(data))

def market_impact_sync(data: Dict[str, np.ndarray], window: int = 20) -> np.ndarray:
    """Synchronous wrapper for Market_Impact"""
    return asyncio.run(Market_Impact(data, window))

def depth_elasticity_sync(data: Dict[str, np.ndarray], window: int = 10) -> np.ndarray:
    """Synchronous wrapper for Depth_Elasticity"""
    return asyncio.run(Depth_Elasticity(data, window))

def taker_dominance_ratio_sync(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Synchronous wrapper for Taker_Dominance_Ratio"""
    return asyncio.run(Taker_Dominance_Ratio(data))

def liquidity_density_sync(data: Dict[str, np.ndarray], tick_range: int = 10) -> float:
    """Synchronous wrapper for Liquidity_Density"""
    return asyncio.run(Liquidity_Density(data, tick_range))


# ==========================================================
# === Export ===============================================
# ==========================================================

__all__ = [
    # Async functions
    "OFI", "CVD", "Microprice_Deviation", "Market_Impact", 
    "Depth_Elasticity", "Taker_Dominance_Ratio", "Liquidity_Density",
    
    # Sync wrappers
    "OFI_sync", "CVD_sync", "Microprice_Deviation_sync", "Market_Impact_sync",
    "Depth_Elasticity_sync", "Taker_Dominance_Ratio_sync", "Liquidity_Density_sync"
]

# ==========================================================
# === Batch Standardization ================================
# ==========================================================

def get_standardized_metrics() -> Dict[str, callable]:
    """Get all metrics standardized with appropriate configuration"""
    
    metrics_dict = {
        'OFI': OFI,
        'CVD': CVD,
        'Microprice_Deviation': Microprice_Deviation,
        'Market_Impact': Market_Impact,
        'Depth_Elasticity': Depth_Elasticity,
        'Taker_Dominance_Ratio': Taker_Dominance_Ratio,
        'Liquidity_Density': Liquidity_Density
    }
    
    # Apply standardization configuration
    standardizer = MetricStandard()
    module_config = {
        'metric_config': {
            'input_type': 'numpy',
            'output_type': 'numpy',
            'min_periods': 5,
            'fillna': True
        },
        'metrics': {
            'Market_Impact': {'min_periods': 20},
            'Depth_Elasticity': {'min_periods': 10},
            'Liquidity_Density': {'min_periods': 10, 'output_type': 'native'}
        }
    }
    
    return standardizer.batch_standardize_metrics(metrics_dict, module_config)
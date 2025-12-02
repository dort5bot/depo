"""
analysis/metrics/microstructure.py
Date: 2024/12/19
Enhanced standard template for microstructure metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "order_book": ["bid_price", "bid_size", "ask_price", "ask_size"],
    "trade_volume": ["buy_volume", "sell_volume"],
    "best_quotes": ["best_bid", "best_ask", "bid_size", "ask_size"],
    "price_volume": ["trade_volume", "price_series"],
    "depth_data": ["depth_price", "depth_volume"],
    "taker_flow": ["taker_buy_volume", "taker_sell_volume"],
    "depth_only": ["depth_volume"],
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "numpy",
    "execution_type": "sync", 
    "category": "microstructure",

    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "ofi": "order_book",
        "cvd": "trade_volume",
        "microprice_deviation": "best_quotes",
        "market_impact": "price_volume",
        "depth_elasticity": "depth_data",
        "taker_dominance_ratio": "taker_flow",
        "liquidity_density": "depth_only",
    },

    # opsiyonel
    "score_profile": {
        "ofi": {
            "method": "zscore",
            "direction": "positive"
        },
        "cvd": {
            "method": "minmax",
            "range": [-1, 1],
            "direction": "positive"
        },
        "microprice_deviation": {
            "method": "zscore", 
            "direction": "positive"
        },
        "market_impact": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "depth_elasticity": {
            "method": "zscore",
            "direction": "negative"
        },
        "taker_dominance_ratio": {
            "method": "minmax", 
            "range": [0, 2],
            "direction": "positive"
        },
        "liquidity_density": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        }
    }
}

# ==================== PURE FUNCTIONS ====================

def ofi(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Order Flow Imbalance (Cont, Stoikov & Talreja, 2014)
    OFI = ΔBid_Size (if bid_price up) - ΔAsk_Size (if ask_price down)
    
    Args:
        data: Dictionary containing 'bid_price', 'bid_size', 'ask_price', 'ask_size'
    
    Returns:
        OFI values as numpy array
    """
    bid_price = data['bid_price']
    bid_size = data['bid_size']
    ask_price = data['ask_price'] 
    ask_size = data['ask_size']
    
    if len(bid_price) != len(ask_price) != len(bid_size) != len(ask_size):
        raise ValueError("All input arrays must have same length")

    d_bid = np.diff(bid_size, prepend=bid_size[0])
    d_ask = np.diff(ask_size, prepend=ask_size[0])

    bid_up = np.full_like(bid_price, False, dtype=bool)
    ask_down = np.full_like(ask_price, False, dtype=bool)
    
    if len(bid_price) > 1:
        bid_up[1:] = bid_price[1:] >= bid_price[:-1]
        ask_down[1:] = ask_price[1:] <= ask_price[:-1]

    ofi = np.where(bid_up, d_bid, 0.0) - np.where(ask_down, d_ask, 0.0)
    ofi = np.where(np.abs(ofi) > 1e10, np.nan, ofi)
    
    return ofi


def cvd(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Cumulative Volume Delta.
    CVD = cumulative sum of (buy_volume - sell_volume)
    
    Args:
        data: Dictionary containing 'buy_volume', 'sell_volume'
    
    Returns:
        CVD values as numpy array
    """
    buy_volume = data['buy_volume']
    sell_volume = data['sell_volume']
    
    if len(buy_volume) != len(sell_volume):
        raise ValueError("Buy and sell volume arrays must have same length")

    delta = buy_volume - sell_volume
    cvd = np.cumsum(delta)
    cvd = np.where(np.abs(cvd) > 1e15, np.nan, cvd)
    
    return cvd


def microprice_deviation(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Microprice deviation from midprice.
    
    Args:
        data: Dictionary containing 'best_bid', 'best_ask', 'bid_size', 'ask_size'
    
    Returns:
        Microprice deviation values as numpy array
    """
    best_bid = data['best_bid']
    best_ask = data['best_ask']
    bid_size = data['bid_size']
    ask_size = data['ask_size']
    
    # Uzunluk kontrolü ve düzeltme
    min_length = min(len(best_bid), len(best_ask), len(bid_size), len(ask_size))
    
    if min_length == 0:
        return np.array([np.nan])
    
    # Array'leri aynı uzunluğa getir
    best_bid = best_bid[:min_length]
    best_ask = best_ask[:min_length]
    bid_size = bid_size[:min_length]
    ask_size = ask_size[:min_length]

    mid = (best_bid + best_ask) / 2.0
    total_size = bid_size + ask_size
    safe_total = np.where(total_size == 0, 1.0, total_size)
    
    micro = (best_ask * bid_size + best_bid * ask_size) / safe_total
    deviation = micro - mid
    deviation = np.where(np.abs(deviation) > 1e10, np.nan, deviation)
    
    return deviation


def market_impact(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Rolling correlation between |ΔP| and trade volume.
    Impact = Corr(|ΔP|, Volume)
    
    Args:
        data: Dictionary containing 'trade_volume', 'price_series'
        window: Rolling window size (optional parameter)
    
    Returns:
        Market impact values as numpy array
    """
    trade_volume = data['trade_volume']
    price_series = data['price_series']
    window = params.get('window', 20)
    
    if len(trade_volume) != len(price_series):
        raise ValueError("Trade volume and price series must have same length")

    n = len(price_series)
    price_change = np.abs(np.diff(price_series, prepend=price_series[0]))
    impact = np.full(n, np.nan, dtype=float)

    for i in range(window, n):
        pv = trade_volume[i - window:i]
        pr = price_change[i - window:i]
        
        mask = ~(np.isnan(pv) | np.isnan(pr))
        pv_clean = pv[mask]
        pr_clean = pr[mask]
        
        if len(pv_clean) < 2 or np.std(pv_clean) == 0 or np.std(pr_clean) == 0:
            impact[i] = np.nan
        else:
            corr_matrix = np.corrcoef(pv_clean, pr_clean)
            impact[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else np.nan

    return impact


def depth_elasticity(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Elasticity of order book depth:
    E = %ΔVolume / %ΔPrice
    
    Args:
        data: Dictionary containing 'depth_price', 'depth_volume'
        window: Smoothing window (optional parameter)
    
    Returns:
        Depth elasticity values as numpy array
    """
    depth_price = data['depth_price']
    depth_volume = data['depth_volume']
    window = params.get('window', 10)
    
    if len(depth_price) != len(depth_volume):
        raise ValueError("Depth price and volume arrays must have same length")

    pct_price = np.diff(depth_price, prepend=depth_price[0]) / (depth_price + 1e-12)
    pct_volume = np.diff(depth_volume, prepend=depth_volume[0]) / (depth_volume + 1e-12)
    
    elasticity = np.divide(pct_volume, pct_price, 
                          out=np.full_like(pct_volume, np.nan), 
                          where=np.abs(pct_price) > 1e-12)

    if window > 1 and len(elasticity) >= window:
        elasticity_smooth = np.full_like(elasticity, np.nan)
        for i in range(window - 1, len(elasticity)):
            window_data = elasticity[i - window + 1: i + 1]
            window_clean = window_data[~np.isnan(window_data)]
            if len(window_clean) > 0:
                elasticity_smooth[i] = np.mean(window_clean)
        elasticity = elasticity_smooth

    return elasticity


def taker_dominance_ratio(data: Dict[str, np.ndarray], **params) -> np.ndarray:
    """
    Aggressive taker dominance ratio.
    > 1 → buyer dominance, < 1 → seller dominance
    
    Args:
        data: Dictionary containing 'taker_buy_volume', 'taker_sell_volume'
    
    Returns:
        Taker dominance ratio values as numpy array
    """
    taker_buy_volume = data['taker_buy_volume']
    taker_sell_volume = data['taker_sell_volume']
    
    if len(taker_buy_volume) != len(taker_sell_volume):
        raise ValueError("Taker buy and sell volume arrays must have same length")

    ratio = np.divide(taker_buy_volume, taker_sell_volume,
                     out=np.full_like(taker_buy_volume, np.nan),
                     where=taker_sell_volume > 1e-12)

    ratio = np.where(ratio > 1e6, np.nan, ratio)
    ratio = np.where(ratio < -1e6, np.nan, ratio)
    
    return ratio


def liquidity_density(data: Dict[str, np.ndarray], **params) -> float:
    """
    Average liquidity per price tick.
    
    Args:
        data: Dictionary containing 'depth_volume'
        tick_range: Number of ticks to average (optional parameter)
    
    Returns:
        Liquidity density as float
    """
    depth_volume = data['depth_volume']
    tick_range = params.get('tick_range', 10)
    
    if len(depth_volume) < tick_range:
        return np.nan

    recent_volume = depth_volume[-tick_range:]
    recent_volume = recent_volume[~np.isnan(recent_volume)]
    
    if len(recent_volume) == 0:
        return np.nan

    density = float(np.mean(recent_volume) / tick_range)
    
    return density if not np.isinf(density) else np.nan


# ==================== MODULE REGISTRY ====================
_METRICS = {
    "ofi": ofi,
    "cvd": cvd,
    "microprice_deviation": microprice_deviation,
    "market_impact": market_impact,
    "depth_elasticity": depth_elasticity,
    "taker_dominance_ratio": taker_dominance_ratio,
    "liquidity_density": liquidity_density
}

def get_metrics() -> List[str]:
    """Composite engine için metric listesi"""
    return list(_METRICS.keys())

def get_function(metric_name: str):
    """Composite engine için fonksiyon döndür"""
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    """Module-level configuration"""
    return _MODULE_CONFIG.copy()

def get_column_groups() -> Dict[str, List[str]]:
    """Column groups for this module"""
    return COLUMN_GROUPS.copy()
"""
analysis/metrics/risk.py
Standard template for all metric modules
Date: 2025/11/29
binace ile tam uyumlu
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "price_data": ["close"],
    "ohlc": ["open", "high", "low", "close"],
    "market_data": ["close", "volume"]
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "risk",
    
    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "volatility_risk": "price_data",
        "spread_risk": "price_data",
        "liquidity_depth_risk": "price_data",
        "price_impact_risk": "price_data",
        "taker_pressure_risk": "market_data",
        "funding_risk": "price_data",
        "open_interest_risk": "price_data"
    },

    # opsiyonel
    "score_profile": {
        "volatility_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "spread_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "liquidity_depth_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "price_impact_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "taker_pressure_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "funding_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "open_interest_risk": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        }
    }
}

# ==================== PURE FUNCTIONS ====================
def volatility_risk(data: Dict[str, Any], **params) -> float:
    """
    Calculate volatility risk based on price returns.
    Returns normalized value between 0-1.
    """
    # For pandas DataFrame input
    if isinstance(data, pd.DataFrame):
        closes = data['close'].values
    else:
        # For backward compatibility with dict input
        klines = data.get("klines", [])
        if not klines:
            return 0.0
        closes = np.array([float(k[4]) for k in klines])
    
    if len(closes) < 2:
        return 0.0
    
    returns = np.diff(np.log(closes))
    vol = np.std(returns)
    return min(1, vol / 0.02)


def spread_risk(data: Dict[str, Any], **params) -> float:
    """
    Calculate bid-ask spread risk.
    Returns normalized value between 0-1.
    """
    depth = data.get("depth", {})
    if not depth or "bids" not in depth or "asks" not in depth:
        return 0.0
    
    bid = float(depth["bids"][0][0])
    ask = float(depth["asks"][0][0])
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    return min(1, spread_pct / 0.002)


def liquidity_depth_risk(data: Dict[str, Any], baseline: float = 500000, **params) -> float:
    """
    Calculate liquidity depth risk.
    Returns normalized value between 0-1.
    """
    depth = data.get("depth", {})
    if not depth or "bids" not in depth or "asks" not in depth:
        return 1.0  # Maximum risk if no depth data
    
    topN = 20

    buy_depth = sum(float(p) * float(q) for p, q in depth["bids"][:topN])
    sell_depth = sum(float(p) * float(q) for p, q in depth["asks"][:topN])
    liquidity = buy_depth + sell_depth

    return 1 - min(1, liquidity / baseline)


def price_impact_risk(data: Dict[str, Any], target: float = 10000, **params) -> float:
    """
    Calculate price impact risk for large orders.
    Returns normalized value between 0-1.
    """
    depth = data.get("depth", {})
    if not depth or "bids" not in depth or "asks" not in depth:
        return 0.0

    def simulate_side(levels):
        total = 0
        for price, qty in levels:
            price = float(price)
            amount_value = price * float(qty)
            total += amount_value
            if total >= target:
                return price
        return float(levels[-1][0])

    bid = float(depth["bids"][0][0])
    ask = float(depth["asks"][0][0])
    mid = (bid + ask) / 2

    buy_price = simulate_side(depth["asks"])
    sell_price = simulate_side(depth["bids"])

    impact_buy = buy_price - mid
    impact_sell = mid - sell_price
    impact_pct = max(impact_buy, impact_sell) / mid

    return min(1, impact_pct / 0.003)


def taker_pressure_risk(data: Dict[str, Any], **params) -> float:
    """
    Calculate taker pressure risk based on trade imbalance.
    Returns normalized value between 0-1.
    """
    aggTrades = data.get("aggTrades", [])
    
    buy_vol = sum(float(t.get("q", 0)) for t in aggTrades if not t.get("m", False))
    sell_vol = sum(float(t.get("q", 0)) for t in aggTrades if t.get("m", False))

    if buy_vol + sell_vol == 0:
        return 0

    imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    return abs(imbalance)


def funding_risk(data: Dict[str, Any], **params) -> float:
    """
    Calculate funding rate risk using z-score.
    Returns normalized value between 0-1.
    """
    funding = data.get("fundingRate", [])

    if not funding or len(funding) < 2:
        return 0

    fr = np.array([float(f.get("fundingRate", 0)) for f in funding])
    z = (fr[-1] - fr.mean()) / (fr.std() + 1e-8)
    return min(1, abs(z) / 3)


def open_interest_risk(data: Dict[str, Any], **params) -> float:
    """
    Calculate open interest change risk.
    Returns normalized value between 0-1.
    """
    oi = data.get("openInterestHist", [])

    if not oi or len(oi) < 2:
        return 0

    prev = float(oi[-2].get("sumOpenInterest", 0))
    cur = float(oi[-1].get("sumOpenInterest", 0))
    
    if prev == 0:
        return 0
    
    change = (cur - prev) / prev
    return min(1, abs(change) / 0.2)


# ==================== REGISTRY ====================
_METRICS = {
    "volatility_risk": volatility_risk,
    "spread_risk": spread_risk,
    "liquidity_depth_risk": liquidity_depth_risk,
    "price_impact_risk": price_impact_risk,
    "taker_pressure_risk": taker_pressure_risk,
    "funding_risk": funding_risk,
    "open_interest_risk": open_interest_risk
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
    """Return column groups for data requirements"""
    return COLUMN_GROUPS.copy()
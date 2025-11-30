"""
analysis/metrics/riskbi.py
Standard template for all metric modules
Date: 2025/11/29
binace ile tam uyumlu
"""

import numpy as np
from typing import Dict, Any, Union, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",      # pandas, numpy, polars
    "execution_type": "sync",    # sync, async
    "category": "risk"           # technical, regime, risk, etc.
}

# ==================== PURE FUNCTIONS ====================
def volatility_risk(data: Dict[str, Any], **params) -> float:
    klines = data["klines"]
    closes = np.array([float(k[4]) for k in klines])
    returns = np.diff(np.log(closes))
    vol = np.std(returns)
    return min(1, vol / 0.02)


def spread_risk(data: Dict[str, Any], **params) -> float:
    depth = data["depth"]
    bid = float(depth["bids"][0][0])
    ask = float(depth["asks"][0][0])
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    return min(1, spread_pct / 0.002)


def liquidity_depth_risk(data: Dict[str, Any], baseline: float = 500000, **params) -> float:
    depth = data["depth"]
    topN = 20

    buy_depth = sum(float(p) * float(q) for p, q in depth["bids"][:topN])
    sell_depth = sum(float(p) * float(q) for p, q in depth["asks"][:topN])
    liquidity = buy_depth + sell_depth

    return 1 - min(1, liquidity / baseline)


def price_impact_risk(data: Dict[str, Any], target: float = 10000, **params) -> float:
    depth = data["depth"]

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
    aggTrades = data["aggTrades"]

    buy_vol = sum(float(t["q"]) for t in aggTrades if not t["m"])
    sell_vol = sum(float(t["q"]) for t in aggTrades if t["m"])

    if buy_vol + sell_vol == 0:
        return 0

    imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    return abs(imbalance)


def funding_risk(data: Dict[str, Any], **params) -> float:
    funding = data.get("fundingRate", [])

    if not funding:
        return 0

    fr = np.array([float(f["fundingRate"]) for f in funding])
    z = (fr[-1] - fr.mean()) / (fr.std() + 1e-8)
    return min(1, abs(z) / 3)


def open_interest_risk(data: Dict[str, Any], **params) -> float:
    oi = data.get("openInterestHist", [])

    if not oi or len(oi) < 2:
        return 0

    prev = float(oi[-2]["sumOpenInterest"])
    cur = float(oi[-1]["sumOpenInterest"])
    change = (cur - prev) / prev

    return min(1, abs(change) / 0.2)


# ==================== MODULE REGISTRY ====================
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

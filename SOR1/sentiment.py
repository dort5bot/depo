# metrics/sentiment_typed.py
"""

funding_rate: "Rolling mean funding rate with enhanced typing, sentiment"
funding_premium: "Funding premium between futures and spot markets with validation, sentiment"
oi_trend: "Trend direction of Open Interest with enhanced return types, sentiment"
"""
# ============================================================
# metrics/sentiment.py
# Standard template for all metric modules
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "sentiment"
}

# ============================================================
# ==================== PURE FUNCTIONS ========================
# ============================================================

def funding_rate(series: pd.Series, window: int = 8) -> pd.Series:
    """
    Pure rolling mean funding rate.
    Input: pd.Series
    Output: pd.Series
    """
    if not isinstance(series, pd.Series):
        raise TypeError("funding_rate: series must be a pandas Series")

    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)

    return series.rolling(window=window, min_periods=1).mean()


def funding_premium(futures_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    """
    Pure funding premium calculation.
    Input: pd.Series, pd.Series
    Output: pd.Series
    """
    if not (
        isinstance(futures_price, pd.Series)
        and isinstance(spot_price, pd.Series)
    ):
        raise TypeError("funding_premium: both inputs must be pandas Series")

    valid = (
        futures_price.notna() &
        spot_price.notna() &
        (spot_price != 0)
    )

    result = pd.Series(np.nan, index=futures_price.index)
    result[valid] = (futures_price[valid] / spot_price[valid] - 1.0) * 100
    return result


def oi_trend(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Linear trend direction of Open Interest.
    Input: pd.Series
    Output: pd.Series
    """
    if not isinstance(series, pd.Series):
        raise TypeError("oi_trend: series must be a pandas Series")

    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)

    def _slope(x):
        if len(x) < 2:
            return np.nan
        x_axis = np.arange(len(x))
        return np.polyfit(x_axis, x, 1)[0]

    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


# ============================================================
# ====================== MODULE REGISTRY ======================
# ============================================================

_METRICS = {
    "funding_rate": funding_rate,
    "funding_premium": funding_premium,
    "oi_trend": oi_trend,
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

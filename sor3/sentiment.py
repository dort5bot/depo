"""
analysis/metrics/sentiment.py
date: 30.11.2025 19:25
Enhanced standard template for sentiment metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "funding_data": ["funding_rate"],
    "price_data": ["futures_price", "spot_price"],
    "oi_data": ["open_interest"],
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "sentiment",

    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "funding_rate": "funding_data",
        "funding_premium": "price_data",
        "oi_trend": "oi_data",
    },

    # opsiyonel
    "score_profile": {
        "funding_rate": {
            "method": "minmax",
            "range": [-0.1, 0.1],  # Typical funding rate range
            "direction": "positive"
        },
        "funding_premium": {
            "method": "zscore",
            "range": [-5, 5],  # Percent range
            "direction": "positive"
        },
        "oi_trend": {
            "method": "zscore",
            "direction": "positive"
        }
    }
}


# ==================== PURE FUNCTIONS ====================
def funding_rate(data: pd.DataFrame, **params) -> pd.Series:
    """
    Pure rolling mean funding rate.
    Input: pd.DataFrame with 'funding_rate' column
    Output: pd.Series
    """
    window = params.get("window", 8)
    
    if "funding_rate" not in data.columns:
        raise ValueError("funding_rate: data must contain 'funding_rate' column")
    
    series = data["funding_rate"]
    
    if not isinstance(series, pd.Series):
        raise TypeError("funding_rate: funding_rate column must be a pandas Series")
    
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    return series.rolling(window=window, min_periods=1).mean()


def funding_premium(data: pd.DataFrame, **params) -> pd.Series:
    """
    Pure funding premium calculation.
    Input: pd.DataFrame with 'futures_price' and 'spot_price' columns
    Output: pd.Series
    """
    futures_col = params.get("futures_column", "futures_price")
    spot_col = params.get("spot_column", "spot_price")
    
    if futures_col not in data.columns:
        raise ValueError(f"funding_premium: data must contain '{futures_col}' column")
    if spot_col not in data.columns:
        raise ValueError(f"funding_premium: data must contain '{spot_col}' column")
    
    futures_price = data[futures_col]
    spot_price = data[spot_col]
    
    if not (isinstance(futures_price, pd.Series) and isinstance(spot_price, pd.Series)):
        raise TypeError("funding_premium: both inputs must be pandas Series")
    
    valid = (
        futures_price.notna() &
        spot_price.notna() &
        (spot_price != 0)
    )
    
    result = pd.Series(np.nan, index=futures_price.index)
    result[valid] = (futures_price[valid] / spot_price[valid] - 1.0) * 100
    return result


def oi_trend(data: pd.DataFrame, **params) -> pd.Series:
    """
    Linear trend direction of Open Interest.
    Input: pd.DataFrame with 'open_interest' column
    Output: pd.Series
    """
    window = params.get("window", 20)
    oi_col = params.get("oi_column", "open_interest")
    
    if oi_col not in data.columns:
        raise ValueError(f"oi_trend: data must contain '{oi_col}' column")
    
    series = data[oi_col]
    
    if not isinstance(series, pd.Series):
        raise TypeError("oi_trend: open_interest column must be a pandas Series")
    
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    def _slope(x):
        if len(x) < 2:
            return np.nan
        x_axis = np.arange(len(x))
        return np.polyfit(x_axis, x, 1)[0]
    
    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


# ==================== REGISTRY ====================
_METRICS = {
    "funding_rate": funding_rate,
    "funding_premium": funding_premium,
    "oi_trend": oi_trend,
}

def get_metrics() -> List[str]:
    return list(_METRICS.keys())

def get_function(metric_name: str):
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    return _MODULE_CONFIG.copy()

def get_column_groups() -> Dict[str, List[str]]:
    return COLUMN_GROUPS.copy()
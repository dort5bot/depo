EKLENEN DOSYAYI BU ŞABLON DOSYAYA GÖRE TAM DÖNÜŞÜMÜNÜ SAĞLA
TAM KODU VER

"""
analysis/metrics/[module_name].py
date: 02/12/2025
Enhanced standard template (Updated: required_columns structure)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "technical",

    # Doğrudan kolon listesi (YENİ TASARIM)
    "required_columns": {
        "abc_metric": ["open", "high", "low", "close"],
        "klm_metric": ["close"],
    },

    # opsiyonel skor profili
    "score_profile": {
        "abc_metric": {
            "method": "minmax",
            "range": [-1, 1],
            "direction": "positive"
        },
        "klm_metric": {
            "method": "zscore",
            "direction": "negative"
        }
    }
}


# ==================== PURE FUNCTIONS ====================
def abc_metric(data: pd.DataFrame, **params) -> pd.Series:
    """
    Example metric using OHLC columns.
    Input: DataFrame with open/high/low/close
    Output: pd.Series
    """
    if not all(c in data.columns for c in ["open", "high", "low", "close"]):
        raise ValueError("abc_metric: required columns open/high/low/close not found")

    # Örnek hesaplama: (high - low) / close
    high = data["high"]
    low = data["low"]
    close = data["close"]

    with np.errstate(divide='ignore', invalid='ignore'):
        result = (high - low) / close.replace(0, np.nan)

    return pd.Series(result, index=data.index)


def klm_metric(data: pd.DataFrame, **params) -> pd.Series:
    """
    Example metric using only close.
    Input: DataFrame with close column
    Output: pd.Series
    """
    if "close" not in data.columns:
        raise ValueError("klm_metric: data must contain 'close' column")

    close = data["close"]

    # Örnek hesaplama: rolling z-score
    window = params.get("window", 14)
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()

    result = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    return pd.Series(result, index=data.index)


# ==================== REGISTRY ====================
_METRICS = {
    "abc_metric": abc_metric,
    "klm_metric": klm_metric,
}


def get_metrics() -> List[str]:
    return list(_METRICS.keys())


def get_function(metric_name: str):
    return _METRICS.get(metric_name)


def get_module_config() -> Dict[str, Any]:
    # Derin kopya gerekmez, dict yeterlidir
    return _MODULE_CONFIG.copy()

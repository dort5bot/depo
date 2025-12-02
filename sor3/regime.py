"""
analysis/metrics/regime.py
date: 30.11.2025 19:25
Enhanced standard template
Regime Analysis Metrics Module
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "close_only": ["close"],
    "price_only": ["close"],  # regime modülü için price serisi yeterli
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "regime",  # category regime olarak kalıyor

    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "advance_decline_line": "close_only",
        "volume_leadership": "close_only",
        "performance_dispersion": "close_only",
    },

    # opsiyonel - skor profilleri
    "score_profile": {
        "advance_decline_line": {
            "method": "raw",  # cumulative sum olduğu için raw bırakıldı
            "range": [-np.inf, np.inf],
            "direction": "positive"  # yükselişler artı yön
        },
        "volume_leadership": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"  # yüksek volatility/volume liderlik göstergesi
        },
        "performance_dispersion": {
            "method": "raw",
            "range": [0, np.inf],
            "direction": "positive"  # yüksek dispersiyon risk artışı
        }
    }
}


# ==================== PURE FUNCTIONS ====================
def advance_decline_line(data: pd.DataFrame, threshold: float = 0.001, **params) -> pd.Series:
    """
    Basitleştirilmiş Advance-Decline Line
    Price series'den advances/declines hesaplar
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        threshold: Advance/decline threshold değeri
        **params: Ek parametreler
    
    Returns:
        Advance-Decline line değerleri (pandas Series)
    """
    # Veri kontrolü
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be pandas DataFrame")
    
    if "close" not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    price_series = data["close"]
    
    if len(price_series) < 20:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = price_series.pct_change()
    
    advances = (returns > threshold).astype(int)   # Yukarı hareketler
    declines = (returns < -threshold).astype(int)  # Aşağı hareketler
    
    ad_line = (advances - declines).cumsum()
    return ad_line


def volume_leadership(data: pd.DataFrame, window: int = 10, **params) -> pd.Series:
    """
    Basitleştirilmiş Volume Leadership
    Volatility'yi volume proxy'si olarak kullanır
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        window: Rolling window boyutu
        **params: Ek parametreler
    
    Returns:
        Volume leadership skorları (0-1 arası, pandas Series)
    """
    # Veri kontrolü
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be pandas DataFrame")
    
    if "close" not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    price_series = data["close"]
    
    if len(price_series) < window:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = np.log(price_series).diff()
    volatility = returns.rolling(window=window, min_periods=1).std()
    
    # Normalize et (0-1 arası)
    min_val = volatility.min()
    max_val = volatility.max()
    
    if pd.isna(min_val) or pd.isna(max_val) or abs(max_val - min_val) < 1e-10:
        leadership = pd.Series(0.0, index=volatility.index)
    else:
        leadership = (volatility - min_val) / (max_val - min_val + 1e-6)
    
    return leadership.fillna(0.0)


def performance_dispersion(data: pd.DataFrame, window: int = 15, **params) -> pd.Series:
    """
    Basitleştirilmiş Performance Dispersion  
    Rolling window'daki return varyansı
    
    Args:
        data: Pandas DataFrame, 'close' kolonu gereklidir
        window: Rolling window boyutu
        **params: Ek parametreler
    
    Returns:
        Performance dispersion değerleri (pandas Series)
    """
    # Veri kontrolü
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be pandas DataFrame")
    
    if "close" not in data.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    price_series = data["close"]
    
    if len(price_series) < window:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = np.log(price_series).diff()
    dispersion = returns.rolling(window=window, min_periods=1).std()
    
    return dispersion.fillna(0.0)


# ==================== REGISTRY ====================
_METRICS = {
    "advance_decline_line": advance_decline_line,
    "volume_leadership": volume_leadership,
    "performance_dispersion": performance_dispersion,
}

def get_metrics() -> List[str]:
    """Mevcut tüm metric'lerin listesini döndür"""
    return list(_METRICS.keys())

def get_function(metric_name: str):
    """Metric adına karşılık gelen fonksiyonu döndür"""
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    """Modül konfigürasyonunu döndür"""
    return _MODULE_CONFIG.copy()

def get_column_groups() -> Dict[str, List[str]]:
    """Kolon gruplarını döndür"""
    return COLUMN_GROUPS.copy()


# ==================== EXPORT ====================
__all__ = [
    "advance_decline_line",
    "volume_leadership", 
    "performance_dispersion",
    "get_metrics",
    "get_function", 
    "get_module_config",
    "get_column_groups"
]
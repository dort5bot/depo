"""
advance_decline_line: "Advance-Decline Line for market breadth analysis, market_breadth"
volume_leadership: "Volume leadership indicator measuring volume concentration, volume_analysis"
performance_dispersion: "Performance dispersion across multiple assets, cross_sectional_analysis"

"""
"""
analysis/metrics/regime.py
Standard template for all metric modules
Date: 2024/12/19
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",      # pandas, numpy, polars
    "execution_type": "sync",    # sync, async
    "category": "regime"         # technical, regime, risk, etc.
}

# ==================== PURE FUNCTIONS ====================

def advance_decline_line(price_series: pd.Series, threshold: float = 0.001) -> pd.Series:
    """
    Basitleştirilmiş Advance-Decline Line
    Price series'den advances/declines hesaplar
    
    Args:
        price_series: Price series input
        threshold: Advance/decline threshold değeri
    
    Returns:
        Advance-Decline line değerleri
    """
    if not isinstance(price_series, pd.Series):
        raise ValueError("Input must be pandas Series")
    
    if len(price_series) < 20:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = price_series.pct_change()
    
    advances = (returns > threshold).astype(int)   # Yukarı hareketler
    declines = (returns < -threshold).astype(int)  # Aşağı hareketler
    
    ad_line = (advances - declines).cumsum()
    return ad_line


def volume_leadership(price_series: pd.Series) -> pd.Series:
    """
    Basitleştirilmiş Volume Leadership
    Volatility'yi volume proxy'si olarak kullanır
    
    Args:
        price_series: Price series input
    
    Returns:
        Volume leadership skorları (0-1 arası)
    """
    if not isinstance(price_series, pd.Series):
        raise ValueError("Input must be pandas Series")
    
    if len(price_series) < 10:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = np.log(price_series).diff()
    volatility = returns.rolling(window=10, min_periods=1).std()
    
    # Normalize et (0-1 arası)
    leadership = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-6)
    return leadership.fillna(0.0)


def performance_dispersion(price_series: pd.Series) -> pd.Series:
    """
    Basitleştirilmiş Performance Dispersion  
    Rolling window'daki return varyansı
    
    Args:
        price_series: Price series input
    
    Returns:
        Performance dispersion değerleri
    """
    if not isinstance(price_series, pd.Series):
        raise ValueError("Input must be pandas Series")
    
    if len(price_series) < 15:
        return pd.Series([np.nan] * len(price_series), index=price_series.index)
    
    returns = np.log(price_series).diff()
    dispersion = returns.rolling(window=15, min_periods=1).std()
    
    return dispersion.fillna(0.0)


# ==================== MODULE REGISTRY ====================
_METRICS = {
    "advance_decline_line": advance_decline_line,
    "volume_leadership": volume_leadership,
    "performance_dispersion": performance_dispersion
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


# ==================== EXPORT ====================
__all__ = [
    "advance_decline_line",
    "volume_leadership", 
    "performance_dispersion",
    "get_metrics",
    "get_function", 
    "get_module_config"
]
# metrics/volatility.py
# -*- coding: utf-8 -*-
"""
version : 1.2.0
Volatility Metrics (Numpy Implementation)
MAPS Framework - Quantitative Volatility Analysis

Includes:
- GARCH(1,1)
- Hurst exponent
- Entropy index
- Variance Ratio test
- Range Expansion index

Author: ysf-bot-framework
Version: 2025.3
Updated: 2025-11-01
"""

import numpy as np
from math import log, sqrt, isfinite
from typing import Union, Dict, Any, Optional, tuple

# ==========================================================
# === Utility: Safe clean ==================================
# ==========================================================

def _safe_clean(arr, fill: float = 0.0) -> np.ndarray:
    """Enhanced safe clean with Pandas Series support"""
    if arr is None or len(arr) == 0:
        return np.array([], dtype=float)
    
    # Handle Pandas Series/DataFrame
    if hasattr(arr, 'values'):
        arr = arr.values
    
    # Extract numeric values if mixed types
    if hasattr(arr, 'dtype') and arr.dtype == object:
        try:
            # try to convert to numeric, coerce errors to NaN
            arr = pd.to_numeric(arr, errors='coerce').astype(float)
        except:
            arr = np.array(arr, dtype=object)
    
    arr = np.asarray(arr, dtype=object)  # First as object to avoid timestamp issues
    
    # Convert to float safely
    result = np.zeros_like(arr, dtype=float)
    for i, val in enumerate(arr):
        try:
            if hasattr(val, 'timestamp'):  # Handle timestamp
                result[i] = float(val.timestamp())
            else:
                result[i] = float(val)
        except (typeError, ValueError):
            result[i] = fill
    
    # Replace remaining NaN/inf
    mask = ~np.isfinite(result)
    result[mask] = fill
    
    return result
    
def _validate_input(data: np.ndarray, min_periods: int = 1) -> bool:
    """Validate input data meets requirements."""
    if data is None or len(data) == 0:
        return False
    return len(data) >= min_periods

# ==========================================================
# === GARCH(1,1) ===========================================
# ==========================================================

# metrics/volatility.py - garch_1_1 DÜZELtMESİ

def garch_1_1(returns: np.ndarray, omega: float = 0.000001, alpha: float = 0.05, 
              beta: float = 0.9, min_periods: int = 10, **kwargs) -> np.ndarray:
    """GARCH(1,1) volatility model - FIXED VERSION"""
    returns = _safe_clean(returns)
    
    if not _validate_input(returns, min_periods):
        return np.full(len(returns) if returns is not None else 0, np.nan)
    
    t = len(returns)
    if t == 0:
        return np.array([])

    var = np.zeros(t)
    var[0] = max(np.var(returns), 1e-12)
    
    for t in range(1, t):
        # ÖNEMLİ: returns[t-1] scalar değilse ilk elemanı al
        prev_return = returns[t - 1]
        if isinstance(prev_return, (np.ndarray, list)) and len(prev_return) > 0:
            prev_return = prev_return[0]  # İlk elemanı al
        elif isinstance(prev_return, (np.ndarray, list)) and len(prev_return) == 0:
            prev_return = 0.0
        
        prev_return = float(prev_return) if np.isscalar(prev_return) else 0.0
        
        var[t] = omega + alpha * (prev_return ** 2) + beta * var[t - 1]
        
        if not np.isfinite(var[t]) or var[t] <= 0:
            var[t] = var[t - 1]
    
    return np.sqrt(np.maximum(var, 1e-12))

# ==========================================================
# === Hurst exponent =======================================
# ==========================================================

def hurst_exponent(price_series: np.ndarray, max_lag: int = 100, 
                   min_periods: int = 50, **kwargs) -> float:
    """Hurst exponent estimation"""
    price_series = _safe_clean(price_series)
    
    if not _validate_input(price_series, min_periods):
        return np.nan
        

    if len(price_series) < max_lag:
        max_lag = min(len(price_series) - 1, 20)
        if max_lag < 5:
            return np.nan

    lags = np.arange(2, max_lag + 1)
    tau = np.array([
        np.std(price_series[lag:] - price_series[:-lag]) for lag in lags
    ])
    
    # Handle zero variance cases
    tau = np.where(tau <= 1e-12, np.nan, tau)
    if np.isnan(tau).all():
        return np.nan

    valid = np.isfinite(tau)
    if valid.sum() < 2:
        return np.nan

    try:
        poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
        hurst = float(poly[0])
        return hurst if np.isfinite(hurst) else np.nan
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


# ==========================================================
# === Entropy index ========================================
# ==========================================================

def entropy_index(price_series: np.ndarray, window: int = 100, bins: int = 30,
                  min_periods: int = 20, **kwargs) -> np.ndarray:
    """Entropy index calculation"""
    price_series = _safe_clean(price_series)
    
    if not _validate_input(price_series, min_periods):
        return np.full(len(price_series) if price_series is not None else 0, np.nan)
        
    
    if len(price_series) < window:
        return np.full(len(price_series), np.nan)

    # Calculate log returns with safe handling
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(np.log(np.maximum(price_series, 1e-9)))
    
    returns = _safe_clean(returns)
    n = len(returns)
    entropy_values = np.full(n + 1, np.nan)  # +1 to match original length
    
    if n < window:
        return entropy_values

    for i in range(window, n):
        segment = returns[i - window:i]
        
        # Skip if all zeros or contains invalid values
        if np.allclose(segment, 0) or not np.all(np.isfinite(segment)):
            continue
            
        try:
            hist, _ = np.histogram(segment, bins=bins, density=true)
            p = hist / (np.sum(hist) + 1e-12)
            p = np.clip(p, 1e-12, 1.0)
            entropy = -np.sum(p * np.log2(p))
            entropy_values[i + 1] = entropy if np.isfinite(entropy) else np.nan
        except (ValueError, ZeroDivisionError):
            continue

    return entropy_values


# ==========================================================
# === Variance Ratio test ==================================
# ==========================================================

def variance_ratio_test(price_series: np.ndarray, lag: int = 10, 
                        min_periods: int = 30) -> float:
    """
    Variance Ratio test for random walk detection.
    vr = Var(r_t aggregated over lag) / (lag * Var(r_t))
    
    Args:
        price_series: Price series
        lag: Aggregation lag
        min_periods: Minimum data points required
    
    Returns:
        float: Variance ratio
    """
    price_series = _safe_clean(price_series)
    
    if not _validate_input(price_series, min_periods):
        return np.nan

    returns = np.diff(price_series)
    returns = _safe_clean(returns)
    
    if len(returns) < lag or lag <= 0:
        return np.nan

    var_1 = np.var(returns, ddof=1)
    if var_1 < 1e-12:
        return np.nan

    try:
        # Aggregate returns over lag periods
        lag_returns = np.add.reduceat(
            returns, 
            np.arange(0, len(returns) - (len(returns) % lag), lag)
        )
        
        if len(lag_returns) < 2:
            return np.nan
            
        var_k = np.var(lag_returns, ddof=1)
        vr = var_k / (lag * var_1 + 1e-12)
        return float(vr) if np.isfinite(vr) else np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


# ==========================================================
# === Range Expansion index ================================
# ==========================================================

def range_expansion_index(high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, window: int = 14,
                          min_periods: int = 10) -> np.ndarray:
    """
    rei = (Close - Mean(Close)) / (High - Low)
    Measures expansion/contraction in price range.
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        window: Rolling window size
        min_periods: Minimum data points required
    
    Returns:
        np.ndarray: rei values
    """
    high = _safe_clean(high)
    low = _safe_clean(low) 
    close = _safe_clean(close)
    
    # Validate all inputs have same length
    if len(high) != len(low) or len(high) != len(close):
        min_len = min(len(high), len(low), len(close))
        high = high[:min_len]
        low = low[:min_len] 
        close = close[:min_len]
    
    if not _validate_input(close, min_periods):
        return np.full(len(close) if close is not None else 0, np.nan)
    
    n = len(close)
    rei = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_close = close[i - window + 1:i + 1]
        mean_close = np.mean(window_close)
        
        denom = (high[i] - low[i])
        if abs(denom) < 1e-9:
            continue
            
        rei_value = (close[i] - mean_close) / denom
        rei[i] = rei_value if np.isfinite(rei_value) else np.nan

    return rei


# ==========================================================
# === Standardized Wrappers ================================
# ==========================================================

def garch_1_1_standardized(data: np.ndarray, **kwargs) -> np.ndarray:
    """Standardized wrapper for garch_1_1"""
    return garch_1_1(data, **kwargs)

def hurst_exponent_standardized(data: np.ndarray, **kwargs) -> float:
    """Standardized wrapper for hurst_exponent"""
    return hurst_exponent(data, **kwargs)

def entropy_index_standardized(data: np.ndarray, **kwargs) -> np.ndarray:
    """Standardized wrapper for entropy_index"""
    return entropy_index(data, **kwargs)

def variance_ratio_test_standardized(data: np.ndarray, **kwargs) -> float:
    """Standardized wrapper for variance_ratio_test"""
    return variance_ratio_test(data, **kwargs)

def range_expansion_index_standardized(high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray, **kwargs) -> np.ndarray:
    """Standardized wrapper for range_expansion_index"""
    return range_expansion_index(high, low, close, **kwargs)


# ==========================================================
# === export ===============================================
# ==========================================================

__all__ = [
    # Core functions
    "garch_1_1",
    "hurst_exponent", 
    "entropy_index",
    "variance_ratio_test",
    "range_expansion_index",
    
    # Standardized wrappers
    "garch_1_1_standardized",
    "hurst_exponent_standardized",
    "entropy_index_standardized", 
    "variance_ratio_test_standardized",
    "range_expansion_index_standardized"
]

# ==========================================================
# === module configuration =================================
# ==========================================================

MODULE_CONFIG = {
    'metric_config': {
        'input_type': 'numpy',
        'output_type': 'native', 
        'min_periods': 10,
        'fillna': True
    },
    'metrics': {
        'garch_1_1': {
            'min_periods': 10,
            'output_type': 'numpy'
        },
        'hurst_exponent': {
            'min_periods': 50,
            'output_type': 'native'
        },
        'entropy_index': {
            'min_periods': 20,
            'output_type': 'numpy'
        },
        'variance_ratio_test': {
            'min_periods': 30, 
            'output_type': 'native'
        },
        'range_expansion_index': {
            'min_periods': 10,
            'output_type': 'numpy'
        }
    }
}
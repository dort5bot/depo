"""
historical_volatility: "Optimized historical volatility, volatility"
bollinger_width: "Optimized Bollinger Band width, volatility"
garch_1_1: "GARCH(1,1) conditional volatility model, volatility_modeling"
hurst_exponent: "Hurst exponent for mean reversion or trend strength, fractal_analysis"
entropy_index: "Entropy-based volatility and randomness measure, information_theory"
variance_ratio_test: "Variance ratio test for random walk detection, statistical_test"
range_expansion_index: "Range Expansion Index measuring price acceleration, volatility_momentum"

"""
"""
analysis/metrics/volatility.py
Standard template for volatility metrics module
Date: 2024/12/19
"""

import numpy as np
from math import log, sqrt
from typing import Dict, Any, Union, List, Optional

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "numpy",      # pandas, numpy, polars
    "execution_type": "sync",   # sync, async
    "category": "volatility"    # technical, regime, risk, etc.
}

# ==================== PURE FUNCTIONS ====================

def _safe_clean_fast(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Ultra-fast array cleaning - NumPy only"""
    if arr is None or arr.size == 0:
        return np.array([], dtype=np.float64)
    
    # Convert to float64 directly
    try:
        result = np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError):
        # Fallback for complex types
        result = np.zeros_like(arr, dtype=np.float64)
        for i in range(len(arr)):
            try:
                result[i] = float(arr[i])
            except:
                result[i] = fill
    
    # Replace inf/nan
    np.nan_to_num(result, copy=False, nan=fill, posinf=fill, neginf=fill)
    return result

def _validate_input_fast(data: np.ndarray, min_periods: int = 1) -> bool:
    """Fast validation"""
    return data is not None and data.size >= min_periods

def historical_volatility(price_series: np.ndarray, window: int = 30, 
                         annualize: bool = True) -> np.ndarray:
    """Pure mathematical function - Optimized historical volatility"""
    price_series = _safe_clean_fast(price_series)
    n = len(price_series)
    
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    
    # Vectorized log returns
    returns = np.diff(np.log(np.maximum(price_series, 1e-9)))
    
    # Manual rolling std for max performance
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(window, n):
        segment = returns[i-window:i]
        if segment.size >= 2:  # Enough for std
            result[i] = np.std(segment, ddof=1)
    
    if annualize:
        result[window:] *= np.sqrt(252)
    
    return result

def bollinger_width(price_series: np.ndarray, window: int = 20, 
                   num_std: float = 2.0) -> np.ndarray:
    """Pure mathematical function - Optimized Bollinger Width"""
    price_series = _safe_clean_fast(price_series)
    n = len(price_series)
    
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(window-1, n):
        window_data = price_series[i-window+1:i+1]
        sma = np.mean(window_data)
        std = np.std(window_data, ddof=1)
        
        if sma > 1e-9:  # Avoid division by zero
            width = (2 * num_std * std) / sma
            result[i] = width
    
    return result

def garch_1_1(price_series: np.ndarray, omega: float = 1e-6, alpha: float = 0.05,
              beta: float = 0.9, min_periods: int = 10) -> np.ndarray:
    """Pure mathematical function - DÜZELTİLMİŞ GARCH(1,1) - PRICE SERIES ALIR"""
    price_series = _safe_clean_fast(price_series)
    
    if not _validate_input_fast(price_series, min_periods):
        return np.full(len(price_series), np.nan, dtype=np.float64)
    
    # 🔥 FIX: Price'dan log returns hesapla
    returns = np.diff(np.log(np.maximum(price_series, 1e-9)))
    
    n = returns.size
    if n < 2:
        return np.full(len(price_series), np.nan, dtype=np.float64)
    
    var = np.full(n, np.nan, dtype=np.float64)
    
    # Initialize with rolling variance (returns üzerinde!)
    first_window = returns[:min(20, n)]
    var[0] = np.var(first_window) if len(first_window) > 1 else 1e-6
    
    # GARCH calculation
    for t in range(1, n):
        var[t] = omega + alpha * (returns[t-1] ** 2) + beta * var[t-1]
        
        # Stability check
        if not np.isfinite(var[t]) or var[t] <= 0:
            var[t] = var[t-1]
    
    # Volatility (std) olarak döndür
    garch_vol = np.sqrt(np.maximum(var, 1e-12))
    
    # 🔥 FIX: Orijinal price_series uzunluğuna uyumla
    result = np.full(len(price_series), np.nan, dtype=np.float64)
    result[1:1+len(garch_vol)] = garch_vol  # Returns bir eksik olduğu için
    
    return result

def hurst_exponent(price_series: np.ndarray, max_lag: int = 100,
                  min_periods: int = 50) -> float:
    """Pure mathematical function - Optimized Hurst Exponent"""
    price_series = _safe_clean_fast(price_series)
    
    if not _validate_input_fast(price_series, min_periods):
        return np.nan
    
    n = len(price_series)
    max_lag = min(max_lag, n // 4)  # Adaptive lag selection
    
    if max_lag < 5:
        return np.nan
    
    lags = np.arange(2, max_lag + 1, dtype=np.int32)
    tau = np.zeros(len(lags), dtype=np.float64)
    
    # Vectorized R/S calculation
    for i, lag in enumerate(lags):
        if lag >= n:
            break
            
        # Calculate rescaled range
        segments = n // lag
        if segments < 2:
            continue
            
        rs_values = np.zeros(segments, dtype=np.float64)
        
        for j in range(segments):
            start = j * lag
            end = start + lag
            segment = price_series[start:end]
            
            if len(segment) < 2:
                continue
                
            mean_val = np.mean(segment)
            deviations = segment - mean_val
            cumulative = np.cumsum(deviations)
            std_val = np.std(segment, ddof=1)
            
            if std_val > 1e-12:
                rs_values[j] = (np.max(cumulative) - np.min(cumulative)) / std_val
        
        valid_rs = rs_values[rs_values > 0]
        if len(valid_rs) > 0:
            tau[i] = np.mean(valid_rs)
        else:
            tau[i] = np.nan
    
    # Linear regression on valid points
    valid_mask = np.isfinite(tau) & (tau > 1e-12)
    if np.sum(valid_mask) < 3:
        return np.nan
    
    try:
        hurst = np.polyfit(np.log(lags[valid_mask]), np.log(tau[valid_mask]), 1)[0]
        return float(hurst) if np.isfinite(hurst) else np.nan
    except:
        return np.nan

def entropy_index(price_series: np.ndarray, window: int = 100, bins: int = 20,
                 min_periods: int = 20) -> np.ndarray:
    """Pure mathematical function - Optimized Entropy Index"""
    price_series = _safe_clean_fast(price_series)
    
    if not _validate_input_fast(price_series, min_periods):
        return np.full(len(price_series), np.nan, dtype=np.float64)
    
    n = len(price_series)
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    
    # Calculate returns
    returns = np.diff(np.log(np.maximum(price_series, 1e-9)))
    returns = _safe_clean_fast(returns)
    
    result = np.full(n, np.nan, dtype=np.float64)
    bin_edges = np.linspace(np.min(returns), np.max(returns), bins + 1)
    
    for i in range(window, len(returns)):
        segment = returns[i-window:i]
        
        if np.std(segment) < 1e-12:  # Zero variance
            continue
            
        # Fast histogram
        hist, _ = np.histogram(segment, bins=bin_edges, density=True)
        hist = hist / (np.sum(hist) + 1e-12)
        hist = np.clip(hist, 1e-12, 1.0)
        
        entropy = -np.sum(hist * np.log2(hist))
        if np.isfinite(entropy):
            result[i+1] = entropy
    
    return result

def variance_ratio_test(price_series: np.ndarray, lag: int = 10,
                       min_periods: int = 30) -> float:
    """Pure mathematical function - Optimized Variance Ratio Test"""
    price_series = _safe_clean_fast(price_series)
    
    if not _validate_input_fast(price_series, min_periods):
        return np.nan
    
    returns = np.diff(price_series)
    n = len(returns)
    
    if n < lag or lag <= 1:
        return np.nan
    
    # Single-pass variance calculation
    var_1 = np.var(returns, ddof=1)
    if var_1 < 1e-12:
        return np.nan
    
    # Efficient lagged returns
    lag_returns = np.zeros(n // lag, dtype=np.float64)
    for i in range(len(lag_returns)):
        start = i * lag
        end = min(start + lag, n)
        lag_returns[i] = np.sum(returns[start:end])
    
    if len(lag_returns) < 2:
        return np.nan
    
    var_k = np.var(lag_returns, ddof=1)
    vr = var_k / (lag * var_1)
    
    return float(vr) if np.isfinite(vr) else np.nan

def range_expansion_index(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         window: int = 14, min_periods: int = 10) -> np.ndarray:
    """Pure mathematical function - Optimized Range Expansion Index"""
    high, low, close = map(_safe_clean_fast, [high, low, close])
    
    n = min(len(high), len(low), len(close))
    high, low, close = high[:n], low[:n], close[:n]
    
    if not _validate_input_fast(close, min_periods):
        return np.full(n, np.nan, dtype=np.float64)
    
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(window-1, n):
        window_close = close[i-window+1:i+1]
        mean_close = np.mean(window_close)
        price_range = high[i] - low[i]
        
        if price_range > 1e-9:
            rei = (close[i] - mean_close) / price_range
            if np.isfinite(rei):
                result[i] = rei
    
    return result

# ==================== MODULE REGISTRY ====================
_METRICS = {
    "historical_volatility": historical_volatility,
    "bollinger_width": bollinger_width,
    "garch_1_1": garch_1_1,
    "hurst_exponent": hurst_exponent,
    "entropy_index": entropy_index,
    "variance_ratio_test": variance_ratio_test,
    "range_expansion_index": range_expansion_index,
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
    "historical_volatility", 
    "bollinger_width", 
    "garch_1_1", 
    "hurst_exponent", 
    "entropy_index", 
    "variance_ratio_test", 
    "range_expansion_index",
    "get_metrics",
    "get_function", 
    "get_module_config"
]
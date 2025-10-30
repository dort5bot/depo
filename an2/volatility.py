# metrics/volatility.py
"""
Volatility and Regime Analysis Metrics for MAPS Framework.
Includes Historical Volatility, ATR, Bollinger Width, Variance Ratio, GARCH(1,1),
Entropy Index, and Hurst Exponent.

Author: ysf-bot-framework
Version: 2025.1
Updated: 2025-10-28

volatility	Numpy	Yoğun matematiksel işlemler	Async + Process
"""

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

from math import sqrt, log


# ==========================================================
# === Historical Volatility ================================
# ==========================================================

def Historical_Volatility(price_series, window: int = 30, annualize: bool = True):
    """
    Computes rolling historical volatility (standard deviation of log returns).
    """
    price_series = np.asarray(price_series)
    log_ret = np.diff(np.log(price_series + 1e-9))
    hv = pd.Series(log_ret).rolling(window).std()
    if annualize:
        hv *= np.sqrt(252)
    return hv.to_numpy()


# ==========================================================
# === Average True Range (ATR) =============================
# ==========================================================

def ATR(high, low, close, window: int = 14):
    """
    Average True Range: measures volatility by decomposing the entire range of an asset price.
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    prev_close = np.roll(close, 1)
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    atr = pd.Series(tr).rolling(window).mean()
    return atr.to_numpy()


# ==========================================================
# === Bollinger Band Width =================================
# ==========================================================

def Bollinger_Width(price_series, window: int = 20, num_std: float = 2.0):
    """
    Computes Bollinger Band Width = (Upper - Lower) / Middle.
    """
    price_series = pd.Series(price_series)
    mid = price_series.rolling(window).mean()
    std = price_series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-9)
    return width.to_numpy()


# ==========================================================
# === Variance Ratio Test ==================================
# ==========================================================

def Variance_Ratio_Test(price_series, lag: int = 10):
    """
    Variance Ratio test for mean reversion or random walk detection.
    VR = Var(r_t, lag k) / (k * Var(r_t))
    """
    price_series = np.asarray(price_series)
    returns = np.diff(price_series)
    n = len(returns)
    mean = np.mean(returns)
    var_1 = np.var(returns, ddof=1)
    lag_ret = np.add.reduceat(returns, np.arange(0, len(returns), lag))
    var_k = np.var(lag_ret, ddof=1)
    vr = var_k / (lag * var_1 + 1e-9)
    return vr


# ==========================================================
# === Range Expansion Index (REI) ==========================
# ==========================================================

def Range_Expansion_Index(high, low, close, window: int = 14):
    """
    REI = (Close - Avg(Close)) / (High - Low)
    Measures expansion/contraction in price range.
    """
    close = pd.Series(close)
    avg_close = close.rolling(window).mean()
    high = pd.Series(high)
    low = pd.Series(low)
    rei = (close - avg_close) / (high - low + 1e-9)
    return rei.to_numpy()


# ==========================================================
# === Hurst Exponent =======================================
# ==========================================================

def Hurst_Exponent(price_series, max_lag: int = 100):
    """
    Hurst exponent measures long-term memory of time series.
    H > 0.5 → trending, H < 0.5 → mean-reverting.
    """
    price_series = np.asarray(price_series)
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


# ==========================================================
# === GARCH(1,1) ===========================================
# ==========================================================

def GARCH_1_1(returns, omega=0.000001, alpha=0.05, beta=0.9):
    """
    Basic GARCH(1,1) volatility model implementation.
    h_t = ω + α * ε_{t-1}² + β * h_{t-1}
    """
    returns = np.asarray(returns)
    T = len(returns)
    var = np.zeros(T)
    var[0] = np.var(returns)
    for t in range(1, T):
        var[t] = omega + alpha * (returns[t-1] ** 2) + beta * var[t-1]
    return np.sqrt(var)


# ==========================================================
# === Entropy Index ========================================
# ==========================================================

def Entropy_Index(price_series, window: int = 100, bins: int = 30):
    """
    Shannon entropy of return distribution in a rolling window.
    High entropy → uncertainty (high volatility regime).
    """
    price_series = np.asarray(price_series)
    returns = np.diff(np.log(price_series + 1e-9))
    entropies = []
    for i in range(len(returns)):
        if i < window:
            entropies.append(np.nan)
            continue
        hist, _ = np.histogram(returns[i - window:i], bins=bins, density=True)
        p = hist / np.sum(hist + 1e-9)
        entropy = -np.sum(p * np.log2(p + 1e-9))
        entropies.append(entropy)
    return np.array(entropies)



# Tüm metrik fonksiyonları için template
def metric_template(series, *args, **kwargs):
    # 1. Input validation
    if series is None or len(series) == 0:
        return np.nan
    
    # 2. NaN check
    if isinstance(series, (pd.Series, pd.DataFrame)):
        if series.isna().all():
            return np.nan
        # Forward fill then backward fill
        series = series.ffill().bfill()
    
    # 3. Length check
    if len(series) < kwargs.get('min_periods', 2):
        return np.nan
    
    # 4. Try-except wrapper
    try:
        # Actual calculation
        result = calculate_metric(series, *args, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        return np.nan
        
# ==========================================================
# === Export ===============================================
# ==========================================================

__all__ = [
    "Historical_Volatility",
    "ATR",
    "Bollinger_Width",
    "Variance_Ratio_Test",
    "Range_Expansion_Index",
    "Hurst_Exponent",
    "GARCH_1_1",
    "Entropy_Index"
]

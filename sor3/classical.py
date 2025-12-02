"""
analysis/metrics/classical.py
date: 02/12/2025
Enhanced standard template (Updated: required_columns structure)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.signal import correlate

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "technical",

    # Doğrudan kolon listesi (YENİ TASARIM)
    "required_columns": {
        "adx": ["high", "low", "close"],
        "atr": ["high", "low", "close"],
        "bollinger_bands": ["close"],
        "cross_correlation": ["close"],
        "conditional_value_at_risk": ["close"],
        "ema": ["close"],
        "futures_roc": ["close"],
        "historical_volatility": ["close"],
        "macd": ["close"],
        "max_drawdown": ["close"],
        "oi_growth_rate": ["open_interest"],
        "oi_price_correlation": ["price", "open_interest"],
        "roc": ["close"],
        "rsi": ["close"],
        "sma": ["close"],
        "spearman_corr": ["close"],
        "stochastic_oscillator": ["high", "low", "close"],
        "value_at_risk": ["close"]
    },

    # opsiyonel skor profili
    "score_profile": {
        "adx": {
            "method": "minmax",
            "range": [0, 100],
            "direction": "positive"
        },
        "atr": {
            "method": "zscore",
            "direction": "negative"
        },
        "bollinger_bands": {
            "method": "minmax",
            "range": [-1, 1],
            "direction": "positive"
        },
        "ema": {
            "method": "minmax",
            "direction": "positive"
        },
        "macd": {
            "method": "minmax",
            "range": [-1, 1],
            "direction": "positive"
        },
        "rsi": {
            "method": "minmax",
            "range": [0, 100],
            "direction": "positive"
        },
        "stochastic_oscillator": {
            "method": "minmax",
            "range": [0, 100],
            "direction": "positive"
        }
    }
}

# ==================== PURE FUNCTIONS ====================

# ==========================================================
# === Trend & Moving averages ==============================
# ==========================================================

def ema(data: pd.DataFrame, **params) -> pd.Series:
    """TÜM EMA SERİSİNİ döndürdüğünden emin ol"""
    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("ema: data must contain 'close' column")
    
    close = data["close"]
    result = close.ewm(span=period, adjust=False).mean()
    return pd.Series(result, index=data.index)


def sma(data: pd.DataFrame, **params) -> pd.Series:
    """Simple Moving Average"""
    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("sma: data must contain 'close' column")
    
    close = data["close"]
    result = close.rolling(window=period, min_periods=1).mean()
    return pd.Series(result, index=data.index)


def macd(data: pd.DataFrame, **params) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal = params.get("signal", 9)
    
    if "close" not in data.columns:
        raise ValueError("macd: data must contain 'close' column")
    
    close = data["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    
    result = pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": hist
    }, index=data.index)
    return result


def rsi(data: pd.DataFrame, **params) -> pd.Series:
    """Relative Strength Index"""
    period = params.get("period", 14)
    if "close" not in data.columns:
        raise ValueError("rsi: data must contain 'close' column")
    
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    result = 100 - (100 / (1 + rs))
    return pd.Series(result, index=data.index)


def adx(data: pd.DataFrame, **params) -> pd.Series:
    """Average Directional Index"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"adx: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    result = dx.rolling(period, min_periods=1).mean()
    return pd.Series(result, index=data.index)


def stochastic_oscillator(data: pd.DataFrame, **params) -> pd.Series:
    """Stochastic Oscillator"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"stochastic_oscillator: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    result = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    return pd.Series(result, index=data.index)


def roc(data: pd.DataFrame, **params) -> pd.Series:
    """Rate of Change - Price change percentage over period"""
    period = params.get("period", 1)
    if "close" not in data.columns:
        raise ValueError("roc: data must contain 'close' column")
    
    close = data["close"]
    result = close.pct_change(periods=period) * 100
    return pd.Series(result, index=data.index)

# ==========================================================
# === Volatility metrics ===================================
# ==========================================================

def atr(data: pd.DataFrame, **params) -> pd.Series:
    """Average True Range"""
    period = params.get("period", 14)
    required_cols = ["high", "low", "close"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"atr: required columns {required_cols} not found")
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    result = tr.rolling(window=period, min_periods=1).mean()
    return pd.Series(result, index=data.index)


def bollinger_bands(data: pd.DataFrame, **params) -> pd.DataFrame:
    """Bollinger Bands"""
    period = params.get("period", 20)
    std_factor = params.get("std_factor", 2.0)
    
    if "close" not in data.columns:
        raise ValueError("bollinger_bands: data must contain 'close' column")
    
    close = data["close"]
    sma_val = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper = sma_val + std_factor * std
    lower = sma_val - std_factor * std
    bandwidth = (upper - lower) / (sma_val + 1e-10)
    
    result = pd.DataFrame({
        "middle": sma_val,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth
    }, index=data.index)
    return result


def historical_volatility(data: pd.DataFrame, **params) -> pd.Series:
    """Annualized Historical Volatility"""
    window = params.get("window", 30)
    annualize = params.get("annualize", True)
    
    if "close" not in data.columns:
        raise ValueError("historical_volatility: data must contain 'close' column")
    
    close = data["close"]
    log_returns = np.log(close / close.shift(1))
    volatility = log_returns.rolling(window=window).std()
    if annualize:
        volatility = volatility * np.sqrt(252)  # Annualization
    
    return pd.Series(volatility, index=data.index)

# ==========================================================
# === Risk metrics =========================================
# ==========================================================

def value_at_risk(data: pd.DataFrame, **params) -> pd.Series:
    """Value at Risk (percentile-based)"""
    confidence = params.get("confidence", 0.95)
    
    if "close" not in data.columns:
        raise ValueError("value_at_risk: data must contain 'close' column")
    
    close = data["close"]
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return pd.Series([np.nan], index=data.index[-1:], name="value_at_risk")
    
    val = np.percentile(returns, (1 - confidence) * 100)
    return pd.Series([val], index=data.index[-1:], name="value_at_risk")


def conditional_value_at_risk(data: pd.DataFrame, **params) -> pd.Series:
    """Conditional Value at Risk (expected shortfall)"""
    confidence = params.get("confidence", 0.95)
    
    if "close" not in data.columns:
        raise ValueError("conditional_value_at_risk: data must contain 'close' column")
    
    close = data["close"]
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return pd.Series([np.nan], index=data.index[-1:], name="conditional_value_at_risk")
    
    var_val = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_val].mean()
    return pd.Series([cvar], index=data.index[-1:], name="conditional_value_at_risk")


def max_drawdown(data: pd.DataFrame, **params) -> pd.Series:
    """Maximum Drawdown"""
    if "close" not in data.columns:
        raise ValueError("max_drawdown: data must contain 'close' column")
    
    close = data["close"]
    roll_max = close.cummax()
    drawdown = (close - roll_max) / (roll_max + 1e-10)
    result = pd.Series([drawdown.min()], index=data.index[-1:], name="max_drawdown")
    return result

# ==========================================================
# === Open Interest & Market structure =====================
# ==========================================================

def oi_growth_rate(data: pd.DataFrame, **params) -> pd.Series:
    """Open Interest Growth Rate"""
    period = params.get("period", 7)
    
    if "open_interest" not in data.columns:
        raise ValueError("oi_growth_rate: data must contain 'open_interest' column")
    
    oi_series = data["open_interest"]
    result = oi_series.pct_change(periods=period).fillna(0)
    return pd.Series(result, index=data.index)


def oi_price_correlation(data: pd.DataFrame, **params) -> pd.Series:
    """Rolling correlation between Open Interest and Price"""
    window = params.get("window", 14)
    required_cols = ["price", "open_interest"]
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"oi_price_correlation: required columns {required_cols} not found")
    
    price_series = data["price"]
    oi_series = data["open_interest"]
    result = oi_series.rolling(window=window, min_periods=1).corr(price_series)
    return pd.Series(result, index=data.index)

# ==========================================================
# === Correlation metrics ==================================
# ==========================================================

def spearman_corr(data: pd.DataFrame, **params) -> pd.Series:
    """Spearman rank correlation coefficient"""
    # Bu fonksiyon özel bir durum: iki seri gerekli
    # Bu yüzden data'nın en az iki sütun içermesi gerekir
    if len(data.columns) < 2:
        raise ValueError("spearman_corr: data must contain at least two columns")
    
    series_x = data.iloc[:, 0]
    series_y = data.iloc[:, 1]
    
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    
    if len(aligned_x) > 0:
        window = min(20, len(aligned_x))
        corr = aligned_x.rolling(window=window).corr(aligned_y)
        return pd.Series(corr, index=aligned_x.index)
    else:
        return pd.Series([np.nan], index=data.index[:1])


def cross_correlation(data: pd.DataFrame, **params) -> pd.Series:
    """Cross-correlation between two series with various lags"""
    max_lag = params.get("max_lag", 10)
    
    if len(data.columns) < 2:
        raise ValueError("cross_correlation: data must contain at least two columns")
    
    series_x = data.iloc[:, 0]
    series_y = data.iloc[:, 1]
    
    aligned_x, aligned_y = series_x.align(series_y, join='inner')

    if len(aligned_x) == 0:
        return pd.Series([], dtype=float)

    # Normalize series
    x_normalized = (aligned_x - aligned_x.mean()) / (aligned_x.std() + 1e-10)
    y_normalized = (aligned_y - aligned_y.mean()) / (aligned_y.std() + 1e-10)

    # Calculate cross-correlation
    correlation = correlate(x_normalized, y_normalized, mode='full')
    lags = np.arange(-len(aligned_x) + 1, len(aligned_x))

    valid_indices = (lags >= -max_lag) & (lags <= max_lag)
    valid_correlations = correlation[valid_indices]
    valid_lags = lags[valid_indices]

    if len(valid_correlations) > 0:
        max_corr_idx = np.argmax(np.abs(valid_correlations))
        best_lag = valid_lags[max_corr_idx]
        best_corr = valid_correlations[max_corr_idx]

        result = pd.Series({
            'best_correlation': best_corr,
            'best_lag': best_lag
        })
        return result
    else:
        return pd.Series([np.nan], index=['cross_correlation'])

# ==========================================================
# === Futures metrics ======================================
# ==========================================================

def futures_roc(data: pd.DataFrame, **params) -> pd.Series:
    """Futures Price Change - Simple roc for futures contracts"""
    period = params.get("period", 1)
    
    if "close" not in data.columns:
        raise ValueError("futures_roc: data must contain 'close' column")
    
    futures_series = data["close"]
    result = futures_series.pct_change(periods=period) * 100
    return pd.Series(result, index=data.index)

# ==================== REGISTRY ====================
_METRICS = {
    "adx": adx,
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    "cross_correlation": cross_correlation,
    "conditional_value_at_risk": conditional_value_at_risk,
    "ema": ema,
    "futures_roc": futures_roc,
    "historical_volatility": historical_volatility,
    "macd": macd,
    "max_drawdown": max_drawdown,
    "oi_growth_rate": oi_growth_rate,
    "oi_price_correlation": oi_price_correlation,
    "roc": roc,
    "rsi": rsi,
    "sma": sma,
    "spearman_corr": spearman_corr,
    "stochastic_oscillator": stochastic_oscillator,
    "value_at_risk": value_at_risk
}


def get_metrics() -> List[str]:
    return list(_METRICS.keys())


def get_function(metric_name: str):
    return _METRICS.get(metric_name)


def get_module_config() -> Dict[str, Any]:
    return _MODULE_CONFIG.copy()
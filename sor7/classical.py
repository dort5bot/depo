# 📁 metrics/classical.py
"""
version : 1.1.0
Classical Technical & Statistical metrics (Pandas-SYNC)
Author: ysf-bot-framework
Version: 2025.1
Data Model: Pandas (pure)
Execution Model: Full SYNC (sync input, sync compute)
Compatible with: MetricStandard Interface
"""

import numpy as np
import pandas as pd
from analysis.metrics.standard import metric_standard


# ==========================================================
# === Trend & Moving averages ==============================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def ema(series: pd.Series, period: int = 14) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def sma(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd_Line": macd_line,
        "Signal_Line": signal_line,
        "Histogram": hist
    })


@metric_standard(input_type="pandas", output_type="pandas")
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


@metric_standard(input_type="pandas", output_type="pandas")
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    plus_dm = (high.diff()).clip(lower=0)
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
    return dx.rolling(period, min_periods=1).mean()


# ==========================================================
# === Volatility metrics ===================================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def bollinger_bands(series: pd.Series, period: int = 20, std_factor: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    sma = sma(series, period)
    std = series.rolling(window=period, min_periods=1).std()
    upper = sma + std_factor * std
    lower = sma - std_factor * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return pd.DataFrame({
        "Middle": sma,
        "Upper": upper,
        "Lower": lower,
        "Bandwidth": bandwidth
    })


@metric_standard(input_type="pandas", output_type="pandas")
def historical_volatility(series: pd.Series, period: int = 30) -> pd.Series:
    """Annualized Historical Volatility"""
    log_returns = np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan)
    vol = log_returns.rolling(window=period, min_periods=1).std() * np.sqrt(252)
    return vol


# ==========================================================
# === Risk metrics =========================================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def va_r(series: pd.Series, confidence: float = 0.95) -> pd.Series:
    """Value at Risk (percentile-based)"""
    val = np.percentile(series.dropna(), (1 - confidence) * 100)
    return pd.Series([val], name="va_r")


@metric_standard(input_type="pandas", output_type="pandas")
def Cva_r(series: pd.Series, confidence: float = 0.95) -> pd.Series:
    """Conditional Value at Risk (expected shortfall)"""
    var_val = np.percentile(series.dropna(), (1 - confidence) * 100)
    cvar = series[series <= var_val].mean()
    return pd.Series([cvar], name="Cva_r")


@metric_standard(input_type="pandas", output_type="pandas")
def max_drawdown(series: pd.Series) -> pd.Series:
    """Maximum Drawdown"""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / (roll_max + 1e-10)
    return pd.Series([drawdown.min()], name="max_drawdown")


# ==========================================================
# === Open Interest & Market structure ======================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def oi_growth_rate(oi_series: pd.Series, period: int = 7) -> pd.Series:
    """Open Interest Growth Rate"""
    return oi_series.pct_change(periods=period).fillna(0)


@metric_standard(input_type="pandas", output_type="pandas")
def oi_price_correlation(oi_series: pd.Series, price_series: pd.Series, window: int = 14) -> pd.Series:
    """Rolling correlation between Open Interest and Price"""
    return oi_series.rolling(window=window, min_periods=1).corr(price_series)

# ==========================================================
# === Yeni Eklenen metrikler ===============================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change - Price change percentage over period"""
    return series.pct_change(periods=period) * 100

@metric_standard(input_type="pandas", output_type="pandas")
def spearman_corr(series_x: pd.Series, series_y: pd.Series) -> pd.Series:
    """Spearman rank correlation coefficient"""
    from scipy.stats import spearmanr
    
    def _rolling_spearman(x, y):
        if len(x) < 3:  # Minimum samples for correlation
            return np.nan
        return spearmanr(x, y)[0]
    
    # Align series and ensure same length
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    
    # Calculate rolling correlation with reasonable window
    if len(aligned_x) > 0:
        # Use a reasonable window size (e.g., 20 or min length)
        window = min(20, len(aligned_x))
        corr = aligned_x.rolling(window=window).corr(aligned_y)
        return corr
    else:
        return pd.Series([np.nan], index=series_x.index[:1])

@metric_standard(input_type="pandas", output_type="pandas")
def cross_correlation(series_x: pd.Series, series_y: pd.Series, max_lag: int = 10) -> pd.Series:
    """Cross-correlation between two series with various lags"""
    from scipy.signal import correlate
    
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    
    if len(aligned_x) == 0:
        return pd.Series([], dtype=float)
    
    # Normalize series
    x_normalized = (aligned_x - aligned_x.mean()) / aligned_x.std()
    y_normalized = (aligned_y - aligned_y.mean()) / aligned_y.std()
    
    # Calculate cross-correlation
    correlation = correlate(x_normalized, y_normalized, mode='full')
    lags = np.arange(-len(aligned_x) + 1, len(aligned_x))
    
    # Find maximum correlation within specified lag range
    valid_indices = (lags >= -max_lag) & (lags <= max_lag)
    valid_correlations = correlation[valid_indices]
    valid_lags = lags[valid_indices]
    
    if len(valid_correlations) > 0:
        max_corr_idx = np.argmax(np.abs(valid_correlations))
        best_lag = valid_lags[max_corr_idx]
        best_corr = valid_correlations[max_corr_idx]
        
        return pd.Series({
            'best_correlation': best_corr,
            'best_lag': best_lag
        })
    else:
        return pd.Series([np.nan], index=['cross_correlation'])

@metric_standard(input_type="pandas", output_type="pandas")
def Futures_roc(futures_series: pd.Series, period: int = 1) -> pd.Series:
    """Futures Price Change - Simple roc for futures contracts"""
    return futures_series.pct_change(periods=period) * 100
    
# ==========================================================
# === export ===============================================
# ==========================================================

__all__ = sorted([
    "adx", "atr", "bollinger_bands", "cross_correlation", "Cva_r",
    "ema", "Futures_roc", "historical_volatility", "macd", "max_drawdown",
    "oi_growth_rate", "oi_price_correlation", "roc", "rsi", 
    "sma", "spearman_corr", "va_r"
])



"""
roc: Zaten mevcut, kategorisi "momentum" olarak ayarlandı

spearman_corr: "correlation" kategorisinde, çıktı türü "float_series"

cross_correlation: "correlation" kategorisinde, çıktı türü "dataframe" (en iyi korelasyon ve lag bilgisi döndüğü için)

Futures_roc: "momentum" kategorisinde, input olarak "futures_price" kullanıyor
"""
# metrics/classical.py
"""
Classical technical and statistical indicator computation module.
Author: ysf-bot-framework
Version: 2025.1
Updated: 2025-10-28

classical	Pandas	Time-series operasyonları için optimize	Async + Thread
"""

import numpy as np
import pandas as pd

# ==========================================================
# === Moving Averages & Trend Indicators ===================
# ==========================================================

def EMA(series: pd.Series, period: int = 14) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def SMA(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def MACD(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        "MACD_Line": macd_line,
        "Signal_Line": signal_line,
        "Histogram": hist
    })

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (low.diff() > 0), minus_dm, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = np.max(np.vstack([tr1, tr2, tr3]), axis=0)
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.rolling(period).mean()

def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def Bollinger_Bands(series: pd.Series, period: int = 20, std_factor: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    sma = SMA(series, period)
    std = series.rolling(period).std()
    upper = sma + std_factor * std
    lower = sma - std_factor * std
    return pd.DataFrame({
        "Middle": sma,
        "Upper": upper,
        "Lower": lower,
        "Bandwidth": (upper - lower) / sma
    })

# ==========================================================
# === Volatility & Risk Metrics =============================
# ==========================================================

def Historical_Volatility(series: pd.Series, period: int = 30) -> pd.Series:
    """Annualized Historical Volatility"""
    log_returns = np.log(series / series.shift(1))
    vol = log_returns.rolling(period).std() * np.sqrt(252)
    return vol

def Variance_Ratio_Test(series: pd.Series, lag: int = 2) -> pd.Series:
    """Variance ratio test (simplified)"""
    log_ret = np.log(series / series.shift(1))
    var1 = log_ret.var()
    varn = log_ret.rolling(lag).sum().var() / lag
    return varn / var1

def Range_Expansion_Index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Range Expansion Index (REI)"""
    rng = (high - low).rolling(period).mean()
    rng_std = rng.rolling(period).std()
    return (rng - rng.mean()) / (rng_std + 1e-10)

def Max_Drawdown(series: pd.Series) -> float:
    """Maximum drawdown (percentage)"""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

# ==========================================================
# === Correlation & Statistical =============================
# ==========================================================

def Pearson_Correlation(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation coefficient"""
    return x.corr(y)

def Rolling_Covariance(x: pd.Series, y: pd.Series, window: int = 30) -> pd.Series:
    """Rolling covariance"""
    return x.rolling(window).cov(y)

def Beta_Coefficient(asset: pd.Series, benchmark: pd.Series, window: int = 30) -> pd.Series:
    """Rolling beta (asset vs benchmark)"""
    cov = asset.rolling(window).cov(benchmark)
    var = benchmark.rolling(window).var()
    return cov / var

# ==========================================================
# === Risk Metrics ==========================================
# ==========================================================

def VaR(series: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk"""
    return np.percentile(series.dropna(), (1 - confidence) * 100)

def CVaR(series: pd.Series, confidence: float = 0.95) -> float:
    """Conditional Value at Risk"""
    var = VaR(series, confidence)
    return series[series <= var].mean()

def Volatility_Targeting(series: pd.Series, target_vol: float = 0.2, window: int = 30) -> pd.Series:
    """Volatility targeting leverage ratio"""
    realized_vol = series.pct_change().rolling(window).std() * np.sqrt(252)
    return target_vol / (realized_vol + 1e-10)



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
# === Generalized Export ===================================
# ==========================================================

__all__ = [
    "EMA", "SMA", "MACD", "RSI", "ADX", "ATR", "Bollinger_Bands",
    "Historical_Volatility", "Variance_Ratio_Test", "Range_Expansion_Index",
    "Max_Drawdown", "Pearson_Correlation", "Rolling_Covariance",
    "Beta_Coefficient", "VaR", "CVaR", "Volatility_Targeting"
]

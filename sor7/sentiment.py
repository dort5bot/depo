# 📁 metrics/sentiment.py
"""
version : 1.2.0
Derivatives Market Sentiment metrics (Pandas)
MAPS Framework - Sentiment Module
Author: ysf-bot-framework
Version: 2025.1
Compatible with: MetricStandard Interface
"""

import numpy as np
import pandas as pd
from analysis.metrics.standard import metric_standard

# ==========================================================
# === Funding metrics ======================================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def funding_rate(series: pd.series, window: int = 8) -> pd.series:
    """
    Rolling mean funding rate.
    positive = long bias, negative = short bias.
    """
    if len(series) < window:
        return pd.series([np.nan] * len(series), index=series.index)
    return series.rolling(window=window, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def funding_rate_Trend(series: pd.series, window: int = 24) -> pd.series:
    """
    Measures directional trend in funding rates (momentum-style).
    positive = funding rising, negative = funding falling.
    """
    if len(series) < window:
        return pd.series([np.nan] * len(series), index=series.index)
    return series.diff(window).rolling(window=window, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def funding_premium(futures_price: pd.series, spot_price: pd.series) -> pd.series:
    """
    Funding premium between futures and spot markets.
    > 0 = futures premium (bullish), < 0 = discount (bearish).
    """
    valid_mask = (futures_price.notna()) & (spot_price.notna()) & (spot_price != 0)
    result = pd.series([np.nan] * len(futures_price), index=futures_price.index)
    result[valid_mask] = (futures_price[valid_mask] / spot_price[valid_mask] - 1.0) * 100
    return result


@metric_standard(input_type="pandas", output_type="pandas")
def funding_momentum(series: pd.series, window: int = 12) -> pd.series:
    """
    Momentum of funding rate changes.
    positive = accelerating funding rate increases, negative = decelerating.
    """
    if len(series) < window + 1:
        return pd.series([np.nan] * len(series), index=series.index)
    
    funding_changes = series.diff()
    return funding_changes.rolling(window=window, min_periods=1).mean()


# ==========================================================
# === Open Interest metrics ================================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def open_interest(series: pd.series) -> pd.series:
    """
    Raw Open Interest series (total notional open positions).
    """
    return series


@metric_standard(input_type="pandas", output_type="pandas")
def oi_change_rate(series: pd.series, window: int = 4) -> pd.series:
    """
    Percentage change in Open Interest over a rolling window.
    """
    if len(series) < window + 1:
        return pd.series([np.nan] * len(series), index=series.index)
    
    result = series.pct_change(periods=window)
    # Handle division by zero and infinite values
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


@metric_standard(input_type="pandas", output_type="pandas")
def oi_momentum(series: pd.series, window: int = 7) -> pd.series:
    """
    Measures acceleration in Open Interest changes.
    High positive = rising interest momentum.
    """
    if len(series) < window + 1:
        return pd.series([np.nan] * len(series), index=series.index)
    
    pct_changes = series.pct_change()
    pct_changes = pct_changes.replace([np.inf, -np.inf], np.nan)
    return pct_changes.rolling(window=window, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def oi_trend(series: pd.series, window: int = 20) -> pd.series:
    """
    Trend direction of Open Interest using linear regression slope.
    positive = increasing OI trend, negative = decreasing OI trend.
    """
    if len(series) < window:
        return pd.series([np.nan] * len(series), index=series.index)
    
    def _linear_trend(x):
        if len(x) < 2:
            return np.nan
        x_axis = np.arange(len(x))
        slope = np.polyfit(x_axis, x, 1)[0]
        return slope
    
    return series.rolling(window=window, min_periods=2).apply(_linear_trend, raw=True)


@metric_standard(input_type="pandas", output_type="pandas")
def oi_price_corr(oi_series: pd.series, price_series: pd.series, window: int = 20) -> pd.series:
    """
    Rolling correlation between Open Interest and Price.
    positive = OI and price move together, negative = divergence.
    """
    if len(oi_series) < window or len(price_series) < window:
        return pd.series([np.nan] * len(oi_series), index=oi_series.index)
    
    valid_mask = oi_series.notna() & price_series.notna()
    
    # Calculate rolling correlation
    oi_returns = oi_series.pct_change().replace([np.inf, -np.inf], np.nan)
    price_returns = price_series.pct_change().replace([np.inf, -np.inf], np.nan)
    
    correlation = oi_returns.rolling(window=window).corr(price_returns)
    return correlation


# ==========================================================
# === Long / Short metrics =================================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def long_short_ratio(long_positions: pd.series, short_positions: pd.series) -> pd.series:
    """
    Ratio of long to short positions.
    > 1 → bullish sentiment, < 1 → bearish sentiment.
    """
    valid_mask = (long_positions.notna()) & (short_positions.notna())
    result = pd.series([np.nan] * len(long_positions), index=long_positions.index)
    
    # Avoid division by zero
    denominator = short_positions[valid_mask] + 1e-10
    result[valid_mask] = long_positions[valid_mask] / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


@metric_standard(input_type="pandas", output_type="pandas")
def long_short_imbalance(long_positions: pd.series, short_positions: pd.series) -> pd.series:
    """
    Normalized imbalance between long and short positions.
    positive → longs dominate, negative → shorts dominate.
    """
    valid_mask = (long_positions.notna()) & (short_positions.notna())
    result = pd.series([np.nan] * len(long_positions), index=long_positions.index)
    
    total = long_positions[valid_mask] + short_positions[valid_mask] + 1e-10
    result[valid_mask] = (long_positions[valid_mask] - short_positions[valid_mask]) / total
    return result


@metric_standard(input_type="pandas", output_type="pandas")
def ls_imbalance(long_positions: pd.series, short_positions: pd.series, window: int = 8) -> pd.series:
    """
    Smoothed long-short imbalance with rolling mean.
    More stable version of long_short_imbalance.
    """
    raw_imbalance = long_short_imbalance(long_positions, short_positions)
    return raw_imbalance.rolling(window=window, min_periods=1).mean()


# ==========================================================
# === Volume & Position metrics ============================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def position_build(oi_series: pd.series, price_series: pd.series, window: int = 10) -> pd.series:
    """
    Identifies position building activity.
    Combines OI changes with price movement to detect accumulation/distribution.
    """
    if len(oi_series) < window or len(price_series) < window:
        return pd.series([np.nan] * len(oi_series), index=oi_series.index)
    
    # OI change and price change
    oi_change = oi_series.pct_change(window).replace([np.inf, -np.inf], np.nan)
    price_change = price_series.pct_change(window).replace([np.inf, -np.inf], np.nan)
    
    # Position building score
    position_score = oi_change * np.sign(price_change)
    return position_score.rolling(window=window, min_periods=1).mean()


@metric_standard(input_type="pandas", output_type="pandas")
def futures_volume(futures_volume: pd.series, spot_volume: pd.series, window: int = 10) -> pd.series:
    """
    Analyzes futures volume relative to spot volume.
    High futures/spot volume ratio indicates derivatives market activity.
    """
    valid_mask = (futures_volume.notna()) & (spot_volume.notna()) & (spot_volume != 0)
    result = pd.series([np.nan] * len(futures_volume), index=futures_volume.index)
    
    volume_ratio = futures_volume[valid_mask] / (spot_volume[valid_mask] + 1e-10)
    result[valid_mask] = volume_ratio.rolling(window=window, min_periods=1).mean()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# ==========================================================
# === Liquidation & Heat metrics ============================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def liquidation_heat(long_liq: pd.series, short_liq: pd.series, window: int = 24) -> pd.series:
    """
    Measures liquidation intensity.
    High heat = elevated forced liquidation pressure.
    """
    valid_mask = (long_liq.notna()) & (short_liq.notna())
    result = pd.series([np.nan] * len(long_liq), index=long_liq.index)
    
    total_liq = long_liq[valid_mask] + short_liq[valid_mask]
    result[valid_mask] = total_liq.rolling(window=window, min_periods=1).mean()
    return result


# ==========================================================
# === Trend & Momentum metrics =============================
# ==========================================================

@metric_standard(input_type="pandas", output_type="pandas")
def funding_trend(series: pd.series, window: int = 20) -> pd.series:
    """
    Linear trend of funding rates over specified window.
    More sophisticated trend analysis than simple differences.
    """
    if len(series) < window:
        return pd.series([np.nan] * len(series), index=series.index)
    
    def _linear_trend(x):
        if len(x) < 2:
            return np.nan
        x_axis = np.arange(len(x))
        slope = np.polyfit(x_axis, x, 1)[0]
        return slope
    
    return series.rolling(window=window, min_periods=2).apply(_linear_trend, raw=True)


@metric_standard(input_type="pandas", output_type="pandas")
def funding_premium_Rate(funding_rate: pd.series, benchmark_rate: pd.series = None) -> pd.series:
    """
    Premium of current funding rate relative to historical average or benchmark.
    """
    if benchmark_rate is None:
        # Use rolling historical average as benchmark
        benchmark_rate = funding_rate.rolling(window=168, min_periods=1).mean()
    
    valid_mask = funding_rate.notna() & benchmark_rate.notna()
    result = pd.series([np.nan] * len(funding_rate), index=funding_rate.index)
    result[valid_mask] = funding_rate[valid_mask] - benchmark_rate[valid_mask]
    return result




# ==========================================================
# === Async-Compatible Wrappers ============================
# ==========================================================

class SentimentMetricsAsync:
    """Async-compatible wrapper for sentiment metrics"""
    
    def __init__(self):
        self.metrics = {
            "Funding_Rate": Funding_Rate,
            "Funding_Rate_Trend": Funding_Rate_Trend,
            "Funding_Trend": Funding_Trend,
            "Funding_Premium": Funding_Premium,
            "Funding_Premium_Rate": Funding_Premium_Rate,
            "Funding_Momentum": Funding_Momentum,
            "Open_Interest": Open_Interest,
            "OI_Change_Rate": OI_Change_Rate,
            "OI_Momentum": OI_Momentum,
            "OI_Trend": OI_Trend,
            "OI_Price_Corr": OI_Price_Corr,
            "Long_Short_Ratio": Long_Short_Ratio,
            "Long_Short_Imbalance": Long_Short_Imbalance,
            "LS_Imbalance": LS_Imbalance,
            "Position_Build": Position_Build,
            "Futures_Volume": Futures_Volume,
            "Liquidation_Heat": Liquidation_Heat,
        }
    
    async def compute_metric(self, metric_name: str, *args, **kwargs):
        """Async wrapper for metric computation"""
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # Simulate async operation (in real usage, this would handle async data loading)
        metric_func = self.metrics[metric_name]
        return metric_func(*args, **kwargs)
    
    async def batch_compute(self, computations: list):
        """Batch compute multiple metrics asynchronously"""
        results = {}
        for computation in computations:
            metric_name = computation['metric']
            args = computation.get('args', [])
            kwargs = computation.get('kwargs', {})
            results[metric_name] = await self.compute_metric(metric_name, *args, **kwargs)
        return results


# ==========================================================
# === Export ===============================================
# ==========================================================

__all__ = [
    # Synchronous functions
    "Funding_Rate",
    "Funding_Rate_Trend",
    "Funding_Trend",
    "Funding_Premium",
    "Funding_Premium_Rate",
    "Funding_Momentum",
    "Open_Interest",
    "OI_Change_Rate",
    "OI_Momentum",
    "OI_Trend",
    "OI_Price_Corr",
    "Long_Short_Ratio",
    "Long_Short_Imbalance",
    "LS_Imbalance",
    "Position_Build",
    "Futures_Volume",
    "Liquidation_Heat",
    # Async class
    "SentimentMetricsAsync",
]

# ==========================================================
# === Usage Examples =======================================
# ==========================================================


"""
OI_Price_Corr: Open Interest ve fiyat hareketleri arasındaki korelasyon

Position_Building: Pozisyon birikim/dagıtım desenlerini tespit eder

Futures_Volume_Analysis: Futures hacminin spot piyasaya göre analizi

Funding_Rate_Premium: Funding rate'in tarihsel ortalamaya göre primi

Funding_Momentum: Funding rate değişim momentumu

Open_Interest_Trend: Open Interest lineer trend yönü

Funding_Rate_Trend: Funding rate lineer trend yönü

OI_Momentum: Open Interest değişim hızlanması

Long_Short_Imbalance: Long-Short pozisyon dengesizliği



# 🔄 SYNCHRONOUS USAGE:
funding_rate = Funding_Rate(funding_series, window=8)
oi_change = OI_Change_Rate(oi_series, window=4)
ls_ratio = Long_Short_Ratio(longs, shorts)
oi_price_corr = OI_Price_Corr(oi_series, price_series, window=20)
position_build = Position_Build(oi_series, price_series, window=10)

# 🔄 ASYNCHRONOUS USAGE:
async def main():
    sentiment_async = SentimentMetricsAsync()
    
    # Single metric
    funding = await sentiment_async.compute_metric(
        "Funding_Rate", 
        funding_series, 
        window=8
    )
    
    # Batch metrics
    results = await sentiment_async.batch_compute([
        {
            'metric': 'Funding_Rate',
            'args': [funding_series],
            'kwargs': {'window': 8}
        },
        {
            'metric': 'OI_Price_Corr',
            'args': [oi_series, price_series],
            'kwargs': {'window': 20}
        },
        {
            'metric': 'Position_Build',
            'args': [oi_series, price_series],
            'kwargs': {'window': 10}
        }
    ])
"""
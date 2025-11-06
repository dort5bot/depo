# 📁 metrics/metrics_regime.py
"""
MAPS Framework - Market Regime Detection Metrics
Author: ysf-bot-framework
Version: 2025.1
"""

from typing import Union, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

from .standard import metric_standard

# ==================== MARKET REGIME METRICS ====================

@metric_standard(input_type="pandas", output_type="pandas", min_periods=20, fillna=True)
def advance_decline(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, float]:
    """
    Advance-Decline Line for market breadth analysis
    
    Args:
        data: DataFrame with 'advances' and 'declines' columns, or Series for ratio calculation
        
    Returns:
        Advance-Decline line values or ratio
    """
    if isinstance(data, pd.DataFrame):
        if 'advances' in data.columns and 'declines' in data.columns:
            advances = data['advances']
            declines = data['declines']
            ad_line = (advances - declines).cumsum()
            return ad_line
        else:
            raise ValueError("DataFrame must contain 'advances' and 'declines' columns")
    else:
        # Assume data is advances/declines ratio
        return data.rolling(window=10).mean()

@metric_standard(input_type="pandas", output_type="pandas", min_periods=10, fillna=True)
def volume_leadership(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, float]:
    """
    Volume leadership indicator measuring volume concentration
    
    Args:
        data: DataFrame with multiple asset volumes, or Series for single asset
        
    Returns:
        Volume leadership score (0-1) where 1 indicates high concentration
    """
    if isinstance(data, pd.DataFrame):
        # Calculate Herfindahl index for volume concentration
        volume_sum = data.sum(axis=1)
        volume_share = data.div(volume_sum, axis=0)
        herfindahl = (volume_share ** 2).sum(axis=1)
        return herfindahl
    else:
        # For single series, return normalized volume
        return (data / data.rolling(window=20).mean()).clip(0, 2)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=15, fillna=True)
def perf_dispersion(data: pd.DataFrame) -> pd.Series:
    """
    Performance dispersion across multiple assets
    
    Args:
        data: DataFrame with returns/performance of multiple assets
        
    Returns:
        Cross-sectional standard deviation of performances
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Performance_Dispersion requires DataFrame input")
    
    # Calculate cross-sectional standard deviation
    dispersion = data.std(axis=1)
    return dispersion

# ==================== METRICS DICTIONARY ====================

REGIME_METRICS = {
    'Advance_Decline': Advance_Decline,
    'Volume_Leadership': Volume_Leadership, 
    'Perf_Dispersion': Perf_Dispersion
}

# ==================== USAGE EXAMPLES ====================

"""
# Kullanım Örnekleri:

# 1. Advance-Decline Line
advances = pd.Series([100, 150, 120, 80, 200])  # Yükselen hisse sayıları
declines = pd.Series([50, 30, 80, 120, 20])     # Düşen hisse sayıları
ad_data = pd.DataFrame({'advances': advances, 'declines': declines})
ad_line = Advance_Decline(ad_data)

# 2. Volume Leadership  
volume_data = pd.DataFrame({
    'BTC': [1000, 1500, 1200, 800, 2000],
    'ETH': [800, 900, 1000, 700, 1500],
    'ADA': [200, 300, 400, 500, 600]
})
leadership = Volume_Leadership(volume_data)

# 3. Performance Dispersion
returns_data = pd.DataFrame({
    'Asset1': [0.01, -0.02, 0.03, -0.01, 0.02],
    'Asset2': [0.02, 0.01, -0.01, 0.03, -0.02], 
    'Asset3': [-0.01, 0.03, 0.02, -0.02, 0.01]
})
dispersion = Perf_Dispersion(returns_data)

# Toplu kullanım
from .standard import MetricStandard
standardizer = MetricStandard()
standardized_metrics = standardizer.batch_standardize_metrics(
    REGIME_METRICS, 
    {
        'metric_config': {
            'input_type': 'pandas',
            'output_type': 'pandas',
            'min_periods': 10,
            'fillna': True
        }
    }
)
"""

# ==================== METRIC CONFIGURATION ====================

REGIME_METRICS_CONFIG = {
    'Advance_Decline': {
        'input_type': 'pandas',
        'output_type': 'pandas', 
        'min_periods': 20,
        'fillna': True
    },
    'Volume_Leadership': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 10, 
        'fillna': True
    },
    'Perf_Dispersion': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 15,
        'fillna': True
    }
}
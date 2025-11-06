# 📁 metrics/metrics_risk.py
"""
MAPS Framework - Risk Assessment Metrics
Author: ysf-bot-framework
Version: 2025.1
"""

from typing import Union, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

try:
    import polars as pl
except ImportError:
    pl = None

from .standard import metric_standard

# ==================== RISK ASSESSMENT METRICS ====================

@metric_standard(input_type="pandas", output_type="pandas", min_periods=50, fillna=True)
def liquidation_clusters(data: pd.DataFrame) -> pd.Series:
    """
    Detect liquidation clusters in market data
    
    Args:
        data: DataFrame with price and liquidation data
              Expected columns: ['price', 'liquidations_long', 'liquidations_short']
        
    Returns:
        Liquidation cluster intensity score
    """
    required_cols = ['price', 'liquidations_long', 'liquidations_short']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    price = data['price']
    liq_long = data['liquidations_long']
    liq_short = data['liquidations_short']
    
    # Total liquidation volume
    total_liq = liq_long + liq_short
    
    # Price volatility
    price_volatility = price.pct_change().rolling(window=20).std()
    
    # Liquidation clustering detection
    liq_zscore = (total_liq - total_liq.rolling(window=50).mean()) / total_liq.rolling(window=50).std()
    cluster_intensity = liq_zscore * price_volatility
    
    return cluster_intensity.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=30, fillna=True)
def cascade_risk(data: pd.DataFrame) -> pd.Series:
    """
    Measure potential for liquidation cascades
    
    Args:
        data: DataFrame with leverage and price data
              Expected columns: ['price', 'leverage_ratio', 'funding_rate']
        
    Returns:
        Cascade risk probability (0-1)
    """
    required_cols = ['price', 'leverage_ratio', 'funding_rate']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    price = data['price']
    leverage = data['leverage_ratio']
    funding = data['funding_rate']
    
    # Price drawdown
    price_drawdown = (price / price.rolling(window=50).max() - 1).abs()
    
    # Leverage stress
    leverage_stress = (leverage / leverage.rolling(window=30).mean() - 1)
    
    # Funding pressure
    funding_pressure = funding.rolling(window=20).std()
    
    # Composite cascade risk
    cascade_risk = (price_drawdown * leverage_stress * funding_pressure).clip(0, 1)
    
    return cascade_risk.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=20, fillna=True)
def sr_impact(data: pd.DataFrame) -> pd.Series:
    """
    Support and Resistance level impact assessment
    
    Args:
        data: DataFrame with price and volume data
              Expected columns: ['price', 'volume', 'high', 'low']
        
    Returns:
        Support/Resistance impact strength
    """
    required_cols = ['price', 'volume', 'high', 'low']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    price = data['price']
    volume = data['volume']
    high = data['high']
    low = data['low']
    
    # Calculate recent support/resistance levels
    resistance = high.rolling(window=20).max()
    support = low.rolling(window=20).min()
    
    # Distance to key levels
    dist_to_resistance = (resistance - price) / price
    dist_to_support = (price - support) / price
    
    # Volume confirmation at levels
    volume_at_levels = volume.rolling(window=10).mean()
    
    # Impact strength (closer to levels with high volume = higher impact)
    impact_strength = (1 / (dist_to_resistance.abs() + dist_to_support.abs() + 1e-6)) * volume_at_levels
    
    return impact_strength / impact_strength.rolling(window=50).max()

@metric_standard(input_type="pandas", output_type="pandas", min_periods=25, fillna=True)
def forced_selling(data: pd.DataFrame) -> pd.Series:
    """
    Measure forced selling pressure in the market
    
    Args:
        data: DataFrame with price, volume, and volatility data
              Expected columns: ['price', 'volume', 'volatility']
        
    Returns:
        Forced selling pressure index
    """
    required_cols = ['price', 'volume', 'volatility']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    price = data['price']
    volume = data['volume']
    volatility = data['volatility']
    
    # Price decline momentum
    price_momentum = price.pct_change(5)
    
    # Volume surge during declines
    volume_surge = volume / volume.rolling(window=20).mean()
    
    # Volatility expansion
    vol_expansion = volatility / volatility.rolling(window=20).mean()
    
    # Forced selling signature (declining price + high volume + high volatility)
    forced_selling = (price_momentum.clip(upper=0).abs() * volume_surge * vol_expansion)
    
    return forced_selling.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=40, fillna=True)
def liquidity_gaps(data: pd.DataFrame) -> pd.Series:
    """
    Detect liquidity gaps in order book data
    
    Args:
        data: DataFrame with order book depth data
              Expected columns: ['bid_depth', 'ask_depth', 'spread']
        
    Returns:
        Liquidity gap risk score
    """
    required_cols = ['bid_depth', 'ask_depth', 'spread']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    bid_depth = data['bid_depth']
    ask_depth = data['ask_depth']
    spread = data['spread']
    
    # Depth imbalance
    depth_imbalance = (bid_depth - ask_depth).abs() / (bid_depth + ask_depth)
    
    # Spread widening
    spread_widening = spread / spread.rolling(window=30).mean()
    
    # Thin liquidity detection
    min_depth = np.minimum(bid_depth, ask_depth)
    depth_threshold = min_depth.rolling(window=50).quantile(0.1)
    thin_liquidity = (min_depth < depth_threshold).astype(float)
    
    # Composite liquidity gap score
    liquidity_gap = (depth_imbalance * spread_widening * thin_liquidity)
    
    return liquidity_gap.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=35, fillna=True)
def futures_liq_risk(data: pd.DataFrame) -> pd.Series:
    """
    Futures liquidation risk assessment
    
    Args:
        data: DataFrame with futures market data
              Expected columns: ['futures_price', 'spot_price', 'open_interest', 'funding_rate']
        
    Returns:
        Futures liquidation risk index
    """
    required_cols = ['futures_price', 'spot_price', 'open_interest', 'funding_rate']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    futures = data['futures_price']
    spot = data['spot_price']
    oi = data['open_interest']
    funding = data['funding_rate']
    
    # Basis risk (futures-spot divergence)
    basis = (futures - spot) / spot
    
    # Open interest changes
    oi_change = oi.pct_change(5)
    
    # Funding rate stress
    funding_stress = funding.rolling(window=10).std()
    
    # Liquidation risk composite
    liq_risk = (basis.abs() * oi_change.abs() * funding_stress).clip(0, 1)
    
    return liq_risk.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=60, fillna=True)
def Liquidation_cascade(data: pd.DataFrame) -> pd.Series:
    """
    Detect active liquidation cascades
    
    Args:
        data: DataFrame with comprehensive market data
              Expected columns: ['price', 'liquidations', 'volume', 'volatility', 'leverage']
        
    Returns:
        Liquidation cascade intensity
    """
    required_cols = ['price', 'liquidations', 'volume', 'volatility', 'leverage']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    price = data['price']
    liquidations = data['liquidations']
    volume = data['volume']
    volatility = data['volatility']
    leverage = data['leverage']
    
    # Multi-factor cascade detection
    price_acceleration = price.pct_change().diff()
    liq_momentum = liquidations.pct_change(3)
    volume_acceleration = volume.pct_change().diff()
    leverage_compression = leverage.pct_change(5)
    
    # Cascade intensity score
    cascade_intensity = (
        price_acceleration.clip(upper=0).abs() *
        liq_momentum.clip(lower=0) *
        volume_acceleration.clip(lower=0) *
        leverage_compression.clip(upper=0).abs()
    )
    
    # Normalize to 0-1 range
    cascade_normalized = cascade_intensity / cascade_intensity.rolling(window=100).max()
    
    return cascade_normalized.fillna(0)

@metric_standard(input_type="pandas", output_type="pandas", min_periods=45, fillna=True)
def market_stress(data: pd.DataFrame) -> pd.Series:
    """
    Comprehensive market stress indicator
    
    Args:
        data: DataFrame with multiple market stress factors
              Expected columns: ['volatility', 'volume', 'liquidity', 'correlation', 'leverage']
        
    Returns:
        Market stress index (0-1)
    """
    required_cols = ['volatility', 'volume', 'liquidity', 'correlation', 'leverage']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    volatility = data['volatility']
    volume = data['volume']
    liquidity = data['liquidity']
    correlation = data['correlation']
    leverage = data['leverage']
    
    # Normalize each component
    vol_stress = volatility / volatility.rolling(window=50).max()
    volume_stress = volume / volume.rolling(window=50).max()
    liquidity_stress = 1 - (liquidity / liquidity.rolling(window=50).max())
    correlation_stress = correlation  # Higher correlation = more stress
    leverage_stress = leverage / leverage.rolling(window=50).max()
    
    # Composite stress index (equal weighting)
    market_stress = (vol_stress + volume_stress + liquidity_stress + correlation_stress + leverage_stress) / 5
    
    return market_stress.clip(0, 1).fillna(0)

# ==================== METRICS DICTIONARY ====================

RISK_METRICS = {
    'Liquidation_Clusters': Liquidation_Clusters,
    'Cascade_Risk': Cascade_Risk,
    'SR_Impact': SR_Impact,
    'Forced_Selling': Forced_Selling,
    'Liquidity_Gaps': Liquidity_Gaps,
    'Futures_Liq_Risk': Futures_Liq_Risk,
    'Liquidation_Cascade': Liquidation_Cascade,
    'Market_Stress': Market_Stress
}

# ==================== METRIC CONFIGURATION ====================

RISK_METRICS_CONFIG = {
    'Liquidation_Clusters': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 50,
        'fillna': True
    },
    'Cascade_Risk': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 30,
        'fillna': True
    },
    'SR_Impact': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 20,
        'fillna': True
    },
    'Forced_Selling': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 25,
        'fillna': True
    },
    'Liquidity_Gaps': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 40,
        'fillna': True
    },
    'Futures_Liq_Risk': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 35,
        'fillna': True
    },
    'Liquidation_Cascade': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 60,
        'fillna': True
    },
    'Market_Stress': {
        'input_type': 'pandas',
        'output_type': 'pandas',
        'min_periods': 45,
        'fillna': True
    }
}


"""
Liquidation Risk: Likidasyon kümelenmeleri ve kaskad riski

Market Stress: Kapsamlı piyasa stres göstergesi

Liquidity Analysis: Likidite boşlukları ve derinlik analizi

Forced Selling: Zorunlu satış baskısı tespiti

Support/Resistance: Kritik seviyelerin etki analizi


"""
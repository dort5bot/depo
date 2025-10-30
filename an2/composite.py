# ✅ metrics/composite.py - YENİ DOSYA
"""
Standardized composite metrics for MAPS framework.
All composites return values in [-1, 1] range.

Author: ysf-bot-framework  
Version: 2025.1
Updated: 2025-10-28
"""

import numpy as np
import pandas as pd
from typing import Union

# ================= STANDARD NORMALIZATION FUNCTIONS =================

def normalize_to_standard(value: float, metric_type: str, scale_factor: float = 1.0) -> float:
    """Convert any metric to standard [-1, 1] range"""
    if metric_type == "rsi":
        return (value - 50) / 50  # RSI [0,100] → [-1,1]
    elif metric_type == "macd":
        return np.tanh(value * 0.005)  # MACD → [-1,1]
    elif metric_type == "oscillator":
        return (value - 50) / 50  # [0,100] → [-1,1]
    elif metric_type == "volatility":
        return np.tanh(value * scale_factor - 1)  # Vol [0,∞] → [-1,1]
    elif metric_type == "ratio":
        if value <= 0: return -1.0
        log_val = np.log(value)
        return np.tanh(log_val * 0.5)  # Ratio → [-1,1]
    elif metric_type == "price_change":
        return np.tanh(value * 10)  # Price % → [-1,1]
    else:
        return np.tanh(value * 0.1)  # Default

# ================= COMPOSITE SCORING FUNCTIONS =================

def Trend_Strength_Composite(df: pd.DataFrame, lookback: int = 50) -> float:
    """Trend strength: -1 (Strong Bear) to +1 (Strong Bull)"""
    try:
        # Calculate base metrics
        rsi_val = RSI(df['close'], 14).iloc[-1]
        macd_val = MACD(df['close'])['MACD_Line'].iloc[-1]
        adx_val = ADX(df['high'], df['low'], df['close']).iloc[-1]
        ema_ratio = (EMA(df['close'], 20).iloc[-1] / df['close'].iloc[-1]) - 1
        
        # Normalize to [-1, 1]
        norm_rsi = normalize_to_standard(rsi_val, "rsi")
        norm_macd = normalize_to_standard(macd_val, "macd") 
        norm_adx = (adx_val / 50) - 1  # ADX [0,100] → [-1,1]
        norm_ema = normalize_to_standard(ema_ratio, "price_change")
        
        # Weighted composite
        trend_score = (
            0.30 * norm_rsi +    # Momentum
            0.35 * norm_macd +   # Trend direction
            0.20 * norm_adx +    # Trend strength  
            0.15 * norm_ema      # Trend confirmation
        )
        
        return float(np.clip(trend_score, -1, 1))
    except Exception:
        return 0.0  # Neutral on error

def Volatility_Composite(df: pd.DataFrame, lookback: int = 20) -> float:
    """Volatility regime: -1 (Very Low Vol) to +1 (Very High Vol)"""
    try:
        hist_vol = Historical_Volatility(df['close'], 20).iloc[-1]
        atr_val = ATR(df['high'], df['low'], df['close']).iloc[-1]
        bb_width = Bollinger_Width(df['close']).iloc[-1]
        
        # Normalize volatility metrics
        norm_hv = normalize_to_standard(hist_vol, "volatility", scale_factor=2.0)
        norm_atr = normalize_to_standard(atr_val, "volatility", scale_factor=10.0)
        norm_bb = normalize_to_standard(bb_width, "volatility", scale_factor=50.0)
        
        vol_score = (norm_hv + norm_atr + norm_bb) / 3
        return float(np.clip(vol_score, -1, 1))
    except Exception:
        return 0.0

def Risk_Composite(df: pd.DataFrame, lookback: int = 30) -> float:
    """Risk level: -1 (Very Low Risk) to +1 (Very High Risk)"""
    try:
        returns = df['close'].pct_change().dropna()
        var_val = VaR(returns, 0.95)
        cvar_val = CVaR(returns, 0.95) 
        max_dd = Max_Drawdown(df['close'])
        
        # Normalize risk metrics (absolute values since risk is magnitude)
        norm_var = normalize_to_standard(abs(var_val), "volatility", scale_factor=20.0)
        norm_cvar = normalize_to_standard(abs(cvar_val), "volatility", scale_factor=20.0)
        norm_dd = normalize_to_standard(abs(max_dd), "volatility", scale_factor=10.0)
        
        risk_score = (norm_var + norm_cvar + norm_dd) / 3
        return float(np.clip(risk_score, -1, 1))
    except Exception:
        return 0.0

def Momentum_Composite(df: pd.DataFrame, lookback: int = 14) -> float:
    """Momentum: -1 (Strong Down) to +1 (Strong Up)"""
    try:
        rsi_val = RSI(df['close'], 14).iloc[-1]
        roc_val = (df['close'].iloc[-1] / df['close'].iloc[-lookback] - 1) * 100
        
        norm_rsi = normalize_to_standard(rsi_val, "rsi")
        norm_roc = normalize_to_standard(roc_val, "price_change")
        
        momentum_score = 0.6 * norm_rsi + 0.4 * norm_roc
        return float(np.clip(momentum_score, -1, 1))
    except Exception:
        return 0.0

def Market_Regime(trend_score: float, vol_score: float, risk_score: float) -> str:
    """Market regime classification based on [-1,1] scores"""
    # Thresholds
    STRONG = 0.6
    MODERATE = 0.3
    
    # Trend classification
    if trend_score >= STRONG:
        trend_str = "STRONG_BULL"
    elif trend_score >= MODERATE:
        trend_str = "MODERATE_BULL"
    elif trend_score <= -STRONG:
        trend_str = "STRONG_BEAR" 
    elif trend_score <= -MODERATE:
        trend_str = "MODERATE_BEAR"
    else:
        trend_str = "NEUTRAL"
    
    # Volatility classification
    if vol_score >= MODERATE:
        vol_str = "HIGH_VOL"
    elif vol_score <= -MODERATE:
        vol_str = "LOW_VOL" 
    else:
        vol_str = "MED_VOL"
    
    return f"{trend_str}_{vol_str}"



# ✅ Composite metrikleri MetricResolver'a kaydet
def register_composite_metrics():
    """Composite metriklerini global registry'e kaydet"""
    from analysis.metric_resolver import MetricResolver
    
    resolver = MetricResolver.get_instance()
    
    # Composite fonksiyonlarını kaydet
    composites = {
        "Trend_Strength_Composite": Trend_Strength_Composite,
        "Volatility_Composite": Volatility_Composite,
        "Risk_Composite": Risk_Composite, 
        "Momentum_Composite": Momentum_Composite,
        "Market_Regime": Market_Regime
    }
    
    for name, func in composites.items():
        resolver.register(name, func)
    
    return resolver

# ✅ Modül yüklendiğinde otomatik kayıt
try:
    register_composite_metrics()
    print("✅ Composite metrics registered successfully")
except Exception as e:
    print(f"⚠️ Composite registration failed: {e}")
    
    
__all__ = [
    "Trend_Strength_Composite",
    "Volatility_Composite", 
    "Risk_Composite",
    "Momentum_Composite",
    "Market_Regime",
    "normalize_to_standard"
]
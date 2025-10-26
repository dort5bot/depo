# analysis/config/c_trend.py
"""
Trend & Momentum Analysis Module Configuration - Analysis Helpers Uyumlu
Version: 1.1.0
deepsek
"""

from analysis.config.cm_base import TrendConfig, ModuleLifecycle, ParallelMode
from analysis.config.cm_base import AnalysisModuleConfig, ModuleLifecycle, ParallelMode
from analysis.analysis_helpers import AnalysisHelpers

# ✅ ANALYSIS_HELPERS UYUMLU TREND CONFIG
TrendConfig = AnalysisModuleConfig(
#TrendConfig = TrendConfig(
    module_name="trend_moment",
    file="analysis/trend_moment.py",
    config="c_trend.py",
    command="/api/analysis/trend",
    api_type="public",
    job_type="batch",
    parallel_mode=ParallelMode.ASYNC,
    output_type="score",
    objective="Trend direction and momentum strength analysis",
    maintainer="deepsek",
    description="Comprehensive trend analysis using classical and advanced technical indicators",
    version="1.1.0",
    lifecycle=ModuleLifecycle.DEVELOPMENT,
    enabled=True,
    
    # ✅ TREND-SPECIFIC PARAMETERS
    parameters={
        # Data parameters
        "window": 100,
        "min_data_points": 50,
        
        # Classic TA parameters
        "ema_periods": [20, 50, 200],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "atr_period": 14,
        "adx_period": 14,
        "stoch_rsi_period": 14,
        "stoch_rsi_smooth": 3,
        "momentum_period": 10,
        
        # Advanced metrics parameters
        "kalman": {
            "process_var": 1e-4,
            "obs_var": 1e-3
        },
        "z_score_window": 21,
        "wavelet_family": "db4",
        "wavelet_level": 3,
        "hilbert_window": 10,
        "fdi_window": 10,
    },
    
    # ✅ ANALYSIS_HELPERS UYUMLU WEIGHTS (normalized)
    weights={
        "ema_trend": 0.15,
        "rsi_momentum": 0.12,
        "macd_trend": 0.13,
        "bollinger_trend": 0.10,
        "atr_volatility": 0.08,
        "adx_strength": 0.10,
        "stoch_rsi_momentum": 0.08,
        "momentum_oscillator": 0.07,
        "kalman_trend": 0.05,
        "z_score_normalization": 0.04,
        "wavelet_trend": 0.03,
        "hilbert_slope": 0.03,
        "fdi_complexity": 0.02
    },
    
    # ✅ THRESHOLDS
    thresholds={
        "bullish": 0.7,
        "bearish": 0.3,
        "strong_trend": 0.6,
        "weak_trend": 0.4
    },
    
    # ✅ NORMALIZATION SETTINGS
    normalization={
        "method": "zscore",
        "clip_z": 3.0,
        "robust_quantile_range": (0.25, 0.75)
    }
)

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "ema_periods": "Exponential Moving Average periods for multi-timeframe trend analysis",
    "rsi_period": "Relative Strength Index period for momentum measurement",
    "macd_fast": "MACD fast EMA period",
    "macd_slow": "MACD slow EMA period", 
    "macd_signal": "MACD signal line period",
    "bollinger_period": "Bollinger Bands moving average period",
    "bollinger_std": "Bollinger Bands standard deviation multiplier",
    "weights": "Component weights for final trend score calculation",
    "thresholds": "Score thresholds for trend classification"
}


# c_trend.py sonuna ekleyin
def get_trend_parameters(self) -> Dict[str, Any]:
    """Config parametrelerini dict olarak döndür"""
    return {
        "parameters": self.parameters,
        "weights": self.weights, 
        "thresholds": self.thresholds,
        "normalization": self.normalization
    }

# Config objesine metod ekle
TrendConfig.get_parameters = get_trend_parameters


# ✅ CONFIG VALIDATION ON IMPORT
if __name__ == "__main__":
    if TrendConfig.validate_config():
        print("✅ Trend config validation passed")
    else:
        print("❌ Trend config validation failed")
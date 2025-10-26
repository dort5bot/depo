# analysis/config/c_trend.py
"""
analysis/config/c_trend.py
Version: 2.0.0
Trend & Momentum Analysis Module Configuration - Analysis Helpers Uyumlu
deepsek
"""

from analysis.config.cm_base import TrendConfig, ModuleLifecycle, ParallelMode
from analysis.config.cm_base import AnalysisModuleConfig, ModuleLifecycle, ParallelMode
from analysis.analysis_helpers import AnalysisHelpers
from typing import Dict, Any

# ✅ ANALYSIS_HELPERS UYUMLU TREND CONFIG
CONFIG = AnalysisModuleConfig(
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
    version="2.0.0",
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
        
        # Polars optimization
        "polars_streaming": True,
        "polars_parallel": True,
        "default_interval": "1h"
    },
    
    # ✅ WEIGHTS FOR COMPOSITE SCORE
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
    
    # ✅ THRESHOLDS FOR SIGNAL CLASSIFICATION
    thresholds={
        "bullish": 0.7,
        "bearish": 0.3,
        "strong_trend": 0.6,
        "weak_trend": 0.4,
        "confidence_threshold": 0.6
    },
    
    # ✅ ANALYSIS_HELPERS INTEGRATION
    helpers_config={
        "cache_ttl": 60,
        "validation_strict": True,
        "fallback_enabled": True,
        "normalization_method": "tanh",
        "output_schema": "AnalysisOutput"
    },
    
    # ✅ PERFORMANCE OPTIMIZATIONS
    performance={
        "batch_size": 50,
        "max_concurrent": 10,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "use_thread_pool": True,
        "thread_pool_size": 4
    },
    
    # ✅ METADATA FOR MONITORING
    metadata={
        "category": "technical_analysis",
        "subcategory": "trend_momentum",
        "tags": ["trend", "momentum", "technical_indicators", "polars", "async"],
        "data_requirements": ["ohlcv"],
        "output_metrics": [
            "score", "signal", "confidence", "components", "explain"
        ],
        "compatibility": {
            "analysis_helpers": ">=1.0.0",
            "polars": ">=0.20.0",
            "async_support": True,
            "multi_user": True
        }
    }
)

# ✅ CONFIG HELPER FUNCTIONS
def get_trend_config() -> Dict[str, Any]:
    """Get trend module configuration as dictionary"""
    return CONFIG.dict()

def get_trend_weights() -> Dict[str, float]:
    """Get trend weights configuration"""
    return CONFIG.weights

def get_trend_thresholds() -> Dict[str, float]:
    """Get trend thresholds configuration"""
    return CONFIG.thresholds

def get_trend_parameters() -> Dict[str, Any]:
    """Get trend parameters configuration"""
    return CONFIG.parameters

def validate_trend_config() -> bool:
    """Validate trend configuration using AnalysisHelpers"""
    helpers = AnalysisHelpers()
    return helpers.validate_config_schema(CONFIG.dict())

# ✅ CONFIG INSTANCE FOR DIRECT IMPORT
trend_config = CONFIG
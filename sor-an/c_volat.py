# analysis/config/c_volat.py
"""
Volatility & Regime Module Configuration - Analysis Helpers Uyumlu
Version: 1.1.0
File: analysis/config/c_volat.py
"""

from analysis.config.cm_base import VolatilityConfig, ModuleLifecycle, ParallelMode
from analysis.analysis_helpers import AnalysisHelpers

# ✅ ANALYSIS_HELPERS UYUMLU VOLATILITY CONFIG
CONFIG = VolatilityConfig(
    # ✅ BASE MODULE CONFIG
    module_name="volat_regime",
    file="analysis/volat_regime.py",
    config="c_volat.py",
    command="/api/analysis/volatility",
    api_type="public",
    job_type="batch",
    parallel_mode=ParallelMode.BATCH,
    output_type="regime_score",
    objective="Volatility regime detection and analysis",
    maintainer="deepsek",
    description="Comprehensive volatility analysis using GARCH, Hurst, entropy and other advanced metrics",
    version="1.1.0",
    lifecycle=ModuleLifecycle.DEVELOPMENT,
    enabled=True,
    
    # ✅ VOLATILITY-SPECIFIC PARAMETERS
    garch_params={
        "p": 1,
        "q": 1, 
        "vol": "GARCH",
        "rescale": False
    },
    hurst_window=100,
    entropy_bins=50,
    
    # ✅ VOLATILITY WEIGHTS (ANALYSIS_HELPERS UYUMLU)
    weights={
        "historical_volatility": 0.15,
        "atr": 0.10,
        "bollinger_width": 0.10,
        "variance_ratio": 0.20,
        "hurst": 0.15,
        "entropy_struct": 0.05,
        "garch_implied_realized_diff": 0.15,
        "premium": 0.05,
        "rei": 0.05
    },
    
    # ✅ PARAMETERS DICT
    parameters={
        # Data windowing
        "ohlcv_limit": 500,
        "annualization": 365,
        "hv_scale": 0.8,
        
        # ATR / Bollinger
        "atr_period": 14,
        "atr_lookback": 50,
        "atr_scale": 0.01,
        "bb_window": 20,
        "bb_scale": 0.02,
        
        # Variance ratio
        "var_lag": 2,
        "var_sensitivity": 4.0,
        
        # Hurst
        "hurst_max_lag": 20,
        
        # Entropy
        "entropy_bins": 50,
        "entropy_scale": 3.5,
        
        # GARCH fallback
        "garch_proxy_span": 20,
        "realized_window": 20,
        "premium_scale": 0.05,
        "rei_lookback": 20,
        "rei_scale": 0.5,
        
        # Parallelism
        "max_workers": 4,
        
        # Scoring thresholds
        "trend_threshold": 0.6,
        "range_threshold": 0.55,
    }
)

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "ohlcv_limit": "Number of OHLCV data points to fetch for analysis",
    "annualization": "Annualization factor for historical volatility",
    "hv_scale": "Scale factor for normalizing historical volatility",
    "atr_period": "ATR calculation period",
    "bb_window": "Bollinger Bands window size",
    "var_lag": "Variance ratio test lag parameter",
    "hurst_max_lag": "Maximum lag for Hurst exponent calculation",
    "entropy_bins": "Number of bins for Shannon entropy calculation",
    "weights": "Component weights for final volatility score calculation"
}

# ✅ CONFIG VALIDATION ON IMPORT
if __name__ == "__main__":
    if CONFIG.validate_config():
        print("✅ Volatility config validation passed")
        # Weight validation
        from analysis.config.cm_base import validate_config_weights
        if validate_config_weights(CONFIG.weights, "volat_regime"):
            print("✅ Volatility weights validation passed")
        else:
            print("❌ Volatility weights validation failed")
    else:
        print("❌ Volatility config validation failed")
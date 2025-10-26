# analysis/config/c_risk.py
"""
Risk & Exposure Module Configuration - Analysis Helpers Uyumlu
Version: 1.1.0
File: analysis/config/c_risk.py
"""

from analysis.config.cm_base import BaseModuleConfig, ModuleLifecycle, ParallelMode
from analysis.analysis_helpers import AnalysisHelpers

# ✅ ANALYSIS_HELPERS UYUMLU RISK CONFIG
CONFIG = BaseModuleConfig(
    # ✅ BASE MODULE CONFIG
    module_name="risk_expos",
    file="analysis/risk_expos.py",
    config="c_risk.py",
    command="/api/analysis/risk",
    api_type="private",  # Risk modülü private data kullanır
    job_type="batch",
    parallel_mode=ParallelMode.BATCH,
    output_type="risk_score",
    objective="Risk exposure analysis and management",
    maintainer="deepsek",
    description="Comprehensive risk analysis including VaR, CVaR, leverage, drawdown and volatility targeting",
    version="1.1.0",
    lifecycle=ModuleLifecycle.DEVELOPMENT,
    enabled=True,
    
    # ✅ RISK-SPECIFIC PARAMETERS
    parameters={
        # OHLCV / ATR
        "ohlcv": {
            "interval": "1h",
            "lookback_bars": 500,
        },
        
        # ATR (Average True Range) adaptive stop multiplier
        "atr": {
            "period": 21,
            "multiplier": 3.0
        },
        
        # VaR / CVaR
        "var": {
            "window": 250,
            "confidence_levels": [0.95, 0.99]
        },
        
        # Volatility targeting
        "vol_target": {
            "target_volatility": 0.12,
            "lookback": 63
        },
        
        # Sharpe / Sortino dynamic params
        "performance": {
            "rolling_window": 63,
            "risk_free_rate": 0.0
        },
        
        # Leverage & maintenance assumptions if exact values unavailable
        "leverage": {
            "default_maintenance_margin": 0.005,
            "min_leverage": 1,
            "max_leverage": 125
        },
        
        # Execution / runtime
        "parallel": {
            "cpu_bound_pool_workers": 2
        }
    },
    
    # ✅ RISK WEIGHTS (ANALYSIS_HELPERS UYUMLU)
    weights={
        "var": 0.30,
        "cvar": 0.25,
        "leverage": 0.15,
        "vol_targeting": 0.10,
        "max_drawdown": 0.10,
        "atr_stop": 0.10
    },
    
    # ✅ THRESHOLDS
    thresholds={
        "high_risk": 0.75,
        "medium_risk": 0.45,
        "low_risk": 0.25
    }
)

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "ohlcv.lookback_bars": "Number of OHLCV bars to fetch for historical analysis",
    "atr.multiplier": "ATR multiplier for stop-loss calculation",
    "var.confidence_levels": "Confidence levels for Value at Risk calculation",
    "vol_target.target_volatility": "Annualized target volatility for position sizing",
    "leverage.max_leverage": "Maximum leverage for risk scoring normalization",
    "weights": "Component weights for final risk score calculation",
    "thresholds": "Risk level thresholds for signal classification"
}

# ✅ CONFIG VALIDATION ON IMPORT
if __name__ == "__main__":
    if CONFIG.validate_config():
        print("✅ Risk config validation passed")
        # Weight validation
        from analysis.config.cm_base import validate_config_weights
        if validate_config_weights(CONFIG.weights, "risk_expos"):
            print("✅ Risk weights validation passed")
        else:
            print("❌ Risk weights validation failed")
    else:
        print("❌ Risk config validation failed")
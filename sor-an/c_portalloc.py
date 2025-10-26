# analysis/config/c_portalloc.py
"""
c_portalloc.py
Version: 1.2.0
Portfolio Allocation Configuration Module - Analysis Helpers + Polars Uyumlu
File: analysis/config/c_portalloc.py
"""

from analysis.config.cm_base import BaseModuleConfig, ModuleLifecycle, ParallelMode
from analysis.analysis_helpers import AnalysisHelpers

# ✅ ANALYSIS_HELPERS + POLARS UYUMLU PORTFOLIO ALLOCATION CONFIG
CONFIG = BaseModuleConfig(
    # ✅ BASE MODULE CONFIG
    module_name="portfolio_allocation",
    file="analysis/port_alloc.py",
    config="c_portalloc.py",
    command="/api/analysis/portfolio",
    api_type="public",
    job_type="batch",
    parallel_mode=ParallelMode.BATCH,
    output_type="allocation_weights",
    objective="Portfolio optimization and asset allocation using advanced methods with Polars support",
    maintainer="deepsek",
    description="Black-Litterman, HRP, Risk Parity portfolio optimization with comprehensive metrics (Polars compatible)",
    version="1.2.0",  # Polars uyumlu versiyon
    lifecycle=ModuleLifecycle.DEVELOPMENT,
    enabled=True,
    
    # ✅ PORTFOLIO-SPECIFIC PARAMETERS
    parameters={
        "optimization_methods": {
            "black_litterman": {
                "enabled": True,
                "tau": 0.05,
                "risk_aversion": 2.5,
                "view_confidence": 0.75
            },
            "hierarchical_risk_parity": {
                "enabled": True,
                "linkage_method": "ward",
                "covariance_estimator": "ledoit_wolf"
            },
            "risk_parity": {
                "enabled": True,
                "max_iter": 1000,
                "tolerance": 1e-8
            }
        },
        
        "metrics": {
            "sharpe_ratio": {"risk_free_rate": 0.02},
            "var": {"confidence_level": 0.95, "time_horizon": 1},
            "sortino_ratio": {"target_return": 0.0},
            "max_drawdown": {"window": 252}
        },
        
        "constraints": {
            "max_allocation_per_asset": 0.3,
            "min_allocation_per_asset": 0.02,
            "total_leverage": 1.0,
            "target_return": 0.15
        },
        
        "data": {
            "lookback_period": 252,  # 1 year daily
            "min_data_points": 100,
            "correlation_threshold": 0.7,
            "dataframe_type": "polars"  # ✅ Polars desteği
        },
        
        "parallel_processing": {
            "enabled": True,
            "max_workers": 4,
            "batch_size": 10
        }
    },
    
    # ✅ PORTFOLIO WEIGHTS (ANALYSIS_HELPERS UYUMLU)
    weights={
        "sharpe": 0.30,
        "sortino": 0.25,
        "var": 0.20,
        "drawdown": 0.15,
        "volatility": 0.10
    },
    
    # ✅ THRESHOLDS
    thresholds={
        "optimal_allocation": 0.7,
        "moderate_allocation": 0.4,
        "suboptimal_allocation": 0.2
    }
)

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "black_litterman.tau": "Uncertainty scaling parameter for Black-Litterman model",
    "black_litterman.risk_aversion": "Risk aversion coefficient for implied returns",
    "hierarchical_risk_parity.linkage_method": "Linkage method for hierarchical clustering",
    "weights.sharpe": "Weight for Sharpe ratio in portfolio scoring",
    "weights.sortino": "Weight for Sortino ratio in portfolio scoring",
    "constraints.max_allocation_per_asset": "Maximum allocation per asset (diversification limit)",
    "data.dataframe_type": "Dataframe library type: 'polars' or 'pandas'"
}

# ✅ CONFIG VALIDATION ON IMPORT
if __name__ == "__main__":
    if CONFIG.validate_config():
        print("✅ Portfolio allocation config validation passed")
        # Weight validation
        from analysis.config.cm_base import validate_config_weights
        if validate_config_weights(CONFIG.weights, "portfolio_allocation"):
            print("✅ Portfolio weights validation passed")
        else:
            print("❌ Portfolio weights validation failed")
    else:
        print("❌ Portfolio allocation config validation failed")
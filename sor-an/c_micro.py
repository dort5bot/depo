"""
analysis/config/c_micro.py
Micro Alpha Factor Configuration
Pydantic model for type safety and validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

class MicroAlphaConfig(BaseModel):
    """Micro Alpha Configuration Model"""
    
    # Module metadata
    module_name: str = Field(default="micro_alpha")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Real-time microstructure alpha factor generator")
    
    # Data sources and endpoints
    data_sources: Dict[str, List[str]] = Field(
        default={
            "rest_endpoints": [
                "/fapi/v1/depth",
                "/fapi/v1/trades", 
                "/fapi/v1/ticker/price",
                "/fapi/v1/ticker/bookTicker",
                "/fapi/v1/aggTrades"
            ],
            "websocket_streams": [
                "<symbol>@trade",
                "<symbol>@depth@100ms", 
                "<symbol>@aggTrade",
                "<symbol>@depth",
                "<symbol>@kline_1m"
            ]
        }
    )
    
    # Core parameters
    parameters: Dict[str, float] = Field(
        default={
            "lookback_window": 1000,
            "update_frequency_ms": 100,
            "min_tick_volume": 10,
            "spread_threshold": 0.0001,
        }
    )
    
    # Metric calculation windows
    windows: Dict[str, int] = Field(
        default={
            "cvd_window": 50,
            "ofi_window": 20,
            "microprice_window": 10,
            "zscore_window": 100
        }
    )
    
    # Weights for final alpha score
    weights: Dict[str, float] = Field(
        default={
            "cvd": 0.25,
            "ofi": 0.25,
            "microprice_deviation": 0.20,
            "market_impact": 0.15,
            "latency_flow_ratio": 0.10,
            "hf_zscore": 0.05
        }
    )
    
    # Thresholds for signal generation
    thresholds: Dict[str, float] = Field(
        default={
            "bullish_threshold": 0.7,
            "bearish_threshold": 0.3,
            "cvd_extreme": 2.0,
            "ofi_extreme": 1.5
        }
    )
    
    # Kalman filter parameters for market impact
    kalman: Dict[str, float] = Field(
        default={
            "process_variance": 1e-6,
            "observation_variance": 1e-4,
            "initial_state": 0.0,
            "initial_covariance": 1.0
        }
    )
    
    # Performance settings
    performance: Dict[str, float] = Field(
        default={
            "cache_ttl": 1,
            "batch_size": 100,
            "max_retries": 3,
            "timeout_seconds": 5
        }
    )
    
    # Module metadata
    lifecycle: str = Field(default="development")
    parallel_mode: str = Field(default="async")
    job_type: str = Field(default="stream")
    output_type: str = Field(default="micro_alpha_score")

    class Config:
        extra = "forbid"  # Extra fields not allowed


# Default configuration instance
CONFIG = MicroAlphaConfig()

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "lookback_window": "Number of recent ticks to maintain in memory for calculations",
    "cvd_window": "Window for Cumulative Volume Delta calculation",
    "ofi_window": "Window for Order Flow Imbalance calculation", 
    "weights": "Component weights for final alpha score aggregation",
    "kalman": "Kalman filter parameters for market impact estimation"
}
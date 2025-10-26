"""
analysis/config/c_order.py
Configuration for Order Flow & Microstructure module (c_order.py)
Pydantic model for type safety and validation.
analysis/analysis_helpers.py ile uyumlu
"""

from pydantic import BaseModel, Field
from typing import Dict, Tuple, List, Optional

class OrderFlowConfig(BaseModel):
    """Order Flow Configuration Model"""
    
    # Sampling / data windows
    depth_levels: int = Field(default=12, ge=1, le=50, description="How many price levels to sum for orderbook imbalance")
    elasticity_levels: int = Field(default=12, ge=1, le=50, description="Levels used to compute depth elasticity")
    trades_limit: int = Field(default=500, ge=1, le=5000, description="How many recent trades to fetch")
    liquidity_window_bps: float = Field(default=10.0, ge=0.1, le=100.0, description="Window around mid to compute liquidity density (in bps)")

    # Normalization ranges (min, max) for metrics
    normalization: Dict[str, Tuple[float, float]] = Field(
        default={
            "orderbook_imbalance": (-1.0, 1.0),
            "spread_bps": (0.0, 80.0),
            "market_pressure": (-1.0, 1.0),
            "trade_aggression": (0.0, 1.5),
            "slippage": (0.0, 0.5),
            "depth_elasticity": (0.0, 50.0),
            "cvd": (-1e6, 1e6),
            "ofi": (-1e6, 1e6),
            "taker_dom_ratio": (0.0, 1.0),
            "liquidity_density": (0.0, 1e6)
        }
    )

    # Which metrics to invert after normalization (higher -> worse)
    invert_metrics: List[str] = Field(default=["spread_bps", "slippage"])

    # Weights (prioritized). Sum will be normalized automatically.
    weights: Dict[str, float] = Field(
        default={
            "orderbook_imbalance": 0.22,
            "spread_bps": 0.15,
            "market_buy_sell_pressure": 0.15,
            "trade_aggression_ratio": 0.08,
            "slippage": 0.08,
            "depth_elasticity": 0.10,
            "cvd": 0.07,
            "ofi": 0.06,
            "taker_dom_ratio": 0.05,
            "liquidity_density": 0.04
        }
    )

    # Thresholds for signal generation
    imbalance_signal_thresh: float = Field(default=0.12, ge=0.0, le=1.0, description="Raw imbalance threshold to consider directional signal")
    bullish_threshold: float = Field(default=0.7, ge=0.5, le=0.9)
    bearish_threshold: float = Field(default=0.3, ge=0.1, le=0.5)


    # Meta / other
    mid_price: float = Field(default=100.0, ge=0.0, description="Fallback mid price if book missing")
    version: str = Field(default="1.0.0")

    # Multi-user settings
    max_connections: int = Field(default=100, ge=1, le=10000, description="Max concurrent user connections")
    requests_per_minute: int = Field(default=60, ge=1, le=3600, description="Rate limit per user per minute")
    session_timeout: int = Field(default=300, ge=60, le=3600, description="User session timeout in seconds")
    multi_user_mode: bool = Field(default=True, description="Enable multi-user thread-safe mode")
    
    
    # Resource limits
    max_memory_mb: int = Field(default=512, ge=64, le=4096, description="Max memory usage per instance")
    cpu_time_limit: float = Field(default=30.0, ge=1.0, le=300.0, description="Max CPU time per request")
    


    class Config:
        extra = "forbid"  # Extra fields not allowed


# Config'de multi-user ayarı
CONFIG = OrderFlowConfig(
    multi_user_mode=True,  # ✅ Yeni config
    # ... diğer ayarlar
)
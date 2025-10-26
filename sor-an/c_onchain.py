# analysis/config/c_onchain.py
"""
On-Chain Analysis Configuration
Pydantic model for configuration validation and type safety
analysis/analysis_helpers.py ile uyumlu
"""

from pydantic import BaseModel, validator
from typing import Dict, Any

class OnChainConfig(BaseModel):
    """Pydantic configuration model for on-chain analysis"""
    
    version: str = "1.0.0"
    windows: Dict[str, int] = {"short_days": 7, "medium_days": 30, "long_days": 90}
    weights: Dict[str, float] = {
        "etf_net_flow": 0.15,
        "stablecoin_flow": 0.15,
        "exchange_netflow": 0.20, 
        "net_realized_pl": 0.15,
        "exchange_whale_ratio": 0.10,
        "mvrv_zscore": 0.10,
        "nupl": 0.05,
        "sopr": 0.10
    }
    thresholds: Dict[str, float] = {"bullish": 0.65, "bearish": 0.35}
    normalization: Dict[str, Any] = {"method": "zscore_clip", "clip_z": 3.0}
    data_timeout_seconds: int = 10
    explain_components_limit: int = 5
    parallel_mode: str = "async"
    prometheus: Dict[str, bool] = {"enable": False}
    
    @validator('weights')
    def validate_weights(cls, v):
        """Validate that weights sum to approximately 1.0"""
        total = sum(v.values())
        if not abs(total - 1.0) < 0.01:  # Allow 1% tolerance
            from warnings import warn
            warn(f"On-chain weights sum to {total:.3f}, normalizing to 1.0")
            # Normalize weights
            v = {k: weight/total for k, weight in v.items()}
        return v
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        """Validate threshold consistency"""
        bullish = v.get('bullish', 0.65)
        bearish = v.get('bearish', 0.35)
        if bullish <= bearish:
            raise ValueError(f"Bullish threshold ({bullish}) must be greater than bearish ({bearish})")
        return v

# Singleton instance for backward compatibility
CONFIG = OnChainConfig().dict()
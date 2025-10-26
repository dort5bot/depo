# analysis/config/c_deriv.py
"""
Derivatives & Sentiment Analysis Configuration
Futures pozisyon verilerine dayalı sentiment analizi için Pydantic config
analysis/analysis_helpers.py ile uyumlu
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

class DerivSentimentConfig(BaseModel):
    """Derivatives Sentiment Configuration Model"""
    
    # Module metadata
    module_name: str = Field(default="derivatives_sentiment")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Futures market positioning and sentiment analysis")
    
    # Ağırlıklar (toplam 1.0 olmalı)
    weights: Dict[str, float] = Field(
        default={
            "funding_rate": 0.15,
            "open_interest": 0.12,
            "long_short_ratio": 0.13,
            "oi_change_rate": 0.12,
            "funding_skew": 0.10,
            "volume_imbalance": 0.10,
            "liquidation_heat": 0.15,
            "oi_delta_divergence": 0.08,
            "volatility_skew": 0.05
        }
    )
    
    # Threshold değerleri
    thresholds: Dict[str, float] = Field(
        default={
            "bullish": 0.6,
            "bearish": 0.4,
            "extreme_bull": 0.8,
            "extreme_bear": 0.2,
            "neutral_upper": 0.55,
            "neutral_lower": 0.45
        }
    )
    
    # Parametreler
    parameters: Dict[str, int] = Field(
        default={
            "oi_lookback": 24,
            "funding_lookback": 8,
            "liquidation_window": 12,
            "volatility_period": 20,
            "min_data_points": 5,
            "cache_ttl": 60
        }
    )
    
    # Normalizasyon ayarları
    normalization: Dict[str, Any] = Field(
        default={
            "method": "tanh",
            "scale_factor": 3,
            "rolling_window": 100,
            "use_robust": True
        }
    )
    
    # API ve Data ayarları
    data_sources: Dict[str, str] = Field(
        default={
            "funding_rate": "/fapi/v1/fundingRate",
            "open_interest": "/fapi/v1/openInterestHist",
            "long_short_ratio": "/fapi/v1/longShortRatio",
            "liquidation_orders": "/fapi/v1/liquidationOrders",
            "taker_ratio": "/fapi/v1/takerlongshortRatio",
            "timeframe": "5m",
            "limit": 100
        }
    )
    
    # Risk ve Sınırlamalar
    limits: Dict[str, int] = Field(
        default={
            "max_symbols_batch": 10,
            "request_timeout": 30,
            "rate_limit_delay": 0,
            "circuit_breaker_failures": 5
        }
    )
    
    # Metrik özellikleri
    metrics_metadata: Dict[str, Dict[str, str]] = Field(
        default={
            "funding_rate": {
                "description": "Funding rate sentiment (-1 to 1)",
                "formula": "tanh((current_funding - avg_funding) * 1000)",
                "interpretation": "Positive = perpetual premium, Negative = discount"
            },
            "open_interest": {
                "description": "Open Interest change sentiment",
                "formula": "tanh(oi_change * 10)",
                "interpretation": "Positive = new positions, Negative = position closing"
            },
            "liquidation_heat": {
                "description": "Liquidation intensity metric",
                "formula": "tanh(total_liquidation / 1e6)",
                "interpretation": "High values indicate market stress"
            }
        }
    )
    
    # Modül yaşam döngüsü
    lifecycle: Dict[str, str] = Field(
        default={
            "stage": "development",
            "stability": "beta",
            "deprecation_date": None
        }
    )
    
    # Paralel işlem ayarları
    parallel_config: Dict[str, Any] = Field(
        default={
            "mode": "async",
            "max_concurrent": 5,
            "batch_size": 3,
            "job_type": "io_bound"
        }
    )

    # Validators
    @validator('weights')
    def validate_weights_sum(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f'Weights must sum to 1.0, got {total:.3f}')
        return v

    @validator('thresholds')
    def validate_threshold_ordering(cls, v):
        if not (v['extreme_bear'] < v['bearish'] < v['bullish'] < v['extreme_bull']):
            raise ValueError('Thresholds must be ordered: extreme_bear < bearish < bullish < extreme_bull')
        return v

    class Config:
        extra = "forbid"  # Extra fields not allowed


# Default configuration instance
CONFIG = DerivSentimentConfig()

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "weights": "Her metrik için ağırlık değerleri (toplam 1.0 olmalı)",
    "thresholds": "Sentiment sinyalleri için eşik değerleri",
    "oi_lookback": "Open Interest analizi için lookback periyodu (saat)",
    "funding_lookback": "Funding rate analizi için lookback",
    "liquidation_window": "Likidasyon analizi için pencere boyutu",
    "normalization.method": "Metrik normalizasyon yöntemi (tanh/zscore/minmax)"
}

# Validasyon kuralları
VALIDATION_RULES = {
    "weights_sum": {"rule": "sum == 1.0", "tolerance": 0.01},
    "threshold_ordering": {"rule": "extreme_bear < bearish < bullish < extreme_bull"},
    "positive_parameters": {"params": ["oi_lookback", "funding_lookback", "cache_ttl"]}
}
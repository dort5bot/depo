# analysis/config/c_deriv.py
"""
analysis/config/c_deriv.py
Version: 2.0.0 - Analysis Helpers Tam Uyumlu

Derivatives & Sentiment Analysis Configuration
Futures pozisyon verilerine dayalı sentiment analizi için Pydantic config
Tam async, multi-user yapı için optimize edilmiş
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

class DerivSentimentConfig(BaseModel):
    """Derivatives Sentiment Configuration Model"""
    
    # Module metadata
    module_name: str = Field(default="derivatives_sentiment")
    version: str = Field(default="2.0.0")
    description: str = Field(default="Futures market positioning and sentiment analysis - Async Multi-User")
    
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
    
    # Parametreler - Async için optimize
    parameters: Dict[str, Any] = Field(
        default={
            "oi_lookback": 24,
            "funding_lookback": 8,
            "liquidation_window": 12,
            "volatility_period": 20,
            "min_data_points": 5,
            "cache_ttl": 60,
            "async_timeout": 30,
            "max_concurrent_requests": 10,
            "batch_size": 5
        }
    )
    
    # Normalizasyon ayarları
    normalization: Dict[str, Any] = Field(
        default={
            "method": "tanh",
            "scale_factor": 3.0,
            "rolling_window": 100,
            "use_robust": True,
            "output_range_min": 0.0,
            "output_range_max": 1.0
        }
    )
    
    # API ve Data ayarları - Async için
    data_sources: Dict[str, Any] = Field(
        default={
            "funding_rate": "/fapi/v1/fundingRate",
            "open_interest": "/fapi/v1/openInterestHist",
            "long_short_ratio": "/fapi/v1/longShortRatio",
            "liquidation_orders": "/fapi/v1/liquidationOrders",
            "taker_ratio": "/fapi/v1/takerlongshortRatio",
            "timeframe": "5m",
            "limit": 100,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
    )
    
    # Risk ve Sınırlamalar - Multi-User için
    limits: Dict[str, int] = Field(
        default={
            "max_symbols_batch": 10,
            "request_timeout": 30,
            "rate_limit_delay": 0.1,
            "circuit_breaker_failures": 5,
            "max_users": 100,
            "user_request_limit": 1000,
            "memory_limit_mb": 512
        }
    )
    
    # Metrik özellikleri
    metrics_metadata: Dict[str, Dict[str, str]] = Field(
        default={
            "funding_rate": {
                "description": "Funding rate sentiment (-1 to 1)",
                "formula": "tanh((current_funding - avg_funding) * 1000)",
                "interpretation": "Positive = perpetual premium, Negative = discount",
                "async_safe": True
            },
            "open_interest": {
                "description": "Open Interest change sentiment",
                "formula": "tanh(oi_change * 10)",
                "interpretation": "Positive = new positions, Negative = position closing",
                "async_safe": True
            },
            "liquidation_heat": {
                "description": "Liquidation intensity metric",
                "formula": "tanh(total_liquidation / 1e6)",
                "interpretation": "High values indicate market stress",
                "async_safe": True
            }
        }
    )
    
    # Modül yaşam döngüsü
    lifecycle: Dict[str, str] = Field(
        default={
            "stage": "production",
            "stability": "stable",
            "async_compatible": "true",
            "multi_user_safe": "true",
            "deprecation_date": None
        }
    )
    
    # Paralel işlem ayarları - Async için optimize
    parallel_config: Dict[str, Any] = Field(
        default={
            "mode": "async",
            "max_concurrent": 5,
            "batch_size": 3,
            "job_type": "io_bound",
            "use_thread_pool": False,
            "prefetch_data": True,
            "connection_pool_size": 10
        }
    )

    # Performance monitoring
    performance: Dict[str, Any] = Field(
        default={
            "enable_metrics": True,
            "metrics_retention": 3600,
            "slow_query_threshold": 5.0,
            "log_level": "INFO"
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

    @validator('parameters')
    def validate_positive_parameters(cls, v):
        positive_params = ['oi_lookback', 'funding_lookback', 'cache_ttl', 'async_timeout']
        for param in positive_params:
            if param in v and v[param] <= 0:
                raise ValueError(f'{param} must be positive')
        return v

    class Config:
        extra = "forbid"  # Extra fields not allowed
        validate_assignment = True


# Default configuration instance
CONFIG = DerivSentimentConfig()

# Parameter descriptions for documentation
PARAM_DESCRIPTIONS = {
    "weights": "Her metrik için ağırlık değerleri (toplam 1.0 olmalı)",
    "thresholds": "Sentiment sinyalleri için eşik değerleri",
    "oi_lookback": "Open Interest analizi için lookback periyodu (saat)",
    "funding_lookback": "Funding rate analizi için lookback",
    "liquidation_window": "Likidasyon analizi için pencere boyutu",
    "normalization.method": "Metrik normalizasyon yöntemi (tanh/zscore/minmax)",
    "async_timeout": "Async işlemler için timeout süresi (saniye)",
    "max_concurrent_requests": "Maksimum eşzamanlı istek sayısı"
}

# Validasyon kuralları
VALIDATION_RULES = {
    "weights_sum": {"rule": "sum == 1.0", "tolerance": 0.01},
    "threshold_ordering": {"rule": "extreme_bear < bearish < bullish < extreme_bull"},
    "positive_parameters": {"params": ["oi_lookback", "funding_lookback", "cache_ttl", "async_timeout"]},
    "async_safe": {"rule": "All metrics must be async safe"}
}
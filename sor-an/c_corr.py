# analysis/config/c_corr.py
"""
Korelasyon & Lead-Lag Analiz Modülü Config - Polars Uyumlu

Pydantic model for type safety and validation.
analysis/analysis_helpers.py ile uyumlu
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any

class CorrelationConfig(BaseModel):
    """Correlation and Lead-Lag Analysis Configuration Model - Polars Compatible"""
    
    # Module metadata
    module_name: str = Field(default="correlation_lead_lag")
    version: str = Field(default="2.1.0")
    description: str = Field(default="Coin'ler arası korelasyon ve liderlik analizi - Polars Optimized")
    
    # Hesaplama parametreleri
    calculation: Dict[str, Any] = Field(
        default={
            "default_interval": "1h",
            "default_limit": 200,
            "max_lag_period": 10,
            "rolling_window": 20,
            "granger_max_lags": 5,
            "var_max_lags": 5,
            "fast_corr_threshold": 0.3,
            "max_pairs": 40,
            "min_samples": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
            "max_related_symbols": 10,
            "polars_optimization": True,
            "chunk_size": 1000
        }
    )

    # Metrik ağırlıkları
    weights: Dict[str, float] = Field(
        default={
            "pearson_corr": 0.15,
            "beta": 0.15,
            "rolling_cov": 0.10,
            "partial_corr": 0.10,
            "lead_lag_delta": 0.20,
            "granger_causality": 0.15,
            "dtw_distance": 0.10,
            "var_impulse": 0.05
        }
    )
    
    # Threshold değerleri
    thresholds: Dict[str, float] = Field(
        default={
            "high_correlation": 0.7,
            "medium_correlation": 0.3,
            "significant_lead": 0.1,
            "strong_causality": 0.05,
            "high_connectivity": 0.6,
            "medium_connectivity": 0.4,
            "bullish_threshold": 0.7,
            "bearish_threshold": 0.3
        }
    )
    
    # Paralel işleme
    parallel_processing: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "max_workers": 10,
            "batch_size": 5,
            "use_polars_parallel": True
        }
    )
    
    # API config
    api_config: Dict[str, Any] = Field(
        default={
            "timeout": 30,
            "retry_attempts": 3,
            "rate_limit_delay": 0.1,
            "max_concurrent_requests": 50
        }
    )
    
    # Cache config - Multi-user için
    cache: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "ttl": 300,
            "max_size": 1000,
            "user_specific": True,
            "partition_by_user": True
        }
    )
    
    # Multi-user config
    multi_user: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "user_session_timeout": 3600,
            "max_users": 1000,
            "data_isolation": True
        }
    )
    
    # Lifecycle
    lifecycle: Dict[str, str] = Field(
        default={
            "stage": "production",
            "stability": "stable",
            "deprecation_date": None
        }
    )

    # Validators
    @validator('weights')
    def validate_weights_sum(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f'Weights must sum to 1.0, got {total:.3f}')
        return v

    @validator('calculation')
    def validate_calculation_params(cls, v):
        allowed_intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if v.get("default_interval") not in allowed_intervals:
            raise ValueError(f"Interval must be one of {allowed_intervals}")
        
        if v.get("default_limit") < 20 or v.get("default_limit") > 1000:
            raise ValueError("Limit must be between 20 and 1000")
        
        return v

    class Config:
        extra = "forbid"  # Extra fields not allowed


# Default configuration instance
CONFIG = CorrelationConfig()

# Parametre açıklamaları
PARAM_DESCRIPTIONS = {
    "default_interval": "Varsayılan zaman aralığı",
    "default_limit": "Varsayılan veri noktası sayısı", 
    "max_lag_period": "Maksimum lead-lag periyodu",
    "rolling_window": "Rolling hesaplamalar için pencere boyutu",
    "granger_max_lags": "Granger testi için maksimum lag",
    "var_max_lags": "VAR modeli için maksimum lag",
    "polars_optimization": "Polars optimizasyonunu aktif et"
}

# Parametre validasyon şeması
PARAM_SCHEMA = {
    "default_interval": {
        "type": "string", 
        "allowed": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "default": "1h"
    },
    "default_limit": {
        "type": "integer",
        "min": 20,
        "max": 1000,
        "default": 100
    },
    "max_lag_period": {
        "type": "integer", 
        "min": 5,
        "max": 50,
        "default": 10
    },
    "rolling_window": {
        "type": "integer",
        "min": 10, 
        "max": 200,
        "default": 20
    }
}
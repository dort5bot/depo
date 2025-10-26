# analysis/config/cm_base.py
"""
Base configuration classes - Analysis Helpers Uyumlu
"""

import logging
from abc import ABC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator, Field
from enum import Enum

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers

logger = logging.getLogger(__name__)

class ModuleLifecycle(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

class ParallelMode(str, Enum):
    BATCH = "batch"
    ASYNC = "async"
    STREAM = "stream"
    EVENT = "event"

class BaseModuleConfig(BaseModel):
    """Tüm modül config'leri için base class - Analysis Helpers Uyumlu"""
    
    module_name: str = Field(..., description="Modül adı")
    file: str = Field(..., description="Modül dosya yolu")
    config: str = Field(..., description="Config dosya adı")
    command: str = Field(..., description="API endpoint command")
    api_type: str = Field(..., description="API tipi (public/private)")
    job_type: str = Field(..., description="İş tipi (batch/stream)")
    parallel_mode: str = Field(..., description="Paralel işlem modu")
    output_type: str = Field(..., description="Çıktı tipi")
    objective: str = Field(..., description="Modül amacı")
    maintainer: str = Field(..., description="Bakımcı")
    description: Optional[str] = Field(None, description="Açıklama")
    version: str = Field(..., description="Versiyon")
    lifecycle: ModuleLifecycle = Field(ModuleLifecycle.DEVELOPMENT, description="Yaşam döngüsü")
    enabled: bool = Field(True, description="Aktif/pasif")
    
    # ✅ ANALYSIS_HELPERS UYUMLU METADATA
    created_at: float = Field(default_factory=AnalysisHelpers.get_timestamp, description="Oluşturulma zamanı")
    updated_at: float = Field(default_factory=AnalysisHelpers.get_timestamp, description="Güncellenme zamanı")

    class Config:
        extra = "forbid"
        validate_assignment = True

    # ✅ ANALYSIS_HELPERS UYUMLU VALIDATION METOTLARI
    
    @validator('module_name')
    def validate_module_name(cls, v):
        """Modül adını validate et"""
        if not v or not v.strip():
            raise ValueError('Module name cannot be empty')
        if not v.replace('_', '').isalnum():
            raise ValueError('Module name can only contain alphanumeric characters and underscores')
        return v.lower()

    @validator('version')
    def validate_version(cls, v):
        """Versiyon formatını validate et"""
        if not v or not isinstance(v, str):
            raise ValueError('Version must be a non-empty string')
        # Semantik versioning kontrolü (opsiyonel)
        parts = v.split('.')
        if len(parts) not in [2, 3]:
            logger.warning(f"Version {v} may not follow semantic versioning")
        return v

    @validator('weights', check_fields=False)
    def validate_weights_dict(cls, v):
        """Weights dict validate et - ANALYSIS_HELPERS UYUMLU"""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError('Weights must be a dictionary')
        
        # ✅ ANALYSIS_HELPERS İLE NORMALIZE ET
        normalized_weights = AnalysisHelpers.normalize_weights(v)
        
        # Log weight değişiklikleri
        if v != normalized_weights:
            logger.info(f"Weights normalized: {v} -> {normalized_weights}")
        
        return normalized_weights

    def update_timestamp(self):
        """Güncelleme zamanını güncelle - ANALYSIS_HELPERS UYUMLU"""
        self.updated_at = AnalysisHelpers.get_timestamp()

    def validate_config(self) -> bool:
        """Config'i validate et - ANALYSIS_HELPERS UYUMLU"""
        try:
            # Temel validasyonlar
            if not self.module_name:
                logger.error("Module name is required")
                return False
            
            if not self.version:
                logger.error("Version is required")
                return False
            
            # Weights validasyonu (eğer varsa)
            if hasattr(self, 'weights') and self.weights:
                total_weight = sum(self.weights.values())
                if not 0.99 <= total_weight <= 1.01:
                    logger.warning(f"Weights sum {total_weight} is not approximately 1.0")
            
            logger.debug(f"Config validation passed for {self.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed for {self.module_name}: {e}")
            return False

    def get_config_info(self) -> Dict[str, Any]:
        """Config bilgilerini getir - ANALYSIS_HELPERS UYUMLU"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "lifecycle": self.lifecycle,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "age_seconds": AnalysisHelpers.get_timestamp() - self.created_at,
            "description": self.description
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Config'i düz dict'e çevir (serialization için)"""
        result = self.dict()
        
        # Özel alanları işle
        if 'weights' in result and result['weights']:
            result['weights_total'] = sum(result['weights'].values())
        
        return result


class TrendConfig(BaseModuleConfig):
    """Trend modülü için specific config - Analysis Helpers Uyumlu"""
    
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Trend parametreleri")
    weights: Dict[str, float] = Field(default_factory=dict, description="Metrik ağırlıkları")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Eşik değerleri")
    normalization: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "zscore", "clip_z": 3.0},
        description="Normalizasyon ayarları"
    )
    
    # ✅ ANALYSIS_HELPERS UYUMLU VALIDATION
    
    @validator('weights')
    def validate_weights(cls, v):
        """Weights validate et - ANALYSIS_HELPERS UYUMLU"""
        if not v:
            return v
            
        # ✅ ANALYSIS_HELPERS İLE NORMALIZE ET
        normalized = AnalysisHelpers.normalize_weights(v)
        
        # Ağırlık dağılımını kontrol et
        max_weight = max(normalized.values()) if normalized else 0
        if max_weight > 0.5:
            logger.warning(f"Maximum weight {max_weight:.3f} is high, consider rebalancing")
        
        return normalized
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        """Thresholds validate et"""
        if not v:
            return v
            
        # Threshold sıralamasını kontrol et
        sorted_values = sorted(v.values())
        if sorted_values != list(v.values()):
            logger.warning("Threshold values should be in ascending order")
        
        return v
    
    @validator('normalization')
    def validate_normalization(cls, v):
        """Normalizasyon ayarlarını validate et"""
        allowed_methods = ['zscore', 'minmax', 'tanh', 'robust']
        method = v.get('method', 'zscore')
        
        if method not in allowed_methods:
            logger.warning(f"Normalization method {method} not in allowed methods: {allowed_methods}")
            v['method'] = 'zscore'  # Fallback
        
        return v

    def get_trend_parameters(self) -> Dict[str, Any]:
        """Trend parametrelerini getir"""
        return {
            "weights": self.weights,
            "thresholds": self.thresholds,
            "normalization": self.normalization,
            "parameters": self.parameters,
            "config_info": self.get_config_info()
        }


class VolatilityConfig(BaseModuleConfig):
    """Volatility modülü için specific config - Analysis Helpers Uyumlu"""
    
    garch_params: Dict[str, Any] = Field(
        default_factory=lambda: {"p": 1, "q": 1, "vol": "GARCH"},
        description="GARCH model parametreleri"
    )
    hurst_window: int = Field(default=100, ge=10, le=1000, description="Hurst window boyutu")
    entropy_bins: int = Field(default=50, ge=10, le=200, description="Entropy bin sayısı")
    weights: Dict[str, float] = Field(default_factory=dict, description="Volatility metrik ağırlıkları")
    
    # ✅ ANALYSIS_HELPERS UYUMLU VALIDATION
    
    @validator('hurst_window')
    def validate_hurst_window(cls, v):
        """Hurst window validate et"""
        if v < 20:
            logger.warning(f"Hurst window {v} may be too small for reliable calculation")
        return v
    
    @validator('entropy_bins')
    def validate_entropy_bins(cls, v):
        """Entropy bins validate et"""
        if v < 20:
            logger.warning(f"Entropy bins {v} may be too few for reliable calculation")
        if v > 100:
            logger.info(f"Entropy bins {v} may be computationally expensive")
        return v
    
    @validator('garch_params')
    def validate_garch_params(cls, v):
        """GARCH parametrelerini validate et"""
        if not v:
            return {"p": 1, "q": 1}  # Default values
        
        p = v.get('p', 1)
        q = v.get('q', 1)
        
        if p + q > 3:
            logger.warning(f"GARCH order p+q={p+q} may be too high for financial time series")
        
        return v

    def get_volatility_parameters(self) -> Dict[str, Any]:
        """Volatility parametrelerini getir"""
        return {
            "garch_params": self.garch_params,
            "hurst_window": self.hurst_window,
            "entropy_bins": self.entropy_bins,
            "weights": self.weights,
            "config_info": self.get_config_info()
        }


# ✅ ANALYSIS_HELPERS UYUMLU CONFIG LOADER FONKSİYONU

def load_config(module_name: str) -> BaseModuleConfig:
    """
    Config yükle - Analysis Helpers Uyumlu
    """
    import importlib
    
    try:
        config_module_name = f"analysis.config.c_{module_name}"
        module = importlib.import_module(config_module_name)
        
        if not hasattr(module, "CONFIG"):
            raise AttributeError(f"CONFIG not found in {config_module_name}")
        
        config_obj = module.CONFIG
        
        if not isinstance(config_obj, BaseModuleConfig):
            raise TypeError(f"CONFIG in {config_module_name} is not a BaseModuleConfig instance")
        
        # ✅ CONFIG VALIDATION
        if not config_obj.validate_config():
            logger.warning(f"Config validation warnings for {module_name}")
        
        logger.info(f"Config loaded successfully: {module_name} v{config_obj.version}")
        return config_obj
        
    except ImportError as e:
        logger.error(f"Config module import failed for {module_name}: {e}")
        raise
    except (AttributeError, TypeError) as e:
        logger.error(f"Config object invalid for {module_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config for {module_name}: {e}")
        raise


def create_fallback_config(module_name: str) -> BaseModuleConfig:
    """
    Fallback config oluştur - Analysis Helpers Uyumlu
    """
    logger.warning(f"Creating fallback config for {module_name}")
    
    return BaseModuleConfig(
        module_name=module_name,
        file=f"analysis_{module_name}.py",
        config=f"c_{module_name}.py", 
        command=f"/api/analysis/{module_name}",
        api_type="public",
        job_type="batch",
        parallel_mode="async",
        output_type="score",
        objective=f"Fallback analysis for {module_name}",
        maintainer="system",
        description=f"Fallback configuration for {module_name}",
        version="1.0.0-fallback",
        lifecycle=ModuleLifecycle.DEVELOPMENT,
        enabled=True
    )


# ✅ CONFIG VALIDATION UTILITIES

def validate_config_weights(weights: Dict[str, float], config_name: str) -> bool:
    """
    Config weights validate et - Analysis Helpers Uyumlu
    """
    if not weights:
        logger.warning(f"No weights defined in {config_name}")
        return True
    
    try:
        normalized = AnalysisHelpers.normalize_weights(weights)
        total = sum(normalized.values())
        
        if abs(total - 1.0) > 0.01:
            logger.error(f"Weights sum {total:.3f} != 1.0 in {config_name}")
            return False
        
        # Ağırlık dağılımı kontrolü
        max_weight = max(normalized.values())
        if max_weight > 0.8:
            logger.warning(f"Dominant weight {max_weight:.3f} in {config_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Weight validation failed for {config_name}: {e}")
        return False


def get_config_statistics(configs: List[BaseModuleConfig]) -> Dict[str, Any]:
    """
    Config istatistiklerini getir - Analysis Helpers Uyumlu
    """
    if not configs:
        return {}
    
    lifecycles = {}
    versions = {}
    
    for config in configs:
        lifecycles[config.lifecycle] = lifecycles.get(config.lifecycle, 0) + 1
        versions[config.version] = versions.get(config.version, 0) + 1
    
    return {
        "total_configs": len(configs),
        "lifecycle_distribution": lifecycles,
        "version_distribution": versions,
        "enabled_count": sum(1 for c in configs if c.enabled),
        "timestamp": AnalysisHelpers.get_timestamp()
    }
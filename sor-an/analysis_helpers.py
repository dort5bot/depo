# analysis/analysis_helpers.py
"""
analysis_helpers.py
Version: 2.0.0 - Tam Async
Merkezi Analysis Helper Sınıfı
Tüm analiz modülleri için ortak fonksiyonlar

"""
# analysis/analysis_helpers.py

import os
import time
import json
import hashlib
import logging
import asyncio

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ValidationError, validator
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# ✅ OUTPUT SCHEMA FOR VALIDATION
class AnalysisOutput(BaseModel):
    """Standardized analysis output schema"""
    score: float
    signal: str
    confidence: float
    components: Dict[str, float]
    explain: str
    timestamp: int
    module: str
    metadata: Dict[str, Any]
    
    @validator('score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v
    
    @validator('signal')
    def validate_signal(cls, v):
        valid_signals = ['bullish', 'bearish', 'neutral']
        if v not in valid_signals:
            raise ValueError(f'Signal must be one of {valid_signals}')
        return v
    
    class Config:
        extra = "forbid"


# ✅ ANALYSIS UTILITIES CLASS
class AnalysisUtilities:
    """
    Saf utility fonksiyonları - stateless
    Matematiksel, validation, normalization işlemleri
    """
    
    @staticmethod
    def validate_output(output: Dict[str, Any]) -> bool:
        """Output formatını validate et"""
        required_keys = {'score', 'signal', 'components', 'timestamp', 'module'}
        return all(key in output for key in required_keys)
    
    @staticmethod
    def normalize_score(score: float, method: str = 'tanh') -> float:
        """Normalize score to 0-1 range"""
        if method == 'tanh':
            return (np.tanh(score) + 1) / 2
        elif method == 'sigmoid':
            return 1 / (1 + np.exp(-score))
        elif method == 'minmax':
            return max(0.0, min(1.0, score))
        else:
            return max(0.0, min(1.0, score))
 
    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Ağırlıkları normalize et (toplam 1.0 yap)"""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0/len(weights) for k in weights}
        return {k: v/total for k, v in weights.items()}
    

    @staticmethod
    def calculate_weighted_average(
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Ağırlıklı ortalama hesapla (güvenli ve tutarlı)"""
        
        # Validation
        if not scores or not weights:
            return 0.5

        total_score = 0.0
        total_weight = 0.0

        for key, score in scores.items():
            weight = weights.get(key)
            if weight is not None:
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        raw_score = total_score / total_weight
        return AnalysisUtilities.normalize_score(raw_score)


        
    @staticmethod
    def create_fallback_output(module_name: str, reason: str = "Error") -> Dict[str, Any]:
        """Standart fallback output oluştur"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": f"Fallback: {reason}",
            "timestamp": time.time(),
            "module": module_name,
            "fallback": True
        }
    

    @staticmethod
    def validate_score_dict(scores: Dict[str, float]) -> bool:
        """Score dict formatını ve değer aralığını kontrol et"""
        
        # Tip kontrolü
        if not isinstance(scores, dict):
            return False
        
        # Boş dict kontrolü
        if not scores:
            return False
        
        # İçerik ve değer aralığı kontrolü
        for key, value in scores.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                return False
            if not 0 <= value <= 1:
                return False

        return True

    @staticmethod
    def exponential_smoothing(values: List[float], alpha: float = 0.3) -> float:
        """Üstel düzeltme ile smoothing"""
        if not values:
            return 0.0
        
        result = values[0]
        for value in values[1:]:
            result = alpha * value + (1 - alpha) * result
        return result
    
    @staticmethod
    def calculate_confidence(scores: List[float], min_samples: int = 3) -> float:
        """Skorların tutarlılığına göre confidence hesapla"""
        if len(scores) < min_samples:
            return len(scores) / min_samples * 0.5
        
        if len(scores) == 1:
            return 0.3
            
        variance = np.var(scores) if len(scores) > 1 else 0.0
        consistency = 1.0 - min(variance, 1.0)
        return consistency * 0.8 + 0.2
 
    @staticmethod
    def detect_outliers(data: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        if len(data) == 0:
            return np.array([])
        
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
        return z_scores > threshold
 
    #VeriModelTakviye:
    @staticmethod
    def dataframe_to_scores(
        df: Any,
        score_columns: List[str],
        dataframe_type: str = 'auto'
    ) -> Dict[str, float]:
        """
        Farklı dataframe türlerinden (pandas, polars, numpy) skor sözlüğüne dönüştürür.

        Args:
            df: pandas.DataFrame, polars.DataFrame veya numpy.ndarray
            score_columns: Kullanılacak skor kolon isimleri
            dataframe_type: 'pandas', 'polars', 'numpy', 'auto' (otomatik algılama)
        Returns:
            Dict[str, float]: Her skor kolonu için son satır değerlerinden oluşan sözlük
        """
        scores = {}

        # ✅ 1. DataFrame türünü tespit et
        if dataframe_type == 'auto':
            if hasattr(df, 'iloc') and hasattr(df, 'columns'):  # pandas
                dataframe_type = 'pandas'
            elif hasattr(df, 'select') and hasattr(df, 'schema'):  # polars
                dataframe_type = 'polars'
            elif isinstance(df, np.ndarray):
                dataframe_type = 'numpy'
            else:
                raise ValueError(f"Unsupported dataframe type: {type(df)}")

        try:
            # ✅ 2. Pandas tipi işleme
            if dataframe_type == 'pandas':
                for col in score_columns:
                    if col in df.columns:
                        scores[col] = float(df[col].iloc[-1])  # Son değeri al
                    else:
                        logger.warning(f"Pandas kolon bulunamadı: {col}")
                        scores[col] = 0.5  # Fallback

            # ✅ 3. Polars tipi işleme
            elif dataframe_type == 'polars':
                for col in score_columns:
                    if col in df.columns:
                        scores[col] = float(df[col][-1])  # Son değeri al
                    else:
                        logger.warning(f"Polars kolon bulunamadı: {col}")
                        scores[col] = 0.5

            # ✅ 4. Numpy tipi işleme
            elif dataframe_type == 'numpy':
                if len(score_columns) != df.shape[1]:
                    raise ValueError(
                        f"Column count mismatch: {len(score_columns)} names, {df.shape[1]} columns"
                    )
                for i, col in enumerate(score_columns):
                    scores[col] = float(df[-1, i])  # Son satırdan değer al

            return scores

        except (IndexError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Dataframe conversion error: {e}")
            
            # ✅ Tüm kolonlara default skor ver (örneğin 0.5)
            return {col: 0.5 for col in score_columns}


# ✅ MEVCUT AnalysisHelpers Class'ı (Güncel - TAM UYUMLU)
class AnalysisHelpers:
    """Stateful helper sınıfı - performance tracking, config loading vb."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize analyzer with optional config and caching"""
        self.config = config or {}
        
        # Performans ölçümleri
        self.performance_metrics: Dict[str, List] = {}
        
        # Utility fonksiyonlarına erişim
        self.utils = AnalysisUtilities()
        
        # Basit cache yapısı
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: int = self.config.get("cache_ttl", 300)  # Varsayılan: 5 dakika

    # ✅ TIME & DATE UTILITIES
    @staticmethod
    def get_timestamp() -> int:
        """Get current UTC timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    @staticmethod
    def get_iso_timestamp() -> str:
        """Get current UTC ISO-8601 timestamp"""
        # timezone import edilmeli veya:
        return datetime.utcnow().isoformat() + 'Z'
        # Veya mevcut implementation için timezone import
        
    @staticmethod
    def timestamp_to_datetime(timestamp: int) -> datetime:
        """Convert milliseconds timestamp to UTC datetime"""
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        
    
    # ✅ DATA VALIDATION & CONVERSION
    def validate_ohlcv_data(self, data: Union[pd.DataFrame, pl.DataFrame]) -> bool:
        """Validate OHLCV data structure"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if isinstance(data, pd.DataFrame):
            return all(col in data.columns for col in required_columns)
        elif isinstance(data, pl.DataFrame):
            return all(col in data.columns for col in required_columns)
        else:
            return False
    
    def convert_to_polars(self, data: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """Convert data to Polars DataFrame"""
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def convert_to_pandas(self, data: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """Convert data to Pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pl.DataFrame):
            return data.to_pandas()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")     
     
     
    # 
    @staticmethod
    def get_module_key(module_file: str) -> str:
        return os.path.splitext(os.path.basename(module_file))[0].lower()
    
    @staticmethod
    def get_circuit_breaker_key(module_file: str) -> str:
        return f"cb_{AnalysisHelpers.get_module_key(module_file)}"
    
    @staticmethod
    def get_module_instance_key(module_file: str) -> str:
        base_key = AnalysisHelpers.get_module_key(module_file)
        return f"{base_key}_{hashlib.md5(module_file.encode()).hexdigest()[:8]}"
    
    @staticmethod
    def resolve_analysis_path(relative_path: str) -> Path:
        base_path = Path(__file__).parent
        resolved = (base_path / relative_path).resolve()
        
        if not str(resolved).startswith(str(base_path)):
            raise PermissionError(f"Unauthorized path access: {resolved}")
        return resolved
    

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            return f"{seconds/60:.1f}m"
    
    # === INSTANCE METHODS === (ORİJİNAL + UTILITY ENTEGRASYONU)
    def update_performance_metrics(self, key: str, value: float, max_history: int = 1000):
        """Performance metriklerini güncelle"""
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []
        
        self.performance_metrics[key].append(value)
        
        if len(self.performance_metrics[key]) > max_history:
            self.performance_metrics[key] = self.performance_metrics[key][-max_history//2:]
    

    # ✅ CONFIGURATION MANAGEMENT
    def load_config_safe(self, module_name: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Safely load configuration with fallback and Pydantic support"""
        try:
            config_module = f"analysis.config.c_{module_name}"
            module = __import__(config_module, fromlist=['CONFIG'])
            config_obj = getattr(module, 'CONFIG', None)
            
            if config_obj is None:
                logger.warning(f"CONFIG not found in {config_module}, using defaults")
                return default_config
                
            # ✅ Pydantic model ve dataclass desteği
            if hasattr(config_obj, 'dict'):
                return config_obj.dict()  # Pydantic model
            elif hasattr(config_obj, '__dict__'):
                return dict(config_obj.__dict__)  # Dataclass/NamedTuple
            elif isinstance(config_obj, dict):
                return config_obj  # Plain dict
            else:
                logger.warning(f"Unsupported config type {type(config_obj)}, using defaults")
                return default_config
                
        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"Config load failed for {module_name}: {e}, using defaults")
            return default_config
            
    def calculate_weights(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Instance versiyonu - utility class'ını kullanır"""
        return self.utils.calculate_weighted_average(scores, weights)  # ✅ Utility class'ından

    def validate_config_schema(self, config: Dict[str, Any]) -> bool:
        """Validate configuration schema"""
        required_sections = ['parameters', 'weights', 'thresholds']
        return all(section in config for section in required_sections)


    # ✅ CACHE MANAGEMENT
    def set_cache(self, key: str, value: Any, ttl: int = None):
        """Set cache value with TTL"""
        if ttl is None:
            ttl = self.cache_ttl
            
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def get_cache(self, key: str) -> Any:
        """Get cache value if not expired"""
        if key in self.cache:
            cache_item = self.cache[key]
            if time.time() < cache_item['expires']:
                return cache_item['value']
            else:
                del self.cache[key]
        return None
    
    def clear_expired_cache(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items() 
            if current_time >= item['expires']
        ]
        for key in expired_keys:
            del self.cache[key]
            
            
    # ✅ FALLBACK & ERROR HANDLING
    def create_fallback_output(self, module_name: str, reason: str = "") -> Dict[str, Any]:
        """Create fallback output when analysis fails"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": f"Fallback mode: {reason}",
            "timestamp": self.get_timestamp(),
            "module": module_name,
            "metadata": {
                "fallback": True,
                "reason": reason,
                "calculation_time": 0
            }
        }
    
    def handle_analysis_error(self, error: Exception, module_name: str) -> Dict[str, Any]:
        """Handle analysis errors gracefully"""
        logger.error(f"Analysis error in {module_name}: {error}")
        return self.create_fallback_output(module_name, str(error))
    
    
    # ✅ DATA PROCESSING UTILITIES
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate price returns"""
        if len(prices) < 2:
            return np.array([])
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, window: int = 20) -> float:
        """Calculate volatility from returns"""
        if len(returns) < window:
            return 0.0
        return np.std(returns[-window:])
    
    @staticmethod
    def detect_trend_direction(prices: np.ndarray, window: int = 10) -> str:
        """Detect basic trend direction"""
        if len(prices) < window:
            return "neutral"
        
        recent_prices = prices[-window:]
        slope = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if slope > 0.02:  # 2% upward movement
            return "bullish"
        elif slope < -0.02:  # 2% downward movement
            return "bearish"
        else:
            return "neutral"
            
    # ✅ ASYNC UTILITIES
    async def run_in_threadpool(self, func, *args, **kwargs):
        """Run CPU-bound function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def batch_process(self, items: List, func, batch_size: int = 10):
        """Process items in batches asynchronously"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        return results        
            
    
    # ✅ PERFORMANCE MONITORING
    class Timer:
        """Context manager for performance timing"""
        def __init__(self, name: str = ""):
            self.name = name
            self.start_time = None
            self.end_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            logger.debug(f"Timer {self.name}: {elapsed:.4f}s")
            
        def elapsed(self) -> float:
            """Get elapsed time in seconds"""
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            elif self.start_time:
                return time.time() - self.start_time
            return 0.0
    
    
    # ✅ PERFORMANCE TRACKING METODLARI
    def track_performance(self, module_name: str, duration: float):
        """Performans metriklerini kaydet"""
        self.update_performance_metrics(module_name, duration)
    
    def get_performance_stats(self, module_name: str) -> Dict[str, float]:
        """Performans istatistiklerini döndür"""
        if module_name not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[module_name]
        if not metrics:
            return {}
            
        return {
            "count": len(metrics),
            "mean": np.mean(metrics),
            "median": np.median(metrics),
            "min": np.min(metrics),
            "max": np.max(metrics),
            "last": metrics[-1] if metrics else 0.0
        }
    
    def clear_performance_metrics(self, module_name: Optional[str] = None):
        """Performans metriklerini temizle"""
        if module_name:
            self.performance_metrics.pop(module_name, None)
        else:
            self.performance_metrics.clear()


# Kullanım kolaylığı için instance'lar
# ✅ GLOBAL INSTANCE FOR CONVENIENCE
analysis_helpers = AnalysisHelpers()
utility_functions = AnalysisUtilities()  # ✅ burası kesin olmalı

# ✅ EXPORT KEY COMPONENTS
__all__ = [
    'AnalysisHelpers',
    'AnalysisUtilities', 
    'AnalysisOutput',
    'analysis_helpers',
    'utility_functions'
]

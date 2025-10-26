# analysis/analysis_base_module.py
"""
Base Analysis Module Abstract Class - Async Optimized
analysis_base_module.py
Version: 2.0.0
=====================================================
Tüm analiz modülleri bu sınıftan türemeli.
AnalysisHelpers ile tam uyumlu, tam async yapı.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import asyncio
import time

# ✅ AnalysisHelpers ile tam entegrasyon
from analysis.analysis_helpers import (
    AnalysisHelpers, 
    AnalysisUtilities, 
    AnalysisOutput,
    analysis_helpers,
    utility_functions
)

logger = logging.getLogger(__name__)

class BaseAnalysisModule(ABC):
    """Analiz modülleri için abstract base class - Tam Async"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ✅ AnalysisHelpers entegrasyonu
        self.helpers = analysis_helpers
        self.utils = utility_functions
        
        # Config yükleme - AnalysisHelpers ile güvenli
        self.module_name = self.__class__.__name__.lower()
        default_config = getattr(self, 'DEFAULT_CONFIG', {})
        self.config = self.helpers.load_config_safe(self.module_name, config or default_config)
        
        self.version = getattr(self, 'version', "1.0.0")
        self.dependencies: List[str] = getattr(self, 'dependencies', [])
        
        # ✅ Performance tracking - AnalysisHelpers ile entegre
        self._module_key = self.helpers.get_module_key(__file__)
        
        # Circuit breaker state
        self._circuit_state = "CLOSED"
        self._error_timestamps: List[float] = []

    # ✅ ABSTRACT METHODS - Tam Async
    @abstractmethod
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Ana metrik hesaplama metodu - Async
        
        Args:
            symbol: Analiz yapılacak sembol (Örnek: "BTCUSDT")
            priority: Öncelik seviyesi ("*", "**", "***")
            
        Returns:
            Dict: Analiz sonuçları
        """
        pass
    
    @abstractmethod 
    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Metrikleri aggregate edip final sonuç üret - Async
        
        Args:
            metrics: Hesaplanan metrikler
            symbol: Sembol
            
        Returns:
            Dict: Aggregate edilmiş sonuç
        """
        pass
    
    @abstractmethod
    async def generate_report(self) -> Dict[str, Any]:
        """
        Modül durum raporu oluştur - Async
        
        Returns:
            Dict: Rapor verisi
        """
        pass

    # ✅ CORE ASYNC METHODS
    async def execute_analysis(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Ana analiz execution metodu - Tam Async
        AnalysisHelpers performance tracking ile entegre
        """
        start_time = time.time()
        execution_success = False
        
        try:
            # ✅ Circuit breaker check
            if not await self._check_circuit_breaker():
                logger.warning(f"Circuit breaker OPEN for {self.module_name}, using fallback")
                return self.utils.create_fallback_output(
                    self.module_name, 
                    "Circuit breaker open"
                )
            
            # ✅ Ana analiz işlemi
            metrics = await self.compute_metrics(symbol, priority)
            
            if not self.utils.validate_score_dict(metrics):
                logger.warning(f"Invalid metrics format from {self.module_name}")
                metrics = {}
            
            # ✅ Aggregate output
            result = await self.aggregate_output(metrics, symbol)
            
            # ✅ Output validation
            if not self.utils.validate_output(result):
                logger.warning(f"Output validation failed for {self.module_name}, using fallback")
                result = self.utils.create_fallback_output(
                    self.module_name, 
                    "Output validation failed"
                )
            
            # ✅ AnalysisOutput schema validation
            try:
                validated_output = AnalysisOutput(**result)
                result = validated_output.dict()
            except ValidationError as e:
                logger.error(f"Pydantic validation failed for {self.module_name}: {e}")
                result = self.utils.create_fallback_output(
                    self.module_name, 
                    f"Schema validation failed: {e}"
                )
            
            execution_success = True
            self._record_success()
            return result
            
        except Exception as e:
            logger.error(f"Analysis execution failed for {self.module_name}: {e}")
            self._record_error()
            return self.utils.create_fallback_output(
                self.module_name, 
                f"Execution error: {e}"
            )
        finally:
            # ✅ Performance tracking - AnalysisHelpers ile
            execution_time = time.time() - start_time
            self.helpers.update_performance_metrics(
                f"{self._module_key}_execution_time", 
                execution_time
            )
            self.helpers.update_performance_metrics(
                f"{self._module_key}_success" if execution_success else f"{self._module_key}_error",
                1.0
            )

    # ✅ ASYNC DATA FETCHING METHODS
    async def _fetch_ohlcv_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        OHLCV verisi çekme - Tam Async
        
        Args:
            symbol: Sembol
            interval: Zaman aralığı
            limit: Veri limiti
            
        Returns:
            pd.DataFrame: OHLCV verisi
        """
        try:
            # ✅ Gerçek async API call simülasyonu
            await asyncio.sleep(0.01)  # Async delay simülasyonu
            
            # Mock data generation - async friendly
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
            data = pd.DataFrame({
                'open': np.random.random(limit) * 1000 + 50000,
                'high': np.random.random(limit) * 1000 + 50100,
                'low': np.random.random(limit) * 1000 + 49900,
                'close': np.random.random(limit) * 1000 + 50000,
                'volume': np.random.random(limit) * 1000
            }, index=dates)
            
            logger.debug(f"Fetched OHLCV data for {symbol}, shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol}: {e}")
            raise

    async def _fetch_multiple_ohlcv(self, symbols: List[str], interval: str = "1h", limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Birden fazla sembol için async OHLCV verisi çekme
        """
        tasks = [self._fetch_ohlcv_data(symbol, interval, limit) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                # Fallback: boş dataframe
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result
                
        return data_dict

    # ✅ CIRCUIT BREAKER PATTERN - Async
    async def _check_circuit_breaker(self) -> bool:
        """Circuit breaker durumunu kontrol et"""
        if self._circuit_state == "OPEN":
            # 30 saniye sonra HALF-OPEN'a geç
            if time.time() - max(self._error_timestamps) > 30:
                self._circuit_state = "HALF_OPEN"
                return True
            return False
        
        # Son 10 hatayı kontrol et
        recent_errors = [ts for ts in self._error_timestamps if time.time() - ts < 60]
        if len(recent_errors) >= 5:
            self._circuit_state = "OPEN"
            return False
            
        return True

    def _record_success(self):
        """Başarılı execution kaydı"""
        if self._circuit_state == "HALF_OPEN":
            self._circuit_state = "CLOSED"
        # Eski error kayıtlarını temizle
        self._error_timestamps = [ts for ts in self._error_timestamps if time.time() - ts < 300]

    def _record_error(self):
        """Hata kaydı - circuit breaker için"""
        self._error_timestamps.append(time.time())
        # 300 saniyeden eski kayıtları temizle
        self._error_timestamps = [ts for ts in self._error_timestamps if time.time() - ts < 300]

    # ✅ ASYNC HEALTH CHECK & REPORTING
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint - Tam Async
        
        Returns:
            Dict: Health status
        """
        try:
            # ✅ Async data check
            test_data = await self._fetch_ohlcv_data("BTCUSDT", limit=5)
            data_healthy = not test_data.empty and len(test_data) > 0
            
            # ✅ Circuit breaker status
            circuit_healthy = await self._check_circuit_breaker()
            
            # ✅ Performance metrics
            perf_metrics = self.get_performance_metrics()
            
            health_status = "healthy" if all([
                data_healthy, 
                circuit_healthy,
                perf_metrics["success_rate"] > 0.8
            ]) else "degraded"
            
            return {
                "module": self.module_name,
                "status": health_status,
                "version": self.version,
                "timestamp": self.helpers.get_iso_timestamp(),
                "data_available": data_healthy,
                "circuit_breaker": self._circuit_state,
                "success_rate": perf_metrics["success_rate"],
                "recent_errors": len(self._error_timestamps),
                "dependencies_healthy": True
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.module_name}: {e}")
            return {
                "module": self.module_name,
                "status": "unhealthy",
                "version": self.version,
                "timestamp": self.helpers.get_iso_timestamp(),
                "error": str(e),
                "circuit_breaker": self._circuit_state
            }

    async def get_detailed_report(self) -> Dict[str, Any]:
        """
        Detaylı modül raporu - Async
        """
        health = await self.health_check()
        performance = self.get_performance_metrics()
        
        return {
            **health,
            "performance": performance,
            "config_keys": list(self.config.keys()),
            "dependencies": self.dependencies,
            "circuit_breaker_errors": len(self._error_timestamps)
        }

    # ✅ PERFORMANCE METRICS - AnalysisHelpers entegre
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Performans metriklerini getir - AnalysisHelpers entegre
        """
        exec_times = self.helpers.performance_metrics.get(
            f"{self._module_key}_execution_time", []
        )
        success_count = len(self.helpers.performance_metrics.get(
            f"{self._module_key}_success", []
        ))
        error_count = len(self.helpers.performance_metrics.get(
            f"{self._module_key}_error", []
        ))
        
        avg_time = np.mean(exec_times) if exec_times else 0
        success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
        
        return {
            "module": self.module_name,
            "version": self.version,
            "total_executions": len(exec_times),
            "average_execution_time": avg_time,
            "execution_time_p95": np.percentile(exec_times, 95) if exec_times else 0,
            "success_rate": success_rate,
            "success_count": success_count,
            "error_count": error_count,
            "dependencies": self.dependencies,
            "circuit_breaker_state": self._circuit_state
        }

# ✅ BACKWARD COMPATIBILITY
def legacy_compatible(cls):
    """
    Eski run() fonksiyonu ile uyumluluk sağlayan decorator
    """
    original_init = cls.__init__
    
    def new_init(self, config=None):
        if config is None:
            # Varsayılan config yükle
            config_module_name = f"config{cls.__name__.replace('Module', '')}"
            try:
                # AnalysisHelpers ile güvenli config loading
                helpers = AnalysisHelpers()
                config = helpers.load_config_safe(
                    cls.__name__.lower(), 
                    getattr(cls, 'DEFAULT_CONFIG', {})
                )
            except:
                config = {}
        original_init(self, config)
    
    cls.__init__ = new_init
    
    # Eski run fonksiyonunu ekle - Async
    async def run(symbol: str, priority: Optional[str] = None):
        instance = cls()
        return await instance.execute_analysis(symbol, priority)
    
    cls.run = staticmethod(run)
    return cls
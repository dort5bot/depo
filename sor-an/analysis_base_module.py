"""
analysis_base_module.py
Base Analysis Module Abstract Class - TAM ASYNC Yapı
===================================
Tüm analiz modülleri bu async sınıftan türemeli.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
import asyncio
import time

# ✅ ASYNC ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import (
    analysis_helpers,  
    utility_functions,
    AnalysisOutput
)

logger = logging.getLogger(__name__)

class BaseAnalysisModule(ABC):
    """✅ TAM ASYNC ANALİZ MODÜLÜ - Tüm metodlar async"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.module_name = self.__class__.__name__
        self.utils = utility_functions  # ✅ UTILITY INSTANCE
        self.helpers = analysis_helpers  # ✅ HELPER INSTANCE
        self.version = getattr(self, 'version', "1.0.0")
        self.dependencies: List[str] = getattr(self, 'dependencies', [])
        
        # ✅ ASYNC PERFORMANCE TRACKING
        self._execution_times: List[float] = []
        self._success_count: int = 0
        self._error_count: int = 0
        self._last_execution_time: float = 0.0
        self._performance_lock = asyncio.Lock()  # ✅ ASYNC LOCK





    # ✅ TAM ASYNC ABSTRACT METHODS
    @abstractmethod
    async def initialize(self) -> None:
        """Async modül başlatma"""
        pass
        
    @abstractmethod
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        ✅ ASYNC ana metrik hesaplama metodu
        
        Returns:
            Dict: AnalysisOutput şemasına uygun analiz sonuçları
        """
        pass
    
    @abstractmethod 
    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """✅ ASYNC metrik aggregation"""
        pass
    
    @abstractmethod
    async def generate_report(self) -> Dict[str, Any]:
        """✅ ASYNC rapor oluşturma"""
        pass
    
    # ✅ TAM ASYNC YARDIMCI METOTLAR
    
    async def _create_output_template(self) -> Dict[str, Any]:
        """✅ ASYNC output şablonu"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": "",
            "timestamp": await self._get_timestamp_async(),
            "module": self.module_name
        }
    
    async def _validate_output(self, output: Dict[str, Any]) -> bool:
        """Async output validation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.utils.validate_output(output))
    
    async def _normalize_score(self, score: float) -> float:
        """Async score normalization"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.utils.normalize_score(score))
    
    async def _create_fallback_output(self, reason: str = "Error") -> Dict[str, Any]:
        """Async fallback output"""
        return self.utils.create_fallback_output(self.module_name, reason)
    
    async def _calculate_weighted_score(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Async weighted average"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.utils.calculate_weighted_average(components, weights)
        )



    async def _fetch_ohlcv_data_async(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        ✅ ASYNC OHLCV verisi çekme
        """
        try:
            # ✅ GERÇEK ASYNC IMPLEMENTASYON İÇİN:
            # Binance async API veya aiohttp kullanılacak
            # Şimdilik mock data ile async simüle ediyoruz
            
            # Simüle async delay
            await asyncio.sleep(0.01)
            
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=interval)
            data = pd.DataFrame({
                'open': np.random.random(limit) * 1000 + 50000,
                'high': np.random.random(limit) * 1000 + 50100,
                'low': np.random.random(limit) * 1000 + 49900,
                'close': np.random.random(limit) * 1000 + 50000,
                'volume': np.random.random(limit) * 1000
            }, index=dates)
            
            logger.debug(f"Fetched ASYNC OHLCV data for {symbol}, shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch ASYNC OHLCV data for {symbol}: {e}")
            raise
    
    async def get_performance_metrics_async(self) -> Dict[str, Any]:
        """✅ ASYNC performans metrikleri"""
        async with self._performance_lock:
            avg_time = sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
            success_rate = (
                self._success_count / (self._success_count + self._error_count) 
                if (self._success_count + self._error_count) > 0 else 0
            )
            
            return {
                "module": self.module_name,
                "version": self.version,
                "total_executions": len(self._execution_times),
                "average_execution_time": avg_time,
                "success_rate": success_rate,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "dependencies": self.dependencies,
                "last_execution_time": self._last_execution_time,
                "timestamp": await self._get_timestamp_async()
            }
    
    async def health_check_async(self) -> Dict[str, Any]:
        """✅ ASYNC health check"""
        try:
            # ✅ ASYNC DATA FETCH
            test_data = await self._fetch_ohlcv_data_async("BTCUSDT", limit=10)
            data_healthy = not test_data.empty and len(test_data) > 0
            
            perf_metrics = await self.get_performance_metrics_async()
            
            return {
                "module": self.module_name,
                "status": "healthy" if data_healthy else "degraded",
                "version": self.version,
                "timestamp": await self._get_timestamp_async(),
                "data_available": data_healthy,
                "dependencies_healthy": True,
                "performance_metrics": perf_metrics
            }
            
        except Exception as e:
            logger.error(f"ASYNC Health check failed for {self.module_name}: {e}")
            return {
                "module": self.module_name,
                "status": "unhealthy",
                "version": self.version,
                "timestamp": await self._get_timestamp_async(),
                "error": str(e)
            }
    
    async def _record_execution_async(self, execution_time: float, success: bool = True):
        """✅ ASYNC execution kaydı"""
        async with self._performance_lock:
            self._execution_times.append(execution_time)
            self._last_execution_time = await self._get_timestamp_async()
            
            if success:
                self._success_count += 1
            else:
                self._error_count += 1
                
            # ✅ ASYNC PERFORMANCE UPDATE
            await self._update_performance_metrics_async(execution_time)
            
            # Execution times dizisini trim et
            if len(self._execution_times) > 1000:
                self._execution_times = self._execution_times[-500:]
    
    async def _update_performance_metrics_async(self, execution_time: float):
        """✅ ASYNC performance metrics update"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: analysis_helpers.update_performance_metrics(
                {'execution_times': self._execution_times}, 
                'execution_times', 
                execution_time
            )
        )
    
    async def _get_timestamp_async(self) -> float:
        """✅ ASYNC timestamp alma"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, time.time)
    
    async def _get_execution_stats_async(self) -> Dict[str, Any]:
        """✅ ASYNC execution istatistikleri"""
        async with self._performance_lock:
            if not self._execution_times:
                return {}
                
            loop = asyncio.get_event_loop()
            
            def calculate_stats():
                return {
                    "count": len(self._execution_times),
                    "average": np.mean(self._execution_times),
                    "min": np.min(self._execution_times),
                    "max": np.max(self._execution_times),
                    "last_execution": self._last_execution_time
                }
            
            return await loop.run_in_executor(None, calculate_stats)

    # ✅ YENİ ASYNC METODLAR - ÖNERİLEN YAPIDA
    
    async def validate_input(self, symbol: str) -> bool:
        """✅ ASYNC input validation"""
        return bool(symbol and isinstance(symbol, str))
    
    async def pre_process(self, symbol: str) -> Dict[str, Any]:
        """✅ ASYNC pre-processing"""
        return {
            "symbol": symbol, 
            "timestamp": await self._get_timestamp_async(),
            "module": self.module_name
        }
    
    async def post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """✅ ASYNC post-processing"""
        # Standart output formatına uygun hale getir
        required_fields = {
            "score": 0.5,
            "signal": "neutral", 
            "confidence": 0.0,
            "components": {},
            "explain": "No explanation provided"
        }
        
        for field, default in required_fields.items():
            if field not in result:
                result[field] = default
        
        result["timestamp"] = await self._get_timestamp_async()
        result["module"] = self.module_name
        
        return result
    
# ✅ ANA ASYNC METODLAR
    async def execute_analysis(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """Tam async analysis pipeline"""
        start_time = await self._get_timestamp()
        
        try:
            # 1. Input validation
            if not await self.validate_input(symbol):
                return await self._create_fallback_output("Invalid input")
            
            # 2. Pre-processing
            pre_processed = await self.pre_process(symbol)
            
            # 3. Main computation
            raw_result = await self.compute_metrics(symbol, priority)
            
            # 4. Post-processing  
            final_result = await self.post_process({**pre_processed, **raw_result})
            
            # 5. Output validation
            if not await self._validate_output(final_result):
                return await self._create_fallback_output("Output validation failed")
            
            # 6. Record success
            execution_time = await self._get_timestamp() - start_time
            await self._record_execution(execution_time, True)
            
            return final_result
            
        except Exception as e:
            execution_time = await self._get_timestamp() - start_time
            await self._record_execution(execution_time, False)
            return await self._create_fallback_output(str(e))
            

# ✅ ASYNC LEGACY COMPATIBILITY
def legacy_compatible(cls):
    """
    ✅ ASYNC legacy compatibility decorator
    """
    original_init = cls.__init__
    
    def new_init(self, config=None):
        if config is None:
            # ✅ ASYNC CONFIG YÜKLEME
            config_module_name = f"config{cls.__name__.replace('Module', '')}"
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop running, sync load
                    config = analysis_helpers.load_config_safe(
                        config_module_name.lower(), 
                        {}
                    )
                else:
                    # Async load
                    async def load_config():
                        return analysis_helpers.load_config_safe(
                            config_module_name.lower(), 
                            {}
                        )
                    config = asyncio.run(load_config())
            except Exception as e:
                logger.warning(f"Async config load failed, using defaults: {e}")
                config = {}
        original_init(self, config)
    
    cls.__init__ = new_init
    
    # ✅ ASYNC RUN FONKSİYONU
    async def run(symbol: str, priority: Optional[str] = None):
        instance = cls()
        await instance.initialize()
        return await instance.execute_analysis(symbol, priority)
    
    cls.run = staticmethod(run)
    
    logger.info(f"Async legacy compatibility applied to {cls.__name__}")
    return cls

# ✅ TAM ASYNC STANDART MODÜL
class StandardAnalysisModule(BaseAnalysisModule):
    """✅ TAM ASYNC standart analiz modülü"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.weights = getattr(self, 'weights', {})
        self.thresholds = getattr(self, 'thresholds', {})
        self._initialized = False
    
    async def initialize(self) -> None:
        """✅ ASYNC başlatma"""
        if not self._initialized:
            logger.info(f"Async initializing {self.module_name} v{self.version}")
            # Async initialization tasks here
            await asyncio.sleep(0.001)  # Simüle async iş
            self._initialized = True
    
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """✅ TAM ASYNC metrik hesaplama"""
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Modül spesifik async hesaplamalar
        components = await self._calculate_components_async(symbol, priority)
        score = await self._calculate_weighted_score_async(components, self.weights)
        signal = await self._determine_signal_async(score)
        
        output = await self._create_output_template()
        output.update({
            "score": await self._normalize_score_async(score),
            "signal": signal,
            "confidence": await self._calculate_confidence_async(components),
            "components": components,
            "explain": f"Async analysis completed for {symbol}",
            "metadata": {
                "symbol": symbol,
                "priority": priority,
                "module_version": self.version
            }
        })
        
        return output
    
    async def _calculate_components_async(self, symbol: str, priority: Optional[str] = None) -> Dict[str, float]:
        """✅ ASYNC bileşen hesaplama - override edilmeli"""
        # Simüle async iş
        await asyncio.sleep(0.001)
        return {"default_component": 0.5}
    
    async def _determine_signal_async(self, score: float) -> str:
        """✅ ASYNC sinyal belirleme"""
        bullish_threshold = self.thresholds.get('bullish', 0.7)
        bearish_threshold = self.thresholds.get('bearish', 0.3)
        
        if score >= bullish_threshold:
            return "bullish"
        elif score <= bearish_threshold:
            return "bearish"
        else:
            return "neutral"
    
    async def _calculate_confidence_async(self, components: Dict[str, float]) -> float:
        """✅ ASYNC güvenilirlik hesaplama"""
        if not components:
            return 0.0
        # Basit async implementasyon
        confidence = min(1.0, len(components) * 0.1)
        return await self._normalize_score_async(confidence)
    
    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """✅ ASYNC aggregation"""
        loop = asyncio.get_event_loop()
        
        def sync_aggregate():
            return {
                "symbol": symbol,
                "aggregated_score": normalize_score(np.mean(list(metrics.values()))),
                "component_scores": metrics,
                "timestamp": time.time(),
                "module": self.module_name
            }
        
        return await loop.run_in_executor(None, sync_aggregate)
    
    async def generate_report(self) -> Dict[str, Any]:
        """✅ ASYNC rapor oluşturma"""
        perf_metrics = await self.get_performance_metrics_async()
        return {
            "module": self.module_name,
            "version": self.version,
            "status": "operational",
            "performance": perf_metrics,
            "dependencies": self.dependencies,
            "timestamp": await self._get_timestamp_async(),
            "report_type": "async_module_health_report"
        }
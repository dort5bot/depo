# analysis/analysis_a.py
# -*- coding: utf-8 -*-
"""
AnalysisAggregator (v2025.1) - OPTIMIZED VERSION
Multi-user async orchestrator + metric aggregator
Integrated with Unified Health Check & Performance Monitoring
"""

import asyncio
import logging
import os
import time
import pandas as pd
from typing import Any, Dict, List, Optional, AsyncGenerator
from functools import lru_cache
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Config entegrasyonu
from config import get_config_sync, BotConfig

from analysis.schema_manager import SchemaManager
from analysis.module_loader import ModuleLoader
from analysis.metric_engine import MetricEngine
from analysis.metric_resolver import MetricResolver
from analysis.health_checker import UnifiedHealthChecker, ComponentType

from utils.binance_api.binance_a import BinanceAggregator
from utils.binance_api.binance_multi_user import UserSessionManager #BinanceMultiUser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# 🔹 ANALYSIS CONFIG CLASS
# ============================================================
class AnalysisConfig:
    """Analysis-specific configuration integrated with main config"""
    
    def __init__(self, main_config: BotConfig = None):
        self.config = main_config or get_config_sync()
        
    @property
    def MAX_CONCURRENT(self) -> int:
        return int(os.getenv("ANALYSIS_MAX_CONCURRENT", 
                   str(self.config.BINANCE.Performance.MAX_CONCURRENT_REQUESTS)))
    
    @property
    def METRIC_TIMEOUT(self) -> int:
        return int(os.getenv("METRIC_TIMEOUT", "30"))
    
    @property
    def CACHE_TTL(self) -> int:
        return int(os.getenv("ANALYSIS_CACHE_TTL", 
                   str(self.config.BINANCE.MarketData.CACHE_TTL)))
    
    @property
    def DEFAULT_INTERVAL(self) -> str:
        return os.getenv("ANALYSIS_DEFAULT_INTERVAL", 
               self.config.ANALYSIS.DEFAULT_TIMEFRAME)
    
    @property
    def MAX_LOOKBACK(self) -> int:
        return self.config.ANALYSIS.MAX_LOOKBACK
    
    @property
    def BATCH_SIZE(self) -> int:
        return int(os.getenv("ANALYSIS_BATCH_SIZE", "5"))
    
    @property
    def ENABLED(self) -> bool:
        return self.config.ANALYSIS.ENABLED
    
    @property
    def SYMBOLS(self) -> List[str]:
        return self.config.SCAN.ALL_SYMBOLS
    
    @property
    def DEFAULT_SYMBOLS(self) -> List[str]:
        return self.config.SCAN.DEFAULT_SYMBOLS
    
    @property
    def USE_PROCESS_POOL(self) -> bool:
        return getattr(self.config.ANALYSIS, 'USE_PROCESS_POOL', True)
    
    @property
    def MAX_WORKERS(self) -> int:
        return min(32, (os.cpu_count() or 1) + 4)


# ============================================================
# 🔹 SUPPORTING CLASSES
# ============================================================
class LRUCache:
    """Simple LRU Cache implementation"""
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.maxsize:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)
    
    def __contains__(self, key):
        return key in self.cache

class CircuitBreaker:
    """Circuit Breaker pattern implementation"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def allow_request(self):
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"


# ============================================================
# 🔹 ANA ORCHESTRATOR SINIFI - OPTIMIZE EDİLMİŞ
# ============================================================
class AnalysisAggregator:
    """
    Optimized Analysis Aggregator with enhanced performance and memory management
    Integrated with Unified Health Check & Performance Monitoring
    """
    
    def __init__(self, config: BotConfig = None, base_path: str = None):  # ✅ base_path parametresi EKLENDİ
        
        # 1. Base path ayarla
        current_dir = os.path.dirname(__file__)
        base_path = os.path.abspath(os.path.join(current_dir, '..', 'utils', 'binance_api'))
    
        
        self.base_path = base_path
        
        # 2. Config yükleme
        self._setup_config(config)
        
        # 3. Core component'ler
        self._setup_core_components()
        
        # 4. Cache ve state management
        self._setup_cache()
        
        # 5. Execution pools
        self._setup_execution_pools()
        
        # 6. Performance monitoring
        self._setup_monitoring()
        
        # 7. Binance aggregator - async olarak initialize edilecek
        self.binance_aggregator = None
        self.data_provider = None
        
        logger.info(f"✅ AnalysisAggregator initialized with base_path: {self.base_path}")
        
    # ============================================================
    # 🔹 INITIALIZATION METHODS
    # ============================================================
    
    def _setup_config(self, config: BotConfig):
        """Config yükleme ve validasyon"""
        self.main_config = config or get_config_sync()
        self.analysis_config = AnalysisConfig(self.main_config)
        
        self.settings = {
            'max_concurrent': self.analysis_config.MAX_CONCURRENT,
            'timeout': self.analysis_config.METRIC_TIMEOUT,
            'cache_ttl': self.analysis_config.CACHE_TTL,
            'batch_size': self.analysis_config.BATCH_SIZE,
            'enabled': self.analysis_config.ENABLED,
            'use_process_pool': self.analysis_config.USE_PROCESS_POOL,
            'max_workers': self.analysis_config.MAX_WORKERS
        }
    
    def _setup_core_components(self):
        """Core component'leri başlat"""
        self.schema = SchemaManager()
        self.loader = ModuleLoader("analysis/schemas/module_registry.yaml")
        self.engine = MetricEngine()
        self.resolver = MetricResolver()
        self.multi_user = UserSessionManager()
        
        # data_provider ve binance_aggregator async olarak initialize edilecek
        self.data_provider = None
        self.binance_aggregator = None
        
        self.health_checker = UnifiedHealthChecker(self)
    
    def _setup_cache(self):
        """Cache ve state management"""
        self._result_cache = {}
        self._cache_expiry = timedelta(minutes=5)
        self._request_deduplication: Dict[str, asyncio.Task] = {}
        self._metric_cache = LRUCache(maxsize=1000)
        self._ohlcv_cache = LRUCache(maxsize=500)
    
    def _setup_execution_pools(self):
        """Execution pools için setup"""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.settings['max_workers'],
            thread_name_prefix="metric_thread"
        )
        
        if self.settings['use_process_pool']:
            self.process_pool = ProcessPoolExecutor(
                max_workers=max(2, self.settings['max_workers'] // 2)
            )
        else:
            self.process_pool = None
    
    def _setup_monitoring(self):
        """Monitoring ve health check setup"""
        self._performance_stats = {
            'total_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0
        }
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )

    async def initialize_data_provider(self):
        """BinanceAggregator'ü async olarak başlat"""
        if self.data_provider is None:
            try:
                # ✅ DÜZELTME: BinanceAggregator'ü base_path ile oluştur
                from utils.binance_api.binance_a import BinanceAggregator
                self.data_provider = await BinanceAggregator.get_instance(base_path=self.base_path)
                self.binance_aggregator = self.data_provider  # Alias for compatibility
                logger.info(f"✅ BinanceAggregator initialized with base_path: {self.base_path}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize BinanceAggregator: {e}")
                raise


    # ============================================================
    # 🔹 PUBLIC API - PERFORMANCE OPTIMIZED
    # ============================================================
    
    async def run_module(self, user_id: str, module_name: str, symbol: str, 
                        interval: str = None) -> Dict[str, Any]:
        """
        Tek modülün metrik hesaplamalarını yürütür (performans izlemeli).
        Optimized with request deduplication and circuit breaker.
        """
        # ✅ Data provider'ı initialize et
        if self.data_provider is None:
            await self.initialize_data_provider()

        
        # Circuit breaker kontrolü
        if not self._circuit_breaker.allow_request():
            raise Exception("Service temporarily unavailable")
        
        start_time = time.time()
        success = True
        cache_key = f"{user_id}_{module_name}_{symbol}_{interval}"

        try:
            # Request deduplication
            if cache_key in self._request_deduplication:
                return await self._request_deduplication[cache_key]

            if not self.settings['enabled']:
                logger.warning("Analysis is disabled in configuration")
                return {module_name: {"error": "Analysis disabled"}}
                
            # Use default interval from config if not provided
            interval = self._get_interval(interval)

            # Symbol validation
            if not self.validate_symbol(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")

            mod_info = self.schema.get_module(module_name)
            if not mod_info:
                raise ValueError(f"Modül bulunamadı: {module_name}")

            # Cache check
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self._performance_stats['cache_hits'] += 1
                return cached_result

            # OHLCV verisini al
            ohlcv = await self._fetch_ohlcv_data(user_id, symbol, interval)

            # Compute intensity'ye göre execution strategy seç
            compute_intensity = mod_info.get("compute_intensity", "medium")
            module_results = await self._execute_metrics_with_strategy(
                module_name, mod_info, ohlcv, compute_intensity
            )

            result = {module_name: module_results}
            
            # Cache'e kaydet
            self._set_cached_result(cache_key, result)
            self._performance_stats['total_requests'] += 1
            self._circuit_breaker.record_success()

            return result

        except Exception as e:
            success = False
            self._performance_stats['failed_requests'] += 1
            self._circuit_breaker.record_failure()
            logger.exception(f"run_module hata: {e}")
            raise

        finally:
            # Request deduplication temizleme
            self._request_deduplication.pop(cache_key, None)
            
            # Performance tracking
            total_duration = time.time() - start_time
            self._update_performance_stats(total_duration)
            
            # Health checker tracking
            self.health_checker.track_performance(
                ComponentType.METRIC_ENGINE,
                total_duration,
                success
            )

    async def run_all_modules(self, user_id: str, symbol: str, interval: str = None) -> Dict[str, Any]:
        """Bir kullanıcı için tüm modülleri async çalıştırır."""
        interval = self._get_interval(interval)
        modules = self.schema.list_modules()
        
        # Compute intensity'ye göre gruplama
        grouped_tasks = self._group_tasks_by_intensity(modules, user_id, symbol, interval)
        
        # Grup bazlı parallel execution
        all_results = []
        for intensity, tasks in grouped_tasks.items():
            if intensity == "high" and self.process_pool:
                # Process pool için
                results = await self._execute_in_process_pool(tasks)
            else:
                # Thread pool veya async için
                results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)
        
        return self._aggregate_results(all_results)

    async def run_multi_user(self, user_ids: List[str], symbol: str, interval: str = None) -> Dict[str, Any]:
        """Birden fazla kullanıcı için aynı anda modül analizlerini çalıştırır."""
        interval = self._get_interval(interval)
        
        # Rate limiting with config
        semaphore = asyncio.Semaphore(self.settings['max_concurrent'])
        
        async def run_with_semaphore(user_id):
            async with semaphore:
                return await self.run_all_modules(user_id, symbol, interval)
        
        tasks = [run_with_semaphore(uid) for uid in user_ids]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._aggregate_multi_user(user_ids, all_results)

    # ============================================================
    # 🔹 STREAMING & BATCH OPERATIONS - MEMORY OPTIMIZED
    # ============================================================
    
    async def run_module_batch(self, user_id: str, modules: List[str], symbol: str, interval: str = None):
        """Batch processing for better performance with memory optimization"""
        interval = self._get_interval(interval)
            
        # Memory-efficient streaming execution
        results = []
        async for result in self.stream_metrics_calculation(user_id, modules, symbol, interval):
            results.append(result)
        
        return self._aggregate_results(results)

    async def stream_metrics_calculation(self, user_id: str, modules: List[str], symbol: str, 
                                       interval: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Memory-efficient streaming calculation
        Büyük modül listeleri için memory-friendly execution
        """
        interval = self._get_interval(interval)
            
        for module_name in modules:
            try:
                result = await self.run_module(user_id, module_name, symbol, interval)
                yield result
                
                # Ara belleği temizle - memory optimization
                if hasattr(self, 'engine'):
                    self.engine.reset_last_valid(module_name)
                    
                # Büyük dataset'ler için incremental garbage collection
                if len(modules) > 20:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Stream calculation failed for {module_name}: {e}")
                yield {module_name: {"error": str(e)}}

    async def stream_metrics_calculation_with_callback(self, user_id: str, modules: List[str], symbol: str, 
                                                     interval: str = None, callback: callable = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming calculation with progress callback
        UI veya progress tracking için ideal
        """
        interval = self._get_interval(interval)
            
        total_modules = len(modules)
        
        for i, module_name in enumerate(modules):
            result = await self.run_module(user_id, module_name, symbol, interval)
            
            # Progress callback
            if callback and callable(callback):
                await callback({
                    'module': module_name,
                    'progress': (i + 1) / total_modules,
                    'current': i + 1,
                    'total': total_modules,
                    'result': result
                })
            
            yield result
            
            # Memory cleanup
            if hasattr(self, 'engine'):
                self.engine.reset_last_valid(module_name)

    async def _stream_modules_execution(self, user_id: str, modules: List[str], symbol: str, interval: str):
        """Memory-efficient streaming module execution"""
        batch_size = self.settings['batch_size']
        
        for i in range(0, len(modules), batch_size):
            batch = modules[i:i + batch_size]
            tasks = [self.run_module(user_id, mod_name, symbol, interval) for mod_name in batch]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                yield result
            
            # Memory cleanup
            if hasattr(self, 'engine'):
                for mod_name in batch:
                    self.engine.reset_last_valid(mod_name)
            
            # Force garbage collection for large batches
            if batch_size > 10:
                import gc
                gc.collect()

    # ============================================================
    # 🔹 ADVANCED FEATURES - PERFORMANCE OPTIMIZED
    # ============================================================
    
    async def run_composite_analysis(self, user_id: str, symbol: str, interval: str = None) -> Dict[str, Any]:
        """Composite metrik analizi çalıştırır - Optimized version"""
        interval = self._get_interval(interval)
        
        cache_key = f"composite_{user_id}_{symbol}_{interval}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Paralel execution
        base_results_task = self.run_all_modules(user_id, symbol, interval)
        composite_task = self._calculate_composites(user_id, symbol, interval)
        
        base_results, composite_results = await asyncio.gather(
            base_results_task, composite_task, return_exceptions=True
        )
        
        # Hata kontrolü
        if isinstance(base_results, Exception):
            logger.error(f"Base results failed: {base_results}")
            base_results = {}
        if isinstance(composite_results, Exception):
            logger.error(f"Composite calculation failed: {composite_results}")
            composite_results = {}

        final_result = {
            "composite_scores": composite_results.get("composites", {}),
            "base_metrics": base_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self._set_cached_result(cache_key, final_result)
        return final_result

    # ============================================================
    # 🔹 UTILITY METHODS
    # ============================================================
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available in configuration"""
        return symbol in self.analysis_config.SYMBOLS
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from configuration"""
        return self.analysis_config.SYMBOLS
    
    def get_default_symbols(self, count: int = None) -> List[str]:
        """Get default symbols with optional count limit"""
        if count is None:
            return self.analysis_config.DEFAULT_SYMBOLS
        return self.analysis_config.DEFAULT_SYMBOLS[:count]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self._performance_stats.copy()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Public health check endpoint"""
        health_data = await self.health_checker.comprehensive_health_check()
        
        # Custom performance metrics ekle
        health_data['performance_stats'] = self.get_performance_stats()
        health_data['cache_effectiveness'] = {
            'cache_hits': self._performance_stats['cache_hits'],
            'total_requests': self._performance_stats['total_requests'],
            'hit_ratio': (
                self._performance_stats['cache_hits'] / 
                max(1, self._performance_stats['total_requests'])
            )
        }
        
        return health_data

    # ============================================================
    # 🔹 INTERNAL HELPERS (PRIVATE) - PERFORMANCE CRITICAL
    # ============================================================
    
    def _get_interval(self, interval: str = None) -> str:
        """Get interval with default fallback"""
        return interval or self.analysis_config.DEFAULT_INTERVAL
    
    def _group_tasks_by_intensity(self, modules: List[str], user_id: str, symbol: str, interval: str) -> Dict[str, List]:
        """Görevleri compute intensity'ye göre grupla"""
        grouped = {"low": [], "medium": [], "high": []}
        
        for mod_name in modules:
            mod_info = self.schema.get_module(mod_name)
            intensity = mod_info.get("compute_intensity", "medium")
            task = self.run_module(user_id, mod_name, symbol, interval)
            grouped[intensity].append(task)
        
        return grouped
    
    async def _execute_metrics_with_strategy(self, module_name: str, mod_info: Dict, ohlcv: Any, intensity: str):
        """Compute intensity'ye göre optimize execution strategy"""
        module_results = {}
        
        for group_name, metric_list in mod_info.get("metrics", {}).items():
            for metric_name in metric_list:
                try:
                    if intensity == "high" and self.process_pool:
                        # Process pool için
                        result = await self._execute_in_process_pool_metric(
                            module_name, metric_name, ohlcv
                        )
                    else:
                        # Thread pool veya async
                        func = self.resolver.resolve(metric_name)
                        result = await self.engine.compute_async(
                            module_name,
                            metric_name,
                            func,
                            ohlcv,
                            use_last_valid=True,
                            default=0.0,
                            timeout=self.settings['timeout']
                        )
                    
                    module_results[metric_name] = result
                    
                except Exception as e:
                    logger.warning(f"[{module_name}.{metric_name}] hesaplama hatası: {e}")
                    module_results[metric_name] = None
        
        return module_results
    
    async def _execute_in_process_pool(self, tasks: List) -> List:
        """Process pool'da execution"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, lambda: [task for task in tasks])
    
    async def _execute_in_process_pool_metric(self, module_name: str, metric_name: str, ohlcv: Any):
        """Tekil metric için process pool execution"""
        loop = asyncio.get_event_loop()
        func = self.resolver.resolve(metric_name)
        
        def compute_wrapper():
            return self.engine.compute(
                module_name,
                metric_name,
                func,
                ohlcv,
                use_last_valid=True,
                default=0.0
            )
        
        return await loop.run_in_executor(self.process_pool, compute_wrapper)
    
    async def _calculate_composites(self, user_id: str, symbol: str, interval: str) -> Dict[str, Any]:
        """Composite hesaplamaları - optimized"""
        try:
            from metrics.composite import (
                Trend_Strength_Composite, 
                Volatility_Composite,
                Risk_Composite,
                Market_Regime
            )
            
            # OHLCV verisini al
            ohlcv = await self._fetch_ohlcv_data(user_id, symbol, interval)
            
            # DataFrame'e çevir
            df = self._ohlcv_to_dataframe(ohlcv)
            
            # Paralel composite hesaplama
            trend_task = asyncio.to_thread(Trend_Strength_Composite, df)
            vol_task = asyncio.to_thread(Volatility_Composite, df)
            risk_task = asyncio.to_thread(Risk_Composite, df)
            
            trend_score, vol_score, risk_score = await asyncio.gather(
                trend_task, vol_task, risk_task, return_exceptions=True
            )
            
            # Hata kontrolü
            if isinstance(trend_score, Exception): trend_score = 0.0
            if isinstance(vol_score, Exception): vol_score = 0.0
            if isinstance(risk_score, Exception): risk_score = 0.0
            
            regime = Market_Regime(trend_score, vol_score, risk_score)
            
            return {
                "composites": {
                    "trend_strength": trend_score,
                    "volatility_regime": vol_score,
                    "risk_level": risk_score,
                    "market_regime": regime
                }
            }
            
        except Exception as e:
            logger.error(f"Composite analysis failed: {e}")
            return {"composites": {}}



    #------------------------------------------
    # binance_a dan veri çekme metotu
    #------------------------------------------
    
    async def _fetch_ohlcv_data(self, user_id: str, symbol: str, interval: str) -> Any:
        """Fetch OHLCV data using binance_a.py maps - YENİ VERSİYON"""
        cache_key = f"ohlcv_{user_id}_{symbol}_{interval}_{self.analysis_config.MAX_LOOKBACK}"
        
        # Cache kontrolü - AYNI KALABİLİR
        cached_data = self._ohlcv_cache.get(cache_key)
        if cached_data:
            return cached_data
            
        try:
            # ✅ DÜZELTME: BinanceAggregator async instance al
            if not hasattr(self.data_provider, 'get_data'):
                self.data_provider = await BinanceAggregator.get_instance()
            
            # ✅ DÜZELTME: Maps tabanlı endpoint çağrısı
            ohlcv_data = await self.data_provider.get_data(
                endpoint_name="klines",  # b_map_public.yaml'de tanımlı
                user_id=user_id,  # None ise public, int ise private
                symbol=symbol,
                interval=interval,
                limit=self.analysis_config.MAX_LOOKBACK
            )
            
            # ✅ Cache'e kaydet - AYNI KALABİLİR
            self._ohlcv_cache.set(cache_key, ohlcv_data)
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"OHLCV data fetch failed: {e}")
            return []  # Fallback: boş liste
          
    def _ohlcv_to_dataframe(self, ohlcv: Any) -> pd.DataFrame:
        """Safely convert OHLCV data to DataFrame"""
        try:
            if isinstance(ohlcv, pd.DataFrame):
                return ohlcv
            elif isinstance(ohlcv, list) and len(ohlcv) > 0:
                # Liste formatını kontrol et
                if isinstance(ohlcv[0], (list, tuple)) and len(ohlcv[0]) >= 6:
                    return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    raise ValueError("Unexpected OHLCV data format")
            else:
                raise ValueError(f"Unsupported OHLCV type: {type(ohlcv)}")
        except Exception as e:
            logger.error(f"OHLCV to DataFrame conversion failed: {e}")
            # Fallback: boş DataFrame döndür
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # ============================================================
    # 🔹 CACHE MANAGEMENT - OPTIMIZED
    # ============================================================
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Optimized cache getter"""
        if cache_key in self._result_cache:
            timestamp, data = self._result_cache[cache_key]
            if datetime.now() - timestamp < self._cache_expiry:
                return data
            else:
                # Expired - remove from cache
                del self._result_cache[cache_key]
        return None

    def _set_cached_result(self, cache_key: str, data: Any):
        """Optimized cache setter with size limit"""
        if len(self._result_cache) > 1000:  # Cache size limit
            # Remove oldest 10% of entries
            keys_to_remove = list(self._result_cache.keys())[:100]
            for key in keys_to_remove:
                del self._result_cache[key]
        
        self._result_cache[cache_key] = (datetime.now(), data)

    # ============================================================
    # 🔹 PERFORMANCE & AGGREGATION HELPERS
    # ============================================================
    
    def _update_performance_stats(self, duration: float):
        """Update performance statistics"""
        self._performance_stats['avg_response_time'] = (
            self._performance_stats['avg_response_time'] * 0.9 + duration * 0.1
        )
    
    def _resolve_metric_conflicts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aynı metric isimleri için namespace çakışmasını çöz"""
        conflicted_metrics = {}
        metric_count = {}
        
        for key, value in results.items():
            if '.' in key:
                module, metric = key.split('.', 1)
                new_key = f"{module}.{metric}"
                
                # Aynı metric ismi kaç kez kullanılmış?
                if new_key in metric_count:
                    metric_count[new_key] += 1
                    # Çakışma varsa module_metric formatını kullan
                    if metric_count[new_key] > 1:
                        new_key = f"{module}_{metric}"
                else:
                    metric_count[new_key] = 1
                    
                conflicted_metrics[new_key] = value
            else:
                conflicted_metrics[key] = value
        
        # Conflict loglama
        conflicts = {k: v for k, v in metric_count.items() if v > 1}
        if conflicts:
            logger.warning(
                f"[MetricConflict] {len(conflicts)} metric conflict(s) resolved: {list(conflicts.keys())}"
            )
  
        return conflicted_metrics

    def _aggregate_results(self, module_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tüm modüllerin çıktısını tek JSON içinde birleştirir - conflict resolution ile"""
        combined: Dict[str, Any] = {}
        
        for mod_res in module_results:
            if isinstance(mod_res, Exception):
                logger.error(f"Module result error: {mod_res}")
                continue
                
            for mod_name, metrics in mod_res.items():
                for metric, value in metrics.items():
                    combined[f"{mod_name}.{metric}"] = value
        
        # Conflict resolution uygula
        return self._resolve_metric_conflicts(combined)
        
    def _aggregate_multi_user(self, user_ids: List[str], all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Multi-user sonuçlarını user bazında gruplar"""
        result = {}
        for uid, res in zip(user_ids, all_results):
            if isinstance(res, Exception):
                result[uid] = {"error": str(res)}
            else:
                result[uid] = res
        return result

    # ============================================================
    # 🔹 RESOURCE MANAGEMENT
    # ============================================================
    
    def cleanup(self):
        """Resource cleanup"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)
        
        # Cache temizleme
        self._result_cache.clear()
        self._request_deduplication.clear()


# ============================================================
# 🔹 Factory function for easy instance creation
# ============================================================

def create_analysis_aggregator(config: BotConfig = None, base_path: str = None) -> AnalysisAggregator:
    """Create AnalysisAggregator instance with optional custom config and base_path"""
    return AnalysisAggregator(config, base_path=base_path)



# ============================================================
# 🔹 Standalone test örneği
# ============================================================
if __name__ == "__main__":
    import asyncio

    async def test_analysis():
        # Create aggregator with config
        agg = create_analysis_aggregator()
        
        try:
            # Test with config values
            user_list = ["user_A", "user_B"]
            symbol = agg.get_default_symbols(1)[0] if agg.get_default_symbols() else "BTCUSDT"
            
            print(f"🔧 Config: {agg.analysis_config.MAX_CONCURRENT} max concurrent, {agg.analysis_config.DEFAULT_INTERVAL} default interval")
            print(f"📊 Available symbols: {len(agg.get_available_symbols())}")
            
            # Performance test
            result = await agg.run_multi_user(user_list, symbol, "1h")
            print("📊 Multi-user sonuçları:")
            for user, res in result.items():
                print(f"\n🧩 {user}:")
                for k, v in list(res.items())[:5]:
                    print(f"   {k}: {v}")
            
            # Health check
            health = await agg.get_system_health()
            print(f"🏥 System Health: {health['overall_status']}")
            print(f"📈 Performance: {health['performance_stats']}")
            
            # Test streaming
            print("\n🔄 Testing streaming calculation:")
            modules = agg.schema.list_modules()[:3]
            async for stream_result in agg.stream_metrics_calculation("user_A", modules, symbol):
                print(f"Stream result: {list(stream_result.keys())}")
            
        finally:
            # Cleanup
            agg.cleanup()

    asyncio.run(test_analysis())
   
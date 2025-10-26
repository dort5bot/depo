# analysis/analysis_core.py

import os
import asyncio
import logging
import importlib.util
import time
import hashlib
import gc
from time import perf_counter
from pydantic import ValidationError
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# Schema imports
from .analysis_schema_manager import (
    load_analysis_schema,
    load_module_run_function,
    resolve_module_path,
    AnalysisModule,
    AnalysisSchema,
    CircuitBreaker
)

from .analysis_base_module import BaseAnalysisModule
from .analysis_base_module import legacy_compatible
from .analysis_helpers import AnalysisHelpers, AnalysisOutput, AnalysisUtilities, analysis_helpers
from analysis.composite.composite_engine import CompositeScoreEngine

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisPriority(Enum):
    BASIC = "basic"
    PRO = "pro" 
    EXPERT = "expert"

class AnalysisStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AnalysisResult:
    module_name: str
    command: str
    status: AnalysisStatus
    data: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    priority: Optional[str] = None

@dataclass
class AggregatedResult:
    symbol: str
    results: List[AnalysisResult]
    total_execution_time: float
    success_count: int
    failed_count: int
    overall_score: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class UserSession:
    user_id: str
    lock: asyncio.Lock
    created_at: float = field(default_factory=time.time)

@dataclass
class UserLimit:
    max_concurrent: int
    max_per_minute: int 
    max_modules_per_request: int
    _current_minute: int = 0
    _minute_count: int = 0
    
    def can_execute(self, module_count: int) -> bool:
        current_minute = time.time() // 60
        if current_minute != self._current_minute:
            self._current_minute = current_minute
            self._minute_count = 0
            
        return (module_count <= self.max_modules_per_request and 
                self._minute_count + module_count <= self.max_per_minute)
    
    def record_execution(self, module_count: int):
        self._minute_count += module_count

class AnalysisAggregator:
    _instance: Optional['AnalysisAggregator'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._lock = asyncio.Lock()
        self.schema: Optional[AnalysisSchema] = None
        self._module_cache: Dict[str, Any] = {}
        self._result_cache: Dict[str, AggregatedResult] = {}
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        
        # Performance monitoring
        self._execution_times: List[float] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # Yeni özellikler
        self._module_instances: Dict[str, Any] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Birleşik skorlar
        self.composite_engine = CompositeScoreEngine(self)
        
        self.helpers = analysis_helpers
        self._initialized = True
        logger.info("AnalysisAggregator initialized successfully")

    # analysis/analysis_helpers.py/ async
    async def validate_and_normalize_output(
        self, 
        raw_data: Dict[str, Any], 
        module_name: str
    ) -> Dict[str, Any]:
        """✅ ASYNC VALIDATION"""
        try:
            # CPU-bound işi thread pool'da yap
            loop = asyncio.get_event_loop()
            
            def sync_validation():
                data = {k: v for k, v in raw_data.items() if k != "module"}
                validated_output = AnalysisOutput(
                    **raw_data,
                    module=module_name,
                    timestamp=time.time()
                )
                return validated_output.dict()
            
            return await loop.run_in_executor(None, sync_validation)
            
        except ValidationError as e:
            logger.warning(f"Output validation failed for {module_name}: {e}")
            return create_fallback_output(module_name, f"Validation error: {str(e)}")
            

    # analysis_core.py - bu metotları helper
    # ✅ Key metotları- tutarlı kullanım için
    def _get_module_key(self, module_file: str) -> str:
        return AnalysisHelpers.get_module_key(module_file)

    def _get_module_instance_key(self, module_file: str) -> str:
        return AnalysisHelpers.get_module_instance_key(module_file)

    def _get_circuit_breaker_key(self, module_file: str) -> str:
        """Circuit breaker anahtarı - TUTARLI KULLANIM İÇİN"""
        return AnalysisHelpers.get_circuit_breaker_key(module_file)




    # ✅ LOCK YÖNETİMİ
    @asynccontextmanager
    async def _get_module_lock(self, module_name: str):
        if module_name not in self._execution_locks:
            self._execution_locks[module_name] = asyncio.Lock()
        lock = self._execution_locks[module_name]
        async with lock:
            yield

    # ✅ LIFECYCLE MANAGEMENT
    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Aggregator started")

    async def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("Aggregator cleanup task was cancelled")
        logger.info("Aggregator stopped")

    # ✅ CACHE MANAGEMENT
    def _get_cache_key(
        self, 
        symbol: str, 
        module_name: str, 
        priority: Optional[str] = None, 
        user_id: Optional[str] = None
    ) -> str:
        """Sembol + modül + öncelik + kullanıcı bazlı cache anahtarı üretir"""
        normalized_name = self._get_module_key(module_name)
        key_string = f"{(symbol or 'unknown').upper()}:{normalized_name}:{priority or 'default'}:{user_id or 'anon'}"
        return hashlib.md5(key_string.encode()).hexdigest()

    # ✅ MODULE LOADING

    async def _load_module_function(self, module_file: str):
        """✅ MODÜL FONKSİYONUNU ASYNC YÜKLE"""
        module_key = self._get_module_key(module_file)
        cache_key = f"module_{module_key}"

        if cache_key in self._module_cache:
            self._cache_hits += 1
            return self._module_cache[cache_key]

        self._cache_misses += 1
        try:
            resolved_path = resolve_module_path(module_file)
            
            # ✅ GÜVENLİ PATH KONTROLÜ
            allowed_prefix = os.path.abspath("analysis/modules/")
            if not resolved_path.startswith(allowed_prefix):
                raise PermissionError(f"Unauthorized module path: {resolved_path}")

            # ✅ ASYNC YÜKLEME
            run_function = await self._load_module_async(resolved_path)
            self._module_cache[cache_key] = run_function
            return run_function
            
        except (ImportError, AttributeError, FileNotFoundError) as e:
            logger.error(f"Module load failed for {module_file}: {str(e)}")
            raise

    async def _load_module_async(self, module_path: str):
        """✅ ASYNC MODÜL YÜKLEME"""
        from .analysis_schema_manager import load_module_run_function
        
        # Mevcut sync fonksiyonu async wrapper'a al
        sync_function = load_module_run_function(module_path)
        
        # Eğer zaten async ise direkt dön
        if asyncio.iscoroutinefunction(sync_function):
            return sync_function
        
        # Sync fonksiyonu async wrapper'a al
        async def async_wrapper(symbol: str, priority: Optional[str] = None):
            # Sync fonksiyonu thread pool'da çalıştır
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: sync_function(symbol, priority)
            )
        
        return async_wrapper



    # ✅ TEKİL ANALİZ ÇALIŞTIRMA/ async
    async def run_single_analysis(
        self, 
        module: AnalysisModule, 
        symbol: str, 
        priority: Optional[str] = None
    ) -> AnalysisResult:
        """✅ TAM ASYNC ANALİZ ÇALIŞTIRMA"""
        start_time = time.time()
        result = AnalysisResult(
            module_name=module.name,
            command=module.command,
            status=AnalysisStatus.PENDING,
            data={},
            execution_time=0.0,
            priority=priority
        )
        
        # ✅ STANDARDİZE CIRCUIT BREAKER
        module_key = self._get_module_key(module.file)
        cb_key = self._get_circuit_breaker_key(module.file)
        
        if cb_key not in self._circuit_breakers: 
            self._circuit_breakers[cb_key] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=(Exception,)
            )

        async def execute_analysis():
            async with self._get_module_lock(module_key):
                logger.info(f"Starting ASYNC analysis: {module.name} for {symbol}")
                
                # ✅ MODÜL INSTANCE KONTROLÜ
                instance_key = self._get_module_instance_key(module.file)
                if instance_key in self._module_instances:
                    module_instance = self._module_instances[instance_key]
                    if hasattr(module_instance, "compute_metrics"):
                        try:
                            # ✅ ASYNC METOD ÇAĞRISI
                            analysis_data = await module_instance.compute_metrics(symbol, priority)
                            return analysis_data
                        except Exception as e:
                            logger.warning(f"Module instance failed: {e}")

                # ✅ ASYNC RUN FONKSİYONU
                run_function = await self._load_module_function(module.file)
                analysis_data = await run_function(symbol=symbol, priority=priority)

                if not isinstance(analysis_data, dict):
                    raise ValueError(f"Invalid result type: {type(analysis_data)}")

                return analysis_data
        
        async def fallback_analysis():
            logger.warning(f"Using async fallback for {module.name}")
            return create_fallback_output(module.name, "Circuit breaker activated")
        
        try:
            result.status = AnalysisStatus.RUNNING
            
            # ✅ ASYNC CIRCUIT BREAKER
            raw_data = await self._circuit_breakers[cb_key].execute_with_fallback(
                execute_analysis, 
                fallback_analysis
            )

            # ✅ ASYNC VALIDATION
            result.data = await self.validate_and_normalize_output(raw_data, module.name)
            result.status = AnalysisStatus.COMPLETED

        except asyncio.CancelledError:
            result.status = AnalysisStatus.CANCELLED
            result.error = "Analysis cancelled"
            logger.warning(f"Analysis cancelled: {module.name}")
            raise
            
        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.error = f"Analysis failed: {str(e)}"
            logger.error(f"Analysis failed for {module.name}: {str(e)}", exc_info=True)
            
        finally:
            result.execution_time = time.time() - start_time
            self._execution_times.append(result.execution_time)

            if result.status == AnalysisStatus.COMPLETED:
                logger.info(f"[{module.name}] → COMPLETED in {result.execution_time:.2f}s")
            else:
                logger.warning(f"[{module.name}] → {result.status.value.upper()}")

        return result
        






    # ✅ MODÜL ANALİZ HELPER/ async

    async def get_module_analysis(self, module_name: str, symbol: str):
        """✅ ASYNC MODÜL ANALİZ"""
        
        if not self.schema:
            self.schema = load_analysis_schema()
        
        cache_key = f"mod:{module_name}:{symbol}"

        # ✅ ASYNC CACHE KONTROLÜ
        if cache_key in self._result_cache:
            cached_result = self._result_cache[cache_key]
            if validate_output(cached_result):
                return cached_result
            else:
                del self._result_cache[cache_key]

        # Modül bulma
        module = next(
            (m for m in self.schema.modules if m.name == module_name or m.command == module_name),
            None
        )
        if not module:
            return create_fallback_output(module_name, "Module not found")

        # ✅ ASYNC ANALİZ
        res = await self.run_single_analysis(module, symbol)
        out = res.data if isinstance(res.data, dict) else {'score': 0.5}

        # Cache'e kaydet
        self._result_cache[cache_key] = out

        return out
        


    # ✅ TOPLU ANALİZ ÇALIŞTIRMA/ async
    async def run_all_analyses(self, symbol: str, priority: Optional[str] = None) -> AggregatedResult:
        """✅ OPTİMİZE ASYNC TOPLU ANALİZ"""
        if not self.schema:
            self.schema = load_analysis_schema()
        
        start = time.time()
        
        # ✅ ADAPTİV SEMAPHORE - sistem yüküne göre
        concurrency_limit = min(10, len(self.schema.modules))
        sem = asyncio.Semaphore(concurrency_limit)
        
        async def safe_run(m):
            async with sem:
                return await self.run_single_analysis(m, symbol, priority)

        # ✅ ASYNC GATHER OPTİMİZASYONU
        tasks = [safe_run(m) for m in self.schema.modules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Hata işleme
        valid_results = [r for r in results if isinstance(r, AnalysisResult)]

        return AggregatedResult(
            symbol=symbol,
            results=valid_results,
            total_execution_time=time.time() - start,
            success_count=sum(1 for r in valid_results if r.status == AnalysisStatus.COMPLETED),
            failed_count=sum(1 for r in valid_results if r.status == AnalysisStatus.FAILED)
        )
        


    # ✅ BİRLEŞİK ANALİZ
    async def get_comprehensive_analysis(self, symbol: str):
        """Kapsamlı analiz sonucu"""
        module_results = await self.run_all_analyses(symbol)
        composite_scores = await self.composite_engine.calculate_composite_scores(symbol)
        
        return {
            'symbol': symbol,
            'module_analyses': module_results,
            'composite_scores': composite_scores['composite_scores'],
            'summary': self._generate_summary(composite_scores),
            'timestamp': composite_scores['timestamp']
        }

    def _generate_summary(self, composite_scores: Dict) -> Dict:
        """Özet bilgi oluştur"""
        return {
            'overall_score': composite_scores.get('overall_score', 0.5),
            'trend_strength': composite_scores.get('trend_strength', {}).get('score', 0.5),
            'timestamp': time.time()
        }

    async def get_trend_strength(self, symbol: str):
        """Sadece trend strength skoru al"""
        return await self.composite_engine.calculate_single_score('trend_strength', symbol)



    # ✅ CLEANUP VE HEALTH CHECK
    async def _periodic_cleanup(self):
        """Geliştirilmiş resource cleanup"""
        while self._is_running:
            try:
                await asyncio.sleep(300)
                current_time = time.time()

                # Result cache cleanup
                for key in list(self._result_cache.keys()):
                    result = self._result_cache[key]
                    if current_time - result.created_at > 600:
                        del self._result_cache[key]

                # ✅ DÜZELTİLDİ: Module ve circuit breaker cleanup - TUTARLI YAPISI
                if self.schema:
                    valid_module_keys = [
                        self._get_module_key(m.file)
                        for m in self.schema.modules
                    ]
                    valid_cb_keys = [
                        self._get_circuit_breaker_key(m.file)
                        for m in self.schema.modules
                    ]

                    # Circuit breaker cleanup
                    for cb_key in list(self._circuit_breakers.keys()):
                        if cb_key not in valid_cb_keys:
                            del self._circuit_breakers[cb_key]

                    # Module instance cleanup
                    for instance_key in list(self._module_instances.keys()):
                        module_base_key = instance_key.split('_')[0]
                        if module_base_key not in valid_module_keys:
                            del self._module_instances[instance_key]

                # Performance tracking cleanup
                if len(self._execution_times) > 1000:
                    self._execution_times = self._execution_times[-500:]

                gc.collect()
                logger.debug("Cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")
                


    # ✅ HEALTH CHECK HELPER METOTLARI
    async def _check_module_health(self) -> bool:
        """Modül sağlık kontrolü"""
        try:
            if not self.schema:
                self.schema = load_analysis_schema()
            return len(self.schema.modules) > 0
        except Exception as e:
            logger.error(f"Module health check failed: {e}")
            return False

    def _check_cache_health(self) -> bool:
        """Cache sağlık kontrolü"""
        cache_size = len(self._result_cache) + len(self._module_cache)
        return cache_size < 1000  # Makul limit

    def _get_memory_usage(self) -> bool:
        """Bellek kullanım kontrolü"""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            return memory_usage < 500  # 500MB limit
        except ImportError:
            logger.warning("psutil not available, memory check skipped")
            return True  # Kritik değil


    async def _check_api_connectivity(self) -> bool:
        """API bağlantı kontrolü"""
        try:
            # Basit bir bağlantı testi
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False

    async def comprehensive_health_check(self):
        """Kapsamlı sistem sağlık kontrolü"""
        checks = {
            "module_health": await self._check_module_health(),
            "cache_health": self._check_cache_health(),
            "memory_usage": self._get_memory_usage(),
            "api_connectivity": await self._check_api_connectivity()
        }
        return all(checks.values()), checks

# ✅ GENİŞLETİLMİŞ AGGREGATOR SINIFLARI
class UserAwareAnalysisAggregator(AnalysisAggregator):
    def __init__(self):
        super().__init__()
        self._user_sessions: Dict[str, UserSession] = {}
        self._user_limits: Dict[str, UserLimit] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

    # Kullanıcı session’ı getir veya oluştur
    def _get_user_session(self, user_id: str) -> UserSession:
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = UserSession(
                user_id=user_id,
                lock=asyncio.Lock()
            )
        return self._user_sessions[user_id]

    # Kullanıcı limiti getir veya oluştur
    def _get_user_limit(self, user_id: str) -> UserLimit:
        if user_id not in self._user_limits:
            self._user_limits[user_id] = UserLimit(
                max_concurrent=5,
                max_per_minute=30,
                max_modules_per_request=10
            )
        return self._user_limits[user_id]

    # Kullanıcı semaphore’u getir veya oluştur
    def _get_semaphore(self, user_id: str) -> asyncio.Semaphore:
        if user_id not in self._semaphores:
            self._semaphores[user_id] = asyncio.Semaphore(3)
        return self._semaphores[user_id]

    # Asıl kullanıcı analizini çalıştır
    async def _run_user_analysis(self, user_session: UserSession, symbol: str, modules: List[str]):
        if not self.schema:
            self.schema = load_analysis_schema()

        # Modülleri filtrele
        active_modules = [m for m in self.schema.modules if m.name in modules]
        if not active_modules:
            raise Exception("No valid modules found for analysis")

        # Async task oluştur
        tasks = [self.run_single_analysis(m, symbol, priority="normal") for m in active_modules]

        start_time = perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = perf_counter()

        valid_results = [r for r in results if isinstance(r, AnalysisResult)]
        failed_count = sum(1 for r in results if isinstance(r, Exception) or getattr(r, 'status', None) == AnalysisStatus.FAILED)
        total_time = end_time - start_time

        return AggregatedResult(
            symbol=symbol,
            results=valid_results,
            total_execution_time=total_time,
            success_count=len(valid_results) - failed_count,
            failed_count=failed_count
        )

    # Kullanıcı bazlı analiz workflow’u
    async def run_analysis_for_user(self, user_id: str, symbol: str, modules: List[str]):
        user_session = self._get_user_session(user_id)
        user_limit = self._get_user_limit(user_id)
        semaphore = self._get_semaphore(user_id)

        # Kullanıcı limiti kontrolü
        if not user_limit.can_execute(len(modules)):
            raise Exception("Too many requests")

        async with user_session.lock:
            async with semaphore:
                # Analizi çalıştır
                result = await self._run_user_analysis(user_session, symbol, modules)

                # Kullanıcının kullanımını kaydet
                user_limit.record_execution(len(modules))
                return result



class PerformanceOptimizedAggregator(AnalysisAggregator):
    async def run_priority_batch(self, symbols: List[str], priority_modules: List[str]):
        """Öncelikli batch processing"""
        semaphore = asyncio.Semaphore(10)
        
        async def process_symbol(symbol):
            async with semaphore:
                tasks = []
                for module_name in priority_modules:
                    module = self._get_module(module_name)
                    if module:
                        task = self.run_single_analysis(module, symbol, "high")
                        tasks.append(task)
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        return await asyncio.gather(*[process_symbol(s) for s in symbols])
    
    def _get_module(self, module_name: str) -> Optional[AnalysisModule]:
        """Module ismine göre modül bul"""
        if not self.schema:
            self.schema = load_analysis_schema()
        return next((m for m in self.schema.modules if m.name == module_name), None)

class EnhancedAnalysisAggregator(AnalysisAggregator):
    def __init__(self):
        super().__init__()
        self._user_limits: Dict[str, UserLimit] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        
    async def run_analysis_for_user(self, user_id: str, symbol: str, module_names: List[str], 
                                  priority: Optional[str] = None) -> AggregatedResult:
        """Kullanıcı bazlı limitli analiz"""
        user_limit = self._get_user_limit(user_id)
        if not user_limit.can_execute(len(module_names)):
            raise Exception("Too many requests")  # HTTPException yerine Exception
            
        semaphore = self._get_semaphore(user_id)
        async with semaphore:
            modules = [m for m in self.schema.modules if m.name in module_names]
            tasks = [self.run_single_analysis(m, symbol, priority) for m in modules]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            user_limit.record_execution(len(modules))
            
            valid_results = [r for r in results if isinstance(r, AnalysisResult)]
            total_time = sum(r.execution_time for r in valid_results)
            
            return AggregatedResult(
                symbol=symbol,
                results=valid_results,
                total_execution_time=total_time,
                success_count=sum(1 for r in valid_results if r.status == AnalysisStatus.COMPLETED),
                failed_count=sum(1 for r in valid_results if r.status == AnalysisStatus.FAILED)
            )
    

    async def calculate_weighted_score(
        self, 
        symbol: str, 
        component_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Ağırlıklı skor hesapla"""
        try:
            # Component analizlerini al
            component_scores = {}
            for comp_name in component_weights.keys():
                analysis_result = await self.get_module_analysis(comp_name, symbol)
                component_scores[comp_name] = analysis_result.get('score', 0.5)
            
            # Ağırlıklı ortalama hesapla
            normalized_weights = AnalysisHelpers.normalize_weights(component_weights)
            final_score = AnalysisHelpers.calculate_weights(component_scores, normalized_weights)
            
            return {
                'score': final_score,
                'components': component_scores,
                'weights': normalized_weights,
                'timestamp': AnalysisHelpers.get_timestamp()
            }
        except Exception as e:
            logger.error(f"Weighted score calculation failed: {e}")
            return AnalysisHelpers.create_fallback_output("weighted_score", str(e))
        
      
    def _get_user_limit(self, user_id: str) -> UserLimit:
        if user_id not in self._user_limits:
            self._user_limits[user_id] = UserLimit(
                max_concurrent=5,
                max_per_minute=30,
                max_modules_per_request=10
            )
        return self._user_limits[user_id]
    
    def _get_semaphore(self, user_id: str) -> asyncio.Semaphore:
        if user_id not in self._semaphores:
            self._semaphores[user_id] = asyncio.Semaphore(3)
        return self._semaphores[user_id]



# ✅ GLOBAL INSTANCE
aggregator = AnalysisAggregator()

async def get_aggregator() -> AnalysisAggregator:
    """Dependency injection için aggregator instance'ı"""
    if not aggregator._is_running:
        await aggregator.start()
    return aggregator
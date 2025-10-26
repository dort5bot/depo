# analysis/analysis_schema_manager.py
# Analysis Helpers ile Tam Uyumlu

"""
Module + EnhancedModule → AnalysisModule olarak birleştirildi.
Analysis Helpers ile tam entegre versiyon
"""
import os
import asyncio
import logging
import yaml
import importlib.util
from pathlib import Path
from enum import Enum
from typing import Dict, Any, List, Optional, Literal, Union, Callable, Type, Tuple
from pydantic import BaseModel, validator


from analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_helpers import AnalysisHelpers  # ✅ EKLENDİ

logger = logging.getLogger(__name__)

# ✅ ANALYSIS_HELPERS UYUMLU PATH TANIMLARI
ANALYSIS_BASE_PATH = AnalysisHelpers.resolve_analysis_path("")  # analysis klasörü
CONFIG_BASE_PATH = ANALYSIS_BASE_PATH / "config"

# -----------------------------#
# 🧩 ENUM & TİP TANIMLARI
# -----------------------------#
PriorityLevel = Literal["*", "**", "***"]

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

class Metric(BaseModel):
    name: str
    priority: PriorityLevel
    
    # ✅ Ek validation
    @validator('name')
    def validate_metric_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Metric name can only contain alphanumeric characters and underscores')
        return v.lower()


# -----------------------------#
# 📦 MODÜL ŞEMASI
# -----------------------------#
class AnalysisModule(BaseModel):
    name: str
    file: str
    command: str
    api_type: str
    endpoints: List[str]
    methods: List[Literal["GET", "POST", "PUT", "DELETE", "WebSocket"]]

    classical_metrics: Optional[List[Union[str, Metric]]] = []
    professional_metrics: Optional[List[Metric]] = []
    composite_metrics: Optional[List[str]] = []

    development_notes: Optional[str] = None
    objective: Optional[str] = None
    output_type: Optional[str] = None

    lifecycle: ModuleLifecycle = ModuleLifecycle.DEVELOPMENT
    parallel_mode: ParallelMode = ParallelMode.BATCH
    config_file: Optional[str] = None
    required_metrics: List[str] = []
    outputs: List[str] = []
    version: str = "1.0.0"
    dependencies: List[str] = []

    # ✅ YAML'den gelen ekstra alanlar:
    config: Optional[str] = None
    command_aliases: List[str] = []
    job_type: Optional[str] = None
    description: Optional[str] = None
    maintainer: Optional[str] = None

    class Config:
        extra = "ignore"  # ✅ Fazla alanlar hataya neden olmaz


class AnalysisSchema(BaseModel):
    modules: List[AnalysisModule]


# ---------Singleton Schema Manager--------------------#
class SchemaManager:
    _instance = None
    _schema = None
    _lock = asyncio.Lock()  # ✅ ASYNC LOCK
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def get_schema(cls) -> AnalysisSchema:
        """✅ ASYNC SCHEMA GETTER"""
        async with cls._lock:
            if cls._schema is None:
                cls._schema = await load_analysis_schema()
            return cls._schema
    
    @classmethod
    async def reload_schema(cls):
        """✅ ASYNC SCHEMA RELOAD"""
        async with cls._lock:
            cls._schema = await load_analysis_schema()
            logger.info("Schema reloaded successfully")

# -----------------------------#
# 🔧 HELPER: Dinamik yükleme - ANALYSIS_HELPERS UYUMLU
# -----------------------------#
def resolve_module_path(module_file: str) -> str:
    """✅ ANALYSIS_HELPERS UYUMLU PATH RESOLUTION"""
    try:
        # Güvenli path resolution
        resolved_path = AnalysisHelpers.resolve_analysis_path(module_file.strip())
        
        # Ek güvenlik kontrolü - sadece .py dosyaları
        if not resolved_path.suffix == '.py':
            raise PermissionError(f"Invalid file type: {resolved_path.suffix}")
            
        return str(resolved_path)
        
    except PermissionError as e:
        logger.error(f"Security violation in module path: {e}")
        raise
    except Exception as e:
        logger.error(f"Module path resolution failed: {e}")
        raise FileNotFoundError(f"Module not found: {module_file}")

def load_python_module(module_path: str):
    """Python modülünü yükle"""
    module_name = AnalysisHelpers.get_module_key(module_path)  # ✅ HELPER KULLANIMI
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            raise ImportError(f"Module could not be loaded: {module_path}")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        logger.debug(f"Module loaded successfully: {module_name}")
        return mod
        
    except Exception as e:
        logger.error(f"Failed to load module {module_name}: {e}")
        raise ImportError(f"Module loading failed: {module_path}")

def find_analysis_module_class(mod) -> Optional[Type[BaseAnalysisModule]]:
    """Analiz modülü sınıfını bul"""
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type) and 
            issubclass(attr, BaseAnalysisModule) and 
            attr != BaseAnalysisModule):
            
            logger.debug(f"Found analysis module class: {attr.__name__}")
            return attr
    
    logger.warning("No analysis module class found")
    return None


# analysis_schema_manager.py -async

async def load_module_run_function(module_file: str) -> Callable:
    """✅ ASYNC MODÜL RUN FONKSİYONU YÜKLE"""
    try:
        module_path = resolve_module_path(module_file)
        mod = await load_python_module_async(module_path)

        # ✅ Önce run fonksiyonunu kontrol et
        if hasattr(mod, "run"):
            run_func = mod.run
            logger.debug(f"Using run() function from {module_file}")
            
            # Eğer async değilse, async wrapper'a al
            if not asyncio.iscoroutinefunction(run_func):
                async def async_run_wrapper(symbol: str, priority: Optional[str] = None):
                    return run_func(symbol, priority)
                return async_run_wrapper
            return run_func

        # ✅ Sonra BaseAnalysisModule sınıfını ara
        cls = find_analysis_module_class(mod)
        if cls:
            logger.debug(f"Using BaseAnalysisModule class from {module_file}")
            
            async def run_wrapper(symbol: str, priority: Optional[str] = None):
                instance = cls()
                # compute_metrics async olmalı
                if hasattr(instance, "compute_metrics") and asyncio.iscoroutinefunction(instance.compute_metrics):
                    return await instance.compute_metrics(symbol, priority)
                else:
                    # Sync ise thread pool'da çalıştır
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: instance.compute_metrics(symbol, priority) if hasattr(instance, "compute_metrics") 
                        else {"score": 0.5, "error": "No compute_metrics method"}
                    )
            
            return run_wrapper

        raise AttributeError(f"'run()' veya BaseAnalysisModule sınıfı bulunamadı: {module_file}")
        
    except Exception as e:
        logger.error(f"Failed to load module run function for {module_file}: {e}")
        raise

async def load_python_module_async(module_path: str):
    """✅ ASYNC PYTHON MODÜL YÜKLEME"""
    module_name = AnalysisHelpers.get_module_key(module_path)
    
    try:
        # Modül yükleme CPU-bound, thread pool'da yap
        loop = asyncio.get_event_loop()
        
        def sync_load():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Module could not be loaded: {module_path}")
            
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        
        mod = await loop.run_in_executor(None, sync_load)
        logger.debug(f"Module loaded successfully: {module_name}")
        return mod
        
    except Exception as e:
        logger.error(f"Failed to load module {module_name}: {e}")
        raise ImportError(f"Module loading failed: {module_path}")
        
        
# -----------------------------#
# 🏭 ModuleFactory + cache - ANALYSIS_HELPERS UYUMLU
# -----------------------------#
class ModuleFactory:
    _module_cache: Dict[str, Type[BaseAnalysisModule]] = {}
    _lock = asyncio.Lock()  # ✅ ASYNC LOCK
    
    @staticmethod
    async def create_module(module_name: str, config: Dict[str, Any]) -> BaseAnalysisModule:
        """✅ ASYNC MODÜL INSTANCE OLUŞTUR"""
        cache_key = AnalysisHelpers.get_module_key(module_name)
        
        async with ModuleFactory._lock:
            if cache_key in ModuleFactory._module_cache:
                logger.debug(f"Using cached module: {cache_key}")
                cls = ModuleFactory._module_cache[cache_key]
                return cls(config)
                
        try:
            module_file = f"analysis_{module_name.lower()}.py"
            module_path = resolve_module_path(module_file)
            mod = await load_python_module_async(module_path)
            cls = find_analysis_module_class(mod)
            
            if cls:
                async with ModuleFactory._lock:
                    ModuleFactory._module_cache[cache_key] = cls
                logger.info(f"Module factory created: {module_name}")
                return cls(config)
                
            raise AttributeError(f"Module class not found in {module_file}")
            
        except Exception as e:
            logger.error(f"Module factory failed for {module_name}: {e}")
            raise

# -----------------------------#
# 🧱 Circuit Breaker - ANALYSIS_HELPERS UYUMLU
# -----------------------------#
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30,
                 expected_exception: Tuple[Type[Exception], ...] = (Exception,)):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()  # ✅ ASYNC LOCK

    async def execute_with_fallback(self, command: callable, fallback: callable):
        """✅ ASYNC CIRCUIT BREAKER EXECUTION"""
        async with self._lock:  # ✅ THREAD-SAFE
            current_time = AnalysisHelpers.get_timestamp()
            
            if self.state == "OPEN":
                if current_time - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker half-open, testing recovery")
                else:
                    self.logger.warning("Circuit breaker OPEN, using fallback")
                    return await fallback()

            try:
                # ✅ COMMAND ASYNC OLMALI
                result = await command()
                await self._on_success()
                return result
            except self.expected_exception as e:
                await self._on_failure(current_time)
                self.logger.warning(f"Circuit breaker failure: {e}")
                return await fallback()

    async def _on_success(self):
        """✅ ASYNC SUCCESS HANDLER"""
        async with self._lock:
            self.failures = 0
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")

    async def _on_failure(self, failure_time: float = None):
        """✅ ASYNC FAILURE HANDLER"""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = failure_time or AnalysisHelpers.get_timestamp()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.error(f"Circuit breaker OPENED after {self.failures} failures")
    
    # ✅ ASYNC CONTEXT MANAGER
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.expected_exception):
            await self._on_failure()
        else:
            await self._on_success()
            

# -----------------------------#
# 📥 Yükleyici + Error Handling -async- ANALYSIS_HELPERS UYUMLU
# -----------------------------#
async def load_analysis_schema(yaml_path: str = "analysis_metric_schema.yaml") -> AnalysisSchema:
    """✅ ASYNC SCHEMA YÜKLE"""
    try:
        # ✅ ASYNC FILE READING
        schema_path = AnalysisHelpers.resolve_analysis_path(yaml_path)
        
        loop = asyncio.get_event_loop()
        
        def sync_load():
            with open(schema_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        
        data = await loop.run_in_executor(None, sync_load)
        
        # ✅ Validation öncesi check
        if not data or "modules" not in data:
            raise ValueError("Invalid schema format: 'modules' key missing")
            
        schema = AnalysisSchema(**data)
        logger.info(f"Analysis schema loaded successfully: {len(schema.modules)} modules")
        return schema
        
    except FileNotFoundError:
        logger.error(f"Schema file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading schema: {e}")
        raise
        

# -----------------------------#
# 🔎 Filtreleme Fonksiyonları
# -----------------------------#

def filter_modules_by_priority(schema: AnalysisSchema, priority: PriorityLevel) -> List[AnalysisModule]:
    return [m for m in schema.modules if any(metric.priority == priority for metric in m.professional_metrics or [])]


def get_metrics_by_priority(module: AnalysisModule, priority: PriorityLevel) -> List[Metric]:
    return [m for m in (module.professional_metrics or []) if m.priority == priority]


def get_module_by_field(schema: AnalysisSchema, field: str, value: str) -> Optional[AnalysisModule]:
    return next((m for m in schema.modules if getattr(m, field, None) == value), None)


# get_module_by_field İçin Yardımcı Metodlar
def get_module_by_name(schema: AnalysisSchema, name: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "name", name)


def get_module_by_command(schema: AnalysisSchema, command: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "command", command)


def get_module_by_file(schema: AnalysisSchema, file: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "file", file)


def get_modules_by_lifecycle(schema: AnalysisSchema, lifecycle: ModuleLifecycle) -> List[AnalysisModule]:
    return [m for m in schema.modules if m.lifecycle == lifecycle]


def get_module_dependencies(schema: AnalysisSchema, module_name: str) -> List[str]:
    module = get_module_by_field(schema, "name", module_name)
    return module.dependencies if module else []


# -----------------------------#
# 👤 Kullanıcı Seviyesi
# -----------------------------#

USER_LEVEL_PRIORITY = {
    "basic": "*",
    "pro": "**", 
    "expert": "***"
}

def get_modules_for_user_level(schema: AnalysisSchema, level: str) -> List[AnalysisModule]:
    priority = USER_LEVEL_PRIORITY.get(level.lower())
    if priority:
        return filter_modules_by_priority(schema, priority)
    return []


# -----------------------------#
# 🧪 ÖRNEK TEST - ANALYSIS_HELPERS UYUMLU
# -----------------------------#

if __name__ == "__main__":
    try:
        schema = load_analysis_schema()
        
        print("📊 Tüm modüller:")
        for module in schema.modules:
            print(f" - [{module.command}] {module.name} (file: {module.file})")

        print("\n🎯 *** öncelikli metriklere sahip modüller:")
        for mod in filter_modules_by_priority(schema, "***"):
            metrics = get_metrics_by_priority(mod, "***")
            print(f"🔍 {mod.name} ({mod.command})")
            for m in metrics:
                print(f"   - {m.name} ({m.priority})")

        print("\n👤 Pro seviye kullanıcı modülleri:")
        for mod in get_modules_for_user_level(schema, "pro"):
            print(f" - {mod.name} ({mod.command})")
            
        print(f"\n✅ Schema test completed at {AnalysisHelpers.get_iso_timestamp()}")
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
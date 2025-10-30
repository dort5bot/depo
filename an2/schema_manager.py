# ===============================================================
# analysis/schema_manager.py
# ---------------------------------------------------------------
#  Module Registry & Schema Manager
#  v2025.1 — YAML yapılarını (schemas/module_registry.yaml) yükler,
#  doğrular ve kolay erişim arayüzü sağlar.
# ===============================================================

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | schema_manager | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ===============================================================
#  🔹 SchemaManager Class
# ===============================================================
class SchemaManager:
    """
    Module registry YAML'ını yükler, doğrular ve erişim sağlar.
    """

    def __init__(self, schema_path: Optional[str] = None):
        default_path = Path(__file__).resolve().parent.parent / "analysis" / "schemas" / "module_registry.yaml"
        self.schema_path = Path(schema_path or default_path)
        self.registry: Dict[str, Any] = {}
        self.modules: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self._load_schema()

    # -----------------------------------------------------------
    #  Schema Loading & Validation
    # -----------------------------------------------------------
    def _load_schema(self) -> None:
        """YAML schema dosyasını yükler ve doğrular"""
        if not self.schema_path.exists():
            logger.error(f"❌ Schema dosyası bulunamadı: {self.schema_path}")
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML yükleme hatası: {e}")
            raise

        if not data or "modules" not in data:
            logger.error("❌ Geçersiz YAML: 'modules' alanı eksik.")
            raise ValueError("Invalid schema YAML: missing 'modules' section.")

        self.meta = data.get("meta", {})
        self.modules = data["modules"]
        self.registry = data  # Tüm veriyi registry'de tut
        self._validate_modules()
        logger.info(f"✅ {len(self.modules)} modül başarıyla yüklendi ({self.meta.get('version', 'unknown')})")

    def _validate_modules(self) -> None:
        """Modül tanımlarını doğrular"""
        required_fields = [
            "name", "multi_user", "data_source", "api_type", "endpoint", 
            "job_type", "parallel_mode", "data_model", "compute_intensity"
        ]
        
        valid_parallel_modes = ["async", "sync", "batch", "stream"]
        valid_data_models = ["pandas", "numpy", "polars"]
        valid_intensities = ["low", "medium", "high"]
        valid_api_types = ["public", "private", "internal"]
        valid_job_types = ["metric_calculation", "composite_scoring", "data_aggregation"]
        
        for mod in self.modules:
            # Eksik alan kontrolü
            missing = [f for f in required_fields if f not in mod]
            if missing:
                logger.warning(f"⚠️ Modül {mod.get('name', 'UNKNOWN')} eksik alanlar: {missing}")
            
            # Geçerli değer kontrolü
            if mod.get('parallel_mode') not in valid_parallel_modes:
                logger.warning(f"⚠️ Modül {mod['name']} geçersiz parallel_mode: {mod.get('parallel_mode')}")
            
            if mod.get('data_model') not in valid_data_models:
                logger.warning(f"⚠️ Modül {mod['name']} geçersiz data_model: {mod.get('data_model')}")
                
            if mod.get('compute_intensity') not in valid_intensities:
                logger.warning(f"⚠️ Modül {mod['name']} geçersiz compute_intensity: {mod.get('compute_intensity')}")
                
            if mod.get('api_type') not in valid_api_types:
                logger.warning(f"⚠️ Modül {mod['name']} geçersiz api_type: {mod.get('api_type')}")
                
            if mod.get('job_type') not in valid_job_types:
                logger.warning(f"⚠️ Modül {mod['name']} geçersiz job_type: {mod.get('job_type')}")

    # -----------------------------------------------------------
    #  Module Access & Filtering
    # -----------------------------------------------------------
    def list_modules(self) -> List[str]:
        """Tüm kayıtlı modül isimlerini döndürür."""
        return [m["name"] for m in self.modules]

    def get_module(self, name: str) -> Optional[Dict[str, Any]]:
        """Modül adını kullanarak yapılandırmayı getirir."""
        for m in self.modules:
            if m["name"] == name:
                return m
        logger.warning(f"Modül bulunamadı: {name}")
        return None

    def get_modules_by_compute_intensity(self, intensity: str) -> List[Dict[str, Any]]:
        """Compute intensity'ye göre modülleri filtreler"""
        return [m for m in self.modules if m.get('compute_intensity') == intensity]

    def filter_by_data_model(self, model_type: str) -> List[Dict[str, Any]]:
        """Veri modeli tipine göre modülleri getirir."""
        return [m for m in self.modules if m.get("data_model", "").lower() == model_type.lower()]

    def filter_by_parallel_mode(self, mode: str) -> List[Dict[str, Any]]:
        """Parallel mode'a göre modülleri getirir."""
        return [m for m in self.modules if m.get("parallel_mode", "").lower() == mode.lower()]

    def group_by_data_source(self) -> Dict[str, List[str]]:
        """Veri kaynağına göre modül isimlerini gruplar."""
        grouped: Dict[str, List[str]] = {}
        for mod in self.modules:
            src = mod.get("data_source", "unknown")
            grouped.setdefault(src, []).append(mod["name"])
        return grouped

    # -----------------------------------------------------------
    #  Performance & Resource Management
    # -----------------------------------------------------------
    def get_performance_profile(self, intensity: str) -> Dict[str, Any]:
        """Compute intensity için performance profile getirir"""
        profiles = self.registry.get('performance_profiles', {})
        return profiles.get(intensity, profiles.get('medium_intensity', {}))

    def get_module_performance_config(self, module_name: str) -> Dict[str, Any]:
        """Belirli modül için performance konfigürasyonu getirir"""
        module = self.get_module(module_name)
        if not module:
            return {}
            
        intensity = module.get('compute_intensity', 'medium')
        profile = self.get_performance_profile(intensity)
        
        return {
            'intensity': intensity,
            'max_workers': self._evaluate_expression(profile.get('max_workers', 'cpu_count')),
            'memory_limit': profile.get('memory_limit', '512MB'),
            'timeout': profile.get('timeout', 30),
            'execution_strategy': profile.get('execution_strategy', 'thread_pool')
        }

    def get_resource_requirements(self) -> Dict[str, Any]:
        """Sistem kaynak gereksinimlerini hesaplar"""
        requirements = {
            'total_modules': len(self.modules),
            'high_intensity_count': len(self.get_modules_by_compute_intensity('high')),
            'medium_intensity_count': len(self.get_modules_by_compute_intensity('medium')),
            'low_intensity_count': len(self.get_modules_by_compute_intensity('low')),
            'estimated_memory_mb': 0,
            'recommended_workers': 0
        }
        
        # Memory estimation
        for mod in self.modules:
            intensity = mod.get('compute_intensity', 'medium')
            if intensity == 'high':
                requirements['estimated_memory_mb'] += 1024
            elif intensity == 'medium':
                requirements['estimated_memory_mb'] += 512
            else:
                requirements['estimated_memory_mb'] += 256
        
        # Worker recommendation
        cpu_count = os.cpu_count() or 1
        high_intensity_mods = len(self.get_modules_by_compute_intensity('high'))
        requirements['recommended_workers'] = min(
            cpu_count * 2 + high_intensity_mods, 
            cpu_count * 4
        )
        
        return requirements

    def get_execution_plan(self) -> Dict[str, Any]:
        """Modül çalıştırma planı oluşturur"""
        plan = {
            'async_modules': self.filter_by_parallel_mode('async'),
            'sync_modules': self.filter_by_parallel_mode('sync'),
            'high_priority': self.get_modules_by_compute_intensity('high'),
            'resource_groups': {}
        }
        
        # Kaynak kullanımına göre gruplama
        for intensity in ['high', 'medium', 'low']:
            plan['resource_groups'][intensity] = {
                'modules': self.get_modules_by_compute_intensity(intensity),
                'suggested_strategy': self.get_performance_profile(intensity).get('execution_strategy'),
                'max_workers': self._evaluate_expression(
                    self.get_performance_profile(intensity).get('max_workers', 'cpu_count')
                )
            }
        
        return plan

    # -----------------------------------------------------------
    #  Utility Methods
    # -----------------------------------------------------------
    def _evaluate_expression(self, expr: str) -> int:
        """'cpu_count * 2' gibi ifadeleri değerlendirir"""
        try:
            if isinstance(expr, int):
                return expr
            if expr == 'cpu_count':
                return os.cpu_count() or 1
            if '*' in expr:
                parts = expr.split('*')
                base = self._evaluate_expression(parts[0].strip())
                multiplier = int(parts[1].strip())
                return base * multiplier
            if '//' in expr:
                parts = expr.split('//')
                base = self._evaluate_expression(parts[0].strip())
                divisor = int(parts[1].strip())
                return base // divisor
            return int(expr)
        except:
            return os.cpu_count() or 1

    def validate_data_sources(self) -> Dict[str, List[str]]:
        """Data source path'lerini doğrular"""
        issues = {}
        for mod in self.modules:
            data_source = mod.get('data_source')
            if data_source and data_source != 'internal':
                if '.' not in data_source or data_source.count('.') < 1:
                    issues.setdefault(mod['name'], []).append(
                        f"Geçersiz data_source formatı: {data_source}"
                    )
        return issues

    def get_meta(self) -> Dict[str, Any]:
        """YAML meta bilgilerini döndürür"""
        return self.meta

    def summary(self) -> str:
        """Tüm modüller için kısa özet oluşturur"""
        lines = [
            f"📘 Module Registry Summary ({self.meta.get('version', 'unknown')})",
            "──────────────────────────────"
        ]
        for mod in self.modules:
            lines.append(f"🔹 {mod['name']} | {mod['parallel_mode']} | {mod['data_model']} | {mod['job_type']}")
        return "\n".join(lines)


# ===============================================================
#  Standalone Test
# ===============================================================
if __name__ == "__main__":
    sm = SchemaManager()
    print(sm.summary())
    print("\n🧩 Polars tabanlı modüller:")
    for m in sm.filter_by_data_model("Polars"):
        print(f" - {m['name']} ({m['parallel_mode']})")

    print("\n📡 Veri Kaynaklarına göre gruplama:")
    for k, v in sm.group_by_data_source().items():
        print(f" - {k}: {', '.join(v)}")
# analysis/metric_resolver.py
# -*- coding: utf-8 -*-
"""
MetricResolver (v2025.1)
YAML veya metrik adı üzerinden ilgili fonksiyonu dinamik olarak çözer.
Desteklenen klasörler: metrics/classical.py, advanced.py, sentiment.py, volatility.py, microstructure.py, onchain.py, composite.py

| Özellik                           | Açıklama                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------ |
| 🔹 **Otomatik keşif**             | `metrics/` altındaki tüm fonksiyonlar runtime’da yüklenir                      |
| 🔹 **Küçük/büyük harf toleransı** | `"rsi"`, `"RSI"`, `"R_S_I"` hepsi aynı fonksiyonu bulur                        |
| 🔹 **Manuel kayıt**               | `resolver.register("MyMetric", my_func)` çağrısıyla özel fonksiyon eklenebilir |
| 🔹 **Listeleme**                  | `resolver.list_metrics()` tüm kayıtlı fonksiyonları verir                      |
| 🔹 **Hot reload uyumu**           | Framework başlarken yalnızca bir kez tarama yapar, hızlıdır                    |



"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MetricResolver:
    """
    MetricResolver:
      - metrics/ altındaki fonksiyonları otomatik keşfeder
      - YAML veya metrik adı üzerinden doğru fonksiyonu döndürür
    """

    def __init__(self, base_package: str = "analysis.metrics"):
        self.base_package = base_package
        self._registry: Dict[str, Callable] = {}
        self._initialized = False

    # ----------------------------------------------------------
    # 🔹 Ana public API
    # ----------------------------------------------------------

    def resolve(self, metric_name: str) -> Callable:
        if not self._initialized:
            self._scan_metrics()
        
        # 1️⃣ Direkt eşleşme
        if metric_name in self._registry:
            return self._registry[metric_name]
        
        # 2️⃣ Normalize edilmiş eşleşme
        normalized_target = self._normalize_name(metric_name)
        for key, func in self._registry.items():
            if self._normalize_name(key) == normalized_target:
                return func
        
        # 3️⃣ Modül bazlı fallback
        for module_name in getattr(self, "_get_available_modules", lambda: [])():
            try:
                module = importlib.import_module(f"{getattr(self, 'base_package', '')}.{module_name}")
                if hasattr(module, metric_name):
                    func = getattr(module, metric_name)
                    self._registry[metric_name] = func  # cache'e ekle
                    return func
            except (ImportError, AttributeError):
                continue
        
        raise ValueError(f"Metric '{metric_name}' not found")

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace("_", "").replace("-", "")



    # ----------------------------------------------------------
    # 🔹 metrics/ klasörünü otomatik tarama
    # ----------------------------------------------------------
    def _scan_metrics(self):
        """metrics/ altındaki tüm .py dosyalarını yükler ve fonksiyonları kaydeder."""
        try:
            base_path = Path(__file__).resolve().parent.parent / "metrics"
            for py_file in base_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                mod_name = py_file.stem  # e.g. classical, advanced
                module = importlib.import_module(f"{self.base_package}.{mod_name}")

                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # fonksiyon public mi
                    if not name.startswith("_"):
                        if name not in self._registry:
                            self._registry[name] = obj

            self._initialized = True
            logger.info(f"✅ MetricResolver: {len(self._registry)} metrik fonksiyonu yüklendi.")
        except Exception as e:
            logger.error(f"❌ MetricResolver initialization error: {e}")
            raise

    # ----------------------------------------------------------
    # 🔹 manuel kayıt (isteğe bağlı)
    # ----------------------------------------------------------
    def register(self, name: str, func: Callable):
        """El ile özel bir metrik fonksiyonu ekler."""
        self._registry[name] = func

    def list_metrics(self) -> Dict[str, str]:
        """Kayıtlı metrikleri (isim → modül) olarak listeler."""
        return {name: f"{func.__module__}.{func.__name__}" for name, func in self._registry.items()}


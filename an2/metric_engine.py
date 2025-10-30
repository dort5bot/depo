# analysis/metric_engine.py

"""

MetricEngine
=============
use_last_valid=True → Son geçerli değer kullanılır.
use_last_valid=False → default döner (None veya 0.0)
async hem sync hesaplamayı destekler (compute_async / compute).
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Optional
from cachetools import LRUCache  # LRU cache için

logger = logging.getLogger(__name__)


class MetricEngine:
    """
    MetricEngine: Tüm metrik hesaplamalarını yönetir.
    Safe compute ile hataya dayanıklı, last_valid veya None destekli.
    """

    def __init__(self):
        # Modül bazlı son geçerli metrik değerleri
        self._last_valid: Dict[str, Dict[str, float]] = {}

    async def compute_async(
        self,
        module_name: str,
        metric_name: str,
        func: Callable[..., Any],
        *args,
        use_last_valid: bool = False,
        default: Optional[float] = None,
        fallback_strategy: str = "default",  # "default", "last_valid", "propagate_nan"
        timeout: int = 30,
        **kwargs
    ) -> Optional[float]:
        """Asenkron metric hesaplama"""
        try:
            # Timeout ile execution
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

            # Result validation
            if self._is_valid_result(result):
                self._update_last_valid(module_name, metric_name, result)
                return result
            else:
                raise ValueError("Invalid result")

        except asyncio.TimeoutError:
            logger.warning(f"Metric {metric_name} timeout after {timeout}s")
            return self._handle_fallback(module_name, metric_name, default, fallback_strategy)

        except Exception as e:
            logger.error(f"Metric {metric_name} failed: {e}")
            return self._handle_fallback(module_name, metric_name, default, fallback_strategy)

    def compute(
        self,
        module_name: str,
        metric_name: str,
        func: Callable[..., Any],
        *args,
        use_last_valid: bool = False,
        default: Optional[float] = None,
        **kwargs
    ) -> Optional[float]:
        """Senkron metric hesaplama"""
        try:
            result = func(*args, **kwargs)
            if result is None:
                raise ValueError(f"{metric_name} returned None")
        except Exception as e:
            logger.warning(f"[{module_name}.{metric_name}] computation failed: {e}")
            if use_last_valid:
                result = self._last_valid.get(module_name, {}).get(metric_name, default)
            else:
                result = default

        # Son geçerli değeri güncelle
        if module_name not in self._last_valid:
            self._last_valid[module_name] = {}
        if result is not None:
            self._last_valid[module_name][metric_name] = result

        return result

    def reset_last_valid(self, module_name: Optional[str] = None):
        """Son geçerli değerleri sıfırlama"""
        if module_name:
            self._last_valid.pop(module_name, None)
        else:
            self._last_valid.clear()

    def _is_valid_result(self, result: Any) -> bool:
        """Sonuç geçerli mi kontrol et"""
        if result is None:
            return False
        if isinstance(result, (int, float)) and (result != result):  # NaN kontrolü
            return False
        return True

    def _update_last_valid(self, module_name: str, metric_name: str, result: Any):
        """Son geçerli değeri güncelle"""
        if module_name not in self._last_valid:
            self._last_valid[module_name] = {}
        self._last_valid[module_name][metric_name] = result

    def _handle_fallback(self, module_name: str, metric_name: str, default: Any, strategy: str):
        """Fallback stratejileri"""
        if strategy == "last_valid":
            return self._last_valid.get(module_name, {}).get(metric_name, default)
        elif strategy == "propagate_nan":
            return float("nan")
        else:  # default
            return default


# ======================================================
# 🔹 Performans odaklı MetricEngine (cache destekli)
# ======================================================

class OptimizedMetricEngine(MetricEngine):
    """Lazy loading + memoization destekli MetricEngine"""

    def __init__(self):
        super().__init__()
        self._function_cache: Dict[str, Callable] = {}
        self._computation_cache = LRUCache(maxsize=1000)

    @lru_cache(maxsize=500)
    def _get_metric_function(self, metric_name: str) -> Callable:
        """Metric fonksiyonunu cache’ten döndür"""
        if metric_name in self._function_cache:
            return self._function_cache[metric_name]
        raise ValueError(f"Metric function not found: {metric_name}")

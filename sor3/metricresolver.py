"""
analysis/metricresolver.py
MetricResolver – Yeni Versiyon
-----------------------------------
Görevi: Core'a metrik tanımını (function, columns, params) döndürmek
• Kategori → Liste yapısı ile çalışır
• Otomatik modül yükleyici (importlib)
• Metric fonksiyon caching
• Error-safe (never crashes core)
• Memory-leak safe (weakref cache)
"""

import importlib
import logging
import weakref
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger("MetricResolver")

# metrik konumu: analysis/metrics/*.py >> örnek analysis/metrics/classical.py
METRICS = {
    "classical": [
        "ema", "sma", "macd", "rsi", "adx", "stochastic_oscillator", 
        "roc", "atr", "bollinger_bands", "value_at_risk",
        "conditional_value_at_risk", "max_drawdown", "oi_growth_rate",
        "oi_price_correlation", "spearman_corr", "cross_correlation", "futures_roc"
    ],
    "advanced": [
        "kalman_filter_trend", "wavelet_transform", "hilbert_transform_slope",
        "hilbert_transform_amplitude", "fractal_dimension_index_fdi", 
        "shannon_entropy", "permutation_entropy", "sample_entropy",
        "granger_causality", "phase_shift_index"
    ],
    "volatility": [
        "historical_volatility", "bollinger_width", "garch_1_1", "hurst_exponent",
        "entropy_index", "variance_ratio_test", "range_expansion_index"
    ],
    "sentiment": [
        "funding_rate", "funding_premium", "oi_trend"
    ],
    "microstructure": [
        "ofi", "cvd", "microprice_deviation", "market_impact", "depth_elasticity",
        "taker_dominance_ratio", "liquidity_density"
    ],
    "onchain": [
        "etf_net_flow", "exchange_netflow", "stablecoin_flow", "net_realized_pl",
        "realized_cap", "nupl", "exchange_whale_ratio", "mvrv_zscore", "sopr"
    ],
    "regime": [
        "advance_decline_line", "volume_leadership", "performance_dispersion"
    ],
    "risk": [
        "volatility_risk", "liquidity_depth_risk", "spread_risk", "price_impact_risk",
        "taker_pressure_risk", "funding_stress", "open_interest_shock"
    ],
    "riskiptaledildi": [
        "liquidation_clusters", "cascade_risk", "sr_impact", "forced_selling",
        "liquidity_gaps", "futures_liq_risk", "liquidation_cascade", "market_stress"
    ]
}

class MetricResolver:
    """
    Core'a metrik tanımlarını döndüren sınıf.
    Metrik hangi dosyada, hangi fonksiyon, hangi kolonlar gerekiyor bilgisi.
    """

    def __init__(self, metric_map: Dict[str, list] = None):
        self.metric_map = metric_map or METRICS

        # memory-safe cache (function refs)
        self._func_cache: Dict[str, weakref.ReferenceType] = {}

        # module cache
        self._module_cache: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # MODULE LOADING
    # ---------------------------------------------------------
    def _load_module(self, category: str) -> Optional[Any]:
        """
        Kategoriye ait modülü yükler ve cache'ler.
        """
        if category in self._module_cache:
            return self._module_cache[category]

        module_path = f"analysis.metrics.{category}"
        try:
            module = importlib.import_module(module_path)
            self._module_cache[category] = module
            logger.debug(f"Module loaded: {module_path}")
            return module
        except Exception as e:
            logger.error(f"[MetricResolver] Modül yüklenemedi: {module_path} → {e}")
            return None

    # ---------------------------------------------------------
    # FUNCTION LOADING
    # ---------------------------------------------------------
    def _get_function(self, category: str, name: str) -> Optional[Callable]:
        """
        Metrik fonksiyonunu memory-safe cache ile yükler.
        """
        key = f"{category}.{name}"

        # Cache'te varsa
        if key in self._func_cache:
            fn_ref = self._func_cache[key]()
            if fn_ref is not None:
                return fn_ref

        # Cache'te yoksa modülden al
        module = self._load_module(category)
        if module is None:
            return None

        try:
            # Önce get_function() ile deneyelim
            if hasattr(module, "get_function"):
                fn = module.get_function(name)
            else:
                # Direkt attribute olarak bak
                fn = getattr(module, name)
            
            if fn is not None:
                # weakref ile memory leak önleniyor
                self._func_cache[key] = weakref.ref(fn)
                return fn
            else:
                logger.error(f"[MetricResolver] Fonksiyon None döndü: {key}")
                return None
                
        except AttributeError:
            logger.error(f"[MetricResolver] Fonksiyon bulunamadı: {key}")
            return None

    # ---------------------------------------------------------
    # MAIN RESOLVER FUNCTION
    # ---------------------------------------------------------
    def resolve_metric_definition(self, metric_name: str) -> Dict[str, Any]:
        """
        Metrik TANIMINI döndürür (veri değil)
        
        Returns: {
            "function": <callable>,
            "required_columns": ["open", "high", "low", "close"],
            "module_config": {...},
            "metadata": {...}
        }
        
        Raises: ValueError if metric not found
        """
        logger.debug(f"Resolving metric definition: {metric_name}")
        
        # 1. Metrik hangi kategoride?
        category = self._find_category(metric_name)
        if category is None:
            raise ValueError(f"Metric '{metric_name}' not found in any category")
        
        # 2. Modülü yükle
        module = self._load_module(category)
        if module is None:
            raise ValueError(f"Module for category '{category}' could not be loaded")
        
        # 3. Metrik fonksiyonunu al
        func = self._get_function(category, metric_name)
        if func is None:
            raise ValueError(f"Function for metric '{metric_name}' not found in module")
        
        # 4. Config'den gerekli kolonları al
        try:
            if hasattr(module, "get_module_config"):
                config = module.get_module_config()
            else:
                config = getattr(module, "_MODULE_CONFIG", {})
            
            # Gerekli kolonları al
            required_cols = []
            if "required_columns" in config and isinstance(config["required_columns"], dict):
                required_cols = config["required_columns"].get(metric_name, [])
            elif "required_columns" in config and isinstance(config["required_columns"], list):
                # Eski format: tüm metrikler aynı kolonları kullanıyor
                required_cols = config["required_columns"]
        
        except Exception as e:
            logger.warning(f"Could not get config for {metric_name}: {e}")
            config = {}
            required_cols = []
        
        # 5. Default parametreleri belirle
        default_params = {}
        
        # Window parametreleri için standart değerler
        if any(keyword in metric_name for keyword in ["rsi", "ema", "sma", "atr", "roc"]):
            default_params["window"] = 14
        elif "macd" in metric_name:
            default_params.update({"fast": 12, "slow": 26, "signal": 9})
        
        # 6. Paketle ve döndür
        return {
            "function": func,
            "required_columns": required_cols,
            "default_params": default_params,
            "module_config": config,
            "metadata": {
                "category": category,
                "module_name": module.__name__,
                "metric_name": metric_name,
                "resolved_at": importlib.__import__("datetime").datetime.now().isoformat()
            }
        }

    # ---------------------------------------------------------
    # BATCH RESOLVER (Multiple metrics)
    # ---------------------------------------------------------
    def resolve_multiple_definitions(self, metric_names: list) -> Dict[str, Dict[str, Any]]:
        """
        Birden fazla metriğin tanımını aynı anda çözer.
        
        Returns: {
            "ema": {function, required_columns, ...},
            "rsi": {function, required_columns, ...},
            ...
        }
        """
        results = {}
        
        for metric_name in metric_names:
            try:
                results[metric_name] = self.resolve_metric_definition(metric_name)
            except Exception as e:
                logger.error(f"Failed to resolve {metric_name}: {e}")
                results[metric_name] = None
        
        return results

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _find_category(self, metric: str) -> Optional[str]:
        """
        Metrik hangi kategoriye ait bulur.
        """
        for category, items in self.metric_map.items():
            if metric in items:
                return category
        return None
    
    def get_available_metrics(self) -> list:
        """
        Tüm kullanılabilir metrikleri döndürür.
        """
        all_metrics = []
        for items in self.metric_map.values():
            all_metrics.extend(items)
        return sorted(set(all_metrics))
    
    def get_metrics_by_category(self, category: str) -> list:
        """
        Belirli bir kategorideki metrikleri döndürür.
        """
        return self.metric_map.get(category, [])


# Global instance for easy access
_default_resolver = None

def get_default_resolver() -> MetricResolver:
    """Global MetricResolver instance döndürür."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = MetricResolver()
    return _default_resolver


def resolve_metric(metric_name: str) -> Dict[str, Any]:
    """
    Convenience function: Tek bir metriği çözer.
    """
    return get_default_resolver().resolve_metric_definition(metric_name)


def resolve_metrics(metric_names: list) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function: Birden fazla metriği çözer.
    """
    return get_default_resolver().resolve_multiple_definitions(metric_names)
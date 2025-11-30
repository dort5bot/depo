"""
analysis/metricresolver.py
MetricResolver – Production Version
-----------------------------------
• Kategori → Liste yapısı ile çalışır (classical, advanced, fr, ml ...)
• Otomatik modül yükleyici (importlib)
• Metric fonksiyon caching
• Error-safe execution (never crashes core)
• Sentetik fallback YOK – veri yoksa NaN döner
• Memory-leak safe (weakref cache)
"""

import importlib
import logging
import math
import weakref
from typing import Callable, Dict, Any

logger = logging.getLogger("MetricResolver")

# metrik konumu: analysis/metrics/*.py >> örnek analysis/metrics/classical.py
METRICS = {# Bu yapı uzun vadede en iyi tasarım olur,
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
        "volatility_risk","liquidity_depth_risk","spread_risk","price_impact_risk",
        "taker_pressure_risk","funding_stress","open_interest_shock"
    ],

    "riskiptaledildi": [
        "liquidation_clusters", "cascade_risk", "sr_impact", "forced_selling",
        "liquidity_gaps", "futures_liq_risk", "liquidation_cascade", "market_stress"
    ]
}

class MetricResolver:
    """
    Core’ın metrik çağrılarını yöneten sınıf.
    Metrik fonksiyonlarını otomatik modüllerden alır.
    Fonksiyonlar memory-safe weakref cache içinde tutulur.
    """

    def __init__(self, metric_map: Dict[str, list]):
        self.metric_map = metric_map  # {"classical": [...], ...}

        # memory-safe cache (function refs)
        self._func_cache: Dict[str, weakref.ReferenceType] = {}

        # module cache
        self._module_cache: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # MODULE LOADING
    # ---------------------------------------------------------
    def _load_module(self, category: str):
        """
        Yüklenmiş modül var mı? Varsa cache’den oku.
        Yoksa import et ve cache’e al.
        """
        if category in self._module_cache:
            return self._module_cache[category]

        module_path = f"analysis.metrics.{category}"
        try:
            module = importlib.import_module(module_path)
            self._module_cache[category] = module
            return module
        except Exception as e:
            logger.error(f"[MetricResolver] Modül yüklenemedi: {module_path} → {e}")
            return None

    # ---------------------------------------------------------
    # FUNCTION LOADING
    # ---------------------------------------------------------
    def _get_function(self, category: str, name: str) -> Callable:
        """
        Metrik fonksiyonunu memory-safe cache ile yükler.
        """
        key = f"{category}.{name}"

        # Cache’te varsa
        if key in self._func_cache:
            fn_ref = self._func_cache[key]()
            if fn_ref is not None:
                return fn_ref

        # Cache’te yoksa modülden al
        module = self._load_module(category)
        if module is None:
            return None

        try:
            fn = getattr(module, name)
            # weakref ile memory leak önleniyor
            self._func_cache[key] = weakref.ref(fn)
            return fn
        except AttributeError:
            logger.error(f"[MetricResolver] Fonksiyon bulunamadı: {key}")
            return None

    # ---------------------------------------------------------
    # MAIN EXECUTION
    # ---------------------------------------------------------
    def run(self, metric: str, data: Any):
        """
        Bir metrik çağırır.
        Örn: run("ema", price_df)
        """
        category = self._find_category(metric)
        if category is None:
            logger.error(f"[MetricResolver] Kategori bulunamadı → {metric}")
            return math.nan

        fn = self._get_function(category, metric)
        if fn is None:
            logger.error(f"[MetricResolver] Fonksiyon çözülemedi → {metric}")
            return math.nan

        # Metrik hesapla — veri yoksa fail etmez
        try:
            if data is None:
                logger.warning(f"[MetricResolver] Veri yok: {metric} → nan")
                return math.nan

            result = fn(data)

            # sonuç None veya beklenmeyen bir şeyse → nan
            if result is None:
                logger.warning(f"[MetricResolver] Metrik None döndü: {metric}")
                return math.nan

            return result

        except Exception as e:
            logger.error(f"[MetricResolver] {metric} çalışırken hata: {e}")
            return math.nan

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _find_category(self, metric: str):
        """
        Metrik hangi kategoriye ait bulur.
        """
        for category, items in self.metric_map.items():
            if metric in items:
                return category
        return None

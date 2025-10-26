# analysis/volat_regime.py
"""
Volatility & Regime Module - TAM ASYNC + POLARS
Version: 1.2.0
File: analysis/volat_regime.py

Beklenen çıktılar:
- score: 0..1 normalize edilmiş overall volatility/regime score
- signal: "trend" / "range" / "neutral"
- components: bileşen skorları (hv, atr, bw, var_ratio, premium)
- explain: kısa metin açıklama
- regime_label: Trend / Range

Özellikler:
- Batch ve async destekli (multi-symbol)
- CPU-bound hesaplamalar için ThreadPoolExecutor kullanımı
- Config üzerinden parametreler (analysis/config/c_volat.py)
- Analysis Helpers ile tam uyumlu output formatı
"""

# analysis/volat_regime.py


from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import polars as pl
from polars import DataFrame as plDataFrame

# ✅ GEREKLİ IMPORTLAR
from scipy.stats import kurtosis
from scipy.signal import detrend
import pandas as pd  # ewma_std için gerekli

# Try importing optional libs
try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers
from analysis.analysis_base_module import BaseAnalysisModule

logger = logging.getLogger(__name__)

# ✅ HELPER FONKSİYONLAR - CLASS DIŞINDA
def ewma_std(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving std"""
    series = pd.Series(x)
    return series.ewm(span=span, adjust=False).std().to_numpy()

def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """Estimate Hurst exponent via rescaled range (R/S) method"""
    N = len(ts)
    lags = np.floor(np.logspace(0.0, np.log10(max_lag), num=20)).astype(int)
    lags = np.unique(lags[lags > 1])
    if len(lags) < 2:
        return 0.5
    rs = []
    for lag in lags:
        n_segments = N // lag
        if n_segments < 2:
            continue
        vals = []
        for i in range(n_segments):
            seg = ts[i * lag:(i + 1) * lag]
            if len(seg) < 2:
                continue
            mean = np.mean(seg)
            Y = np.cumsum(seg - mean)
            R = np.max(Y) - np.min(Y)
            S = np.std(seg, ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append(np.mean(vals))
    if len(rs) < 2:
        return 0.5
    import scipy.stats as sps
    slope, _, _, _, _ = sps.linregress(np.log(lags[:len(rs)]), np.log(rs))
    return slope

def shannon_entropy(x: np.ndarray, bins: int = 50) -> float:
    """Calculate Shannon entropy"""
    p, _ = np.histogram(x, bins=bins, density=True)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def variance_ratio_test(returns: np.ndarray, lag: int = 2) -> float:
    """Lo-MacKinlay variance ratio statistic simplified"""
    if len(returns) < lag + 1:
        return 1.0
    var_q = np.var(np.sum(returns.reshape(-1, lag), axis=1), ddof=1)
    var_1 = np.var(returns, ddof=1)
    if var_1 == 0:
        return 1.0
    return var_q / (lag * var_1)

class VolatRegimeModulePolars(BaseAnalysisModule):
    """
    Volatility & Regime analysis - TAM ASYNC + POLARS
    """
    
    version = "1.2.1"
    module_name = "volat_regime_polars"

    def __init__(self, config: Optional[Dict[str, Any]] = None, executor: Optional[ThreadPoolExecutor] = None):
        super().__init__(config=config or {})
        
        self.helpers = AnalysisHelpers
        
        # Config yükleme
        if config is None:
            from analysis.config.cm_loader import config_manager
            config_obj = config_manager.get_config("volat")
            if config_obj:
                self.config_dict = config_obj.get_volatility_parameters()
            else:
                self.config_dict = self._get_default_config()
        else:
            self.config_dict = config
            
        self.weights = self.config_dict.get("weights", {})
        self.parameters = self.config_dict.get("parameters", {})
        
        self._executor = executor or ThreadPoolExecutor(
            max_workers=self.parameters.get("max_workers", 4)
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config"""
        logger.warning("Using default config for VolatRegimeModulePolars")
        return {
            "weights": {
                "historical_volatility": 0.15,
                "atr": 0.10,
                "bollinger_width": 0.10,
                "variance_ratio": 0.20,
                "hurst": 0.15,
                "entropy_struct": 0.05,
                "garch_implied_realized_diff": 0.15,
                "premium": 0.05,
                "rei": 0.05
            },
            "parameters": {
                "ohlcv_limit": 500,
                "annualization": 365,
                "hv_scale": 0.8,
                "atr_period": 14,
                "atr_lookback": 50,
                "atr_scale": 0.01,
                "bb_window": 20,
                "bb_scale": 0.02,
                "var_lag": 2,
                "var_sensitivity": 4.0,
                "hurst_max_lag": 20,
                "entropy_bins": 50,
                "entropy_scale": 3.5,
                "garch_proxy_span": 20,
                "realized_window": 20,
                "premium_scale": 0.05,
                "rei_lookback": 20,
                "rei_scale": 0.5,
                "max_workers": 4,
                "trend_threshold": 0.6,
                "range_threshold": 0.55,
            }
        }

    def _polars_atr(self, high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
        """POLARS ATR implementation"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pl.max_horizontal(tr1, tr2, tr3)
        atr = tr.rolling_mean(window_size=period)
        return atr

    def _polars_bollinger_bands(self, close: pl.Series, window: int = 20) -> Tuple[pl.Series, pl.Series]:
        """POLARS Bollinger Bands"""
        ma = close.rolling_mean(window_size=window)
        std = close.rolling_std(window_size=window)
        return ma, std

    def _compute_components_polars(self, df: plDataFrame) -> Dict[str, float]:
        """POLARS ile bileşen hesaplama"""
        cfg = self.parameters
        
        # POLARS Series'leri al
        close_series = df["close"]
        high_series = df["high"] 
        low_series = df["low"]
        
        # ✅ DÜZELTİLDİ: Polars Series → numpy array
        close_np = close_series.to_numpy()
        high_np = high_series.to_numpy()
        low_np = low_series.to_numpy()
        
        # 1) Historical Volatility
        logret = np.diff(np.log(close_np + 1e-12))
        hv_daily = np.std(logret, ddof=1) * math.sqrt(cfg.get("annualization", 365))
        hv_norm = min(1.0, hv_daily / (cfg.get("hv_scale", 0.8)))

        # 2) ATR - POLARS
        atr_series = self._polars_atr(high_series, low_series, close_series, cfg.get("atr_period", 14))
        last_atr = atr_series[-1] if len(atr_series) > 0 else 0.0
        atr_norm = min(1.0, (last_atr / (close_series.tail(cfg.get("atr_lookback", 50)).mean() + 1e-12)) / cfg.get("atr_scale", 0.01))

        # 3) Bollinger Width - POLARS
        ma, std = self._polars_bollinger_bands(close_series, cfg.get("bb_window", 20))
        last_bw = (std[-1] * 2) / (ma[-1] + 1e-12)
        bw_norm = min(1.0, last_bw / cfg.get("bb_scale", 0.02))

        # 4) Variance Ratio test
        vr = variance_ratio_test(logret, lag=cfg.get("var_lag", 2))
        vr_norm = 1 / (1 + math.exp(- (vr - 1) * cfg.get("var_sensitivity", 4)))

        # 5) Hurst exponent
        try:
            hurst = hurst_exponent(logret, max_lag=cfg.get("hurst_max_lag", 20))
        except Exception:
            hurst = 0.5
        hurst_norm = min(1.0, max(0.0, (hurst - 0.2) / 0.8))

        # 6) Entropy
        ent = shannon_entropy(logret, bins=cfg.get("entropy_bins", 50))
        ent_norm = 1 - min(1.0, ent / cfg.get("entropy_scale", 3.5))

        # 7) GARCH(1,1) conditional volatility vs realized
        garch_score = 0.0
        try:
            if _HAS_ARCH:
                am = arch_model(logret * 100, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", last_obs=len(logret) - 1)
                cond_vol = res.conditional_volatility / 100.0
                garch_pred = np.mean(cond_vol[-5:]) if len(cond_vol) > 0 else np.std(logret)
                realized = np.std(logret[-cfg.get("realized_window", 20):]) if len(logret) >= cfg.get("realized_window", 20) else np.std(logret)
                if realized > 0:
                    garch_score = min(1.0, abs(garch_pred - realized) / (realized + 1e-12))
                else:
                    garch_score = 0.0
            else:
                ew = ewma_std(logret, span=cfg.get("garch_proxy_span", 20))
                garch_score = min(1.0, abs(ew[-1] - np.std(logret[-cfg.get("realized_window", 20):])) / (np.std(logret[-cfg.get("realized_window", 20):]) + 1e-12))
        except Exception:
            garch_score = 0.0

        # 8) Premium / implied-realized proxy
        premium_norm = 0.0
        # ✅ DÜZELTİLDİ: Polars uyumlu premium kontrolü
        if "premium" in df.columns:
            prem_series = df["premium"]
            prem = prem_series[-1] if len(prem_series) > 0 else 0.0
            premium_norm = min(1.0, abs(prem) / cfg.get("premium_scale", 0.05))

        # 9) Range Expansion Index (REI)
        # ✅ DÜZELTİLDİ: Polars uyumlu ranges hesaplama
        ranges_np = high_np - low_np
        rei_lookback = cfg.get("rei_lookback", 20)
        if len(ranges_np) >= rei_lookback:
            rei = ranges_np[-1] / (np.mean(ranges_np[-rei_lookback:]) + 1e-12)
        else:
            rei = 1.0
        rei_norm = min(1.0, (rei - 1) / cfg.get("rei_scale", 0.5) if rei > 1 else 0.0)

        components = {
            "historical_volatility": float(hv_norm),
            "atr": float(atr_norm),
            "bollinger_width": float(bw_norm),
            "variance_ratio": float(vr_norm),
            "hurst": float(hurst_norm),
            "entropy_struct": float(ent_norm),
            "garch_implied_realized_diff": float(garch_score),
            "premium": float(premium_norm),
            "rei": float(rei_norm),
        }
        return components

    async def run_batch(self, symbols: List[str], data_provider, interval: str = "1h") -> Dict[str, Any]:
        """Batch processing multiple symbols"""
        tasks = [self.run_symbol(s, data_provider, interval) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.exception("Error computing symbol %s", sym)
                out[sym] = self.helpers.create_fallback_output(self.module_name, str(res))
            else:
                out[sym] = res
        return out

    async def run_symbol(self, symbol: str, data_provider, interval: str = "1h") -> Dict[str, Any]:
        """POLARS uyumlu async sembol analizi"""
        try:
            ohlcv_limit = self.parameters.get("ohlcv_limit", 500)
            ohlcv = await data_provider.get_ohlcv(
                symbol=symbol, interval=interval, limit=ohlcv_limit
            )
            
            if ohlcv is None or ohlcv.empty:
                raise ValueError(f"No OHLCV data for {symbol}")

            # Pandas → Polars conversion
            df_pl = pl.from_pandas(ohlcv)
            
            # ThreadPool'da POLARS hesaplamaları
            loop = asyncio.get_running_loop()
            components = await loop.run_in_executor(
                self._executor, self._compute_components_polars, df_pl
            )

            score, signal, explain, regime_label = self._aggregate_components(components)

            result = {
                "score": float(score),
                "signal": signal,
                "confidence": self._calculate_confidence(components),
                "components": components,
                "explain": explain,
                "timestamp": self.helpers.get_timestamp(),
                "module": self.module_name,
                "symbol": symbol,
                "regime_label": regime_label,
                "interval": interval,
                "engine": "polars"
            }

            if not self.helpers.validate_output(result):
                return self.helpers.create_fallback_output(self.module_name, "Output validation failed")

            return result

        except Exception as e:
            logger.error(f"POLARS async error for {symbol}: {str(e)}")
            return self.helpers.create_fallback_output(self.module_name, str(e))

    def _aggregate_components(self, components: Dict[str, float]) -> Tuple[float, str, str, str]:
        """Weight and combine component scores"""
        if self.weights and self.helpers.validate_score_dict(components):
            score = self.helpers.calculate_weights(components, self.weights)
        else:
            score = np.mean(list(components.values())) if components else 0.5
        
        score = self.helpers.normalize_score(score)

        trend_strength = (components.get("variance_ratio", 0) * 0.6 +
                          components.get("hurst", 0) * 0.3 +
                          components.get("rei", 0) * 0.1)

        volatility_strength = (components.get("historical_volatility", 0) * 0.5 +
                               components.get("atr", 0) * 0.2 +
                               components.get("bollinger_width", 0) * 0.2 +
                               components.get("garch_implied_realized_diff", 0) * 0.1)

        t_trend = self.parameters.get("trend_threshold", 0.6)
        t_range = self.parameters.get("range_threshold", 0.55)

        if trend_strength >= t_trend and components.get("variance_ratio", 0) > 0.6:
            signal = "trend"
            regime_label = "Trend"
        elif volatility_strength >= t_range and components.get("variance_ratio", 0) < 0.45:
            signal = "range" 
            regime_label = "Range"
        else:
            signal = "neutral"
            regime_label = "Neutral"

        explain = f"trend_strength={trend_strength:.3f}, vol_strength={volatility_strength:.3f}, score={score:.3f}"

        return score, signal, explain, regime_label

    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence score"""
        if not components:
            return 0.0

        comp_vals = list(components.values())
        variance = np.var(comp_vals)
        mean_val = np.mean(comp_vals)
        max_possible_var = 0.25

        consistency = 1.0 - min(1.0, variance / max_possible_var)
        stability_bonus = 1.0 - abs(mean_val - 0.5) * 2
        stability_bonus = max(0.0, stability_bonus)

        w_cons, w_stab = 0.7, 0.3
        conf = w_cons * consistency + w_stab * stability_bonus
        return float(max(0.0, min(1.0, conf)))

    async def run(self, symbol: str, priority: str = "normal", data_provider=None) -> Dict[str, Any]:
        """Main execution method for backward compatibility"""
        if data_provider is None:
            raise ValueError("data_provider must be provided to run()")
        return await self.run_symbol(symbol=symbol, data_provider=data_provider)

    def get_metadata(self) -> Dict[str, Any]:
        """Return module metadata"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "description": "Volatility regime detection and analysis",
            "metrics": list(self.weights.keys()),
            "parallel_mode": "batch",
            "lifecycle": "development",
            "analysis_helpers_compatible": True,
            "garch_support": _HAS_ARCH,
            "engine": "polars"
        }

# ✅ FACTORY FONKSİYONLARI - CLASS DIŞINDA
def create_volat_regime_module(config: Dict = None) -> VolatRegimeModulePolars:
    """Factory function to create VolatRegimeModulePolars instance"""
    return VolatRegimeModulePolars(config=config)

# ✅ COMPATIBILITY HELPER - CLASS DIŞINDA
async def run(symbol: str, priority: int = 5, config: Optional[Dict[str, Any]] = None, data_provider=None):
    """Backward-compatible run function"""
    if data_provider is None:
        raise ValueError("data_provider must be provided to run()")
    module = VolatRegimeModulePolars(config=config)
    return await module.run_symbol(symbol=symbol, data_provider=data_provider)
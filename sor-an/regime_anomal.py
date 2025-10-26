# analysis/regime_anomal.py
"""
Regime Change Detection & Anomaly Module
File: regime_anomal.py
config dosyası yok

Purpose:
- Detect sudden regime changes and anomalies using spot + futures data
- Computes rolling z-score, rolling skewness/kurtosis, cumulative return deviation
- Uses CUSUM (changepoint), IsolationForest (anomaly), and spectral-residual
- Async, vectorized with pandas/numpy and designed for BaseAnalysisModule interface

Key Metrics:
- CUSUM Change Point Detection
- Isolation Forest Anomaly Score  
- Rolling Z-Score Deviation
- Cumulative Return Deviation
- Spectral Residual Signal

Output: Regime Anomaly Score (0-1) with change/anomaly classification

============----------------
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Sequence, Callable

import numpy as np
import pandas as pd

# Pydantic model import
try:
    from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers
    _HAVE_HELPERS = True
except ImportError:
    _HAVE_HELPERS = False
    # Fallback Pydantic model
    from pydantic import BaseModel
    class AnalysisOutput(BaseModel):
        score: float = 0.5
        signal: str = "neutral"
        confidence: Optional[float] = 0.0
        components: Dict[str, float] = {}
        explain: str = ""
        timestamp: float = 0.0
        module: str = "regime_anomal"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try optional imports
try:
    from sklearn.ensemble import IsolationForest
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    from scipy import signal
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Try to import BaseAnalysisModule from project
try:
    from .analysis_base_module import BaseAnalysisModule
except Exception:
    BaseAnalysisModule = object

# ---------------------------
# Configuration Model
# ---------------------------
try:
    from pydantic import BaseModel as PydanticBaseModel
    
    class RegimeAnomalyConfig(PydanticBaseModel):
        """Pydantic configuration model for regime anomaly detection"""
        klines_interval: str = "1h"
        klines_limit: int = 500
        oi_hist_period: str = "1h"
        oi_hist_limit: int = 500
        funding_limit: int = 500
        rolling_window: int = 50
        zscore_threshold: float = 3.0
        cusum_h: float = 5.0
        cusum_k: float = 0.5
        iso_n_estimators: int = 100
        iso_contamination: float = 0.01
        combine_weights: Dict[str, float] = {"cusum": 0.4, "iso": 0.3, "zscore": 0.2, "cumret": 0.1}
        min_points: int = 100
        normalize_bounds: tuple = (0.0, 1.0)
        
        class Config:
            extra = "forbid"
            
except ImportError:
    # Fallback without Pydantic
    class RegimeAnomalyConfig:
        def __init__(self, **kwargs):
            self.klines_interval = kwargs.get("klines_interval", "1h")
            self.klines_limit = kwargs.get("klines_limit", 500)
            self.oi_hist_period = kwargs.get("oi_hist_period", "1h")
            self.oi_hist_limit = kwargs.get("oi_hist_limit", 500)
            self.funding_limit = kwargs.get("funding_limit", 500)
            self.rolling_window = kwargs.get("rolling_window", 50)
            self.zscore_threshold = kwargs.get("zscore_threshold", 3.0)
            self.cusum_h = kwargs.get("cusum_h", 5.0)
            self.cusum_k = kwargs.get("cusum_k", 0.5)
            self.iso_n_estimators = kwargs.get("iso_n_estimators", 100)
            self.iso_contamination = kwargs.get("iso_contamination", 0.01)
            self.combine_weights = kwargs.get("combine_weights", {"cusum": 0.4, "iso": 0.3, "zscore": 0.2, "cumret": 0.1})
            self.min_points = kwargs.get("min_points", 100)
            self.normalize_bounds = kwargs.get("normalize_bounds", (0.0, 1.0))

# ---------------------------
# Default configuration
# ---------------------------
DEFAULT_CONFIG = {
    "klines_interval": "1h",
    "klines_limit": 500,
    "oi_hist_period": "1h",
    "oi_hist_limit": 500,
    "funding_limit": 500,
    "rolling_window": 50,
    "zscore_threshold": 3.0,
    "cusum_h": 5.0,
    "cusum_k": 0.5,
    "iso_n_estimators": 100,
    "iso_contamination": 0.01,
    "combine_weights": {"cusum": 0.4, "iso": 0.3, "zscore": 0.2, "cumret": 0.1},
    "min_points": 100,
    "normalize_bounds": (0.0, 1.0),
}

# ---------------------------
# Helper algorithms
# ---------------------------
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std"""
    rm = series.rolling(window=window, min_periods=1).mean()
    rs = series.rolling(window=window, min_periods=1).std(ddof=0)
    zs = (series - rm) / rs.replace(0, np.nan)
    zs = zs.fillna(0.0)
    return zs.abs()

def cumulative_return_deviation(close: pd.Series, window: int) -> pd.Series:
    """Compute cumulative return deviation from rolling mean cumulative return"""
    r = np.log(close).diff().fillna(0.0)
    cr = r.rolling(window=window, min_periods=1).sum()
    cr_mean = cr.rolling(window=window, min_periods=1).mean()
    cr_std = cr.rolling(window=window, min_periods=1).std(ddof=0).replace(0, np.nan)
    dev = ((cr - cr_mean).abs() / cr_std).fillna(0.0)
    return dev

def simple_cusum(series: Sequence[float], h: float = 5.0, k: float = 0.5) -> Dict[str, Any]:
    """Basic two-sided CUSUM change detection"""
    s_pos = 0.0
    s_neg = 0.0
    pos_idx = []
    neg_idx = []

    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return {"pos": pos_idx, "neg": neg_idx, "cusum_score": 0.0}

    mu = np.nanmedian(arr)
    for i, x in enumerate(arr):
        diff = x - mu - k
        s_pos = max(0.0, s_pos + diff)
        diffn = - (x - mu) - k
        s_neg = max(0.0, s_neg + diffn)
        if s_pos > h:
            pos_idx.append(i)
            s_pos = 0.0
        if s_neg > h:
            neg_idx.append(i)
            s_neg = 0.0

    total = len(pos_idx) + len(neg_idx)
    norm = min(1.0, (total / max(1, arr.size / 10.0)))
    return {"pos": pos_idx, "neg": neg_idx, "cusum_score": float(norm)}

def spectral_residual_score(series: pd.Series) -> pd.Series:
    """Lightweight spectral-residual placeholder"""
    x = np.asarray(series.fillna(method="ffill").fillna(0.0), dtype=float)
    if x.size < 3:
        return pd.Series(np.zeros_like(x), index=series.index)

    if _HAVE_SCIPY:
        detr = signal.detrend(x)
        res = np.abs(detr)
        res = (res - res.min()) / (res.max() - res.min() + 1e-12)
        return pd.Series(res, index=series.index)
    else:
        d2 = np.abs(np.diff(x, n=2))
        pad = np.zeros(x.size)
        if d2.size > 0:
            pad[2:] = d2
        pad = (pad - pad.min()) / (pad.max() - pad.min() + 1e-12)
        return pd.Series(pad, index=series.index)

def isolation_forest_score(series: pd.Series, n_estimators=100, contamination=0.01) -> pd.Series:
    """Return anomaly score 0..1 per sample"""
    x = np.asarray(series.fillna(method="ffill").fillna(0.0), dtype=float).reshape(-1, 1)
    if x.shape[0] == 0:
        return pd.Series([], dtype=float)

    if _HAVE_SKLEARN:
        try:
            iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            iso.fit(x)
            raw_scores = -iso.decision_function(x)
            scaled = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-12)
            return pd.Series(scaled, index=series.index)
        except Exception as e:
            logger.warning("IsolationForest failed: %s. Falling back to median-deviation.", e)

    # fallback: rolling median absolute deviation
    med = pd.Series(x.flatten()).rolling(window=50, min_periods=1).median().values
    mad = np.abs(x.flatten() - med)
    scaled = (mad - mad.min()) / (mad.max() - mad.min() + 1e-12)
    return pd.Series(scaled, index=series.index)

def normalize_score(val: float, minv=0.0, maxv=1.0) -> float:
    """Normalize score with bounds"""
    if math.isnan(val):
        return 0.0
    return float(min(maxv, max(minv, val)))

# ---------------------------
# Module Implementation
# ---------------------------
class RegimeAnomalyModule(BaseAnalysisModule if BaseAnalysisModule is not object else object):
    """
    Regime & Anomaly detection module with Pydantic integration
    """

    version = "2025.1"
    module_name = "regime_anomal"

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None, metrics_client: Optional[Any] = None):
        """Initialize with Pydantic config validation"""
        self.config_dict = {**DEFAULT_CONFIG, **(config or {})}
        
        # Use Pydantic config model if available
        try:
            self.config = RegimeAnomalyConfig(**self.config_dict)
            self.config_dict = self.config.dict()  # Backward compatibility
        except Exception:
            self.config = RegimeAnomalyConfig(**self.config_dict) if hasattr(RegimeAnomalyConfig, '__annotations__') else None
        
        self.data_provider = data_provider
        self.metrics = metrics_client
        
        # Use AnalysisHelpers if available
        if _HAVE_HELPERS:
            self.helpers = AnalysisHelpers()
        else:
            self.helpers = None
            
        self._validate_config()

    def _validate_config(self):
        """Validate configuration with helper normalization"""
        w = self.config_dict.get("combine_weights", {})
        if not math.isclose(sum(w.values()), 1.0, rel_tol=1e-4):
            if self.helpers:
                w = self.helpers.normalize_weights(w)
            else:
                s = sum(w.values()) or 1.0
                w = {k: v / s for k, v in w.items()}
            self.config_dict["combine_weights"] = w
            logger.warning("combine_weights normalized to: %s", w)

    async def _fetch_all(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Orchestrates required data fetches concurrently"""
        async def _safe_call(func: Callable, *args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception("Data provider call failed: %s", e)
                return None

        dp = self.data_provider
        if dp is None:
            try:
                from ..utils.binance_api.futuresclient import FuturesClient as _FClient
                dp = _FClient()
                self.data_provider = dp
            except Exception:
                logger.warning("No data_provider provided and default FuturesClient not available.")
                dp = None

        tasks = {}
        results = {}

        if dp is None:
            raise RuntimeError("No data provider available for regime_anomal module.")

        tasks["klines"] = asyncio.create_task(_safe_call(dp.get_klines, symbol, self.config_dict["klines_interval"], self.config_dict["klines_limit"]))
        tasks["oi_hist"] = asyncio.create_task(_safe_call(dp.get_open_interest_hist, symbol, self.config_dict["oi_hist_period"], self.config_dict["oi_hist_limit"]))
        tasks["funding"] = asyncio.create_task(_safe_call(dp.get_funding_rate, symbol, self.config_dict["funding_limit"]))
        tasks["mark_klines"] = asyncio.create_task(_safe_call(dp.get_markprice_klines, symbol, self.config_dict["klines_interval"], self.config_dict["klines_limit"]))
        tasks["open_interest"] = asyncio.create_task(_safe_call(dp.get_open_interest, symbol))

        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for k, v in zip(tasks.keys(), done):
            results[k] = v

        return results

    async def compute_metrics(self, symbol: str) -> AnalysisOutput:
        """Main computation with Pydantic output"""
        try:
            raw = await self._fetch_all(symbol)

            # Data validation and processing
            klines = raw.get("klines")
            mark_klines = raw.get("mark_klines")
            oi_hist = raw.get("oi_hist")
            funding = raw.get("funding")
            open_interest = raw.get("open_interest")

            if klines is None or (not hasattr(klines, "close") and "close" not in getattr(klines, "columns", [])):
                raise ValueError(f"klines data missing or malformed for {symbol}")

            if not isinstance(klines, pd.DataFrame):
                klines = pd.DataFrame(klines)

            close = pd.to_numeric(klines["close"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

            window = int(self.config_dict["rolling_window"])
            min_points = int(self.config_dict["min_points"])
            if close.size < min_points:
                logger.warning("Not enough data points (%d) for robust analysis, min_points=%d", close.size, min_points)

            # Compute features
            zscore_series = rolling_zscore(close, window)
            cumret_dev_series = cumulative_return_deviation(close, window)

            # Prepare series for detection algorithms
            oi_series = None
            if isinstance(oi_hist, pd.DataFrame):
                if "openInterest" in oi_hist.columns:
                    oi_series = pd.to_numeric(oi_hist["openInterest"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)
                elif "value" in oi_hist.columns:
                    oi_series = pd.to_numeric(oi_hist["value"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

            mark_series = None
            if isinstance(mark_klines, pd.DataFrame) and "close" in mark_klines.columns:
                mark_series = pd.to_numeric(mark_klines["close"].astype(float), errors="coerce").fillna(method="ffill").fillna(0.0)

            price_ret = np.log(close).diff().fillna(0.0)
            combined_series = price_ret.copy()
            if oi_series is not None and len(oi_series) >= len(price_ret):
                oi_aligned = oi_series.iloc[-len(price_ret):].astype(float)
                oi_change = pd.Series(oi_aligned).pct_change().fillna(0.0).values
                combined_series = price_ret + pd.Series(oi_change, index=price_ret.index) * 0.1
            elif mark_series is not None:
                mark_ret = np.log(mark_series).diff().fillna(0.0)
                if len(mark_ret) >= len(price_ret):
                    mr = mark_ret.iloc[-len(price_ret):].values
                    combined_series = price_ret + pd.Series(mr, index=price_ret.index) * 0.2

            # Algorithm computations
            cusum_res = simple_cusum(combined_series.values, h=float(self.config_dict["cusum_h"]), k=float(self.config_dict["cusum_k"]))
            cusum_score = normalize_score(cusum_res.get("cusum_score", 0.0))

            iso_series = isolation_forest_score(close, n_estimators=int(self.config_dict["iso_n_estimators"]), contamination=float(self.config_dict["iso_contamination"]))
            iso_score_val = float(iso_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not iso_series.empty else 0.0
            iso_score_val = normalize_score(iso_score_val)

            zscore_val = float(zscore_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not zscore_series.empty else 0.0
            z_thr = float(self.config_dict["zscore_threshold"])
            zscore_norm = normalize_score(min(1.0, zscore_val / (z_thr + 1e-12)))

            cumret_val = float(cumret_dev_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not cumret_dev_series.empty else 0.0
            cumret_norm = normalize_score(min(1.0, cumret_val))

            spectral_series = spectral_residual_score(close)
            spectral_val = float(spectral_series.rolling(window=window, min_periods=1).max().iloc[-1]) if not spectral_series.empty else 0.0

            # Combine scores
            weights = self.config_dict.get("combine_weights", {})
            combined = (
                cusum_score * weights.get("cusum", 0.0)
                + iso_score_val * weights.get("iso", 0.0)
                + zscore_norm * weights.get("zscore", 0.0)
                + cumret_norm * weights.get("cumret", 0.0)
            )
            combined = normalize_score(combined)

            # Calculate confidence based on component consistency
            confidence = self._calculate_confidence([cusum_score, iso_score_val, zscore_norm, cumret_norm])

            # Derive signal
            signal_label = "neutral"
            if combined >= 0.75:
                dominant = max(
                    ("cusum", cusum_score),
                    ("iso", iso_score_val),
                    ("zscore", zscore_norm),
                    ("cumret", cumret_norm),
                    key=lambda x: x[1],
                )
                signal_label = "regime_change" if dominant[0] in ("cusum", "cumret") else "anomaly"
            elif combined >= 0.45:
                signal_label = "anomaly"

            # Build components and explanation
            components = {
                "cusum": round(float(cusum_score), 6),
                "iso_forest": round(float(iso_score_val), 6),
                "zscore": round(float(zscore_norm), 6),
                "cumret_dev": round(float(cumret_norm), 6),
                "spectral": round(float(spectral_val), 6),
            }

            explain_dict = {
                "window": window,
                "zscore_threshold": z_thr,
                "cusum_hits": {"pos": cusum_res.get("pos", []), "neg": cusum_res.get("neg", [])},
                "iso_summary": {
                    "last_max_window": float(iso_score_val),
                    "n_points": int(len(iso_series)) if hasattr(iso_series, "__len__") else 0,
                    "method": "IsolationForest" if _HAVE_SKLEARN else "median-deviation-fallback",
                },
                "zscore_recent": float(zscore_val),
                "cumret_recent": float(cumret_val),
                "spectral_recent": float(spectral_val),
            }

            # Create Pydantic output
            output = AnalysisOutput(
                score=float(round(combined, 6)),
                signal=signal_label,
                confidence=confidence,
                components=components,
                explain=str(explain_dict),
                timestamp=time.time() if self.helpers is None else self.helpers.get_timestamp(),
                module=self.module_name
            )

            # Metrics observation
            self._observe_metrics(output)

            return output

        except Exception as e:
            logger.error("Error in compute_metrics: %s", e)
            # Return fallback output using helpers
            if self.helpers:
                fallback = self.helpers.create_fallback_output(self.module_name, str(e))
                return AnalysisOutput(**fallback)
            else:
                return AnalysisOutput(
                    score=0.5,
                    signal="neutral",
                    confidence=0.0,
                    components={},
                    explain=f"Error: {str(e)}",
                    timestamp=time.time(),
                    module=self.module_name
                )

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on component consistency"""
        if not scores:
            return 0.0
        # Higher confidence when components agree (low variance)
        variance = np.var(scores) if len(scores) > 1 else 0.0
        base_confidence = np.mean(scores)
        confidence = base_confidence * (1.0 - variance)
        return normalize_score(confidence)

    def _observe_metrics(self, output: AnalysisOutput):
        """Observe metrics if client provided"""
        try:
            if self.metrics is not None:
                try:
                    self.metrics.observe("regime_anomaly.score", output.score)
                    self.metrics.observe("regime_anomaly.confidence", output.confidence or 0.0)
                except Exception:
                    try:
                        self.metrics.set("regime_anomaly.score", output.score)
                        self.metrics.set("regime_anomaly.confidence", output.confidence or 0.0)
                    except Exception:
                        logger.debug("metrics client has no observe/set interface")
        except Exception:
            logger.exception("Metrics observation failed")

    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """Report wrapper with human-friendly summary"""
        metrics = await self.compute_metrics(symbol)
        
        # Convert to dict for report
        metrics_dict = metrics.dict() if hasattr(metrics, 'dict') else {
            "score": metrics.score,
            "signal": metrics.signal,
            "confidence": metrics.confidence,
            "components": metrics.components,
            "explain": metrics.explain,
            "timestamp": metrics.timestamp,
            "module": metrics.module
        }
        
        friendly = {
            "title": f"Regime/Anomaly Report - {symbol}",
            "summary": "",
            "score": metrics_dict["score"],
            "signal": metrics_dict["signal"],
            "short_explain": "",
        }
        
        score = metrics_dict["score"]
        if score >= 0.75:
            friendly["summary"] = f"High-confidence {metrics_dict['signal']} detected for {symbol} (score={score:.3f})."
        elif score >= 0.45:
            friendly["summary"] = f"Medium confidence anomaly signals for {symbol} (score={score:.3f})."
        else:
            friendly["summary"] = f"No strong anomaly/regime change detected for {symbol} (score={score:.3f})."

        friendly["short_explain"] = f"Top components: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(metrics_dict["components"].items(), key=lambda x: -x[1])[:3]
        )

        return {"report": friendly, "metrics": metrics_dict}

    async def run(self, symbol: str, priority: Optional[int] = None) -> Dict[str, Any]:
        """Backward-compatible entry point"""
        return await self.generate_report(symbol)

# Expose top-level helper run() for legacy support
async def run(symbol: str, priority: Optional[int] = None, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None) -> Dict[str, Any]:
    """Convenience function expected by some aggregators"""
    module = RegimeAnomalyModule(config=config, data_provider=data_provider)
    return await module.run(symbol, priority)

# Quick local test
if __name__ == "__main__":
    import asyncio

    async def _smoke():
        class DummyDP:
            async def get_klines(self, symbol, interval, limit):
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                base = np.cumsum(np.random.randn(limit) * 0.01) + 100.0
                if limit > 200:
                    base[-150:] += np.linspace(0, 5.0, 150)
                df = pd.DataFrame({"close": base}, index=idx)
                return df

            async def get_open_interest_hist(self, symbol, period, limit):
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                oi = np.abs(1000 + np.cumsum(np.random.randn(limit) * 10))
                return pd.DataFrame({"openInterest": oi}, index=idx)

            async def get_funding_rate(self, symbol, limit):
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
                fr = np.random.randn(limit) * 0.0001
                return pd.DataFrame({"fundingRate": fr}, index=idx)

            async def get_markprice_klines(self, symbol, interval, limit):
                return await self.get_klines(symbol, interval, limit)

            async def get_open_interest(self, symbol):
                return float(1000.0 + np.random.randn() * 10)

        dp = DummyDP()
        mod = RegimeAnomalyModule(data_provider=dp)
        out = await mod.run("BTCUSDT")
        import json, pprint
        pprint.pprint(out)

    asyncio.run(_smoke())
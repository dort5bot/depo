# analysis/onchain.py
"""
On-Chain & Macro Analysis Module
- File: onchain.py
- Config: analysis/config/c_onchain.py

Purpose:
- Analyze on-chain liquidity & macro trends to generate Macro Score (0-1)
- Integrates ETF flows, exchange netflows, stablecoin movements, and key blockchain metrics
- Uses circuit breaker for external API resilience with configurable weights and thresholds

Key Metrics:
- ETF Net Flow (Bullish indicator)
- Exchange Netflow (Bearish when positive) 
- Stablecoin Flow (Bearish when inflow to exchanges)
- Net Realized Profit/Loss
- Exchange Whale Ratio
- MVRV Z-Score (Market Value to Realized Value)
- NUPL (Net Unrealized Profit/Loss)
- SOPR (Spent Output Profit Ratio)

Output: Macro Score (0-1) with bullish/bearish/neutral classification

Pydantic model for configuration validation and type safety
analysis/analysis_helpers.py ile uyumlu
============----------------
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import time

import numpy as np
import pandas as pd

# Pydantic and helper imports
try:
    from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers, analysis_helpers
    _HAVE_HELPERS = True
except ImportError:
    _HAVE_HELPERS = False
    from pydantic import BaseModel
    class AnalysisOutput(BaseModel):
        score: float = 0.5
        signal: str = "neutral"
        confidence: Optional[float] = 0.0
        components: Dict[str, float] = {}
        explain: str = ""
        timestamp: float = 0.0
        module: str = "onchain"

# Try to import config with Pydantic model
try:
    from analysis.config.c_onchain import OnChainConfig
    _HAVE_PYDANTIC_CONFIG = True
except ImportError:
    _HAVE_PYDANTIC_CONFIG = False
    # Fallback to dict config
    try:
        from analysis.config.c_onchain import CONFIG as OnChainConfig
    except ImportError:
        OnChainConfig = {}

# Base module import
try:
    from analysis.analysis_base_module import BaseAnalysisModule
except ImportError:
    BaseAnalysisModule = object

# Data provider import
try:
    from utils.data_sources.data_provider import DataProvider
except ImportError:
    class DataProvider:
        """Fallback data provider"""
        async def get_etf_flows(self, symbol): return None
        async def get_exchange_netflow(self, symbol): return None
        async def get_metric(self, source, symbol, metric): return None

_LOG = logging.getLogger(__name__)

# ---------------------------
# Pydantic Configuration Model
# ---------------------------
try:
    from pydantic import BaseModel as PydanticBaseModel, validator
    
    class OnChainConfigModel(PydanticBaseModel):
        """Pydantic configuration model for on-chain analysis"""
        version: str = "1.0.0"
        windows: Dict[str, int] = {"short_days": 7, "medium_days": 30, "long_days": 90}
        weights: Dict[str, float] = {
            "etf_net_flow": 0.15,
            "stablecoin_flow": 0.15, 
            "exchange_netflow": 0.20,
            "net_realized_pl": 0.15,
            "exchange_whale_ratio": 0.10,
            "mvrv_zscore": 0.10,
            "nupl": 0.05,
            "sopr": 0.10
        }
        thresholds: Dict[str, float] = {"bullish": 0.65, "bearish": 0.35}
        normalization: Dict[str, Any] = {"method": "zscore_clip", "clip_z": 3.0}
        data_timeout_seconds: int = 10
        explain_components_limit: int = 5
        parallel_mode: str = "async"
        prometheus: Dict[str, bool] = {"enable": False}
        
        @validator('weights')
        def validate_weights(cls, v):
            """Validate that weights sum to approximately 1.0"""
            total = sum(v.values())
            if not abs(total - 1.0) < 0.01:  # Allow 1% tolerance
                _LOG.warning(f"On-chain weights sum to {total}, normalizing to 1.0")
                # Normalize weights
                v = {k: weight/total for k, weight in v.items()}
            return v
            
except ImportError:
    # Fallback without Pydantic
    class OnChainConfigModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# ---------------------------
# Circuit Breaker with Helper Integration
# ---------------------------
class CircuitBreaker:
    """Circuit breaker with helper integration"""
    
    def __init__(self, failure_threshold: int = 3, recovery_time: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_ts = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_ts = time.time()

    def record_success(self):
        self.failures = 0
        self.last_failure_ts = 0

    def allow(self) -> bool:
        if self.failures < self.failure_threshold:
            return True
        if time.time() - self.last_failure_ts > self.recovery_time:
            self.failures = 0
            return True
        return False

# ---------------------------
# Utility Functions with Helper Integration
# ---------------------------
def zscore_clip(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score normalization with clipping"""
    if series.isnull().all():
        return series
    s = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0)
    s = s.clip(-clip, clip)
    s = (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) != 0 else (s * 0) + 0.5
    return s

def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalization"""
    if series.isnull().all():
        return series
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

def normalize(series: pd.Series, method: str = "zscore_clip", clip: float = 3.0) -> pd.Series:
    """Normalize series using specified method"""
    if method == "minmax":
        return minmax_norm(series)
    return zscore_clip(series, clip=clip)

def safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Safely get last value from series"""
    try:
        v = series.dropna()
        return float(v.iloc[-1]) if len(v) > 0 else default
    except Exception:
        return default

# ---------------------------
# Main Module Class
# ---------------------------
class OnChainModule(BaseAnalysisModule):
    """
    On-Chain analysis module with Pydantic integration
    Implements standardized AnalysisOutput format
    """

    module_name = "onchain"

    def __init__(self, config: Optional[Dict[str, Any]] = None, data_provider: Optional[DataProvider] = None):
        # Initialize configuration with Pydantic model
        config_dict = {**OnChainConfig, **(config or {})}
        try:
            self.config = OnChainConfigModel(**config_dict)
            self.config_dict = self.config.dict()  # For backward compatibility
        except Exception as e:
            _LOG.warning(f"Pydantic config failed: {e}, using dict config")
            self.config = OnChainConfigModel(**config_dict)
            self.config_dict = config_dict

        self.dp = data_provider or DataProvider()
        self.cb = CircuitBreaker()
        
        # Use AnalysisHelpers if available
        if _HAVE_HELPERS:
            self.helpers = AnalysisHelpers()
        else:
            self.helpers = None
            
        # Normalize weights using helper if available
        if self.helpers:
            self.weights = self.helpers.normalize_weights(self.config.weights)
        else:
            self.weights = self._normalize_weights(self.config.weights)

    def _normalize_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total = sum(w.values()) if w else 0.0
        if total == 0:
            return {k: 1.0 / max(len(w), 1) for k in w}
        return {k: float(v) / float(total) for k, v in w.items()}

    async def _safe_fetch(self, fetch_coro, *args, **kwargs):
        """Wrapper for external data calls with circuit breaker"""
        if not self.cb.allow():
            _LOG.warning("Circuit breaker open - skipping data fetch")
            return None
        try:
            timeout = self.config.data_timeout_seconds if hasattr(self.config, 'data_timeout_seconds') else 10
            return await asyncio.wait_for(fetch_coro(*args, **kwargs), timeout=timeout)
        except Exception as e:
            _LOG.exception("Data fetch failed: %s", e)
            self.cb.record_failure()
            return None

    # -------------------------
    # Metric fetch helpers
    # -------------------------
    async def _fetch_etf_net_flow(self, symbol: str) -> Optional[pd.Series]:
        data = await self._safe_fetch(self.dp.get_etf_flows, symbol)
        if not data:
            return None
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    async def _fetch_exchange_netflow(self, symbol: str) -> Optional[pd.Series]:
        data = await self._safe_fetch(self.dp.get_exchange_netflow, symbol)
        if not data:
            return None
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    async def _fetch_metric_generic(self, source: str, metric: str, symbol: str) -> Optional[pd.Series]:
        data = await self._safe_fetch(self.dp.get_metric, source, symbol, metric)
        if not data:
            return None
        try:
            ser = pd.Series(data).sort_index()
            self.cb.record_success()
            return ser
        except Exception:
            return None

    # -------------------------
    # Metric computations
    # -------------------------
    def _compute_rolling_sum(self, series: pd.Series, window_days: int) -> pd.Series:
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        try:
            return series.rolling(window=window_days, min_periods=1).sum()
        except Exception:
            arr = np.asarray(series.fillna(0.0))
            if len(arr) == 0:
                return pd.Series(dtype=float)
            kernel = np.ones(min(window_days, len(arr)))
            conv = np.convolve(arr, kernel, mode="same")
            return pd.Series(conv, index=series.index)

    def _score_component_from_series(self, series: pd.Series, method: str) -> float:
        if series is None or series.empty:
            return 0.5  # neutral fallback
            
        norm_config = self.config.normalization if hasattr(self.config, 'normalization') else {}
        norm_method = norm_config.get("method", "zscore_clip")
        clip_val = norm_config.get("clip_z", 3.0)
        
        norm = normalize(series, method=norm_method, clip=clip_val)
        return safe_last(norm, default=0.5)

    async def compute_metrics(self, symbol: str) -> AnalysisOutput:
        """
        Main metric computation with standardized AnalysisOutput
        """
        try:
            # Fetch all data in parallel
            tasks = {
                "etf_net_flow": asyncio.create_task(self._fetch_etf_net_flow(symbol)),
                "exchange_netflow": asyncio.create_task(self._fetch_exchange_netflow(symbol)),
                "stablecoin_flow": asyncio.create_task(self._fetch_metric_generic("glassnode", "stablecoin_netflow", symbol)),
                "realized_cap": asyncio.create_task(self._fetch_metric_generic("glassnode", "realized_cap", symbol)),
                "nupl": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "nupl", symbol)),
                "net_realized_pl": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "net_realized_profit_loss", symbol)),
                "exchange_whale_ratio": asyncio.create_task(self._fetch_metric_generic("cryptoquant", "exchange_whale_ratio", symbol)),
                "mvrv_zscore": asyncio.create_task(self._fetch_metric_generic("glassnode", "mvrv_zscore", symbol)),
                "sopr": asyncio.create_task(self._fetch_metric_generic("glassnode", "sopr", symbol)),
            }

            results = {}
            for key, t in tasks.items():
                try:
                    results[key] = await t
                except Exception as e:
                    _LOG.exception("Task %s failed: %s", key, e)
                    results[key] = None

            # Compute component scores
            components = {}
            windows = self.config.windows if hasattr(self.config, 'windows') else {"medium_days": 30}
            medium = windows.get("medium_days", 30)

            try:
                # ETF net flow: more positive is bullish
                etf_series = results.get("etf_net_flow")
                if etf_series is not None:
                    etf_roll = self._compute_rolling_sum(etf_series.fillna(0), medium)
                    components["etf_net_flow"] = self._score_component_from_series(etf_roll, "zscore_clip")
                else:
                    components["etf_net_flow"] = 0.5

                # Stablecoin flow: inflow to exchanges -> bearish (invert)
                sc_series = results.get("stablecoin_flow")
                if sc_series is not None:
                    sc_roll = self._compute_rolling_sum(sc_series.fillna(0), medium)
                    sc_score = 1.0 - self._score_component_from_series(sc_roll, "zscore_clip")
                    components["stablecoin_flow"] = float(np.clip(sc_score, 0.0, 1.0))
                else:
                    components["stablecoin_flow"] = 0.5

                # Exchange netflow: positive inflow -> bearish (invert)
                ex_series = results.get("exchange_netflow")
                if ex_series is not None:
                    ex_roll = self._compute_rolling_sum(ex_series.fillna(0), medium)
                    ex_score = 1.0 - self._score_component_from_series(ex_roll, "zscore_clip")
                    components["exchange_netflow"] = float(np.clip(ex_score, 0.0, 1.0))
                else:
                    components["exchange_netflow"] = 0.5

                # Net Realized P/L: large realized profit -> bearish (invert)
                nrp = results.get("net_realized_pl")
                if nrp is not None:
                    nrp_roll = self._compute_rolling_sum(nrp.fillna(0), medium)
                    nrp_score = 1.0 - self._score_component_from_series(nrp_roll, "zscore_clip")
                    components["net_realized_pl"] = float(np.clip(nrp_score, 0.0, 1.0))
                else:
                    components["net_realized_pl"] = 0.5

                # Exchange whale ratio: higher -> bearish (invert)
                ewr = results.get("exchange_whale_ratio")
                if ewr is not None:
                    ewr_score = 1.0 - self._score_component_from_series(ewr.fillna(0), "zscore_clip")
                    components["exchange_whale_ratio"] = float(np.clip(ewr_score, 0.0, 1.0))
                else:
                    components["exchange_whale_ratio"] = 0.5

                # MVRV Z-Score: higher -> overvalued -> bearish (invert)
                mvrv = results.get("mvrv_zscore")
                if mvrv is not None:
                    mvrv_score = 1.0 - self._score_component_from_series(mvrv.fillna(0), "zscore_clip")
                    components["mvrv_zscore"] = float(np.clip(mvrv_score, 0.0, 1.0))
                else:
                    components["mvrv_zscore"] = 0.5

                # NUPL: high positive -> euphoria -> bearish (invert)
                nupl = results.get("nupl")
                if nupl is not None:
                    nupl_score = 1.0 - self._score_component_from_series(nupl.fillna(0), "zscore_clip")
                    components["nupl"] = float(np.clip(nupl_score, 0.0, 1.0))
                else:
                    components["nupl"] = 0.5

                # SOPR: >1 profit being realized -> bearish (invert)
                sopr = results.get("sopr")
                if sopr is not None:
                    sopr_score = 1.0 - self._score_component_from_series(sopr.fillna(0), "zscore_clip")
                    components["sopr"] = float(np.clip(sopr_score, 0.0, 1.0))
                else:
                    components["sopr"] = 0.5

            except Exception as e:
                _LOG.exception("Error computing components: %s", e)

            # Calculate weighted score
            weighted_sum = 0.0
            total_weight = 0.0
            for comp_name, weight in self.weights.items():
                comp_val = components.get(comp_name, 0.5)
                weighted_sum += comp_val * weight
                total_weight += weight

            score = weighted_sum / total_weight if total_weight > 0 else 0.5
            score = float(np.clip(score, 0.0, 1.0))

            # Determine signal
            thresholds = self.config.thresholds if hasattr(self.config, 'thresholds') else {"bullish": 0.65, "bearish": 0.35}
            if score >= thresholds.get("bullish", 0.65):
                signal = "bullish"
            elif score <= thresholds.get("bearish", 0.35):
                signal = "bearish"
            else:
                signal = "neutral"

            # Calculate confidence based on component consistency
            confidence = self._calculate_confidence(list(components.values()))

            # Build explanation
            explain = self._build_explanation(components, results)

            # Create standardized output
            timestamp = self.helpers.get_timestamp() if self.helpers else time.time()
            
            return AnalysisOutput(
                score=float(round(score, 4)),
                signal=signal,
                confidence=confidence,
                components=components,
                explain=explain,
                timestamp=timestamp,
                module=self.module_name
            )

        except Exception as e:
            _LOG.error("Error in compute_metrics: %s", e)
            # Return fallback output
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
        # Higher confidence when components have clear signals (away from 0.5)
        avg_distance = np.mean([abs(score - 0.5) for score in scores])
        confidence = avg_distance * 2.0  # Scale to 0-1
        return float(np.clip(confidence, 0.0, 1.0))

    def _build_explanation(self, components: Dict[str, float], results: Dict[str, Any]) -> str:
        """Build human-readable explanation"""
        explain_limit = self.config.explain_components_limit if hasattr(self.config, 'explain_components_limit') else 5
        
        # Sort components by weight * value impact
        impactful_components = []
        for name, value in components.items():
            weight = self.weights.get(name, 0.0)
            impact = abs(value - 0.5) * weight
            impactful_components.append((name, value, weight, impact))
        
        impactful_components.sort(key=lambda x: -x[3])  # Sort by impact descending
        top_components = impactful_components[:explain_limit]
        
        explanations = []
        for name, value, weight, impact in top_components:
            direction = "bullish" if value > 0.6 else "bearish" if value < 0.4 else "neutral"
            explanations.append(f"{name}({direction}:{value:.2f})")
        
        available_metrics = [k for k, v in results.items() if v is not None]
        return f"Top factors: {', '.join(explanations)} | Data sources: {len(available_metrics)}/{len(results)}"

    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """Generate report with standardized format"""
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
        
        # Add friendly summary
        friendly = {
            "title": f"On-Chain Analysis Report - {symbol}",
            "summary": self._generate_summary(metrics_dict),
            "score": metrics_dict["score"],
            "signal": metrics_dict["signal"],
            "confidence": metrics_dict.get("confidence", 0.0),
        }
        
        return {"report": friendly, "metrics": metrics_dict}

    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-friendly summary"""
        score = metrics["score"]
        signal = metrics["signal"]
        
        if signal == "bullish":
            return f"Strong bullish on-chain signals for {metrics.get('symbol', 'asset')} (score: {score:.3f})"
        elif signal == "bearish":
            return f"Bearish on-chain pressure detected for {metrics.get('symbol', 'asset')} (score: {score:.3f})"
        else:
            return f"Mixed or neutral on-chain signals for {metrics.get('symbol', 'asset')} (score: {score:.3f})"

    async def run(self, symbol: Union[str, List[str]], priority: Optional[int] = None) -> Dict[str, Any]:
        """Backward-compatible entrypoint"""
        if isinstance(symbol, str):
            report = await self.generate_report(symbol)
            return {symbol: report}
        elif isinstance(symbol, list):
            tasks = {s: asyncio.create_task(self.generate_report(s)) for s in symbol}
            out = {}
            for s, t in tasks.items():
                try:
                    out[s] = await t
                except Exception as e:
                    _LOG.exception("Report generation failed for %s: %s", s, e)
                    out[s] = {"error": str(e)}
            return out
        else:
            raise ValueError("symbol must be str or list of str")

# Module-level helper function
async def run(symbol: Union[str, List[str]], priority: Optional[int] = None, 
              config: Optional[Dict[str, Any]] = None, 
              data_provider: Optional[DataProvider] = None) -> Dict[str, Any]:
    """Convenience function for backward compatibility"""
    mod = OnChainModule(config=config, data_provider=data_provider)
    return await mod.run(symbol, priority=priority)
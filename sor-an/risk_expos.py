# analysis/risk_expos.py
"""
Risk & Exposure Management Module - Analysis Helpers Uyumlu
Version: 1.1.0

Exports:
    - class RiskExposureModule(BaseAnalysisModule)
    - run(symbols, priority) function for backward compatibility

Purpose:
    Calculate risk metrics for spot + futures (public + private data if available)
    Metrics computed:
      - ATR-based adaptive stop
      - Liquidation zone estimation
      - Max Drawdown
      - Volatility targeting (position sizing guidance)
      - Position Leverage Ratio
      - VaR (historical simulation) at configurable confidence
      - Expected Shortfall / CVaR
      - Dynamic Sharpe & Sortino
      - Final normalized Risk Score (0..1) with components & explainability

Design notes:
    - Analysis Helpers ile tam uyumlu output formatı
    - Merkezi config yönetimi ile entegrasyon
    - Standart fallback mekanizması
"""

from __future__ import annotations

import asyncio
import math
import time
import logging

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput
from analysis.analysis_base_module import BaseAnalysisModule

logger = logging.getLogger(__name__)

# --- Helpers & numeric functions ---

def ensure_df_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has columns: ['open','high','low','close','volume'] and index is datetime.
    Accepts dict-like or DataFrame and returns DataFrame copy.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    df = df.copy()
    cols = [c.lower() for c in df.columns]
    mapping = {}
    if 'open' not in cols and 'Open' in df.columns:
        mapping['Open'] = 'open'
    # best-effort mapping
    candidates = {'open': ['open', 'o'], 'high': ['high', 'h'], 'low': ['low', 'l'], 'close': ['close', 'c'], 'volume': ['volume', 'v']}
    colmap = {}
    for std, names in candidates.items():
        for n in names:
            if n in df.columns:
                colmap[n] = std
                break
    df = df.rename(columns=colmap)
    # ensure required columns exist
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan
    # index to datetime if possible
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            try:
                df.index = pd.to_datetime(df.index, unit='ms', utc=True)
            except Exception:
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    return df[['open', 'high', 'low', 'close', 'volume']]

def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range series"""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False).mean()

def max_drawdown(series: pd.Series) -> float:
    """Returns maximum drawdown (positive number e.g. 0.25 for 25%)"""
    if series is None or len(series) == 0:
        return 0.0
    roll_max = series.cummax()
    drawdown = (roll_max - series) / roll_max
    return float(drawdown.max())

def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical VaR (percentile). returns is negative for losses if standard convention,
    but we assume returns as pct returns; VaR returned as positive loss magnitude.
    """
    if len(returns) == 0:
        return 0.0
    q = np.quantile(returns, 1 - confidence)
    return float(max(0.0, -q))

def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) == 0:
        return 0.0
    threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(max(0.0, -tail.mean()))

def annualize_vol(std_per_period: float, periods_per_year: int) -> float:
    return std_per_period * math.sqrt(periods_per_year)

def rolling_sharpe(returns: pd.Series, window: int, rf_per_period: float = 0.0) -> pd.Series:
    excess = returns - rf_per_period
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return roll_mean / (roll_std.replace(0, np.nan))

def rolling_sortino(returns: pd.Series, window: int, rf_per_period: float = 0.0) -> pd.Series:
    # downside deviation uses negative returns only
    excess = returns - rf_per_period
    def sortino_s(arr):
        arr = np.asarray(arr)
        if len(arr) == 0:
            return np.nan
        downside = arr[arr < 0]
        if len(downside) == 0:
            return np.nan
        mean = arr.mean()
        dd = np.sqrt((downside ** 2).mean())
        return (mean) / dd if dd > 0 else np.nan
    return returns.rolling(window).apply(lambda x: sortino_s(x - rf_per_period), raw=False)

# --- Module Implementation ---

class RiskExposureModule(BaseAnalysisModule):
    """
    Risk & Exposure Module - Analysis Helpers Uyumlu
    
    Constructor:
        RiskExposureModule(config: dict, data_provider: Optional[object] = None, metrics_client: Optional[object] = None)
    """

    def __init__(self, config: Dict[str, Any], data_provider: Optional[Any] = None, metrics_client: Optional[Any] = None):
        super().__init__(config)
        
        # ✅ ANALYSIS_HELPERS INTEGRATION
        self.helpers = AnalysisHelpers
        self.module_name = "risk_expos"
        self.version = "1.1.0"
        
        # Load configuration - ANALYSIS_HELPERS UYUMLU
        if config is None:
            from analysis.config.cm_loader import config_manager
            config_obj = config_manager.get_config("risk")
            if config_obj:
                self.config_dict = config_obj.to_flat_dict()
            else:
                self.config_dict = self._get_default_config()
        else:
            self.config_dict = config
            
        self.weights = self.config_dict.get("weights", {})
        self.parameters = self.config_dict.get("parameters", {})
        self.thresholds = self.config_dict.get("thresholds", {})
        
        self.data_provider = data_provider
        self.metrics_client = metrics_client
        self._loop = asyncio.get_event_loop()

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config oluştur"""
        logger.warning("Using default config for RiskExposureModule")
        return {
            "weights": {
                "var": 0.30,
                "cvar": 0.25,
                "leverage": 0.15,
                "vol_targeting": 0.10,
                "max_drawdown": 0.10,
                "atr_stop": 0.10
            },
            "thresholds": {
                "high_risk": 0.75,
                "medium_risk": 0.45,
                "low_risk": 0.25
            },
            "parameters": {
                "ohlcv": {"interval": "1h", "lookback_bars": 500},
                "atr": {"period": 21, "multiplier": 3.0},
                "var": {"confidence_levels": [0.95, 0.99]},
                "vol_target": {"target_volatility": 0.12},
                "leverage": {"max_leverage": 125}
            }
        }

    async def _safe_fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Wrapper to call the data provider's get_klines. Graceful fallback to empty DataFrame.
        """
        if self.data_provider is None:
            return pd.DataFrame()
        try:
            if asyncio.iscoroutinefunction(self.data_provider.get_klines):
                res = await self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                res = await self._loop.run_in_executor(None, lambda: self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit))
            
            if isinstance(res, pd.DataFrame):
                df = res
            else:
                df = pd.DataFrame(res)
            df = ensure_df_ohlcv(df)
            return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    async def _safe_fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch futures position risk or account positions if available.
        """
        if self.data_provider is None:
            return []
        try:
            if hasattr(self.data_provider, "get_futures_position_risk"):
                fn = self.data_provider.get_futures_position_risk
                if asyncio.iscoroutinefunction(fn):
                    res = await fn()
                else:
                    res = await self._loop.run_in_executor(None, fn)
                if isinstance(res, dict):
                    return [res]
                return list(res or [])
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def _estimate_liquidation_price(self, entry_price: float, leverage: float, side: str = "LONG", maintenance: float = 0.005) -> float:
        """
        Conservative approximation of liquidation price for isolated margin.
        """
        if leverage is None or leverage <= 1:
            return float(entry_price)
        if side.upper() == "LONG":
            liq = entry_price * (1 - (1.0 / max(leverage, 1.0)) - maintenance)
        else:
            liq = entry_price * (1 + (1.0 / max(leverage, 1.0)) + maintenance)
        return float(max(0.0, liq))

    def _score_from_component(self, value: float, higher_is_riskier: bool = True, clip: Tuple[float, float] = (0.0, 1.0)) -> float:
        """
        Map a raw metric value into [0..1] risk score for that component.
        """
        lo, hi = clip
        if hi == lo:
            return 0.0
        v = float(value)
        normalized = (v - lo) / (hi - lo)
        if not higher_is_riskier:
            normalized = 1.0 - normalized
        return float(max(0.0, min(1.0, normalized)))

    async def compute_metrics(self, symbol: str, interval: Optional[str] = None, lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute risk metrics for a single symbol - ANALYSIS_HELPERS UYUMLU
        """
        try:
            cfg = self.parameters
            interval = interval or cfg.get("ohlcv", {}).get("interval", "1h")
            lookback = lookback or cfg.get("ohlcv", {}).get("lookback_bars", 500)

            # fetch historical price data
            df = await self._safe_fetch_klines(symbol=symbol, interval=interval, limit=lookback)
            if df is None or df.empty:
                # ✅ ANALYSIS_HELPERS UYUMLU FALLBACK
                return self.helpers.create_fallback_output(self.module_name, "No price data available")

            # compute returns (periodic pct returns)
            close = df['close'].astype(float)
            period_returns = close.pct_change().dropna()
            periods = len(period_returns)
            
            # interval mapping for annualization
            if 'm' in interval and interval.endswith('m'):
                minutes = int(interval[:-1])
                periods_per_year = int((60 / minutes) * 24 * 365)
            elif 'h' in interval:
                hours = int(interval[:-1])
                periods_per_year = int((24 / hours) * 365)
            elif 'd' in interval:
                days = int(interval[:-1])
                periods_per_year = int(365 / days)
            else:
                periods_per_year = 24 * 365

            # volatility (std)
            std_per_period = float(period_returns.std(ddof=1) or 0.0)
            ann_vol = annualize_vol(std_per_period, periods_per_year)

            # Max drawdown
            md = max_drawdown(close)

            # ATR stop
            atr_period = cfg.get("atr", {}).get("period", 21)
            atr_multiplier = cfg.get("atr", {}).get("multiplier", 3.0)
            atr_series = atr(df, period=atr_period)
            latest_atr = float(atr_series.dropna().iloc[-1]) if len(atr_series.dropna()) > 0 else 0.0
            latest_price = float(close.iloc[-1])
            atr_stop = max(0.0, latest_price - atr_multiplier * latest_atr)

            # VaR / CVaR
            var_cfg = cfg.get("var", {})
            var_conf_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
            var_results = {}
            cvar_results = {}
            for c in var_conf_levels:
                v = historical_var(period_returns.values, confidence=c)
                cv = historical_cvar(period_returns.values, confidence=c)
                var_results[f"VaR_{int(c*100)}"] = v
                cvar_results[f"CVaR_{int(c*100)}"] = cv

            # Volatility targeting
            vt_cfg = cfg.get("vol_target", {})
            target_vol = vt_cfg.get("target_volatility", 0.12)
            vol_targeting_factor = 1.0
            if ann_vol > 0:
                vol_targeting_factor = float(min(3.0, target_vol / ann_vol))
            else:
                vol_targeting_factor = 1.0

            # positions & leverage
            positions = await self._safe_fetch_positions()
            leverage_ratios = []
            liquidation_infos = []
            maintenance_default = cfg.get("leverage", {}).get("default_maintenance_margin", 0.005)
            
            for pos in positions:
                try:
                    entry_price = float(pos.get("entryPrice", pos.get("entry_price", latest_price)))
                    leverage = float(pos.get("leverage", 1)) if pos.get("leverage") is not None else 1.0
                    side = "LONG" if float(pos.get("positionAmt", 0)) > 0 else "SHORT"
                    liq = pos.get("liquidationPrice") or pos.get("liquidation_price") or None
                    if liq is None:
                        liq = self._estimate_liquidation_price(entry_price=entry_price, leverage=leverage, side=side, maintenance=maintenance_default)
                    liquidation_infos.append({
                        "symbol": pos.get("symbol", symbol),
                        "entry_price": entry_price,
                        "leverage": leverage,
                        "side": side,
                        "estimated_liq_price": float(liq)
                    })
                    leverage_ratios.append(float(leverage))
                except Exception:
                    continue

            avg_leverage = float(np.mean(leverage_ratios)) if len(leverage_ratios) > 0 else 1.0
            max_leverage = float(np.max(leverage_ratios)) if len(leverage_ratios) > 0 else 1.0

            # dynamic Sharpe/Sortino
            perf_cfg = cfg.get("performance", {})
            rolling_window = perf_cfg.get("rolling_window", 63)
            rf = perf_cfg.get("risk_free_rate", 0.0)
            sharpe_series = rolling_sharpe(period_returns, window=rolling_window, rf_per_period=rf)
            sortino_series = rolling_sortino(period_returns, window=rolling_window, rf_per_period=rf)
            latest_sharpe = float(sharpe_series.dropna().iloc[-1]) if len(sharpe_series.dropna()) > 0 else 0.0
            latest_sortino = float(sortino_series.dropna().iloc[-1]) if len(sortino_series.dropna()) > 0 else 0.0

            # Compose component scores
            v95 = var_results.get("VaR_95", 0.0)
            cv95 = cvar_results.get("CVaR_95", 0.0)

            comp_var = self._score_from_component(v95, higher_is_riskier=True, clip=(0.0, 0.10))
            comp_cvar = self._score_from_component(cv95, higher_is_riskier=True, clip=(0.0, 0.15))
            cap_lev = cfg.get("leverage", {}).get("max_leverage", 125)
            comp_leverage = self._score_from_component(avg_leverage, higher_is_riskier=True, clip=(1.0, min(cap_lev, 50.0)))
            comp_vol_target = self._score_from_component(1.0 / (vol_targeting_factor + 1e-9), higher_is_riskier=True, clip=(0.0, 2.0))
            comp_mdd = self._score_from_component(md, higher_is_riskier=True, clip=(0.0, 0.5))
            
            if latest_price > 0:
                atr_dist = (latest_price - atr_stop) / latest_price
            else:
                atr_dist = 1.0
            comp_atr = self._score_from_component(atr_dist, higher_is_riskier=False, clip=(0.0, 0.5))

            # ✅ ANALYSIS_HELPERS ILE AGIRLIKLI ORTALAMA
            component_scores = {
                "var": comp_var,
                "cvar": comp_cvar,
                "leverage": comp_leverage,
                "vol_targeting": comp_vol_target,
                "max_drawdown": comp_mdd,
                "atr_stop": comp_atr
            }

            if self.weights and self.helpers.validate_score_dict(component_scores):
                score = self.helpers.calculate_weights(component_scores, self.weights)
            else:
                # Fallback: simple average
                score = np.mean(list(component_scores.values())) if component_scores else 0.5
            
            # ✅ NORMALIZE SCORE
            score = self.helpers.normalize_score(score)

            # signal interpretation
            t_high = self.thresholds.get("high_risk", 0.75)
            t_med = self.thresholds.get("medium_risk", 0.45)
            
            if score >= t_high:
                signal = "high"
            elif score >= t_med:
                signal = "medium"
            else:
                signal = "low"

            # ✅ ANALYSIS_HELPERS UYUMLU EXPLAIN
            explain = {
                "summary": f"Risk analysis indicates {signal} risk level",
                "confidence": self._calculate_confidence(component_scores),
                "key_metrics": {
                    "annual_volatility": ann_vol,
                    "max_drawdown": md,
                    "avg_leverage": avg_leverage,
                    "var_95": v95,
                    "cvar_95": cv95
                },
                "interpretation": self._interpret_risk_score(score, signal),
                "recommendation": self._generate_risk_recommendation(score, signal)
            }

            # ✅ ANALYSIS_HELPERS UYUMLU OUTPUT FORMATI
            output = {
                "score": float(score),
                "signal": signal,
                "confidence": explain["confidence"],
                "components": component_scores,
                "explain": explain,
                "timestamp": self.helpers.get_timestamp(),
                "module": self.module_name,
                "symbol": symbol,
                "detailed_metrics": {
                    "latest_price": latest_price,
                    "ann_vol": ann_vol,
                    "atr": latest_atr,
                    "atr_stop": atr_stop,
                    "var_results": var_results,
                    "cvar_results": cvar_results,
                    "vol_targeting_factor": vol_targeting_factor,
                    "leverage_info": {
                        "avg_leverage": avg_leverage,
                        "max_leverage": max_leverage,
                        "positions_count": len(positions)
                    },
                    "performance_metrics": {
                        "sharpe": latest_sharpe,
                        "sortino": latest_sortino
                    }
                }
            }

            # ✅ OUTPUT VALIDATION
            if not self.helpers.validate_output(output):
                logger.warning("Output validation failed, using fallback")
                return self.helpers.create_fallback_output(self.module_name, "Output validation failed")

            return output

        except Exception as e:
            logger.error(f"Error computing risk for {symbol}: {str(e)}")
            return self.helpers.create_fallback_output(self.module_name, str(e))


    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """
        Gelişmiş confidence:
          - component_variance: düşük varyans = yüksek güven
          - component_coverage: yeterli sayıda risk faktörü varsa güven artar
        """
        if not components:
            return 0.0

        comp_vals = list(components.values())
        n_factors = len(comp_vals)
        variance = np.var(comp_vals)
        max_possible_var = 0.25  # [0,1] ölçeğinde beklenen sınır

        consistency = 1.0 - min(1.0, variance / max_possible_var)
        coverage = min(1.0, n_factors / 5.0)  # 5+ faktör varsa tam puan

        w_cons, w_cov = 0.7, 0.3
        conf = w_cons * consistency + w_cov * coverage
        return float(max(0.0, min(1.0, conf)))



    def _interpret_risk_score(self, score: float, signal: str) -> str:
        """Generate human-readable interpretation of risk score"""
        if score >= 0.8:
            return "Very high risk exposure with significant drawdown potential"
        elif score >= 0.7:
            return "High risk level requiring immediate attention"
        elif score >= 0.5:
            return "Elevated risk with moderate exposure"
        elif score >= 0.3:
            return "Moderate risk within acceptable parameters"
        else:
            return "Low risk exposure with comfortable margin"

    def _generate_risk_recommendation(self, score: float, signal: str) -> str:
        """Generate risk management recommendations"""
        if signal == "high":
            return "Consider reducing position sizes, increasing stops, or hedging exposure"
        elif signal == "medium":
            return "Monitor positions closely and consider partial profit taking"
        else:
            return "Current risk levels are manageable, maintain existing risk controls"

    async def run(self, symbols: Sequence[str], priority: int = 0) -> Dict[str, Any]:
        """
        Backwards compatible run method
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        tasks = [self.compute_metrics(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                output[symbol] = self.helpers.create_fallback_output(self.module_name, str(result))
            else:
                output[symbol] = result
                
        return output

    def get_metadata(self) -> Dict[str, Any]:
        """Return module metadata"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "description": "Risk exposure analysis and management",
            "metrics": list(self.weights.keys()),
            "parallel_mode": "batch",
            "lifecycle": "development",
            "analysis_helpers_compatible": True,
            "requires_private_data": True
        }

# Factory function for module creation
def create_risk_exposure_module(config: Dict = None, data_provider: Any = None) -> RiskExposureModule:
    """
    Factory function to create RiskExposureModule instance
    """
    return RiskExposureModule(config=config, data_provider=data_provider)

# Backwards-compatible function
async def run(symbols: Sequence[str], priority: int = 0, config: Optional[Dict[str, Any]] = None, data_provider: Optional[Any] = None):
    """
    Convenience top-level run when module used standalone
    """
    if data_provider is None:
        raise ValueError("data_provider must be provided to run()")
    
    mod = RiskExposureModule(config=config, data_provider=data_provider)
    return await mod.run(symbols, priority=priority)
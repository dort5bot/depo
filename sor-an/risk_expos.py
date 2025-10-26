# analysis/risk_expos.py
"""
analysis/risk_expos.py
Version: 1.9.9
Risk & Exposure Management Module - Analysis Helpers Uyumlu + Polars Optimized

Exports:
    - class RiskExposureModule(BaseAnalysisModule)
    - run(symbols, priority) function for backward compatibility

Purpose:
    Calculate risk metrics for spot + futures using Polars for performance
    Full async support with multi-user isolation
"""

from __future__ import annotations

import asyncio
import math
import time
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from polars import col, lit

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput
from analysis.analysis_base_module import BaseAnalysisModule

logger = logging.getLogger(__name__)

# --- Polars Helpers & numeric functions ---

def ensure_pl_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure Polars DataFrame has columns: ['open','high','low','close','volume'] and timestamp index.
    """
    if df is None or df.is_empty():
        return pl.DataFrame(schema={
            'timestamp': pl.Datetime(time_zone='UTC'),
            'open': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'close': pl.Float64,
            'volume': pl.Float64
        })
    
    # Convert to Polars if needed
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if 'datetime' in df.columns:
            df = df.rename({'datetime': 'timestamp'})
        else:
            # Assume first column is timestamp or use index
            df = df.with_columns(pl.first().alias('timestamp'))
    
    # Standardize column names
    column_mapping = {}
    for col_name in df.columns:
        lower_col = col_name.lower()
        if lower_col in ['open', 'o']:
            column_mapping[col_name] = 'open'
        elif lower_col in ['high', 'h']:
            column_mapping[col_name] = 'high'
        elif lower_col in ['low', 'l']:
            column_mapping[col_name] = 'low'
        elif lower_col in ['close', 'c']:
            column_mapping[col_name] = 'close'
        elif lower_col in ['volume', 'v', 'vol']:
            column_mapping[col_name] = 'volume'
        elif lower_col in ['timestamp', 'datetime', 'time']:
            column_mapping[col_name] = 'timestamp'
    
    if column_mapping:
        df = df.rename(column_mapping)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for req_col in required_cols:
        if req_col not in df.columns:
            if req_col == 'timestamp':
                df = df.with_columns(pl.arange(0, df.height).cast(pl.Datetime).alias('timestamp'))
            else:
                df = df.with_columns(pl.lit(0.0).alias(req_col))
    
    return df.select(required_cols).sort('timestamp')

def true_range_pl(df: pl.DataFrame) -> pl.Series:
    """Compute True Range using Polars"""
    high = col("high")
    low = col("low")
    prev_close = col("close").shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    return df.select(
        pl.max_horizontal(tr1, tr2, tr3).alias("true_range")
    )["true_range"]

def atr_pl(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Compute ATR using Polars"""
    tr = true_range_pl(df)
    return tr.ewm_mean(span=period, adjust=False)

def max_drawdown_pl(close_series: pl.Series) -> float:
    """Returns maximum drawdown using Polars"""
    if close_series.is_empty():
        return 0.0
    
    cumulative_max = close_series.cum_max()
    drawdown = (cumulative_max - close_series) / cumulative_max
    return float(drawdown.max())

def historical_var_pl(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical VaR using Polars"""
    if returns.is_empty():
        return 0.0
    
    q = returns.quantile(1 - confidence)
    return float(max(0.0, -q))

def historical_cvar_pl(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical CVaR using Polars"""
    if returns.is_empty():
        return 0.0
    
    threshold = returns.quantile(1 - confidence)
    tail = returns.filter(returns <= threshold)
    
    if tail.is_empty():
        return 0.0
    
    return float(max(0.0, -tail.mean()))

def rolling_sharpe_pl(returns: pl.Series, window: int, rf_per_period: float = 0.0) -> pl.Series:
    """Rolling Sharpe ratio using Polars"""
    excess = returns - rf_per_period
    roll_mean = excess.rolling_mean(window_size=window, min_periods=1)
    roll_std = excess.rolling_std(window_size=window, min_periods=1)
    return roll_mean / roll_std.replace(0.0, None)

def rolling_sortino_pl(returns: pl.Series, window: int, rf_per_period: float = 0.0) -> pl.Series:
    """Rolling Sortino ratio using Polars"""
    def sortino_calculation(window_returns: pl.Series) -> float:
        if window_returns.is_empty():
            return float('nan')
        
        arr = window_returns.to_numpy()
        downside = arr[arr < 0]
        if len(downside) == 0:
            return float('nan')
        
        mean_return = np.mean(arr)
        downside_std = np.std(downside, ddof=1)
        return mean_return / downside_std if downside_std > 0 else float('nan')
    
    excess_returns = returns - rf_per_period
    return excess_returns.rolling_map(sortino_calculation, window_size=window)

# --- User Session Management for Multi-User Support ---

@dataclass
class UserSession:
    """User session data for multi-user isolation"""
    user_id: str
    created_at: datetime
    last_activity: datetime
    symbols: Set[str]
    resource_usage: Dict[str, Any]
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds() < timeout_seconds
    
    def update_activity(self):
        self.last_activity = datetime.now(timezone.utc)

# --- Module Implementation ---

class RiskExposureModule(BaseAnalysisModule):
    """
    Risk & Exposure Module - Polars Optimized + Multi-User Async
    """

    def __init__(self, config: Dict[str, Any], data_provider: Optional[Any] = None, metrics_client: Optional[Any] = None):
        super().__init__(config)
        
        # ✅ ANALYSIS_HELPERS INTEGRATION
        self.helpers = AnalysisHelpers
        self.module_name = "risk_expos"
        self.version = "2.0.0"
        
        # Load configuration
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
        self.multi_user_config = self.config_dict.get("multi_user", {})
        
        self.data_provider = data_provider
        self.metrics_client = metrics_client
        
        # ✅ MULTI-USER SESSION MANAGEMENT
        self.user_sessions: Dict[str, UserSession] = {}
        self.session_lock = asyncio.Lock()
        
        # ✅ POLARS OPTIMIZED EXECUTOR
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.parameters.get("parallel", {}).get("cpu_bound_pool_workers", 4)
        )
        self._loop = asyncio.get_event_loop()

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config with Polars optimizations"""
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
                "ohlcv": {"interval": "1h", "lookback_bars": 1000, "batch_size": 50},
                "atr": {"period": 21, "multiplier": 3.0},
                "var": {"confidence_levels": [0.95, 0.99], "batch_calculation": True},
                "vol_target": {"target_volatility": 0.12},
                "leverage": {"max_leverage": 125},
                "polars": {"streaming": True, "lazy_optimization": True}
            },
            "multi_user": {
                "user_isolation": True,
                "max_users_per_instance": 100,
                "user_timeout_seconds": 300
            }
        }

    async def _get_user_session(self, user_id: str) -> UserSession:
        """Get or create user session with thread-safe access"""
        async with self.session_lock:
            now = datetime.now(timezone.utc)
            
            # Cleanup expired sessions
            expired_users = [
                uid for uid, session in self.user_sessions.items() 
                if not session.is_active(self.multi_user_config.get("user_timeout_seconds", 300))
            ]
            for uid in expired_users:
                del self.user_sessions[uid]
            
            # Check user limit
            max_users = self.multi_user_config.get("max_users_per_instance", 100)
            if user_id not in self.user_sessions and len(self.user_sessions) >= max_users:
                # Remove oldest session
                oldest_user = min(self.user_sessions.keys(), 
                                key=lambda uid: self.user_sessions[uid].last_activity)
                del self.user_sessions[oldest_user]
            
            # Create or update session
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = UserSession(
                    user_id=user_id,
                    created_at=now,
                    last_activity=now,
                    symbols=set(),
                    resource_usage={"memory_mb": 0, "compute_time": 0, "symbol_count": 0}
                )
            else:
                self.user_sessions[user_id].update_activity()
            
            return self.user_sessions[user_id]

    async def _check_user_quota(self, user_id: str, symbols: List[str]) -> bool:
        """Check if user has sufficient quota for requested symbols"""
        if not self.multi_user_config.get("user_isolation", True):
            return True
            
        session = await self._get_user_session(user_id)
        quota = self.multi_user_config.get("resource_quota_per_user", {})
        
        max_symbols = quota.get("max_symbols", 50)
        if len(symbols) > max_symbols:
            logger.warning(f"User {user_id} exceeded symbol quota: {len(symbols)} > {max_symbols}")
            return False
            
        return True

    async def _safe_fetch_klines(self, symbol: str, interval: str, limit: int) -> pl.DataFrame:
        """
        Async fetch klines with Polars DataFrame return
        """
        if self.data_provider is None:
            return pl.DataFrame()
        
        try:
            if asyncio.iscoroutinefunction(self.data_provider.get_klines):
                res = await self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                res = await self._loop.run_in_executor(
                    self._thread_pool,
                    lambda: self.data_provider.get_klines(symbol=symbol, interval=interval, limit=limit)
                )
            
            return ensure_pl_ohlcv(res)
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pl.DataFrame()

    async def _safe_fetch_positions(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Fetch futures position risk with user isolation
        """
        if self.data_provider is None:
            return []
        
        try:
            # User-specific position fetching if supported
            if hasattr(self.data_provider, "get_futures_position_risk"):
                fn = self.data_provider.get_futures_position_risk
                if asyncio.iscoroutinefunction(fn):
                    res = await fn(user_id=user_id)
                else:
                    res = await self._loop.run_in_executor(
                        self._thread_pool,
                        lambda: fn(user_id=user_id)
                    )
                if isinstance(res, dict):
                    return [res]
                return list(res or [])
            return []
        except Exception as e:
            logger.error(f"Error fetching positions for user {user_id}: {e}")
            return []

    async def compute_metrics(self, symbol: str, interval: Optional[str] = None, 
                            lookback: Optional[int] = None, user_id: str = "default") -> Dict[str, Any]:
        """
        Compute risk metrics for a single symbol - Polars Optimized + Async
        """
        start_time = time.time()
        
        try:
            # ✅ USER QUOTA CHECK
            if not await self._check_user_quota(user_id, [symbol]):
                return self.helpers.create_fallback_output(
                    self.module_name, 
                    f"User quota exceeded for symbol {symbol}"
                )

            cfg = self.parameters
            interval = interval or cfg.get("ohlcv", {}).get("interval", "1h")
            lookback = lookback or cfg.get("ohlcv", {}).get("lookback_bars", 1000)

            # ✅ ASYNC DATA FETCHING
            df = await self._safe_fetch_klines(symbol=symbol, interval=interval, limit=lookback)
            if df.is_empty():
                return self.helpers.create_fallback_output(self.module_name, "No price data available")

            # ✅ POLARS OPTIMIZED CALCULATIONS
            # Compute returns
            close_prices = df["close"].cast(pl.Float64)
            returns = close_prices.pct_change().drop_nulls()
            
            if returns.is_empty():
                return self.helpers.create_fallback_output(self.module_name, "Insufficient data for returns calculation")

            # Volatility calculation
            std_per_period = returns.std()
            ann_vol = annualize_vol(std_per_period, self._get_periods_per_year(interval))

            # Max drawdown
            md = max_drawdown_pl(close_prices)

            # ATR stop
            atr_period = cfg.get("atr", {}).get("period", 21)
            atr_multiplier = cfg.get("atr", {}).get("multiplier", 3.0)
            atr_series = atr_pl(df, period=atr_period)
            latest_atr = atr_series[-1] if not atr_series.is_empty() else 0.0
            latest_price = close_prices[-1]
            atr_stop = max(0.0, latest_price - atr_multiplier * latest_atr)

            # VaR / CVaR
            var_cfg = cfg.get("var", {})
            var_conf_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
            var_results = {}
            cvar_results = {}
            
            for confidence in var_conf_levels:
                var_val = historical_var_pl(returns, confidence=confidence)
                cvar_val = historical_cvar_pl(returns, confidence=confidence)
                var_results[f"VaR_{int(confidence*100)}"] = var_val
                cvar_results[f"CVaR_{int(confidence*100)}"] = cvar_val

            # Volatility targeting
            vt_cfg = cfg.get("vol_target", {})
            target_vol = vt_cfg.get("target_volatility", 0.12)
            vol_targeting_factor = target_vol / ann_vol if ann_vol > 0 else 1.0
            vol_targeting_factor = min(3.0, vol_targeting_factor)

            # Positions & leverage with user isolation
            positions = await self._safe_fetch_positions(user_id)
            leverage_ratios = []
            liquidation_infos = []
            maintenance_default = cfg.get("leverage", {}).get("default_maintenance_margin", 0.005)
            
            for pos in positions:
                try:
                    if pos.get("symbol") != symbol:
                        continue
                        
                    entry_price = float(pos.get("entryPrice", pos.get("entry_price", latest_price)))
                    leverage = float(pos.get("leverage", 1)) if pos.get("leverage") is not None else 1.0
                    side = "LONG" if float(pos.get("positionAmt", 0)) > 0 else "SHORT"
                    
                    liq = pos.get("liquidationPrice") or pos.get("liquidation_price")
                    if liq is None:
                        liq = self._estimate_liquidation_price(
                            entry_price=entry_price, 
                            leverage=leverage, 
                            side=side, 
                            maintenance=maintenance_default
                        )
                    
                    liquidation_infos.append({
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "leverage": leverage,
                        "side": side,
                        "estimated_liq_price": float(liq)
                    })
                    leverage_ratios.append(float(leverage))
                except Exception as e:
                    logger.warning(f"Error processing position for {symbol}: {e}")
                    continue

            avg_leverage = float(np.mean(leverage_ratios)) if leverage_ratios else 1.0
            max_leverage = float(np.max(leverage_ratios)) if leverage_ratios else 1.0

            # Performance metrics with Polars
            perf_cfg = cfg.get("performance", {})
            rolling_window = perf_cfg.get("rolling_window", 63)
            rf = perf_cfg.get("risk_free_rate", 0.0)
            
            sharpe_series = rolling_sharpe_pl(returns, window=rolling_window, rf_per_period=rf)
            sortino_series = rolling_sortino_pl(returns, window=rolling_window, rf_per_period=rf)
            
            latest_sharpe = sharpe_series[-1] if not sharpe_series.is_empty() else 0.0
            latest_sortino = sortino_series[-1] if not sortino_series.is_empty() else 0.0

            # ✅ COMPONENT SCORING
            v95 = var_results.get("VaR_95", 0.0)
            cv95 = cvar_results.get("CVaR_95", 0.0)

            component_scores = {
                "var": self._score_from_component(v95, higher_is_riskier=True, clip=(0.0, 0.10)),
                "cvar": self._score_from_component(cv95, higher_is_riskier=True, clip=(0.0, 0.15)),
                "leverage": self._score_from_component(
                    avg_leverage, 
                    higher_is_riskier=True, 
                    clip=(1.0, min(cfg.get("leverage", {}).get("max_leverage", 125), 50.0))
                ),
                "vol_targeting": self._score_from_component(
                    1.0 / (vol_targeting_factor + 1e-9), 
                    higher_is_riskier=True, 
                    clip=(0.0, 2.0)
                ),
                "max_drawdown": self._score_from_component(md, higher_is_riskier=True, clip=(0.0, 0.5)),
                "atr_stop": self._score_from_component(
                    (latest_price - atr_stop) / latest_price if latest_price > 0 else 1.0,
                    higher_is_riskier=False, 
                    clip=(0.0, 0.5)
                )
            }

            # ✅ FINAL SCORE CALCULATION
            if self.weights and self.helpers.validate_score_dict(component_scores):
                score = self.helpers.calculate_weights(component_scores, self.weights)
            else:
                score = np.mean(list(component_scores.values())) if component_scores else 0.5
            
            score = self.helpers.normalize_score(score)

            # ✅ SIGNAL INTERPRETATION
            signal = self._classify_risk_signal(score)

            # ✅ EXPLAINABLE OUTPUT
            explain = {
                "summary": f"Risk analysis indicates {signal} risk level for {symbol}",
                "confidence": self._calculate_confidence(component_scores),
                "key_metrics": {
                    "annual_volatility": ann_vol,
                    "max_drawdown": md,
                    "avg_leverage": avg_leverage,
                    "var_95": v95,
                    "cvar_95": cv95
                },
                "interpretation": self._interpret_risk_score(score, signal),
                "recommendation": self._generate_risk_recommendation(score, signal),
                "user_id": user_id,
                "computation_time": time.time() - start_time
            }

            # ✅ ANALYSIS_HELPERS UYUMLU OUTPUT
            output = {
                "score": float(score),
                "signal": signal,
                "confidence": explain["confidence"],
                "components": component_scores,
                "explain": explain,
                "timestamp": self.helpers.get_timestamp(),
                "module": self.module_name,
                "symbol": symbol,
                "user_id": user_id,
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

            # ✅ UPDATE USER RESOURCE USAGE
            if self.multi_user_config.get("user_isolation", True):
                async with self.session_lock:
                    if user_id in self.user_sessions:
                        session = self.user_sessions[user_id]
                        session.resource_usage["compute_time"] += time.time() - start_time
                        session.resource_usage["symbol_count"] = len(session.symbols)
                        session.symbols.add(symbol)

            # ✅ OUTPUT VALIDATION
            if not self.helpers.validate_output(output):
                logger.warning(f"Output validation failed for {symbol}, using fallback")
                return self.helpers.create_fallback_output(self.module_name, "Output validation failed")

            return output

        except Exception as e:
            logger.error(f"Error computing risk for {symbol}: {str(e)}")
            return self.helpers.create_fallback_output(self.module_name, str(e))

    def _get_periods_per_year(self, interval: str) -> int:
        """Calculate periods per year for volatility annualization"""
        if 'm' in interval and interval.endswith('m'):
            minutes = int(interval[:-1])
            return int((60 / minutes) * 24 * 365)
        elif 'h' in interval:
            hours = int(interval[:-1])
            return int((24 / hours) * 365)
        elif 'd' in interval:
            days = int(interval[:-1])
            return int(365 / days)
        else:
            return 24 * 365  # Default to hourly

    def _estimate_liquidation_price(self, entry_price: float, leverage: float, side: str = "LONG", maintenance: float = 0.005) -> float:
        """Conservative liquidation price estimation"""
        if leverage <= 1:
            return float(entry_price)
        
        if side.upper() == "LONG":
            liq = entry_price * (1 - (1.0 / max(leverage, 1.0)) - maintenance)
        else:
            liq = entry_price * (1 + (1.0 / max(leverage, 1.0)) + maintenance)
        
        return float(max(0.0, liq))

    def _score_from_component(self, value: float, higher_is_riskier: bool = True, clip: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Map raw metric to [0..1] risk score"""
        lo, hi = clip
        if hi == lo:
            return 0.0
        
        normalized = (float(value) - lo) / (hi - lo)
        if not higher_is_riskier:
            normalized = 1.0 - normalized
        
        return float(max(0.0, min(1.0, normalized)))

    def _classify_risk_signal(self, score: float) -> str:
        """Classify risk score into signal category"""
        t_high = self.thresholds.get("high_risk", 0.75)
        t_med = self.thresholds.get("medium_risk", 0.45)
        t_extreme = self.thresholds.get("extreme_risk", 0.90)
        
        if score >= t_extreme:
            return "extreme"
        elif score >= t_high:
            return "high"
        elif score >= t_med:
            return "medium"
        else:
            return "low"

    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence score based on component consistency"""
        if not components:
            return 0.0

        comp_vals = list(components.values())
        n_factors = len(comp_vals)
        variance = np.var(comp_vals)
        max_possible_var = 0.25

        consistency = 1.0 - min(1.0, variance / max_possible_var)
        coverage = min(1.0, n_factors / 5.0)

        w_cons, w_cov = 0.7, 0.3
        conf = w_cons * consistency + w_cov * coverage
        return float(max(0.0, min(1.0, conf)))

    def _interpret_risk_score(self, score: float, signal: str) -> str:
        """Generate human-readable interpretation"""
        if score >= 0.9:
            return "Extreme risk exposure - Immediate action required"
        elif score >= 0.8:
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
        if signal == "extreme":
            return "CRITICAL: Immediately reduce exposure, increase stops, consider hedging or exiting positions"
        elif signal == "high":
            return "URGENT: Consider reducing position sizes, increasing stops, or hedging exposure"
        elif signal == "medium":
            return "CAUTION: Monitor positions closely and consider partial profit taking"
        else:
            return "NORMAL: Current risk levels are manageable, maintain existing risk controls"

    async def run(self, symbols: Sequence[str], priority: int = 0, user_id: str = "default") -> Dict[str, Any]:
        """
        Main run method with multi-user support and Polars optimization
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # ✅ USER QUOTA VALIDATION
        if not await self._check_user_quota(user_id, list(symbols)):
            return {
                symbol: self.helpers.create_fallback_output(
                    self.module_name, 
                    f"User quota exceeded for {symbol}"
                )
                for symbol in symbols
            }

        # ✅ BATCH PROCESSING WITH POLARS OPTIMIZATION
        cfg = self.parameters.get("parallel", {})
        chunk_size = cfg.get("chunk_size", 25)
        max_concurrent = cfg.get("max_concurrent_tasks", 10)

        # Process symbols in optimized chunks
        results = {}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            
            # Limit concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(symbol: str):
                async with semaphore:
                    return await self.compute_metrics(symbol, user_id=user_id)
            
            tasks = [process_with_semaphore(symbol) for symbol in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(chunk, chunk_results):
                if isinstance(result, Exception):
                    results[symbol] = self.helpers.create_fallback_output(
                        self.module_name, 
                        f"Error processing {symbol}: {str(result)}"
                    )
                else:
                    results[symbol] = result

        # ✅ AGGREGATE RESULTS
        output = {
            "module": self.module_name,
            "version": self.version,
            "timestamp": self.helpers.get_timestamp(),
            "user_id": user_id,
            "symbols_processed": len(symbols),
            "results": results,
            "summary": {
                "total_symbols": len(symbols),
                "successful_symbols": len([r for r in results.values() if r.get("score") is not None]),
                "average_risk_score": np.mean([r.get("score", 0.5) for r in results.values()]),
                "risk_distribution": {
                    "extreme": len([r for r in results.values() if r.get("signal") == "extreme"]),
                    "high": len([r for r in results.values() if r.get("signal") == "high"]),
                    "medium": len([r for r in results.values() if r.get("signal") == "medium"]),
                    "low": len([r for r in results.values() if r.get("signal") == "low"])
                }
            }
        }

        return output

    async def close(self):
        """Cleanup resources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        # Clear user sessions
        async with self.session_lock:
            self.user_sessions.clear()

# ✅ BACKWARD COMPATIBILITY FUNCTION
async def run(symbols: Sequence[str], priority: int = 0, user_id: str = "default") -> Dict[str, Any]:
    """
    Standalone run function for backward compatibility
    """
    module = RiskExposureModule(config=None)
    try:
        result = await module.run(symbols, priority, user_id)
        return result
    finally:
        await module.close()

# ✅ POLARS HELPER FUNCTIONS

def annualize_vol(period_vol: float, periods_per_year: int) -> float:
    """Annualize volatility"""
    return float(period_vol * math.sqrt(periods_per_year))

def calculate_returns_pl(close_prices: pl.Series) -> pl.Series:
    """Calculate returns using Polars"""
    return close_prices.pct_change().drop_nulls()

if __name__ == "__main__":
    # Test the module
    async def test():
        module = RiskExposureModule(config=None)
        result = await module.run(["BTCUSDT", "ETHUSDT"])
        print("Test completed successfully")
        print(f"Processed {len(result['results'])} symbols")
        await module.close()
    
    asyncio.run(test())
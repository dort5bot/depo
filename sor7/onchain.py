# 📁 metrics/onchain.py
"""
MAPS Framework - On-Chain Metrics (Polars Optimized, Async-Safe)
Version: 1.2.0
Author: ysf-bot-framework

Standardized on-chain metrics with full async support and NaN protection.
Integrated with MetricStandard interface for seamless data type conversion.

Veri Modeli : Polars (saf)
Yapı Tipi   : tam - ASYNC  
Giriş/Çıkış : async input, async compute
"""

import polars as pl
import numpy as np
import asyncio
from typing import Optional, Callable, Dict, Any, Union
from .standard import MetricStandard, metric_standard

# ==========================================================
# === Core Async-Safe Computation Engine ===================
# ==========================================================

class OnChainComputeEngine:
    """Async-safe computation engine for on-chain metrics"""
    
    @staticmethod
    def ensure_series(data) -> pl.Series:
        """Ensure input is Polars Series with async compatibility"""
        if isinstance(data, pl.Series):
            return data
        if isinstance(data, (list, np.ndarray, tuple)):
            return pl.Series(data)
        if isinstance(data, pl.DataFrame):
            return data.to_series(0)
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return pl.from_pandas(data).to_series(0)
        raise TypeError(f"Unsupported type: {type(data)}")

    @staticmethod
    def sanitize_series(s: pl.Series) -> pl.Series:
        """Replace NaN, Inf, None with 0 for async safety"""
        if s.is_empty():
            return s
            
        s = s.fill_nan(0)
        s = s.fill_null(0)
        s = s.replace(float("inf"), 0)
        s = s.replace(float("-inf"), 0)
        return s

    @staticmethod
    def safe_compute(func: Callable, *args, **kwargs) -> pl.Series:
        """Safe compute wrapper with comprehensive NaN/Inf protection"""
        try:
            # Convert and sanitize all inputs
            args = [OnChainComputeEngine.sanitize_series(
                OnChainComputeEngine.ensure_series(a)
            ) for a in args]
            
            # Execute computation
            result = func(*args, **kwargs)
            
            # Standardize output
            if isinstance(result, pl.Series):
                return OnChainComputeEngine.sanitize_series(result)
            elif isinstance(result, pl.DataFrame):
                return OnChainComputeEngine.sanitize_series(result.to_series(0))
            elif np.isscalar(result):
                return pl.Series([float(result)])
            else:
                return OnChainComputeEngine.sanitize_series(pl.Series(result))
                
        except Exception as e:
            # Return safe fallback series
            n = len(args[0]) if args else 1
            return pl.Series([0.0] * n)  # Use 0 instead of NaN for async safety

    @staticmethod
    async def compute_async(func: Callable, *args, **kwargs) -> pl.Series:
        """Async wrapper for metric compute with thread safety"""
        return await asyncio.to_thread(OnChainComputeEngine.safe_compute, func, *args, **kwargs)


# ==========================================================
# === Standardized Metric Functions ========================
# ==========================================================

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def etf_net_flow(inflow: pl.Series, outflow: pl.Series) -> pl.Series:
    """ETF Net Flow = Inflow - Outflow"""
    return inflow - outflow

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def exchange_netflow(deposits: pl.Series, withdrawals: pl.Series) -> pl.Series:
    """Exchange Netflow = Deposits - Withdrawals"""
    return deposits - withdrawals

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def stablecoin_flow(stable_in: pl.Series, stable_out: pl.Series) -> pl.Series:
    """Stablecoin Flow = Stable In - Stable Out"""
    return stable_in - stable_out

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def net_realized_pl(realized_profit: pl.Series, realized_loss: pl.Series) -> pl.Series:
    """Net Realized Profit/Loss = Profit - Loss"""
    return realized_profit - realized_loss

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def realized_cap(price_series: pl.Series, realized_price: pl.Series) -> pl.Series:
    """Realized Cap = mean(price * realized_price)"""
    product = price_series * realized_price
    return pl.Series([product.mean()]) if len(product) > 0 else pl.Series([0.0])

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def nupl(market_cap: pl.Series, realized_cap: pl.Series) -> pl.Series:
    """NUPL = (Market Cap - Realized Cap) / Market Cap"""
    denominator = market_cap.replace(0, 1e-9)  # Avoid division by zero
    return (market_cap - realized_cap) / denominator

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def exchange_whale_ratio(whale_deposits: pl.Series, total_deposits: pl.Series) -> pl.Series:
    """Whale Ratio = Top 10 inflow wallets / Total inflow"""
    denominator = total_deposits.replace(0, 1e-9)  # Avoid division by zero
    return whale_deposits / denominator

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def mvrv_zscore(market_cap: pl.Series, realized_cap: pl.Series, std_dev: pl.Series) -> pl.Series:
    """MVRV Z-Score = (Market Cap - Realized Cap) / Std(Market Cap)"""
    denominator = std_dev.replace(0, 1e-9)  # Avoid division by zero
    return (market_cap - realized_cap) / denominator

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def sopr(realized_value: pl.Series, spent_value: pl.Series) -> pl.Series:
    """SOPR = Realized Value / Spent Value"""
    denominator = spent_value.replace(0, 1e-9)  # Avoid division by zero
    return realized_value / denominator

@metric_standard(input_type="polars", output_type="polars", min_periods=1, fillna=True)
def etf_flow_composite(
    etf_inflow: pl.Series,
    etf_outflow: pl.Series,
    stable_in: pl.Series,
    stable_out: pl.Series,
    exchange_in: pl.Series,
    exchange_out: pl.Series
) -> pl.Series:
    """ETF Flow Composite = 0.5*NetETF + 0.3*NetStable + 0.2*(-NetExchange)"""
    net_etf = etf_net_flow(etf_inflow, etf_outflow)
    net_stable = stablecoin_flow(stable_in, stable_out)
    net_exchange = exchange_netflow(exchange_in, exchange_out)
    return (0.5 * net_etf) + (0.3 * net_stable) + (0.2 * (-net_exchange))


# ==========================================================
# === Async Computation Interface ==========================
# ==========================================================

class AsyncOnChainMetrics:
    """Async interface for on-chain metrics computation"""
    
    def __init__(self):
        self.compute_engine = OnChainComputeEngine()
    
    async def compute_metric_async(self, metric_func: Callable, *args, **kwargs) -> pl.Series:
        """Compute any metric function asynchronously with safety"""
        return await self.compute_engine.compute_async(metric_func, *args, **kwargs)
    
    # Individual async metric methods
    async def etf_net_flow_async(self, inflow: pl.Series, outflow: pl.Series) -> pl.Series:
        return await self.compute_metric_async(etf_net_flow, inflow, outflow)
    
    async def exchange_netflow_async(self, deposits: pl.Series, withdrawals: pl.Series) -> pl.Series:
        return await self.compute_metric_async(exchange_netflow, deposits, withdrawals)
    
    async def stablecoin_flow_async(self, stable_in: pl.Series, stable_out: pl.Series) -> pl.Series:
        return await self.compute_metric_async(stablecoin_flow, stable_in, stable_out)
    
    async def net_realized_pl_async(self, realized_profit: pl.Series, realized_loss: pl.Series) -> pl.Series:
        return await self.compute_metric_async(net_realized_pl, realized_profit, realized_loss)
    
    async def realized_cap_async(self, price_series: pl.Series, realized_price: pl.Series) -> pl.Series:
        return await self.compute_metric_async(realized_cap, price_series, realized_price)
    
    async def nupl_async(self, market_cap: pl.Series, realized_cap: pl.Series) -> pl.Series:
        return await self.compute_metric_async(nupl, market_cap, realized_cap)
    
    async def exchange_whale_ratio_async(self, whale_deposits: pl.Series, total_deposits: pl.Series) -> pl.Series:
        return await self.compute_metric_async(exchange_whale_ratio, whale_deposits, total_deposits)
    
    async def mvrv_zscore_async(self, market_cap: pl.Series, realized_cap: pl.Series, std_dev: pl.Series) -> pl.Series:
        return await self.compute_metric_async(mvrv_zscore, market_cap, realized_cap, std_dev)
    
    async def sopr_async(self, realized_value: pl.Series, spent_value: pl.Series) -> pl.Series:
        return await self.compute_metric_async(sopr, realized_value, spent_value)
    
    async def etf_flow_composite_async(
        self,
        etf_inflow: pl.Series,
        etf_outflow: pl.Series,
        stable_in: pl.Series,
        stable_out: pl.Series,
        exchange_in: pl.Series,
        exchange_out: pl.Series
    ) -> pl.Series:
        return await self.compute_metric_async(
            etf_flow_composite,
            etf_inflow, etf_outflow,
            stable_in, stable_out,
            exchange_in, exchange_out
        )


# ==========================================================
# === Batch Processing Interface ===========================
# ==========================================================

class OnChainBatchProcessor:
    """Batch processing for multiple on-chain metrics"""
    
    def __init__(self):
        self.async_processor = AsyncOnChainMetrics()
        self.standardizer = MetricStandard()
    
    async def compute_batch_async(self, metrics_config: Dict[str, Dict]) -> Dict[str, pl.Series]:
        """
        Compute multiple metrics asynchronously in batch
        
        Args:
            metrics_config: {
                'metric_name': {
                    'function': metric_function,
                    'args': [arg1, arg2, ...],
                    'kwargs': {'param': value}
                }
            }
        """
        tasks = {}
        
        for metric_name, config in metrics_config.items():
            func = config['function']
            args = config.get('args', [])
            kwargs = config.get('kwargs', {})
            
            tasks[metric_name] = self.async_processor.compute_metric_async(func, *args, **kwargs)
        
        # Execute all async tasks
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results with error handling
        final_results = {}
        for metric_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                # Return safe fallback for failed computations
                n = len(metrics_config[metric_name].get('args', [{}])[0]) if metrics_config[metric_name].get('args') else 1
                final_results[metric_name] = pl.Series([0.0] * n)
            else:
                final_results[metric_name] = result
                
        return final_results


# ==========================================================
# === Export Registry ======================================
# ==========================================================

__all__ = [
    # Core computation
    "OnChainComputeEngine",
    "AsyncOnChainMetrics", 
    "OnChainBatchProcessor",
    
    # Metric functions
    "etf_net_flow",
    "exchange_netflow", 
    "stablecoin_flow",
    "net_realized_pl",
    "realized_cap",
    "nupl",
    "exchange_whale_ratio",
    "mvrv_zscore", 
    "sopr",
    "etf_flow_composite",
    
    # Async computation methods
    "compute_async",  # Backward compatibility
]

# Backward compatibility alias
compute_async = OnChainComputeEngine.compute_async
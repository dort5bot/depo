# 📁 metrics/onchain.py
"""
burdakiler Binance API üzerinden doğrudan alınamaz, sadece hesaplanabilir:
❌ 1. stablecoin_flow
Binance bunu sağlamaz → zincir üstü veri gerekir (Glassnode, CryptoQuant, DefiLlama).
❌ 2. exchange_netflow
Spot/futures netflow verisi (deposit/withdraw aggregate) Binance’te yok.
❌ 3. etf_net_flow
Grayscale, Blackrock ETF verisi → Binance sağlamaz.
❌ 4. cascade_risk
Likidasyon zincirleme risk metriği Binance’te yok.
Sadece bireysel liquidation events var → yetersiz.


stablecoin_flow: "Stablecoin Flow = Stable In - Stable Out, stablecoin_flow"
net_realized_pl: "Net Realized Profit/Loss = Profit - Loss, profitability"
realized_cap: "Realized Cap = mean(price * realized_price), valuation"
nupl: "Net Unrealized Profit/Loss (NUPL) = (Market Cap - Realized Cap) / Market Cap, profitability"
exchange_whale_ratio: "Whale Ratio = Top 10 inflow wallets / Total inflow, exchange_behavior"
mvrv_zscore: "MVRV Z-Score = (Market Cap - Realized Cap) / Std(Market Cap), valuation"
sopr: "SOPR = Realized Value / Spent Value, profitability"
etf_flow_composite: "ETF Flow Composite = 0.5*NetETF + 0.3*NetStable + 0.2*(-NetExchange), composite"

"""
"""
analysis/metrics/onchain.py
Standard template for all metric modules
Date: 2024/12/19
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Any, Union, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "polars",      # pandas, numpy, polars
    "execution_type": "async",   # sync, async
    "category": "onchain"        # technical, regime, risk, etc.
}

# ==================== PURE FUNCTIONS ====================
def etf_net_flow(inflow: pl.Series, outflow: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    ETF Net Flow = Inflow - Outflow
    """
    return inflow - outflow

def exchange_netflow(deposits: pl.Series, withdrawals: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    Exchange Netflow = Deposits - Withdrawals
    """
    return deposits - withdrawals

def stablecoin_flow(stable_in: pl.Series, stable_out: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    Stablecoin Flow = Stable In - Stable Out
    """
    return stable_in - stable_out

def net_realized_pl(realized_profit: pl.Series, realized_loss: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    Net Realized Profit/Loss = Profit - Loss
    """
    return realized_profit - realized_loss

def realized_cap(price_series: pl.Series, realized_price: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    Realized Cap = mean(price * realized_price)
    """
    product = price_series * realized_price
    return pl.Series([product.mean()]) if len(product) > 0 else pl.Series([0.0])

def nupl(market_cap: pl.Series, realized_cap: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    NUPL = (Market Cap - Realized Cap) / Market Cap
    """
    denominator = market_cap.replace(0, 1e-9)
    return (market_cap - realized_cap) / denominator

def exchange_whale_ratio(whale_deposits: pl.Series, total_deposits: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    Whale Ratio = Top 10 inflow wallets / Total inflow
    """
    denominator = total_deposits.replace(0, 1e-9)
    return whale_deposits / denominator

def mvrv_zscore(market_cap: pl.Series, realized_cap: pl.Series, std_dev: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    MVRV Z-Score = (Market Cap - Realized Cap) / Std(Market Cap)
    """
    denominator = std_dev.replace(0, 1e-9)
    return (market_cap - realized_cap) / denominator

def sopr(realized_value: pl.Series, spent_value: pl.Series) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    SOPR = Realized Value / Spent Value
    """
    denominator = spent_value.replace(0, 1e-9)
    return realized_value / denominator

def etf_flow_composite(
    etf_inflow: pl.Series,
    etf_outflow: pl.Series,
    stable_in: pl.Series,
    stable_out: pl.Series,
    exchange_in: pl.Series,
    exchange_out: pl.Series
) -> pl.Series:
    """
    Pure mathematical function - NO standardization
    ETF Flow Composite = 0.5*NetETF + 0.3*NetStable + 0.2*(-NetExchange)
    """
    net_etf = etf_inflow - etf_outflow
    net_stable = stable_in - stable_out
    net_exchange = exchange_in - exchange_out
    return (0.5 * net_etf) + (0.3 * net_stable) + (0.2 * (-net_exchange))

# ==================== MODULE REGISTRY ====================
_METRICS = {
    "etf_net_flow": etf_net_flow,
    "exchange_netflow": exchange_netflow,
    "stablecoin_flow": stablecoin_flow,
    "net_realized_pl": net_realized_pl,
    "realized_cap": realized_cap,
    "nupl": nupl,
    "exchange_whale_ratio": exchange_whale_ratio,
    "mvrv_zscore": mvrv_zscore,
    "sopr": sopr,
    "etf_flow_composite": etf_flow_composite,
}

def get_metrics() -> List[str]:
    """Composite engine için metric listesi"""
    return list(_METRICS.keys())

def get_function(metric_name: str):
    """Composite engine için fonksiyon döndür"""
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    """Module-level configuration"""
    return _MODULE_CONFIG.copy()
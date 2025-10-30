# metrics/onchain.py
"""
On-Chain Metrics for MAPS framework.
Includes ETF net flow, stablecoin flow, exchange netflow, NUPL, SOPR, MVRV, and whale ratio.

Author: ysf-bot-framework
Version: 2025.1
Updated: 2025-10-28

onchain	Polars	Büyük veri setleri için	Async
"""

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


# ==========================================================
# === ETF Net Flow =========================================
# ==========================================================

def etf_net_flow(inflow, outflow):
    """
    ETF Net Flow = Inflow - Outflow
    Indicates institutional capital entering or exiting crypto ETFs.
    """
    inflow = np.asarray(inflow)
    outflow = np.asarray(outflow)
    return inflow - outflow


# ==========================================================
# === Stablecoin Flow ======================================
# ==========================================================

def stablecoin_flow(stable_in, stable_out):
    """
    Stablecoin Flow = Net change in exchange stablecoin balances.
    Positive → inflow to exchanges (potential selling pressure).
    """
    stable_in = np.asarray(stable_in)
    stable_out = np.asarray(stable_out)
    return stable_in - stable_out


# ==========================================================
# === Exchange Netflow =====================================
# ==========================================================

def exchange_netflow(deposits, withdrawals):
    """
    Exchange Netflow = Deposits - Withdrawals
    > 0 → capital moving into exchanges (bearish)
    < 0 → outflow (bullish)
    """
    deposits = np.asarray(deposits)
    withdrawals = np.asarray(withdrawals)
    return deposits - withdrawals


# ==========================================================
# === Net Realized Profit/Loss =============================
# ==========================================================

def net_realized_pl(realized_profit, realized_loss):
    """
    Net Realized P/L = Profit - Loss
    Used to gauge market participants’ net realized outcome.
    """
    realized_profit = np.asarray(realized_profit)
    realized_loss = np.asarray(realized_loss)
    return realized_profit - realized_loss


# ==========================================================
# === Realized Cap =========================================
# ==========================================================

def realized_cap(price_series, realized_price):
    """
    Realized Cap = Σ(Supply_i * Realized_Price_i)
    Approximation using total supply and realized price mean.
    """
    price_series = np.asarray(price_series)
    realized_price = np.asarray(realized_price)
    return np.mean(price_series * realized_price)


# ==========================================================
# === NUPL (Net Unrealized Profit/Loss) ====================
# ==========================================================

def nupl(market_cap, realized_cap):
    """
    NUPL = (Market Cap - Realized Cap) / Market Cap
    > 0 → market in profit, < 0 → market in loss.
    """
    market_cap = np.asarray(market_cap)
    realized_cap = np.asarray(realized_cap)
    return (market_cap - realized_cap) / (market_cap + 1e-9)


# ==========================================================
# === Exchange Whale Ratio ================================
# ==========================================================

def exchange_whale_ratio(whale_deposits, total_deposits):
    """
    Whale Ratio = Top 10 inflow wallets / Total inflow
    High ratio → whales dominating deposits.
    """
    whale_deposits = np.asarray(whale_deposits)
    total_deposits = np.asarray(total_deposits)
    return whale_deposits / (total_deposits + 1e-9)


# ==========================================================
# === MVRV Z-Score =========================================
# ==========================================================

def mvrv_zscore(market_cap, realized_cap, std_dev):
    """
    MVRV Z-Score = (Market Cap - Realized Cap) / Std(Market Cap)
    Indicates overvaluation / undervaluation of the network.
    """
    market_cap = np.asarray(market_cap)
    realized_cap = np.asarray(realized_cap)
    std_dev = np.asarray(std_dev)
    return (market_cap - realized_cap) / (std_dev + 1e-9)


# ==========================================================
# === SOPR (Spent Output Profit Ratio) =====================
# ==========================================================

def sopr(realized_value, spent_value):
    """
    SOPR = Realized Value / Spent Value
    SOPR > 1 → coins sold in profit, < 1 → in loss.
    """
    realized_value = np.asarray(realized_value)
    spent_value = np.asarray(spent_value)
    return realized_value / (spent_value + 1e-9)


# ==========================================================
# === ETF Flow Aggregator ==================================
# ==========================================================

def etf_flow_composite(etf_inflow, etf_outflow, stable_in, stable_out, exchange_in, exchange_out):
    """
    Composite ETF Flow Score combining multiple liquidity streams.
    """
    net_etf = etf_net_flow(etf_inflow, etf_outflow)
    net_stable = stablecoin_flow(stable_in, stable_out)
    net_exchange = exchange_netflow(exchange_in, exchange_out)
    score = 0.5 * net_etf + 0.3 * net_stable + 0.2 * (-net_exchange)
    return score


# Tüm metrik fonksiyonları için template
def metric_template(series, *args, **kwargs):
    # 1. Input validation
    if series is None or len(series) == 0:
        return np.nan
    
    # 2. NaN check
    if isinstance(series, (pd.Series, pd.DataFrame)):
        if series.isna().all():
            return np.nan
        # Forward fill then backward fill
        series = series.ffill().bfill()
    
    # 3. Length check
    if len(series) < kwargs.get('min_periods', 2):
        return np.nan
    
    # 4. Try-except wrapper
    try:
        # Actual calculation
        result = calculate_metric(series, *args, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        return np.nan

# ==========================================================
# === Export ===============================================
# ==========================================================

__all__ = [
    "etf_net_flow",
    "stablecoin_flow",
    "exchange_netflow",
    "net_realized_pl",
    "realized_cap",
    "nupl",
    "exchange_whale_ratio",
    "mvrv_zscore",
    "sopr",
    "etf_flow_composite"
]

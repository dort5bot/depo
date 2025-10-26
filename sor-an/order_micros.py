"""
analysis/order_micros.py
Order Flow & Microstructure module
File: order_micros.py
Config file: c_order.py
analysis/order_micros.py
Order Flow & Microstructure module
Analysis Helpers Uyumlu Versiyon
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import time

import numpy as np
import pandas as pd

from utils.data_sources.data_provider import DataProvider
from analysis.analysis_base_module import BaseAnalysisModule, legacy_compatible
from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers

# Import config
from analysis.config.c_order import OrderFlowConfig, CONFIG

# ---- Data provider interface -------------------------------------------------
class DataProviderInterface:
    """
    Provide snapshot and small-history data to the module.
    In production, implement using utils.binance_api.binance_a or WebSocket stream handlers.
    """

    async def get_order_book(self, symbol: str) -> dict:
        raise NotImplementedError

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[dict]:
        raise NotImplementedError

    async def get_book_ticker(self, symbol: str) -> dict:
        raise NotImplementedError


# ---- Mock Provider for testing/demo ----------------------------------
class MockDataProvider(DataProviderInterface):
    """Generate synthetic orderbook/trade snapshots for local testing - MULTI-USER SUPPORT"""

    def __init__(self, mid_price: float = 100.0, depth_levels: int = 20, thread_safe: bool = True):
        self.mid = mid_price
        self.depth = depth_levels
        self.thread_safe = thread_safe
        np.random.seed(42)
        
        # Multi-user için state management
        self._user_books: Dict[str, Dict] = {}
        self._user_trades: Dict[str, List[dict]] = {}
        self._book_lock = asyncio.Lock()
        self._trade_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Price evolution tracking
        self._price_histories: Dict[str, List[float]] = defaultdict(list)

    async def get_order_book(self, symbol: str) -> dict:
        """Get order book with multi-user isolation"""
        if self.thread_safe:
            return await self._get_order_book_thread_safe(symbol)
        else:
            return await self._get_order_book_basic(symbol)
    
    async def _get_order_book_basic(self, symbol: str) -> dict:
        """Original single-user implementation"""
        bids = []
        asks = []
        step = 0.01 * self.mid
        for i in range(self.depth):
            price_bid = round(self.mid - (i + 0.5) * step, 8)
            price_ask = round(self.mid + (i + 0.5) * step, 8)
            bids.append({"price": price_bid, "qty": float(np.abs(np.random.exponential(scale=2.0)))})
            asks.append({"price": price_ask, "qty": float(np.abs(np.random.exponential(scale=2.0)))})
        return {"bids": bids, "asks": asks, "timestamp": int(time.time() * 1000)}
    
    async def _get_order_book_thread_safe(self, symbol: str) -> dict:
        """Multi-user thread-safe implementation"""
        async with self._book_lock:
            if symbol not in self._user_books:
                self._user_books[symbol] = await self._generate_user_book(symbol)
            return await self._update_user_book(symbol)

    async def get_recent_trades(self, symbol: str, limit: int = 200) -> List[dict]:
        """Get recent trades with multi-user isolation"""
        if self.thread_safe:
            return await self._get_recent_trades_thread_safe(symbol, limit)
        else:
            return await self._get_recent_trades_basic(symbol, limit)
    
    async def _get_recent_trades_basic(self, symbol: str, limit: int = 200) -> List[dict]:
        """Original single-user trades implementation"""
        trades = []
        for i in range(limit):
            price = float(self.mid + np.random.normal(scale=0.02 * self.mid))
            qty = float(abs(np.random.exponential(scale=1.0)))
            isBuyerMaker = bool(np.random.rand() > 0.5)
            trades.append({"price": price, "qty": qty, "isBuyerMaker": isBuyerMaker, "timestamp": int(time.time() * 1000) - i * 100})
        trades = sorted(trades, key=lambda x: x["timestamp"], reverse=False)
        return trades
    
    async def _get_recent_trades_thread_safe(self, symbol: str, limit: int = 200) -> List[dict]:
        """Multi-user thread-safe trades"""
        async with self._trade_locks[symbol]:
            if symbol not in self._user_trades:
                self._user_trades[symbol] = await self._generate_user_trades(symbol, limit)
            return await self._update_user_trades(symbol, limit)

    async def get_book_ticker(self, symbol: str) -> dict:
        """Get book ticker - automatically uses thread-safe version if enabled"""
        ob = await self.get_order_book(symbol)
        return {"bidPrice": ob["bids"][0]["price"], "askPrice": ob["asks"][0]["price"], "timestamp": ob["timestamp"]}

    # --- MULTI-USER SPECIFIC METHODS ---
    
    async def _generate_user_book(self, symbol: str) -> dict:
        """User-specific initial book generation"""
        # Symbol-based unique price variation
        user_hash = hash(symbol) % 1000 / 1000.0  # 0-1 arası
        user_mid = self.mid * (0.99 + 0.02 * user_hash)  # ±%1 varyasyon
        
        # Geçici olarak mid price'ı değiştir
        original_mid = self.mid
        self.mid = user_mid
        book = await self._get_order_book_basic(symbol)
        self.mid = original_mid  # Geri resetle
        
        # Price history'yi başlat
        self._price_histories[symbol] = [user_mid]
        
        return book
    
    async def _update_user_book(self, symbol: str) -> dict:
        """Mevcut book'u hafifçe güncelle - realistic random walk"""
        current = self._user_books[symbol]
        current_mid = (current["bids"][0]["price"] + current["asks"][0]["price"]) / 2
        
        # Realistic price random walk (volatility simulation)
        volatility = 0.002  # %0.2 volatility
        price_change = np.random.normal(0, volatility * current_mid)
        new_mid = max(0.01, current_mid + price_change)
        
        # Spread variation
        spread_ratio = np.random.uniform(0.8, 1.2)
        current_spread = current["asks"][0]["price"] - current["bids"][0]["price"]
        new_spread = current_spread * spread_ratio
        
        updated_bids = []
        updated_asks = []
        
        # Update bids with new mid price and spread
        for i, bid in enumerate(current["bids"]):
            price_level_ratio = (i + 1) / len(current["bids"])
            new_price = new_mid - (new_spread / 2) - (price_level_ratio * 0.01 * new_mid)
            new_qty = max(0.001, bid["qty"] * np.random.uniform(0.9, 1.1))  # ±%10 quantity variation
            updated_bids.append({"price": round(new_price, 8), "qty": new_qty})
        
        # Update asks
        for i, ask in enumerate(current["asks"]):
            price_level_ratio = (i + 1) / len(current["asks"])
            new_price = new_mid + (new_spread / 2) + (price_level_ratio * 0.01 * new_mid)
            new_qty = max(0.001, ask["qty"] * np.random.uniform(0.9, 1.1))
            updated_asks.append({"price": round(new_price, 8), "qty": new_qty})
        
        updated_book = {
            "bids": updated_bids,
            "asks": updated_asks, 
            "timestamp": int(time.time() * 1000)
        }
        
        self._user_books[symbol] = updated_book
        self._price_histories[symbol].append(new_mid)
        
        # Keep only last 100 price points
        if len(self._price_histories[symbol]) > 100:
            self._price_histories[symbol].pop(0)
            
        return updated_book
    
    async def _generate_user_trades(self, symbol: str, limit: int) -> List[dict]:
        """User-specific initial trades generation"""
        current_book = self._user_books.get(symbol)
        if not current_book:
            current_book = await self._generate_user_book(symbol)
            
        current_mid = (current_book["bids"][0]["price"] + current_book["asks"][0]["price"]) / 2
        trades = []
        
        for i in range(limit):
            # More realistic trade generation based on current book
            spread = current_book["asks"][0]["price"] - current_book["bids"][0]["price"]
            price = float(current_mid + np.random.normal(0, spread * 0.5))
            qty = float(abs(np.random.exponential(scale=1.0)))
            
            # Determine if trade is aggressive (outside spread)
            is_aggressive_buy = price >= current_book["asks"][0]["price"]
            is_aggressive_sell = price <= current_book["bids"][0]["price"]
            isBuyerMaker = not is_aggressive_buy if (is_aggressive_buy or is_aggressive_sell) else bool(np.random.rand() > 0.5)
            
            trades.append({
                "price": price, 
                "qty": qty, 
                "isBuyerMaker": isBuyerMaker, 
                "timestamp": int(time.time() * 1000) - (limit - i) * 100
            })
        
        return sorted(trades, key=lambda x: x["timestamp"], reverse=False)
    
    async def _update_user_trades(self, symbol: str, limit: int) -> List[dict]:
        """Update user trades with new trades"""
        current_trades = self._user_trades[symbol]
        current_book = self._user_books.get(symbol)
        
        if not current_book:
            return current_trades[-limit:]
        
        # Add 1-3 new trades
        new_trades_count = np.random.randint(1, 4)
        current_mid = (current_book["bids"][0]["price"] + current_book["asks"][0]["price"]) / 2
        
        for _ in range(new_trades_count):
            spread = current_book["asks"][0]["price"] - current_book["bids"][0]["price"]
            price = float(current_mid + np.random.normal(0, spread * 0.3))
            qty = float(abs(np.random.exponential(scale=0.8)))
            
            is_aggressive_buy = price >= current_book["asks"][0]["price"]
            is_aggressive_sell = price <= current_book["bids"][0]["price"]
            isBuyerMaker = not is_aggressive_buy if (is_aggressive_buy or is_aggressive_sell) else bool(np.random.rand() > 0.5)
            
            new_trade = {
                "price": price, 
                "qty": qty, 
                "isBuyerMaker": isBuyerMaker, 
                "timestamp": int(time.time() * 1000)
            }
            
            current_trades.append(new_trade)
        
        # Keep only latest trades and sort
        current_trades = sorted(current_trades, key=lambda x: x["timestamp"], reverse=False)
        self._user_trades[symbol] = current_trades[-limit:]
        
        return self._user_trades[symbol]

    # --- UTILITY METHODS ---
    
    def get_user_count(self) -> int:
        """Get number of unique users/symbols"""
        return len(self._user_books)
    
    def get_price_history(self, symbol: str) -> List[float]:
        """Get price history for a symbol"""
        return self._price_histories.get(symbol, [])
    
    def reset_user_data(self, symbol: str = None):
        """Reset data for specific user or all users"""
        if symbol:
            self._user_books.pop(symbol, None)
            self._user_trades.pop(symbol, None)
            self._price_histories.pop(symbol, None)
        else:
            self._user_books.clear()
            self._user_trades.clear()
            self._price_histories.clear()

# ---- Utility functions ------------------------------------------------------
def _safe_div(a, b):
    return a / b if (b is not None and b != 0) else 0.0


def _min_max_scale(x: float, lo: float, hi: float):
    if math.isfinite(x):
        if hi == lo:
            return 0.0
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))
    return 0.0


# ---- Core Module ------------------------------------------------------------
@legacy_compatible
class OrderMicrosModule(BaseAnalysisModule):
    """
    Order Flow & Microstructure analysis module.
    Analysis Helpers uyumlu standart analiz modülü
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # ✅ CONFIG YÜKLEME - her instance kendi config'ine sahip
        self.cfg = OrderFlowConfig(**(config or CONFIG))
        self.module_name = "order_micros"
        self.version = "2.0.0"
        self.dependencies = ["binance_api"]
        
        # ✅ WEIGHTS VE THRESHOLDS - instance level
        self.weights = self.cfg.weights
        self.thresholds = {
            "bullish": getattr(self.cfg, 'bullish_threshold', 0.7),
            "bearish": getattr(self.cfg, 'bearish_threshold', 0.3)
        }
        
        # ✅ Data provider - her instance kendi provider'ı
        # ✅ BİRLEŞTİRİLMİŞ DATA PROVIDER
        self.data_provider = MockDataProvider(
            mid_price=getattr(self.cfg, 'mid_price', 100.0),
            depth_levels=getattr(self.cfg, 'depth_levels', 20),
            thread_safe=getattr(self.cfg, 'multi_user_mode', True)  # Config'den kontrol
        )
        
        
        
        # ✅ Session-specific state
        self._session_state = {
            "execution_count": 0,
            "total_time": 0.0,
            "last_execution": None
        }
        
        # ✅ Thread-safe lock for state updates
        self._state_lock = asyncio.Lock()
        
        
        # Normalize weights on initialization
        self.normalized_weights = self.helpers.normalize_weights(self.cfg.weights)
                
        # Connection pool simulation
        self._connection_pool = {}
        self._max_connections = getattr(self.cfg, 'max_connections', 100)
        
        # Rate limiting
        self._user_requests = defaultdict(list)
        self._rate_limit = getattr(self.cfg, 'requests_per_minute', 60)
        
        logger.info(f"OrderMicrosModule initialized - MultiUser: {self.data_provider.thread_safe}")
        
    async def initialize(self):
        """Initialize module resources"""
        logger.info("OrderMicrosModule initialized successfully")

    # Multi-user uyumlu compute_metrics
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None, 
                         user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
      
        # User context extraction
        user_id = user_context.get("user_id", "default") if user_context else "default"
        session_id = user_context.get("session_id", "default")
        
        # ✅ RATE LIMIT KONTROLÜ EKLE:
        if not await self._check_rate_limit(user_id):
            return self._create_fallback_output("Rate limit exceeded", user_context)
            
        start_time = AnalysisHelpers.get_timestamp()
        
        try:        
            # Fetch data
            order_book = await self.data_provider.get_order_book(symbol)
            trades = await self.data_provider.get_recent_trades(symbol, limit=getattr(self.cfg, 'trades_limit', 200))
            book_ticker = await self.data_provider.get_book_ticker(symbol)

            # Compute raw metrics
            raw_metrics = await self.compute_metrics_from_snapshot(order_book, trades, book_ticker)
            
            # User-specific state update
            async with self._state_lock:
                self._session_state["execution_count"] += 1
                self._session_state["total_time"] += (AnalysisHelpers.get_timestamp() - start_time)
                self._session_state["last_execution"] = AnalysisHelpers.get_timestamp()
            

            # Normalize metrics
            normalized = self._normalize_metrics(raw_metrics)
            
            # Prepare components for scoring
            components = {}
            for metric, weight in self.normalized_weights.items():
                norm_key = metric
                if metric == "market_buy_sell_pressure":
                    norm_key = "market_pressure"
                if metric == "trade_aggression_ratio":
                    norm_key = "trade_aggression"
                    
                components[metric] = normalized.get(norm_key, 0.0)

            # Calculate weighted score
            score = self._calculate_weighted_score(components, self.normalized_weights)
            
            # Create output
            output = self._create_output_template()
            output.update({
                "score": self._normalize_score(score),
                "signal": self._determine_signal(score, raw_metrics),
                "confidence": self._calculate_confidence(raw_metrics),
                "components": components,
                "explain": self._create_explanation(raw_metrics),
                "metadata": {
                    "symbol": symbol,
                    "user_id": user_id,
                    "session_id": session_id,
                    "priority": priority,
                    "calculation_time": AnalysisHelpers.get_timestamp() - start_time,
                    "spread_bps": raw_metrics.get("spread_bps", 0)
                }
            })
            
            # Validate output
            if not self._validate_output(output):
                return self._create_fallback_output("Output validation failed")
            
            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, True)
            return output
            
        except Exception as e:
            logger.error(f"Compute metrics failed for {symbol}: {e}")
            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, False)
            return self._create_fallback_output(str(e))

    async def compute_metrics_from_snapshot(self, order_book: dict, trades: List[dict], book_ticker: dict) -> Dict[str, Any]:
        """Compute raw microstructure metrics from a single snapshot."""

        # Convert order book to DataFrame
        bids = pd.DataFrame(order_book.get("bids", []))
        asks = pd.DataFrame(order_book.get("asks", []))

        # Ensure sorted
        bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
        asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

        # Top-of-book
        best_bid = float(bids.loc[0, "price"]) if not bids.empty else float(book_ticker.get("bidPrice", np.nan))
        best_ask = float(asks.loc[0, "price"]) if not asks.empty else float(book_ticker.get("askPrice", np.nan))
        mid_price = (best_bid + best_ask) / 2.0 if (np.isfinite(best_bid) and np.isfinite(best_ask)) else float(getattr(self.cfg, 'mid_price', 100.0))
        spread_bps = 10000.0 * _safe_div((best_ask - best_bid), mid_price)

        # Orderbook Imbalance
        depth_levels = getattr(self.cfg, 'depth_levels', 10)
        bid_qty = bids.head(depth_levels)["qty"].sum() if not bids.empty else 0.0
        ask_qty = asks.head(depth_levels)["qty"].sum() if not asks.empty else 0.0
        orderbook_imbalance = _safe_div((bid_qty - ask_qty), (bid_qty + ask_qty))

        # Depth Elasticity
        def depth_elasticity_side(df_side, direction="bid"):
            if df_side.empty:
                return 0.0
            df = df_side.head(getattr(self.cfg, 'elasticity_levels', 5)).copy()
            df["dist"] = np.abs(df["price"] - mid_price) / mid_price
            df["cum_qty"] = df["qty"].cumsum()
            if df["dist"].nunique() < 2:
                return 0.0
            coef = np.polyfit(df["dist"].values, df["cum_qty"].values, 1)[0]
            return float(np.abs(coef))
        
        depth_el_bid = depth_elasticity_side(bids, "bid")
        depth_el_ask = depth_elasticity_side(asks, "ask")
        depth_elasticity = (depth_el_bid + depth_el_ask) / 2.0

        # Liquidity Density Map
        near_window = getattr(self.cfg, 'liquidity_window_bps', 10)
        price_tol = (near_window / 10000.0) * mid_price
        liq_bid = bids[bids["price"] >= (mid_price - price_tol)]["qty"].sum() if not bids.empty else 0.0
        liq_ask = asks[asks["price"] <= (mid_price + price_tol)]["qty"].sum() if not asks.empty else 0.0
        liquidity_density = liq_bid + liq_ask

        # Process trades DataFrame
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame([{"price": mid_price, "qty": 0.0, "isBuyerMaker": False, "timestamp": int(time.time() * 1000)}])

        # CVD - cumulative buy minus sell volume
        trades_df["taker_side"] = trades_df["isBuyerMaker"].apply(lambda x: "sell" if x else "buy")
        trades_df["signed_qty"] = trades_df.apply(lambda r: r["qty"] if r["taker_side"] == "buy" else -r["qty"], axis=1)
        cvd = float(trades_df["signed_qty"].sum())

        # Order Flow Imbalance (OFI)
        trades_df["price_diff"] = trades_df["price"].diff().fillna(0.0)
        trades_df["price_sign"] = trades_df["price_diff"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        trades_df["ofi_component"] = trades_df["signed_qty"] * trades_df["price_sign"]
        ofi = float(trades_df["ofi_component"].sum())

        # Taker Dominance Ratio
        taker_buy_vol = trades_df[trades_df["taker_side"] == "buy"]["qty"].sum()
        taker_sell_vol = trades_df[trades_df["taker_side"] == "sell"]["qty"].sum()
        taker_dom_ratio = _safe_div(taker_buy_vol, (taker_buy_vol + taker_sell_vol))

        # Market Buy/Sell Pressure
        total_trade_vol = trades_df["qty"].sum()
        market_pressure = _safe_div((taker_buy_vol - taker_sell_vol), total_trade_vol) if total_trade_vol > 0 else 0.0

        # Trade Aggression Ratio
        aggressive_buys = trades_df[trades_df["price"] >= best_ask]["qty"].sum()
        aggressive_sells = trades_df[trades_df["price"] <= best_bid]["qty"].sum()
        trade_aggression_ratio = _safe_div((aggressive_buys + aggressive_sells), total_trade_vol) if total_trade_vol > 0 else 0.0

        # Slippage estimate
        trades_df["slippage_abs_bps"] = 10000.0 * np.abs(trades_df["price"] - mid_price) / mid_price
        slippage = float(trades_df["slippage_abs_bps"].mean()) / 10000.0

        raw_metrics = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "orderbook_imbalance": orderbook_imbalance,
            "depth_elasticity": depth_elasticity,
            "liquidity_density": liquidity_density,
            "cvd": cvd,
            "ofi": ofi,
            "taker_dom_ratio": taker_dom_ratio,
            "market_buy_sell_pressure": market_pressure,
            "trade_aggression_ratio": trade_aggression_ratio,
            "slippage": slippage,
            "bid_qty_top": bid_qty,
            "ask_qty_top": ask_qty,
            "timestamp": int(time.time() * 1000),
        }

        return raw_metrics

    async def _check_rate_limit(self, user_id: str) -> bool:
        """User-specific rate limiting"""
        now = time.time()
        user_requests = self._user_requests[user_id]
        
        # 1 dakikadan eski istekleri temizle
        user_requests[:] = [req_time for req_time in user_requests 
                          if now - req_time < 60]
        
        if len(user_requests) >= self._rate_limit:
            return False
        
        user_requests.append(now)
        return True
        


    def _normalize_metrics(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Normalize raw metrics to [0,1] according to normalization config."""
        normed = {}
        normalization_config = getattr(self.cfg, 'normalization', {})
        invert_metrics = getattr(self.cfg, 'invert_metrics', [])
        
        for k, (lo, hi) in normalization_config.items():
            val = float(raw.get(k, 0.0))
            normed[k] = _min_max_scale(val, lo, hi)
        
        # Invert metrics where higher is worse
        for m in invert_metrics:
            if m in normed:
                normed[m] = 1.0 - normed[m]
        return normed

    def _calculate_confidence(self, raw_metrics: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and signal strength."""
        confidence_factors = []
        
        # Data quality factors
        if raw_metrics.get("bid_qty_top", 0) > 0 and raw_metrics.get("ask_qty_top", 0) > 0:
            confidence_factors.append(0.3)  # Good order book data
        
        if raw_metrics.get("spread_bps", 100) < 50:  # Reasonable spread
            confidence_factors.append(0.2)
            
        # Signal strength factors
        imbalance_strength = abs(raw_metrics.get("orderbook_imbalance", 0))
        pressure_strength = abs(raw_metrics.get("market_buy_sell_pressure", 0))
        confidence_factors.append(min(0.5, imbalance_strength + pressure_strength))
        
        return min(1.0, sum(confidence_factors))

    def _create_explanation(self, raw_metrics: Dict[str, Any]) -> str:
        """Create human-readable explanation for the analysis result."""
        explanations = []
        
        imbalance = raw_metrics.get("orderbook_imbalance", 0)
        pressure = raw_metrics.get("market_buy_sell_pressure", 0)
        spread = raw_metrics.get("spread_bps", 0)
        imbalance_thresh = getattr(self.cfg, 'imbalance_signal_thresh', 0.1)
        
        if abs(imbalance) > imbalance_thresh:
            side = "buy" if imbalance > 0 else "sell"
            explanations.append(f"Strong {side}-side orderbook imbalance ({imbalance:.3f})")
            
        if abs(pressure) > 0.05:
            side = "buy" if pressure > 0 else "sell"
            explanations.append(f"Market {side} pressure present ({pressure:.3f})")
            
        if spread < 10:
            explanations.append("Tight spreads")
        elif spread > 50:
            explanations.append("Wide spreads affecting liquidity")
            
        if not explanations:
            explanations.append("Neutral market conditions")
            
        return f"Order Flow Analysis: {', '.join(explanations)}"

    def _determine_signal(self, score: float, raw_metrics: Dict[str, Any]) -> str:
        """Convert alpha score to trading signal with raw metrics context."""
        imbalance = raw_metrics.get("orderbook_imbalance", 0)
        pressure = raw_metrics.get("market_buy_sell_pressure", 0)
        imbalance_thresh = getattr(self.cfg, 'imbalance_signal_thresh', 0.1)
        
        # Use both score and raw metrics for signal determination
        if (score >= self.thresholds["bullish"] and 
            imbalance > imbalance_thresh and pressure > 0.05):
            return "bullish"
        elif (score <= self.thresholds["bearish"] and 
              imbalance < -imbalance_thresh and pressure < -0.05):
            return "bearish"
        else:
            return "neutral"

    async def _get_user_connection(self, user_id: str):
        """User-specific connection/get resource"""
        if user_id not in self._connection_pool:
            if len(self._connection_pool) >= self._max_connections:
                # LRU cleanup
                oldest_user = min(self._connection_pool.keys(), 
                                key=lambda k: self._connection_pool[k]["last_used"])
                del self._connection_pool[oldest_user]
            
            self._connection_pool[user_id] = {
                "created": time.time(),
                "last_used": time.time(),
                "data": f"connection_{user_id}"
            }
        
        self._connection_pool[user_id]["last_used"] = time.time()
        return self._connection_pool[user_id]["data"]
        


    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """✅ ANALYSIS_HELPERS UYUMLU AGGREGATE"""
        return {
            "symbol": symbol,
            "aggregated_score": self._normalize_score(np.mean(list(metrics.values()))),
            "component_scores": metrics,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "module": self.module_name
        }

    def generate_report(self) -> Dict[str, Any]:
        """✅ ANALYSIS_HELPERS UYUMLU RAPOR"""
        perf_metrics = self.get_performance_metrics()
        return {
            "module": self.module_name,
            "version": self.version,
            "status": "operational",
            "performance": perf_metrics,
            "dependencies": self.dependencies,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "report_type": "order_microstructure_report"
        }

# ✅ UYUMLU FACTORY FUNCTION
# order_micros.py - Enhanced factory
_module_instances: Dict[str, OrderMicrosModule] = {}

def create_module(config: Dict[str, Any] = None, instance_id: str = "default") -> OrderMicrosModule:
    """Multi-instance factory function"""
    if instance_id not in _module_instances:
        _module_instances[instance_id] = OrderMicrosModule(config)
    
    return _module_instances[instance_id]

def get_module_instance(instance_id: str = "default") -> Optional[OrderMicrosModule]:
    """Get existing module instance"""
    return _module_instances.get(instance_id)

def cleanup_instance(instance_id: str):
    """Cleanup specific instance"""
    if instance_id in _module_instances:
        del _module_instances[instance_id]

# If used as standalone script, demonstrate on synthetic data
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def demo():
        module = OrderMicrosModule()
        result = await module.compute_metrics("BTCUSDT")
        
        print("=== Order Microstructure Module Demo ===")
        print(f"Module: {result.get('module', 'N/A')}")
        print(f"Score: {result.get('score', 0):.4f}")
        print(f"Signal: {result.get('signal', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        print(f"Explanation: {result.get('explain', 'N/A')}")
        print("Components:")
        for k, v in result.get('components', {}).items():
            print(f"  {k}: {v:.4f}")

    asyncio.run(demo())
"""
analysis/microalpha.py
Version: 2.0.0
Micro Alpha Factor Module
Real-time tick-level microstructure analysis for high-frequency alpha generation

Key Metrics:
- Cumulative Volume Delta (CVD)
- Order Flow Imbalance (OFI) 
- Microprice Deviation
- Market Impact Model (Kyle's λ)
- Latency Adjusted Flow Ratio
- High-Frequency Z-score
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import deque
import time
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# ✅ ANALYSIS_HELPERS UYUMLU IMPORT
from analysis.analysis_base_module import BaseAnalysisModule, legacy_compatible
from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers
from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result
from analysis.config.c_micro import MicroAlphaConfig, CONFIG

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data structure"""
    symbol: str
    timestamp: float
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    is_maker: bool

@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, quantity)]
    asks: List[Tuple[float, float]]
    spread: float

@legacy_compatible
class MicroAlphaModule(BaseAnalysisModule):
    """
    Microstructure Alpha Factor Generator
    Analysis Helpers uyumlu standart analiz modülü
    Multi-user uyumlu async yapı
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # ✅ CONFIG YÜKLEME
        self.cfg = MicroAlphaConfig(**(config or CONFIG))
        self.module_name = "micro_alpha"
        self.version = "2.0.0"
        self.dependencies = ["binance_api"]
        
        # ✅ WEIGHTS VE THRESHOLDS
        self.weights = self.cfg.weights
        self.thresholds = self.cfg.thresholds
        
        # Multi-user için instance-specific data storage
        self.user_buffers: Dict[str, Dict] = {}  # {user_id: {tick_buffer, order_book_buffer, metric_history}}
        
        # Global state variables (user-specific olacak)
        self.user_states: Dict[str, Dict] = {}  # {user_id: {last_microprice, cumulative_delta, etc.}}
        
        # Initialize Binance client
        self.binance_aggregator = BinanceAggregator()
        
        # Kalman filter state (user-specific olacak)
        self.user_kalman_states: Dict[str, Dict] = {}
        
        logger.info(f"MicroAlphaModule initialized: {self.module_name} v{self.version}")

    def _get_user_buffers(self, user_id: str = "default") -> Dict[str, deque]:
        """Get or create user-specific buffers"""
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = {
                "tick_buffer": deque(maxlen=int(self.cfg.parameters["lookback_window"])),
                "order_book_buffer": deque(maxlen=100),
                "metric_history": deque(maxlen=500)
            }
        return self.user_buffers[user_id]

    def _get_user_state(self, user_id: str = "default") -> Dict[str, Any]:
        """Get or create user-specific state"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "last_microprice": 0.0,
                "cumulative_delta": 0.0,
                "last_trade_side": None
            }
        return self.user_states[user_id]

    def _get_user_kalman_state(self, user_id: str = "default") -> Dict[str, float]:
        """Get or create user-specific Kalman filter state"""
        if user_id not in self.user_kalman_states:
            self.user_kalman_states[user_id] = {
                "state": self.cfg.kalman["initial_state"],
                "covariance": self.cfg.kalman["initial_covariance"]
            }
        return self.user_kalman_states[user_id]

    async def initialize(self):
        """Initialize module resources"""
        await self.binance_aggregator.initialize()
        logger.info("MicroAlphaModule initialized successfully")

    async def compute_metrics(self, symbol: str, priority: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
        """
        Ana metrik hesaplama metodu - Multi-user uyumlu
        """
        start_time = AnalysisHelpers.get_timestamp()

        try:
            # 1. Realtime veri al
            tick_data, order_book = await self._fetch_realtime_data(symbol)
            
            # 2. Buffer'ları güncelle
            self._update_buffers(tick_data, order_book, user_id)
            
            # 3. Tüm metrikleri hesapla
            metrics = await self._calculate_all_metrics(symbol, tick_data, order_book, user_id)
            
            # 4. Alpha skoru oluştur
            alpha_score, components = self._aggregate_alpha_score(metrics, user_id)
            
            # 5. Sinyal ve açıklama oluştur
            signal = self._determine_signal(alpha_score)
            explanation = self._generate_explanation(alpha_score, components, metrics)
            confidence = self._calculate_confidence(metrics, components, user_id)
            
            # 6. Çıktıyı formatla
            output = self._create_output_template()
            output.update({
                "score": self._normalize_score(alpha_score),
                "signal": signal,
                "confidence": confidence,
                "components": components,
                "explain": explanation,
                "metadata": {
                    "symbol": symbol,
                    "priority": priority,
                    "user_id": user_id,
                    "calculation_time": AnalysisHelpers.get_timestamp() - start_time,
                    "timestamp": AnalysisHelpers.get_timestamp(),
                    "tick_count": len(self._get_user_buffers(user_id)["tick_buffer"])
                }
            })

            if not self._validate_output(output):
                return self._create_fallback_output("Output validation failed")

            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, True)
            return output

        except Exception as e:
            logger.exception(f"Compute metrics failed for {symbol} (user:{user_id}): {e}")
            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, False)
            return self._create_fallback_output(str(e))

    async def _fetch_realtime_data(self, symbol: str) -> Tuple[Optional[TickData], Optional[OrderBookSnapshot]]:
        """Fetch real-time tick and order book data"""
        try:
            # Get recent trades
            trades_data = await self.binance_aggregator.get_recent_trades(symbol=symbol, limit=10)
            # Get order book
            order_book_data = await self.binance_aggregator.get_order_book(symbol=symbol, limit=20)
            
            if not trades_data or not order_book_data:
                return None, None
            
            # Convert to internal format
            latest_trade = trades_data[-1] if trades_data else None
            if latest_trade:
                tick_data = TickData(
                    symbol=symbol,
                    timestamp=latest_trade.get('time', time.time()),
                    price=float(latest_trade['price']),
                    quantity=float(latest_trade['qty']),
                    side='buy' if latest_trade['isBuyerMaker'] else 'sell',
                    is_maker=latest_trade['isBuyerMaker']
                )
            else:
                tick_data = None
            
            # Create order book snapshot
            order_book_snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=time.time(),
                bids=[(float(bid[0]), float(bid[1])) for bid in order_book_data.get('bids', [])],
                asks=[(float(ask[0]), float(ask[1])) for ask in order_book_data.get('asks', [])],
                spread=0.0
            )
            
            # Calculate spread
            if order_book_snapshot.bids and order_book_snapshot.asks:
                best_bid = order_book_snapshot.bids[0][0]
                best_ask = order_book_snapshot.asks[0][0]
                order_book_snapshot.spread = best_ask - best_bid
            
            return tick_data, order_book_snapshot
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None, None

    def _update_buffers(self, tick_data: TickData, order_book: OrderBookSnapshot, user_id: str = "default"):
        """Update internal data buffers - user specific"""
        buffers = self._get_user_buffers(user_id)
        
        if tick_data:
            buffers["tick_buffer"].append(tick_data)
        if order_book:
            buffers["order_book_buffer"].append(order_book)

    async def _calculate_all_metrics(self, symbol: str, tick_data: TickData, order_book: OrderBookSnapshot, user_id: str = "default") -> Dict[str, float]:
        """Calculate all micro-structure metrics - user specific"""
        metrics = {}
        
        # 1. Cumulative Volume Delta (CVD)
        metrics['cvd'] = self._calculate_cvd(user_id)
        
        # 2. Order Flow Imbalance (OFI)
        metrics['ofi'] = self._calculate_order_flow_imbalance(order_book)
        
        # 3. Microprice and Deviation
        microprice, microprice_dev = self._calculate_microprice_deviation(order_book, user_id)
        metrics['microprice'] = microprice
        metrics['microprice_deviation'] = microprice_dev
        
        # 4. Market Impact (Kyle's Lambda)
        metrics['market_impact'] = self._calculate_market_impact(tick_data, order_book, user_id)
        
        # 5. Latency Adjusted Flow Ratio
        metrics['latency_flow_ratio'] = self._calculate_latency_flow_ratio(user_id)
        
        # 6. High-Frequency Z-score
        metrics['hf_zscore'] = self._calculate_hf_zscore(metrics, user_id)
        
        return metrics

    def _calculate_cvd(self, user_id: str = "default") -> float:
        """Calculate Cumulative Volume Delta - user specific"""
        buffers = self._get_user_buffers(user_id)
        
        if len(buffers["tick_buffer"]) < 2:
            return 0.0
        
        cvd = 0.0
        for tick in list(buffers["tick_buffer"])[-int(self.cfg.windows["cvd_window"]):]:
            if tick.side == 'buy':
                cvd += tick.quantity
            else:
                cvd -= tick.quantity
        
        # Normalize by recent volume
        total_volume = sum(tick.quantity for tick in list(buffers["tick_buffer"])[-int(self.cfg.windows["cvd_window"]):])
        if total_volume > 0:
            cvd_normalized = cvd / total_volume
        else:
            cvd_normalized = 0.0
            
        return cvd_normalized

    def _calculate_order_flow_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate Order Flow Imbalance (OFI)"""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        bid_size = sum(qty for _, qty in order_book.bids[:5])
        ask_size = sum(qty for _, qty in order_book.asks[:5])
        
        if bid_size + ask_size == 0:
            return 0.0
            
        ofi = (bid_size - ask_size) / (bid_size + ask_size)
        return ofi

    def _calculate_microprice_deviation(self, order_book: OrderBookSnapshot, user_id: str = "default") -> Tuple[float, float]:
        """Calculate Microprice and its deviation from current price"""
        if not order_book.bids or not order_book.asks:
            return 0.0, 0.0
        
        best_bid_price, best_bid_size = order_book.bids[0]
        best_ask_price, best_ask_size = order_book.asks[0]
        
        if best_bid_size + best_ask_size == 0:
            return 0.0, 0.0
        
        microprice = (best_bid_size * best_ask_price + best_ask_size * best_bid_price) / (best_bid_size + best_ask_size)
        mid_price = (best_bid_price + best_ask_price) / 2
        
        if mid_price > 0:
            deviation = (microprice - mid_price) / mid_price
        else:
            deviation = 0.0
        
        # Update user state
        user_state = self._get_user_state(user_id)
        user_state["last_microprice"] = microprice
        
        return microprice, deviation

    def _calculate_market_impact(self, tick_data: TickData, order_book: OrderBookSnapshot, user_id: str = "default") -> float:
        """Calculate Market Impact using Kalman filter - user specific"""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        buffers = self._get_user_buffers(user_id)
        kalman_state = self._get_user_kalman_state(user_id)
        
        price_change = 0.0
        if len(buffers["tick_buffer"]) >= 2:
            recent_ticks = list(buffers["tick_buffer"])[-2:]
            if len(recent_ticks) == 2:
                price_change = recent_ticks[1].price - recent_ticks[0].price
        
        order_flow = tick_data.quantity if tick_data and tick_data.side == 'buy' else -tick_data.quantity if tick_data else 0.0
        
        if order_flow != 0:
            process_var = self.cfg.kalman["process_variance"]
            kalman_state["covariance"] += process_var
            
            obs_var = self.cfg.kalman["observation_variance"]
            kalman_gain = kalman_state["covariance"] / (kalman_state["covariance"] + obs_var)
            
            predicted_change = kalman_state["state"] * order_flow
            innovation = price_change - predicted_change
            
            kalman_state["state"] += kalman_gain * innovation / order_flow if order_flow != 0 else 0
            kalman_state["covariance"] *= (1 - kalman_gain)
        
        return abs(kalman_state["state"])

    def _calculate_latency_flow_ratio(self, user_id: str = "default") -> float:
        """Calculate Latency Adjusted Flow Ratio - user specific"""
        buffers = self._get_user_buffers(user_id)
        
        if len(buffers["tick_buffer"]) < 10:
            return 0.5
        
        recent_ticks = list(buffers["tick_buffer"])[-10:]
        timestamps = [tick.timestamp for tick in recent_ticks]
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        if not time_diffs:
            return 0.5
            
        avg_time_diff = np.mean(time_diffs)
        if avg_time_diff == 0:
            return 0.5
            
        volumes = [tick.quantity for tick in recent_ticks]
        avg_volume = np.mean(volumes)
        
        flow_ratio = avg_volume / avg_time_diff if avg_time_diff > 0 else 0
        normalized_ratio = min(flow_ratio / 1000, 1.0)
        
        return normalized_ratio

    def _calculate_hf_zscore(self, current_metrics: Dict[str, float], user_id: str = "default") -> float:
        """Calculate High-Frequency Z-score for anomaly detection - user specific"""
        buffers = self._get_user_buffers(user_id)
        
        if len(buffers["metric_history"]) < self.cfg.windows["zscore_window"]:
            return 0.0
        
        cvd_values = [metric.get('cvd', 0) for metric in list(buffers["metric_history"])[-int(self.cfg.windows["zscore_window"]):]]
        
        if len(cvd_values) < 2:
            return 0.0
        
        current_cvd = current_metrics.get('cvd', 0)
        mean_cvd = np.mean(cvd_values)
        std_cvd = np.std(cvd_values)
        
        if std_cvd > 0:
            zscore = (current_cvd - mean_cvd) / std_cvd
            normalized_zscore = 1 / (1 + np.exp(-zscore))
        else:
            normalized_zscore = 0.5
            
        return normalized_zscore

    def _aggregate_alpha_score(self, metrics: Dict[str, float], user_id: str = "default") -> Tuple[float, Dict[str, float]]:
        """Aggregate individual metrics into final alpha score - user specific"""
        components = {}
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                normalized_value = self._normalize_metric(metric_name, metrics[metric_name])
                components[metric_name] = normalized_value
        
        alpha_score = self._calculate_weighted_score(components, self.weights)
        alpha_score = self._normalize_score(alpha_score)
        
        # Update metric history
        buffers = self._get_user_buffers(user_id)
        buffers["metric_history"].append(metrics)
            
        return alpha_score, components

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric values to 0-1 range"""
        if metric_name == 'cvd':
            return (value + 1) / 2
        elif metric_name == 'ofi':
            return (value + 1) / 2
        elif metric_name == 'microprice_deviation':
            return min(max((value * 100) + 0.5, 0), 1)
        elif metric_name == 'market_impact':
            return min(value * 1000, 1.0)
        elif metric_name == 'latency_flow_ratio':
            return value
        elif metric_name == 'hf_zscore':
            return value
        else:
            return min(max(value, 0), 1)

    def _calculate_confidence(self, metrics: Dict[str, float], components: Dict[str, float], user_id: str = "default") -> float:
        """Calculate confidence score - user specific"""
        if not components:
            return 0.0

        buffers = self._get_user_buffers(user_id)
        
        tick_quality = min(1.0, len(buffers["tick_buffer"]) / 20.0)
        signal_strength = 1.0 if metrics.get('microprice', 0) > 0 else 0.3

        comp_vals = list(components.values())
        if len(comp_vals) <= 1:
            consistency = 0.5
        else:
            consistency = max(0.0, 1.0 - (np.std(comp_vals) / (np.mean(comp_vals) + 1e-6)))

        w_tick, w_sig, w_cons = 0.4, 0.3, 0.3
        conf = w_tick * tick_quality + w_sig * signal_strength + w_cons * consistency
        return float(max(0.0, min(1.0, conf)))

    def _generate_explanation(self, alpha_score: float, components: Dict[str, float], 
                            metrics: Dict[str, float]) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        key_drivers = [f"{driver}({value:.3f})" for driver, value in sorted_components[:2]]
        
        if key_drivers:
            explanations.append(f"Key drivers: {', '.join(key_drivers)}")
        
        if alpha_score > self.thresholds.get("bullish_threshold", 0.7):
            explanations.append("Strong buying pressure with positive order flow")
        elif alpha_score < self.thresholds.get("bearish_threshold", 0.3):
            explanations.append("Strong selling pressure with negative order flow")
        else:
            explanations.append("Balanced order flow with neutral microstructure")
            
        microprice_dev = metrics.get('microprice_deviation', 0)
        if abs(microprice_dev) > 0.001:
            direction = "above" if microprice_dev > 0 else "below"
            explanations.append(f"Microprice {direction} mid price")
        
        return " | ".join(explanations)

    def _determine_signal(self, score: float) -> str:
        """Convert alpha score to trading signal"""
        bullish_thresh = self.thresholds.get("bullish_threshold", 0.7)
        bearish_thresh = self.thresholds.get("bearish_threshold", 0.3)

        if score >= bullish_thresh:
            return "bullish"
        elif score <= bearish_thresh:
            return "bearish"
        else:
            return "neutral"

    def _calculate_weighted_score(self, components: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted score from components"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in components:
                weighted_sum += components[metric_name] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range"""
        return max(0.0, min(1.0, score))

    def _create_output_template(self) -> Dict[str, Any]:
        """Create output template"""
        return {
            "score": 0.0,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": "",
            "metadata": {}
        }

    def _create_fallback_output(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback output on error"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": f"Error: {error_msg}",
            "metadata": {"error": error_msg}
        }

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output structure"""
        required_fields = ["score", "signal", "confidence", "components", "explain", "metadata"]
        return all(field in output for field in required_fields)

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
        
        # User statistics
        user_stats = {}
        for user_id, buffers in self.user_buffers.items():
            user_stats[user_id] = {
                "tick_buffer_size": len(buffers["tick_buffer"]),
                "order_book_buffer_size": len(buffers["order_book_buffer"]),
                "metric_history_size": len(buffers["metric_history"])
            }
        
        return {
            "module": self.module_name,
            "version": self.version,
            "status": "operational",
            "performance": perf_metrics,
            "dependencies": self.dependencies,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "report_type": "microstructure_analysis_report",
            "user_statistics": user_stats,
            "total_users": len(self.user_buffers)
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.binance_aggregator.close()
        
        # Clear user data
        self.user_buffers.clear()
        self.user_states.clear()
        self.user_kalman_states.clear()
        
        logger.info("MicroAlphaModule cleanup completed")

# ✅ UYUMLU FACTORY FUNCTION
def create_module(config: Dict[str, Any] = None) -> MicroAlphaModule:
    """Factory function for creating MicroAlphaModule instances"""
    return MicroAlphaModule(config)
	
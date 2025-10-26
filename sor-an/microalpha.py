"""
analysis/microalpha.py
Micro Alpha Factor Module
Real-time tick-level microstructure analysis for high-frequency alpha generation

Key Metrics:
- Cumulative Volume Delta (CVD)
- Order Flow Imbalance (OFI) 
- Microprice Deviation
- Market Impact Model (Kyle's λ)
- Latency Adjusted Flow Ratio
- High-Frequency Z-score


analysis/microalpha.py
Micro Alpha Factor Module - Analysis Helpers Uyumlu
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import deque
import time

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
        
        # Data storage
        self.tick_buffer = deque(maxlen=int(self.cfg.parameters["lookback_window"]))
        self.order_book_buffer = deque(maxlen=100)
        self.metric_history = deque(maxlen=500)
        
        # State variables
        self.last_microprice = 0.0
        self.cumulative_delta = 0.0
        self.last_trade_side = None
        
        # Initialize Binance client
        #self.binance_client = BinanceAggregator()
        self.binance_aggregator = BinanceAggregator()
        
        # Kalman filter state for market impact
        self.kalman_state = self.cfg.kalman["initial_state"]
        self.kalman_covariance = self.cfg.kalman["initial_covariance"]
        
        logger.info(f"MicroAlphaModule initialized: {self.module_name} v{self.version}")

    async def initialize(self):
        """Initialize module resources"""
        await self.binance_aggregator.initialize()
        logger.info("MicroAlphaModule initialized successfully")


    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Güçlendirilmiş compute_metrics:
         - veri hizalama (resample + log-returns)
         - hızlı korelasyon eşiği ile ön-filtre
         - istatistiksel p-value kontrolü + multiple-testing correction
         - exception logging per-pair
        """
        start_time = AnalysisHelpers.get_timestamp()

        try:
            # 1) sembolleri topla
            symbols = await self._get_related_symbols(symbol)
            interval = getattr(self.cfg.calculation, "default_interval", "1h")
            limit = getattr(self.cfg.calculation, "default_limit", 200)

            # 2) fiyat verilerini çek
            price_data = await self.fetch_price_data(symbols, interval, limit)
            if not price_data:
                return self._create_fallback_output("Fiyat verisi alınamadı")

            # 3) preprocess: ensure pandas Series with datetime index, resample, log-returns
            def _prepare_series(s):
                # expects s as DataFrame/Series-like with 'timestamp' index or column
                ser = pd.Series(s).astype(float).dropna()
                if not isinstance(ser.index, pd.DatetimeIndex):
                    try:
                        ser.index = pd.to_datetime(ser.index, unit='ms', utc=True)
                    except Exception:
                        ser.index = pd.to_datetime(ser.index, utc=True, errors='coerce')
                # resample to interval
                try:
                    ser = ser.resample(interval).last().ffill().bfill()
                except Exception:
                    # fallback: if resample fails, just ensure monotonic index
                    ser = ser.asfreq('1H').ffill().bfill()
                # return log-returns
                returns = np.log(ser).diff().dropna()
                return returns

            prepared = {}
            for sym, raw in price_data.items():
                try:
                    prepared[sym] = _prepare_series(raw)
                except Exception as e:
                    logger.debug(f"Prepare series failed for {sym}: {e}")

            if len(prepared) < 2:
                return self._create_fallback_output("Yeterli hazırlanmış seri yok")

            # 4) fast correlation matrix (pair filter)
            keys = list(prepared.keys())
            series_mat = pd.DataFrame({k: prepared[k] for k in keys})
            corr = series_mat.corr(method='pearson').abs()

            # choose candidate pairs: top correlated per symbol or global threshold
            THRESH = getattr(self.cfg, 'fast_corr_threshold', 0.3)
            candidate_pairs = []
            for i, s1 in enumerate(keys):
                for s2 in keys[i+1:]:
                    cval = corr.loc[s1, s2] if s1 in corr and s2 in corr else 0.0
                    if cval >= THRESH:
                        candidate_pairs.append((s1, s2))

            # cap number of pairs to avoid explosion
            MAX_PAIRS = getattr(self.cfg, 'max_pairs', 40)
            candidate_pairs = candidate_pairs[:MAX_PAIRS]

            # 5) compute detailed metrics per pair (parallel)
            async def _compute_for_pair(s1, s2):
                try:
                    x = prepared[s1].align(prepared[s2], join='inner')[0]
                    y = prepared[s1].align(prepared[s2], join='inner')[1]
                    if len(x) < 30:
                        return {'pair': (s1, s2), 'skipped': True, 'reason': 'small_sample'}

                    # pearson + pvalue
                    r, p = stats.pearsonr(x, y)
                    # cross-correlation for lead-lag
                    c = np.correlate(x - x.mean(), y - y.mean(), mode='full')
                    lag = c.argmax() - (len(x) - 1)

                    # try Granger causality if enough data and config allows
                    granger_p = None
                    try:
                        maxlag = getattr(self.cfg.calculation, 'granger_max_lags', 5)
                        test_res = grangercausalitytests(pd.concat([x, y], axis=1), maxlag=maxlag, verbose=False)
                        # choose minimal p-value across lags for x->y
                        pvals = [test_res[l][0]['ssr_ftest'][1] for l in test_res if 'ssr_ftest' in test_res[l][0]]
                        if pvals:
                            granger_p = min(pvals)
                    except Exception:
                        granger_p = None

                    return {
                        'pair': (s1, s2),
                        'pearson': float(r),
                        'pearson_p': float(p),
                        'lag': int(lag),
                        'granger_p': float(granger_p) if granger_p is not None else None,
                        'n': len(x),
                        'raw_metrics': {
                            'pearson_correlation': float(r)
                        },
                        'components': {
                            'pearson': abs(float(r)),
                            'lead_lag': abs(int(lag))
                        }
                    }
                except Exception as e:
                    logger.exception(f"Pair metrics failed {s1}-{s2}: {e}")
                    return {'pair': (s1, s2), 'error': str(e)}

            tasks = [_compute_for_pair(a, b) for a, b in candidate_pairs]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # 6) filter valid and apply simple multiple-testing correction for pearson p-values
            pvals = [r['pearson_p'] for r in results if isinstance(r, dict) and 'pearson_p' in r]
            # Benjamini-Hochberg simple implementation
            def bh_filter(p_list, alpha=0.05):
                if not p_list:
                    return set()
                sorted_idx = np.argsort(p_list)
                m = len(p_list)
                thresh = [ (i+1)/m * alpha for i in range(m) ]
                accepted = set()
                for rank, idx in enumerate(sorted_idx[::-1]): # highest to lowest
                    if p_list[idx] <= thresh[idx]:
                        accepted.update(sorted_idx[:idx+1].tolist())
                        break
                return accepted

            accepted_idx = bh_filter(pvals, alpha=0.05)
            # map results to keep only accepted (or large effect size)
            valid_results = []
            p_iter = iter(pvals)
            for r in results:
                if not isinstance(r, dict):
                    continue
                if 'pearson_p' in r:
                    idx = next(p_iter)
                    # keep if accepted or |r|>0.6
                    if (pvals and (r['pearson_p'] in pvals and pvals.index(r['pearson_p']) in accepted_idx)) or abs(r.get('pearson',0))>0.6:
                        valid_results.append(r)
                elif 'error' in r:
                    logger.debug(f"Pair error: {r['pair']} -> {r['error']}")

            # 7) aggregate
            score, signal, components, explanation = self._aggregate_results(valid_results, symbols)

            confidence = self._calculate_confidence(valid_results, components)
            output = self._create_output_template()
            output.update({
                "score": self._normalize_score(score),
                "signal": signal,
                "confidence": confidence,
                "components": components,
                "explain": explanation,
                "metadata": {
                    "symbol": symbol,
                    "priority": priority,
                    "calculation_time": AnalysisHelpers.get_timestamp() - start_time,
                    "pairs_analyzed": len(valid_results),
                    "candidate_pairs": len(candidate_pairs),
                    "total_symbols": len(symbols)
                }
            })

            if not self._validate_output(output):
                return self._create_fallback_output("Output validation failed")

            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, True)
            return output

        except Exception as e:
            logger.exception(f"Compute metrics failed for {symbol}: {e}")
            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, False)
            return self._create_fallback_output(str(e))



    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        ✅ ANALYSIS_HELPERS UYUMLU AGGREGATE
        """
        return {
            "symbol": symbol,
            "aggregated_score": self._normalize_score(np.mean(list(metrics.values()))),
            "component_scores": metrics,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "module": self.module_name
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        ✅ ANALYSIS_HELPERS UYUMLU RAPOR
        """
        perf_metrics = self.get_performance_metrics()
        return {
            "module": self.module_name,
            "version": self.version,
            "status": "operational",
            "performance": perf_metrics,
            "dependencies": self.dependencies,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "report_type": "microstructure_analysis_report",
            "buffer_sizes": {
                "tick_buffer": len(self.tick_buffer),
                "order_book_buffer": len(self.order_book_buffer),
                "metric_history": len(self.metric_history)
            }
        }

    # ✅ MEVCUT METOTLARI KORU (sadece küçük adaptasyonlar)
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

    def _update_buffers(self, tick_data: TickData, order_book: OrderBookSnapshot):
        """Update internal data buffers"""
        if tick_data:
            self.tick_buffer.append(tick_data)
        if order_book:
            self.order_book_buffer.append(order_book)

    async def _calculate_all_metrics(self, symbol: str, tick_data: TickData, order_book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate all micro-structure metrics"""
        metrics = {}
        
        # 1. Cumulative Volume Delta (CVD)
        metrics['cvd'] = self._calculate_cvd()
        
        # 2. Order Flow Imbalance (OFI)
        metrics['ofi'] = self._calculate_order_flow_imbalance(order_book)
        
        # 3. Microprice and Deviation
        microprice, microprice_dev = self._calculate_microprice_deviation(order_book)
        metrics['microprice'] = microprice
        metrics['microprice_deviation'] = microprice_dev
        
        # 4. Market Impact (Kyle's Lambda)
        metrics['market_impact'] = self._calculate_market_impact(tick_data, order_book)
        
        # 5. Latency Adjusted Flow Ratio
        metrics['latency_flow_ratio'] = self._calculate_latency_flow_ratio()
        
        # 6. High-Frequency Z-score
        metrics['hf_zscore'] = self._calculate_hf_zscore(metrics)
        
        return metrics

    # ✅ MEVCUT HESAPLAMA METOTLARINI KORU
    def _calculate_cvd(self) -> float:
        """Calculate Cumulative Volume Delta"""
        if len(self.tick_buffer) < 2:
            return 0.0
        
        cvd = 0.0
        for tick in list(self.tick_buffer)[-int(self.cfg.windows["cvd_window"]):]:
            if tick.side == 'buy':
                cvd += tick.quantity
            else:
                cvd -= tick.quantity
        
        # Normalize by recent volume
        total_volume = sum(tick.quantity for tick in list(self.tick_buffer)[-int(self.cfg.windows["cvd_window"]):])
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

    def _calculate_microprice_deviation(self, order_book: OrderBookSnapshot) -> Tuple[float, float]:
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
            
        self.last_microprice = microprice
        return microprice, deviation

    def _calculate_market_impact(self, tick_data: TickData, order_book: OrderBookSnapshot) -> float:
        """Calculate Market Impact using Kalman filter"""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        price_change = 0.0
        if len(self.tick_buffer) >= 2:
            recent_ticks = list(self.tick_buffer)[-2:]
            if len(recent_ticks) == 2:
                price_change = recent_ticks[1].price - recent_ticks[0].price
        
        order_flow = tick_data.quantity if tick_data.side == 'buy' else -tick_data.quantity
        
        if order_flow != 0:
            process_var = self.cfg.kalman["process_variance"]
            self.kalman_covariance += process_var
            
            obs_var = self.cfg.kalman["observation_variance"]
            kalman_gain = self.kalman_covariance / (self.kalman_covariance + obs_var)
            
            predicted_change = self.kalman_state * order_flow
            innovation = price_change - predicted_change
            
            self.kalman_state += kalman_gain * innovation / order_flow if order_flow != 0 else 0
            self.kalman_covariance *= (1 - kalman_gain)
        
        return abs(self.kalman_state)

    def _calculate_latency_flow_ratio(self) -> float:
        """Calculate Latency Adjusted Flow Ratio"""
        if len(self.tick_buffer) < 10:
            return 0.5
        
        recent_ticks = list(self.tick_buffer)[-10:]
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

    def _calculate_hf_zscore(self, current_metrics: Dict[str, float]) -> float:
        """Calculate High-Frequency Z-score for anomaly detection"""
        if len(self.metric_history) < self.cfg.windows["zscore_window"]:
            return 0.0
        
        cvd_values = [metric.get('cvd', 0) for metric in list(self.metric_history)[-int(self.cfg.windows["zscore_window"]):]]
        
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

    def _aggregate_alpha_score(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Aggregate individual metrics into final alpha score"""
        components = {}
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                normalized_value = self._normalize_metric(metric_name, metrics[metric_name])
                components[metric_name] = normalized_value
        
        alpha_score = self._calculate_weighted_score(components, self.weights)
        alpha_score = self._normalize_score(alpha_score)
        
        self.metric_history.append(metrics)
            
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


    def _calculate_confidence(self, metrics: Dict[str, float], components: Dict[str, float]) -> float:
        """
        Gelişmiş confidence:
          - data_quality: yeterli tick sayısı
          - signal_strength: microprice/metrik değerlerinin anlamlılığı
          - consistency: bileşenlerin standart sapmasına göre güven
        """
        if not components:
            return 0.0

        tick_quality = min(1.0, len(self.tick_buffer) / 20.0)
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


    async def cleanup(self):
        """Cleanup resources"""
        await self.binance_aggregator.close()
        logger.info("MicroAlphaModule cleanup completed")

# ✅ UYUMLU FACTORY FUNCTION
def create_module(config: Dict[str, Any] = None) -> MicroAlphaModule:
    """Factory function for creating MicroAlphaModule instances"""
    return MicroAlphaModule(config)
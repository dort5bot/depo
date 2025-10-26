# analysis/trend_moment.py
"""
Trend & Momentum Analysis Module - Analysis Helpers Uyumlu
Version: 1.1.0
deepsek
Purpose: Analyze price direction and momentum strength using classical and advanced technical indicators
Output: Trend Score (0-1) with detailed component breakdown
Metrics:
- Classic: EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, Momentum Oscillator
- Advanced: Kalman Filter, Z-Score Normalization, Wavelet Transform, Hilbert Transform, Fractal Dimension
Dependencies: numpy, pandas, scipy, pykalman, pywt
"""

import os
import logging
import asyncio
import pywt
import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.signal import hilbert, detrend

from pykalman import KalmanFilter

from analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput

from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result

logger = logging.getLogger(__name__)


class TrendModule(BaseAnalysisModule):
    """
    Trend and Momentum Analysis Module - Analysis Helpers Uyumlu
    
    Computes comprehensive trend score using multiple technical indicators
    and advanced signal processing techniques.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.module_name = "trend_moment"
        self.version = "1.1.0"
        
        # ✅ ANALYSIS_HELPERS INTEGRATION
        self.helpers = AnalysisHelpers
        
        
        
        # Load configuration
        # Load configuration - DÜZELTİLMİŞ
        # ÖNERİ: Daha robust config yükleme
        if config is None:
            try:
                from analysis.config.cm_loader import config_manager
                config_obj = config_manager.get_config("trend_moment")
                if config_obj:
                    # ✅ Multiple fallback options
                    self.config = getattr(config_obj, 'to_dict', 
                                 getattr(config_obj, 'get_parameters', 
                                 getattr(config_obj, 'parameters', 
                                 lambda: self._get_default_config())))()
                else:
                    self.config = self._get_default_config()
            except Exception as e:
                logger.warning(f"Config load failed: {e}, using defaults")
                self.config = self._get_default_config()
                
            
            
        self.weights = self.config.get("weights", {})
        self.thresholds = self.config.get("thresholds", {})
        
        # Initialize state
        self._kalman_filters = {}
        self._cache_ttl = 60  # 1 minute cache
        
        # ✅ CPU offloading için thread pool
        # ✅ Thread pool initialization
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, (os.cpu_count() or 2))
        )
        self._loop = asyncio.get_event_loop()
        
        
        logger.info(f"TrendModule initialized with {len(self.weights)} components")

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config oluştur"""
        logger.warning("Using default config for TrendModule")
        return {
            "weights": {
                "ema_trend": 0.15, "rsi_momentum": 0.12, "macd_trend": 0.13,
                "bollinger_trend": 0.10, "atr_volatility": 0.08, "adx_strength": 0.10,
                "stoch_rsi_momentum": 0.08, "momentum_oscillator": 0.07,
                "kalman_trend": 0.05, "z_score_normalization": 0.04,
                "wavelet_trend": 0.03, "hilbert_slope": 0.03, "fdi_complexity": 0.02
            },
            "thresholds": {
                "bullish": 0.7, "bearish": 0.3,
                "strong_trend": 0.6, "weak_trend": 0.4
            },
            "parameters": {
                "window": 100, "ema_periods": [20, 50, 200], "rsi_period": 14
            }
        }


        r"""
        Güçlendirilmiş compute_metrics:
         - veri hizalama (resample + log-returns)
         - hızlı korelasyon eşiği ile ön-filtre
         - istatistiksel p-value kontrolü + multiple-testing correction
         - exception logging per-pair
        """
    
    def _create_output_template(self) -> Dict[str, Any]:
        """Create standardized output template"""
        return {
            "module": self.module_name,
            "version": self.version,
            "timestamp": self.helpers.get_timestamp(),
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": {},
            "metadata": {}
        }

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output structure"""
        required = ["score", "signal", "confidence", "components", "explain"]
        return all(key in output for key in required) and 0 <= output["score"] <= 1

    def _record_execution(self, duration: float, success: bool):
        """Record execution metrics"""
        logger.debug(f"Trend analysis completed in {duration:.2f}s - Success: {success}")
        

            
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """Trend ve momentum analizi yapan ana metod"""
        start_time = AnalysisHelpers.get_timestamp()

        try:
            # 1) OHLCV verisini al - DÜZELTİLMİŞ
            interval = self.config.get("parameters", {}).get("default_interval", "1h")  # ✅ self.cfg yerine self.config
            lookback = self.config.get("parameters", {}).get("window", 100)
            
            data = await self._get_ohlcv_data(symbol, interval, lookback)
            min_data_points = self.config.get("parameters", {}).get("min_data_points", 50)
            if data.empty or len(data) < min_data_points:
                return await self._fallback_computation(symbol)

            # 2) Tüm metrikleri async olarak hesapla
            metrics = await self._compute_all_metrics_async(data)
            
            # 3) Sonuçları aggregate et
            aggregated = self.aggregate_output(metrics)
            
            # 4) Detaylı rapor oluştur
            explanation = self.generate_report(aggregated, metrics)
            
            output = self._create_output_template()
            output.update({
                "score": aggregated["score"],
                "signal": aggregated["signal"],
                "strength": aggregated["strength"],
                "confidence": aggregated["confidence"],
                "components": metrics,
                "explain": explanation,
                "metadata": {
                    "symbol": symbol,
                    "priority": priority,
                    "calculation_time": AnalysisHelpers.get_timestamp() - start_time,
                    "data_points": len(data),
                    "interval": interval
                }
            })

            if not self._validate_output(output):
                return await self._fallback_computation(symbol)

            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, True)
            return output

        except Exception as e:
            logger.exception(f"Trend analysis failed for {symbol}: {e}")
            self._record_execution(AnalysisHelpers.get_timestamp() - start_time, False)
            return await self._fallback_computation(symbol)
            

    async def _compute_all_metrics_async(self, data: pd.DataFrame) -> Dict[str, float]:
        """Tüm metrikleri async olarak hesapla - CPU offloading ile"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # ✅ CPU-bound işlemleri thread pool'a taşı
        compute_tasks = []
        
        # Classic metrics - thread pool'da çalıştır
        classic_methods = [
            ('ema_trend', partial(self._compute_ema_trend, close_prices)),
            ('rsi_momentum', partial(self._compute_rsi_momentum, close_prices)),
            ('macd_trend', partial(self._compute_macd_trend, close_prices)),
            ('bollinger_trend', partial(self._compute_bollinger_trend, close_prices)),
            ('atr_volatility', partial(self._compute_atr_volatility, high_prices, low_prices, close_prices)),
            ('adx_strength', partial(self._compute_adx_strength, high_prices, low_prices, close_prices)),
        ]
        
        for name, method in classic_methods:
            task = self._loop.run_in_executor(self._thread_pool, method)
            compute_tasks.append((name, task))
        
        # Advanced metrics - daha hafif olduğu için direkt async
        advanced_metrics = await asyncio.gather(
            self._compute_kalman_trend_async(close_prices),
            self._compute_z_score_normalization_async(close_prices),
            self._compute_wavelet_trend_async(close_prices),
            self._compute_hilbert_slope_async(close_prices),
            self._compute_fdi_complexity_async(close_prices),
            return_exceptions=True
        )
        
        # Sonuçları topla
        metrics = {}
        
        # Classic metrics sonuçlarını bekle
        for name, task in compute_tasks:
            try:
                result = await task
                metrics.update(result)
            except Exception as e:
                logger.warning(f"Metric {name} failed: {e}")
                metrics[name] = 0.5  # fallback
                
        # Advanced metrics sonuçlarını ekle
        advanced_names = ['kalman_trend', 'z_score_normalization', 'wavelet_trend', 'hilbert_slope', 'fdi_complexity']
        for name, result in zip(advanced_names, advanced_metrics):
            if not isinstance(result, Exception):
                metrics.update(result)
            else:
                logger.warning(f"Advanced metric {name} failed: {result}")
                metrics[name] = 0.5
                
        return metrics

    # ✅ Async metodlar için wrapper'lar
    async def _compute_kalman_trend_async(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_kalman_trend, prices
        )
    
    async def _compute_z_score_normalization_async(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_z_score_normalization, prices
        )
    
    async def _compute_wavelet_trend_async(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_wavelet_trend, prices
        )
    
    async def _compute_hilbert_slope_async(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_hilbert_slope, prices
        )
    
    async def _compute_fdi_complexity_async(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_fdi_complexity, prices
        )

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)





    @cache_result(ttl=60)
    async def _get_ohlcv_data(self, symbol: str, interval: str, lookback: int = None) -> pd.DataFrame:
        """Get OHLCV data with caching"""
        if lookback is None:
            lookback = self.config.get("parameters", {}).get("window", 100)
            
        binance = BinanceAggregator.get_instance()
        data = await binance.get_klines(symbol, interval, limit=lookback)
        
        if data.empty:
            raise ValueError(f"No OHLCV data for {symbol}")
            
        return data

    async def _compute_all_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute all trend and momentum metrics"""
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        volumes = data['volume'].values
        
        metrics = {}
        
        # Classic TA Metrics
        metrics.update(self._compute_ema_trend(close_prices))
        metrics.update(self._compute_rsi_momentum(close_prices))
        metrics.update(self._compute_macd_trend(close_prices))
        metrics.update(self._compute_bollinger_trend(close_prices))
        metrics.update(self._compute_atr_volatility(high_prices, low_prices, close_prices))
        metrics.update(self._compute_adx_strength(high_prices, low_prices, close_prices))
        metrics.update(self._compute_stoch_rsi_momentum(close_prices))
        metrics.update(self._compute_momentum_oscillator(close_prices))
        
        # Advanced Metrics
        metrics.update(self._compute_kalman_trend(close_prices))
        metrics.update(self._compute_z_score_normalization(close_prices))
        metrics.update(self._compute_wavelet_trend(close_prices))
        metrics.update(self._compute_hilbert_slope(close_prices))
        metrics.update(self._compute_fdi_complexity(close_prices))
        
        return metrics

    # [Mevcut teknik indicator metodları aynı kalacak...]
    # _compute_ema_trend, _compute_rsi_momentum, vb. metodlar değişmeden kalır
    # ✅ INDICATOR METODLARI

    def _compute_ema_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """EMA trend analizi"""
        try:
            ema_periods = self.config.get("parameters", {}).get("ema_periods", [20, 50, 200])
            current_price = prices[-1]
            
            ema_scores = []
            for period in ema_periods:
                ema = self._exponential_moving_average(prices, period)
                if len(ema) > period and not np.isnan(ema[-1]):
                    # Price > EMA is bullish
                    trend_strength = (current_price - ema[-1]) / ema[-1]
                    ema_scores.append(np.tanh(trend_strength * 10))  # normalize
            
            score = np.mean(ema_scores) if ema_scores else 0.0
            return {"ema_trend": float((score + 1) / 2)}  # normalize to 0-1

        except Exception as e:
                logger.warning(f"EMA trend calculation failed for {len(prices)} data points: {e}")
                return {"ema_trend": 0.5}



    def _compute_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """RSI momentum analizi"""
        try:
            period = self.config.get("parameters", {}).get("rsi_period", 14)
            rsi = self._relative_strength_index(prices, period)
            
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                # RSI 30-70 arası normalize
                rsi_value = rsi[-1]
                if rsi_value > 70:
                    score = 1.0  # overbought - bearish
                elif rsi_value < 30:
                    score = 0.0  # oversold - bullish  
                else:
                    # 30-70 arasını 0-1'e normalize
                    score = (rsi_value - 30) / 40
            else:
                score = 0.5
                
            return {"rsi_momentum": float(score)}
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return {"rsi_momentum": 0.5}

    def _compute_stoch_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """Basit Stoch RSI implementasyonu"""
        try:
            period = self.config.get("parameters", {}).get("stoch_rsi_period", 14)
            rsi = self._relative_strength_index(prices, period)
            
            if len(rsi) > period:
                # Simple Stoch RSI calculation
                current_rsi = rsi[-1]
                min_rsi = np.min(rsi[-period:])
                max_rsi = np.max(rsi[-period:])
                
                if max_rsi != min_rsi:
                    stoch_rsi = (current_rsi - min_rsi) / (max_rsi - min_rsi)
                    score = float(stoch_rsi / 100)  # normalize to 0-1
                else:
                    score = 0.5
            else:
                score = 0.5
                
            return {"stoch_rsi_momentum": score}
        except Exception as e:
            logger.warning(f"Stoch RSI calculation failed: {e}")
            return {"stoch_rsi_momentum": 0.5}
            

    def _compute_macd_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """MACD trend analizi"""
        try:
            fast = self.config.get("parameters", {}).get("macd_fast", 12)
            slow = self.config.get("parameters", {}).get("macd_slow", 26)
            signal = self.config.get("parameters", {}).get("macd_signal", 9)
            
            exp1 = pd.Series(prices).ewm(span=fast).mean()
            exp2 = pd.Series(prices).ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            if len(histogram) > 0 and not np.isnan(histogram.iloc[-1]):
                # Histogram pozitif ise bullish
                trend_strength = np.tanh(histogram.iloc[-1] / (prices[-1] * 0.01))
                score = (trend_strength + 1) / 2
            else:
                score = 0.5
                
            return {"macd_trend": float(score)}
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return {"macd_trend": 0.5}

    def _compute_bollinger_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Bollinger Bands trend analizi"""
        try:
            period = self.config.get("parameters", {}).get("bollinger_period", 20)
            std_dev = self.config.get("parameters", {}).get("bollinger_std", 2)
            
            sma = pd.Series(prices).rolling(period).mean()
            std = pd.Series(prices).rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = prices[-1]
            if len(sma) > period and not np.isnan(sma.iloc[-1]):
                # Price position in bands
                if current_price > upper_band.iloc[-1]:
                    score = 0.0  # overbought - bearish
                elif current_price < lower_band.iloc[-1]:
                    score = 1.0  # oversold - bullish
                else:
                    # Normalize position between bands
                    band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
                    if band_width > 0:
                        position = (current_price - lower_band.iloc[-1]) / band_width
                        score = 1.0 - position  # lower = more bullish
                    else:
                        score = 0.5
            else:
                score = 0.5
                
            return {"bollinger_trend": float(score)}
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            return {"bollinger_trend": 0.5}

    # Diğer teknik indicator metodları için placeholder'lar
    def _compute_atr_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """ATR volatility analizi"""
        return {"atr_volatility": 0.5}

    def _compute_adx_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """ADX trend strength analizi"""
        return {"adx_strength": 0.5}

    def _compute_stoch_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """Stochastic RSI momentum"""
        return {"stoch_rsi_momentum": 0.5}

    def _compute_momentum_oscillator(self, prices: np.ndarray) -> Dict[str, float]:
        """Momentum oscillator"""
        return {"momentum_oscillator": 0.5}

    def _compute_kalman_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Kalman filter trend"""
        return {"kalman_trend": 0.5}

    def _compute_z_score_normalization(self, prices: np.ndarray) -> Dict[str, float]:
        """Z-score normalization"""
        return {"z_score_normalization": 0.5}

    def _compute_wavelet_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Wavelet trend analysis"""
        return {"wavelet_trend": 0.5}

    def _compute_hilbert_slope(self, prices: np.ndarray) -> Dict[str, float]:
        """Hilbert transform slope"""
        return {"hilbert_slope": 0.5}

    def _compute_fdi_complexity(self, prices: np.ndarray) -> Dict[str, float]:
        """Fractal Dimension Index complexity"""
        return {"fdi_complexity": 0.5}
        


    def aggregate_output(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Aggregate individual metrics into final trend score
        
        Args:
            metrics: Dictionary of computed metric scores
            
        Returns:
            Aggregated output with final score and signal
        """
        # ✅ ANALYSIS_HELPERS ILE AGIRLIKLI ORTALAMA
        if self.weights and self.helpers.validate_score_dict(metrics):
            final_score = self.helpers.calculate_weights(metrics, self.weights)
        else:
            # Fallback: simple average
            final_score = np.mean(list(metrics.values())) if metrics else 0.5
        
        # ✅ NORMALIZE SCORE
        final_score = self.helpers.normalize_score(final_score)
            
        # Determine trend signal
        bullish_threshold = self.thresholds.get("bullish", 0.7)
        bearish_threshold = self.thresholds.get("bearish", 0.3)
        
        if final_score >= bullish_threshold:
            signal = "bullish"
        elif final_score <= bearish_threshold:
            signal = "bearish"
        else:
            signal = "neutral"
            
        # Trend strength classification
        strong_threshold = self.thresholds.get("strong_trend", 0.6)
        weak_threshold = self.thresholds.get("weak_trend", 0.4)
        
        if final_score >= strong_threshold or final_score <= (1 - strong_threshold):
            strength = "strong"
        elif final_score >= weak_threshold or final_score <= (1 - weak_threshold):
            strength = "moderate"
        else:
            strength = "weak"
            
        return {
            "score": float(final_score),
            "signal": signal,
            "strength": strength,
            "components": metrics,
            "timestamp": self.helpers.get_timestamp(),
            "confidence": min(final_score, 1 - final_score) * 2  # Distance from 0.5
        }

    def generate_report(self, aggregated: Dict, metrics: Dict) -> Dict[str, Any]:
        """
        Generate detailed explanation for the trend analysis
        
        Args:
            aggregated: Aggregated output from aggregate_output
            metrics: Raw metric scores
            
        Returns:
            Detailed explanation dictionary
        """
        score = aggregated["score"]
        signal = aggregated["signal"]
        strength = aggregated["strength"]
        
        # Key contributors
        components = aggregated.get("components", {})
        top_contributors = sorted(
            [(name, data if isinstance(data, (int, float)) else data.get('score', 0)) 
             for name, data in components.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        explanation = {
            "summary": f"Trend analysis indicates {strength} {signal} bias",
            "confidence": aggregated.get("confidence", 0.5),
            "key_metrics": {
                name: {
                    "score": score_val,
                    "contribution": score_val * self.weights.get(name, 0)
                }
                for name, score_val in top_contributors
            },
            "interpretation": self._interpret_trend_score(score, strength),
            "recommendation": self._generate_recommendation(score, signal, strength)
        }
        
        return explanation

    def _interpret_trend_score(self, score: float, strength: str) -> str:
        """Generate human-readable interpretation of trend score"""
        if score >= 0.8:
            return "Very strong upward momentum with clear bullish trend"
        elif score >= 0.7:
            return "Strong upward trend with positive momentum"
        elif score >= 0.6:
            return "Moderate upward bias with developing trend"
        elif score >= 0.4:
            return "Neutral to slightly directional, awaiting clearer trend"
        elif score >= 0.3:
            return "Moderate downward bias with developing trend"
        elif score >= 0.2:
            return "Strong downward trend with negative momentum"
        else:
            return "Very strong downward momentum with clear bearish trend"

    def _generate_recommendation(self, score: float, signal: str, strength: str) -> str:
        """Generate trading recommendation based on trend analysis"""
        if signal == "bullish" and strength == "strong":
            return "Consider long positions with tight stop-loss"
        elif signal == "bullish" and strength == "moderate":
            return "Potential long opportunities, monitor for confirmation"
        elif signal == "bearish" and strength == "strong":
            return "Consider short positions or reducing long exposure"
        elif signal == "bearish" and strength == "moderate":
            return "Potential short opportunities, await confirmation"
        else:
            return "Wait for clearer trend direction before taking positions"

    async def _fallback_computation(self, symbol: str) -> Dict[str, Any]:
        """Fallback computation when primary method fails"""
        logger.warning(f"Using fallback computation for {symbol}")
        
        # ✅ ANALYSIS_HELPERS UYUMLU FALLBACK
        fallback_data = self.helpers.create_fallback_output(self.module_name, "Insufficient data or computation error")
        
        return {
            **fallback_data,
            "symbol": symbol,
            "strength": "weak",
            "explain": {
                "summary": "Fallback analysis: insufficient data or computation error",
                "confidence": 0.1,
                "key_metrics": {},
                "interpretation": "Unable to determine clear trend direction",
                "recommendation": "Wait for more data or verify symbol"
            }
        }

    # Technical indicator helper methods (mevcut kod aynı kalır)
    def _exponential_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _simple_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(period).mean().values

    def _relative_strength_index(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    async def run(self, symbol: str, priority: str = "normal") -> Dict[str, Any]:
        """
        Main execution method for backward compatibility
        
        Args:
            symbol: Trading symbol to analyze
            priority: Execution priority
            
        Returns:
            Analysis results
        """
        return await self.compute_metrics(symbol)

    def get_metadata(self) -> Dict[str, Any]:
        """Return module metadata"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "description": "Trend direction and momentum strength analysis",
            "metrics": list(self.weights.keys()),
            "parallel_mode": "batch",
            "lifecycle": "development",
            "analysis_helpers_compatible": True,
            "config_source": "cm_loader" if hasattr(self, 'config') else "default"
        }



         
# Factory function for module creation
def create_trend_module(config: Dict = None) -> TrendModule:
    """
    Factory function to create TrendModule instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TrendModule instance
    """
    return TrendModule(config)
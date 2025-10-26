# analysis/trend_moment.py
"""
Trend & Momentum Analysis Module - Analysis Helpers Uyumlu
trend_moment.py
Version: 2.0.0 - Polars + Async + Multi-user
"""

import os
import logging
import asyncio
import pywt
import numpy as np
import pandas as pd
import polars as pl
import concurrent.futures
from functools import partial
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.signal import hilbert, detrend

from pykalman import KalmanFilter
from pydantic import ValidationError

from analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput, AnalysisUtilities

from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result

logger = logging.getLogger(__name__)


class TrendModule(BaseAnalysisModule):
    """
    Trend and Momentum Analysis Module - Tam Uyumlu
    Polars + Async + Multi-user support
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.module_name = "trend_moment"
        self.version = "2.0.0"
        self.dependencies = ["numpy", "pandas", "polars", "scipy", "pykalman", "pywt"]
        
        # ✅ ANALYSIS_HELPERS TAM ENTEGRASYON
        self.helpers = AnalysisHelpers()
        self.utils = AnalysisUtilities()
        
        # Load configuration - AnalysisHelpers ile
        if config is None:
            self.config = self.helpers.load_config_safe(
                self.module_name, 
                self._get_default_config()
            )
        else:
            self.config = config
            
        self.weights = self.config.get("weights", {})
        self.thresholds = self.config.get("thresholds", {})
        
        # Initialize state
        self._kalman_filters = {}
        self._cache_ttl = 60
        
        # ✅ POLARS OPTIMIZATION SETTINGS
        self._polars_streaming = self.config.get("polars_streaming", True)
        self._polars_parallel = self.config.get("polars_parallel", True)
        
        # ✅ Thread pool initialization
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, (os.cpu_count() or 2))
        )
        self._loop = asyncio.get_event_loop()
        
        logger.info(f"TrendModule v{self.version} initialized - Polars+Async")

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config with Polars support"""
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
                "window": 100, 
                "ema_periods": [20, 50, 200], 
                "rsi_period": 14,
                "min_data_points": 50,
                "default_interval": "1h"
            },
            "polars_streaming": True,
            "polars_parallel": True
        }

    async def execute_analysis(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """✅ BASE CLASS ABSTRACT METHOD IMPLEMENTATION - İSİM KORUNDU"""
        return await self.compute_metrics(symbol, priority)

    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """✅ BASE CLASS ABSTRACT METHOD IMPLEMENTATION"""
        start_time = self.helpers.get_timestamp()

        try:
            # 1) OHLCV verisini al - Polars destekli
            interval = self.config.get("parameters", {}).get("default_interval", "1h")
            lookback = self.config.get("parameters", {}).get("window", 100)
            
            data = await self._get_ohlcv_data(symbol, interval, lookback)
            min_data_points = self.config.get("parameters", {}).get("min_data_points", 50)
            
            if data.is_empty() or len(data) < min_data_points:
                return await self._fallback_computation(symbol)

            # 2) ✅ POLARS DATA PROCESSING
            close_prices = data['close'].to_numpy()
            high_prices = data['high'].to_numpy()
            low_prices = data['low'].to_numpy()

            # 3) Tüm metrikleri async olarak hesapla
            metrics = await self._compute_all_metrics(
                close_prices, high_prices, low_prices
            )
            
            # 4) Sonuçları aggregate et - ASYNC
            aggregated = await self.aggregate_output(metrics, symbol)
            
            # 5) Detaylı rapor oluştur - ASYNC
            explanation = await self.generate_report()
            
            output = {
                "score": aggregated["score"],
                "signal": aggregated["signal"],
                "confidence": aggregated["confidence"],
                "components": metrics,
                "explain": explanation,
                "timestamp": self.helpers.get_timestamp(),
                "module": self.module_name,
                "metadata": {
                    "symbol": symbol,
                    "priority": priority,
                    "calculation_time": self.helpers.get_timestamp() - start_time,
                    "data_points": len(data),
                    "interval": interval,
                    "data_type": "polars"
                }
            }

            # ✅ ANALYSIS_OUTPUT VALIDATION
            try:
                validated_output = AnalysisOutput(**output)
                return validated_output.dict()
            except ValidationError as e:
                logger.error(f"Output validation failed: {e}")
                return await self._fallback_computation(symbol)

        except Exception as e:
            logger.exception(f"Trend analysis failed for {symbol}: {e}")
            return await self._fallback_computation(symbol)

    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """✅ BASE CLASS ABSTRACT METHOD IMPLEMENTATION - ASYNC"""
        
        # ✅ ANALYSIS_HELPERS ILE AGIRLIKLI ORTALAMA
        if self.weights and self.utils.validate_score_dict(metrics):
            final_score = self.utils.calculate_weighted_average(metrics, self.weights)
        else:
            # Fallback: simple average
            final_score = np.mean(list(metrics.values())) if metrics else 0.5
        
        # ✅ NORMALIZE SCORE
        final_score = self.utils.normalize_score(final_score)
            
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

    async def generate_report(self) -> str:
        """✅ BASE CLASS ABSTRACT METHOD IMPLEMENTATION - ASYNC - STRING DÖNDÜRÜR"""
        # Basit string raporu - AnalysisOutput schema ile uyumlu
        return f"Trend analysis completed at {self.helpers.get_iso_timestamp()} using {len(self.weights)} technical indicators."

    async def _compute_all_metrics(self, close_prices: np.ndarray, 
                                       high_prices: np.ndarray, 
                                       low_prices: np.ndarray) -> Dict[str, float]:
        """Tüm metrikleri async olarak hesapla - Polars optimized"""
        
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
            ('stoch_rsi_momentum', partial(self._compute_stoch_rsi_momentum, close_prices)),
            ('momentum_oscillator', partial(self._compute_momentum_oscillator, close_prices)),
        ]
        
        for name, method in classic_methods:
            task = self._loop.run_in_executor(self._thread_pool, method)
            compute_tasks.append((name, task))
        
        # Advanced metrics - async wrapper'lar
        advanced_metrics = await asyncio.gather(
            self._compute_kalman_trend(close_prices),
            self._compute_z_score_normalization(close_prices),
            self._compute_wavelet_trend(close_prices),
            self._compute_hilbert_slope(close_prices),
            self._compute_fdi_complexity(close_prices),
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

    # ✅ POLARS DATA CONVERSION METHODS
    def _convert_to_polars(self, data: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
        """Convert pandas DataFrame to Polars DataFrame"""
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # ✅ ASYNC METOD WRAPPERS
    async def _compute_kalman_trend(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_kalman_trend, prices
        )
    
    async def _compute_z_score_normalization(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_z_score_normalization, prices
        )
    
    async def _compute_wavelet_trend(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_wavelet_trend, prices
        )
    
    async def _compute_hilbert_slope(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_hilbert_slope, prices
        )
    
    async def _compute_fdi_complexity(self, prices: np.ndarray) -> Dict[str, float]:
        return await self._loop.run_in_executor(
            self._thread_pool, self._compute_fdi_complexity, prices
        )

    # ✅ TAM METOD IMPLEMENTASYONLARI
    def _compute_ema_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """EMA trend analizi"""
        try:
            ema_periods = self.config.get("parameters", {}).get("ema_periods", [20, 50, 200])
            current_price = prices[-1]
            
            ema_scores = []
            for period in ema_periods:
                ema = self._exponential_moving_average(prices, period)
                if len(ema) > period and not np.isnan(ema[-1]):
                    trend_strength = (current_price - ema[-1]) / ema[-1]
                    ema_scores.append(np.tanh(trend_strength * 10))
            
            score = np.mean(ema_scores) if ema_scores else 0.0
            return {"ema_trend": float((score + 1) / 2)}

        except Exception as e:
            logger.warning(f"EMA trend calculation failed: {e}")
            return {"ema_trend": 0.5}

    def _compute_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """RSI momentum analizi"""
        try:
            period = self.config.get("parameters", {}).get("rsi_period", 14)
            rsi = self._relative_strength_index(prices, period)
            
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                rsi_value = rsi[-1]
                if rsi_value > 70:
                    score = 1.0
                elif rsi_value < 30:
                    score = 0.0
                else:
                    score = (rsi_value - 30) / 40
            else:
                score = 0.5
                
            return {"rsi_momentum": float(score)}
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return {"rsi_momentum": 0.5}

    def _compute_macd_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """MACD trend analizi"""
        try:
            fast = self.config.get("parameters", {}).get("macd_fast", 12)
            slow = self.config.get("parameters", {}).get("macd_slow", 26)
            signal = self.config.get("parameters", {}).get("macd_signal", 9)
            
            exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
            exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            if len(histogram) > 0 and not np.isnan(histogram.iloc[-1]):
                hist_value = histogram.iloc[-1]
                # Histogram pozitifse bullish, negatifse bearish
                score = (np.tanh(hist_value * 10) + 1) / 2
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
            
            series = pd.Series(prices)
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = prices[-1]
            sma_current = sma.iloc[-1] if not sma.isna().iloc[-1] else current_price
            
            if not np.isnan(upper_band.iloc[-1]) and not np.isnan(lower_band.iloc[-1]):
                # Price position relative to bands
                if current_price > upper_band.iloc[-1]:
                    score = 1.0  # Strong bullish
                elif current_price < lower_band.iloc[-1]:
                    score = 0.0  # Strong bearish
                else:
                    # Normalize position between bands
                    band_width = upper_band.iloc[-1] - lower_band.iloc[-1]
                    if band_width > 0:
                        position = (current_price - lower_band.iloc[-1]) / band_width
                        score = position
                    else:
                        score = 0.5
            else:
                score = 0.5
                
            return {"bollinger_trend": float(score)}
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}")
            return {"bollinger_trend": 0.5}

    def _compute_atr_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """ATR volatility analizi - volatility yüksekse trend güvenilirliği düşer"""
        try:
            period = self.config.get("parameters", {}).get("atr_period", 14)
            
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            if len(atr) > period and not np.isnan(atr.iloc[-1]):
                current_atr = atr.iloc[-1]
                current_price = close[-1]
                # ATR/Price ratio for normalized volatility
                volatility_ratio = current_atr / current_price
                # High volatility -> lower confidence in trend (score approaches 0.5)
                score = 1.0 - min(volatility_ratio * 10, 0.5)
            else:
                score = 0.5
                
            return {"atr_volatility": float(score)}
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return {"atr_volatility": 0.5}

    def _compute_adx_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """ADX trend strength analizi"""
        try:
            period = self.config.get("parameters", {}).get("adx_period", 14)
            
            # Basit ADX benzeri hesaplama
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            plus_dm = high_series.diff()
            minus_dm = -low_series.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = true_range = pd.concat([
                high_series - low_series,
                abs(high_series - close_series.shift(1)),
                abs(low_series - close_series.shift(1))
            ], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            if len(adx) > period and not np.isnan(adx.iloc[-1]):
                adx_value = adx.iloc[-1]
                # ADX > 25 strong trend, < 20 weak trend
                score = min(adx_value / 50, 1.0)  # Normalize to 0-1
            else:
                score = 0.5
                
            return {"adx_strength": float(score)}
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            return {"adx_strength": 0.5}

    def _compute_stoch_rsi_momentum(self, prices: np.ndarray) -> Dict[str, float]:
        """Stochastic RSI momentum analizi"""
        try:
            period = self.config.get("parameters", {}).get("stoch_rsi_period", 14)
            smooth = self.config.get("parameters", {}).get("stoch_rsi_smooth", 3)
            
            rsi = self._relative_strength_index(prices, period)
            rsi_series = pd.Series(rsi)
            
            stoch_rsi = (rsi_series - rsi_series.rolling(period).min()) / \
                       (rsi_series.rolling(period).max() - rsi_series.rolling(period).min())
            stoch_rsi_smooth = stoch_rsi.rolling(smooth).mean()
            
            if len(stoch_rsi_smooth) > 0 and not np.isnan(stoch_rsi_smooth.iloc[-1]):
                stoch_value = stoch_rsi_smooth.iloc[-1] * 100  # Convert to 0-100 scale
                if stoch_value > 80:
                    score = 1.0
                elif stoch_value < 20:
                    score = 0.0
                else:
                    score = (stoch_value - 20) / 60
            else:
                score = 0.5
                
            return {"stoch_rsi_momentum": float(score)}
        except Exception as e:
            logger.warning(f"Stochastic RSI calculation failed: {e}")
            return {"stoch_rsi_momentum": 0.5}

    def _compute_momentum_oscillator(self, prices: np.ndarray) -> Dict[str, float]:
        """Momentum oscillator analizi"""
        try:
            period = self.config.get("parameters", {}).get("momentum_period", 10)
            
            momentum = (prices[-1] / prices[-period] - 1) * 100
            
            # Normalize momentum to 0-1 scale
            score = (np.tanh(momentum / 10) + 1) / 2
            return {"momentum_oscillator": float(score)}
        except Exception as e:
            logger.warning(f"Momentum oscillator calculation failed: {e}")
            return {"momentum_oscillator": 0.5}

    def _compute_kalman_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Kalman filter trend analizi"""
        try:
            process_var = self.config.get("parameters", {}).get("kalman", {}).get("process_var", 1e-4)
            obs_var = self.config.get("parameters", {}).get("kalman", {}).get("obs_var", 1e-3)
            
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=prices[0],
                initial_state_covariance=1,
                observation_covariance=obs_var,
                transition_covariance=process_var
            )
            
            state_means, _ = kf.filter(prices)
            state_means = state_means.flatten()
            
            if len(state_means) > 1:
                # Kalman slope
                recent_slope = state_means[-1] - state_means[-2]
                normalized_slope = recent_slope / prices[-1]
                score = (np.tanh(normalized_slope * 100) + 1) / 2
            else:
                score = 0.5
                
            return {"kalman_trend": float(score)}
        except Exception as e:
            logger.warning(f"Kalman filter calculation failed: {e}")
            return {"kalman_trend": 0.5}

    def _compute_z_score_normalization(self, prices: np.ndarray) -> Dict[str, float]:
        """Z-score normalization trend analizi"""
        try:
            window = self.config.get("parameters", {}).get("z_score_window", 21)
            
            if len(prices) < window:
                return {"z_score_normalization": 0.5}
                
            recent_prices = prices[-window:]
            z_scores = (recent_prices - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
            current_z = z_scores[-1]
            
            # Z-score > 0 bullish, < 0 bearish
            score = (np.tanh(current_z) + 1) / 2
            return {"z_score_normalization": float(score)}
        except Exception as e:
            logger.warning(f"Z-score calculation failed: {e}")
            return {"z_score_normalization": 0.5}

    def _compute_wavelet_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Wavelet transform trend analizi"""
        try:
            wavelet_family = self.config.get("parameters", {}).get("wavelet_family", "db4")
            level = self.config.get("parameters", {}).get("wavelet_level", 3)
            
            # Detrend the data
            detrended = detrend(prices)
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(detrended, wavelet_family, level=level)
            
            # Use approximation coefficients for trend
            trend_coeffs = coeffs[0]
            
            if len(trend_coeffs) > 1:
                trend_slope = trend_coeffs[-1] - trend_coeffs[0]
                normalized_slope = trend_slope / (np.max(trend_coeffs) - np.min(trend_coeffs) + 1e-8)
                score = (np.tanh(normalized_slope * 10) + 1) / 2
            else:
                score = 0.5
                
            return {"wavelet_trend": float(score)}
        except Exception as e:
            logger.warning(f"Wavelet calculation failed: {e}")
            return {"wavelet_trend": 0.5}

    def _compute_hilbert_slope(self, prices: np.ndarray) -> Dict[str, float]:
        """Hilbert transform trend analizi"""
        try:
            window = self.config.get("parameters", {}).get("hilbert_window", 10)
            
            if len(prices) < window:
                return {"hilbert_slope": 0.5}
                
            analytic_signal = hilbert(prices[-window:])
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            if len(instantaneous_frequency) > 0:
                freq_trend = np.mean(instantaneous_frequency[-5:]) if len(instantaneous_frequency) >= 5 else instantaneous_frequency[-1]
                score = (np.tanh(freq_trend * 100) + 1) / 2
            else:
                score = 0.5
                
            return {"hilbert_slope": float(score)}
        except Exception as e:
            logger.warning(f"Hilbert transform calculation failed: {e}")
            return {"hilbert_slope": 0.5}

    def _compute_fdi_complexity(self, prices: np.ndarray) -> Dict[str, float]:
        """Fractal Dimension Index complexity analizi"""
        try:
            window = self.config.get("parameters", {}).get("fdi_window", 10)
            
            if len(prices) < window:
                return {"fdi_complexity": 0.5}
                
            # Basit FD hesaplama
            n = len(prices)
            l = np.sum(np.abs(np.diff(prices[-window:])))
            d = np.max(prices[-window:]) - np.min(prices[-window:])
            
            if d > 0:
                fd = 1 + (np.log(l) / np.log(d)) if l > 0 and d > 0 else 1.0
                # FD ~1 smooth trend, FD ~2 noisy/chaotic
                score = 2.0 - fd  # Smooth trend -> higher score
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5
                
            return {"fdi_complexity": float(score)}
        except Exception as e:
            logger.warning(f"FDI calculation failed: {e}")
            return {"fdi_complexity": 0.5}

    # ✅ DATA FETCHING WITH POLARS SUPPORT
    @cache_result(ttl=60)
    async def _get_ohlcv_data(self, symbol: str, interval: str, lookback: int = None) -> pl.DataFrame:
        """Get OHLCV data with Polars support"""
        if lookback is None:
            lookback = self.config.get("parameters", {}).get("window", 100)
            
        binance = BinanceAggregator.get_instance()
        data = await binance.get_klines(symbol, interval, limit=lookback)
        
        if data.empty:
            raise ValueError(f"No OHLCV data for {symbol}")
            
        # ✅ POLARS CONVERSION
        return self._convert_to_polars(data)

    async def _fallback_computation(self, symbol: str) -> Dict[str, Any]:
        """Fallback computation when primary method fails"""
        logger.warning(f"Using fallback computation for {symbol}")
        
        fallback_data = self.utils.create_fallback_output(
            self.module_name, 
            "Insufficient data or computation error"
        )
        
        return fallback_data

    # Technical indicator helper methods
    def _exponential_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _relative_strength_index(self, prices: np.ndarray, period: int) -> np.ndarray:
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    async def run(self, symbol: str, priority: str = "normal") -> Dict[str, Any]:
        """Backward compatibility method"""
        return await self.execute_analysis(symbol, priority)

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
            "polars_support": True,
            "async_support": True,
            "multi_user_ready": True,
            "config_source": "cm_loader" if hasattr(self, 'config') else "default"
        }

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


# Factory function for module creation
def create_trend_module(config: Dict = None) -> TrendModule:
    """Factory function to create TrendModule instance"""
    return TrendModule(config)
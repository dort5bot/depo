
# analysis/regime_anomal.py
"""
Regime Change Detection & Anomaly Module - TAM ASYNC
File: regime_anomal.py
Version: 2.0.0 - Async Optimized
not: ayrı config dosyası yok

Purpose:
- Detect sudden regime changes and anomalies using spot + futures data
- Computes rolling z-score, rolling skewness/kurtosis, cumulative return deviation
- Uses CUSUM (changepoint), IsolationForest (anomaly), and spectral-residual
- Full async, vectorized with pandas/numpy/polars support
- BaseAnalysisModule ile tam uyumlu, AnalysisHelpers entegrasyonlu

Key Metrics:
- CUSUM Change Point Detection
- Isolation Forest Anomaly Score  
- Rolling Z-Score Deviation
- Cumulative Return Deviation
- Spectral Residual Signal

Output: Regime Anomaly Score (0-1) with change/anomaly classification
"""

import asyncio
import logging
import math
import time
from typing import Dict, Any, Optional, List, Sequence, Callable
import numpy as np
import pandas as pd

# ✅ Analysis sistem import'ları
from analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_helpers import (
    AnalysisOutput, 
    AnalysisHelpers, 
    AnalysisUtilities,
    analysis_helpers,
    utility_functions
)

logger = logging.getLogger(__name__)

# ✅ Optional imports with fallbacks
try:
    from sklearn.ensemble import IsolationForest
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False
    logger.warning("scikit-learn not available, using fallback methods")

try:
    from scipy import signal
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    logger.warning("scipy not available, using fallback methods")

try:
    import polars as pl
    _HAVE_POLARS = True
except ImportError:
    _HAVE_POLARS = False
    logger.warning("polars not available, using pandas only")

# ✅ DEFAULT CONFIG - BaseModule ile uyumlu
DEFAULT_CONFIG = {
    "klines_interval": "1h",
    "klines_limit": 500,
    "oi_hist_period": "1h",
    "oi_hist_limit": 500,
    "funding_limit": 500,
    "rolling_window": 50,
    "zscore_threshold": 3.0,
    "cusum_h": 5.0,
    "cusum_k": 0.5,
    "iso_n_estimators": 100,
    "iso_contamination": 0.01,
    "combine_weights": {
        "cusum": 0.4, 
        "iso": 0.3, 
        "zscore": 0.2, 
        "cumret": 0.1
    },
    "min_points": 100,
    "normalize_bounds": (0.0, 1.0),
    "dataframe_type": "auto"  # auto, pandas, polars
}

class RegimeAnomalyModule(BaseAnalysisModule):
    """
    Regime & Anomaly Detection Module - Tam Async
    BaseAnalysisModule ile tam uyumlu, AnalysisHelpers entegrasyonlu
    """
    
    version = "2.0.0"
    module_name = "regime_anomal"
    dependencies = ["binance_api", "data_provider"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ✅ BaseAnalysisModule initialization
        super().__init__(config)
        
        # ✅ Module-specific initialization
        self.data_provider = None
        self._initialize_data_provider()
        
        # ✅ Config validation
        self._validate_config()

    def _initialize_data_provider(self):
        """Async data provider initialization"""
        try:
            from utils.binance_api.futuresclient import FuturesClient
            self.data_provider = FuturesClient()
        except ImportError as e:
            logger.warning(f"Data provider not available: {e}")
            self.data_provider = None

    def _validate_config(self):
        """Config validation with helper normalization"""
        weights = self.config.get("combine_weights", {})
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-4):
            normalized_weights = self.utils.normalize_weights(weights)
            self.config["combine_weights"] = normalized_weights
            logger.info(f"Combined weights normalized: {normalized_weights}")

    # ✅ CORE ASYNC METHODS - BaseAnalysisModule abstract methods
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Ana metrik hesaplama - Tam Async
        BaseAnalysisModule abstract method implementation
        """
        try:
            # ✅ Data fetching
            data_dict = await self._fetch_all_data(symbol)
            
            if not data_dict or data_dict.get("klines") is None:
                logger.warning(f"No data available for {symbol}")
                return self._create_fallback_metrics("No data available")
            
            # ✅ Data processing based on configured dataframe type
            dataframe_type = self.config.get("dataframe_type", "auto")
            processed_data = await self._process_data(data_dict, dataframe_type)
            
            # ✅ Feature computation
            features = await self._compute_features(processed_data, dataframe_type)
            
            # ✅ Anomaly detection algorithms
            anomaly_scores = await self._compute_anomaly_scores(features, dataframe_type)
            
            # ✅ Score aggregation
            final_metrics = await self._aggregate_scores(anomaly_scores, symbol)
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error in compute_metrics for {symbol}: {e}")
            return self._create_fallback_metrics(str(e))

    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Metrikleri aggregate edip final output oluştur - Tam Async
        BaseAnalysisModule abstract method implementation
        """
        try:
            # ✅ Score calculation
            combined_score = self._calculate_combined_score(metrics)
            
            # ✅ Signal determination
            signal = self._determine_signal(combined_score, metrics)
            
            # ✅ Confidence calculation
            confidence = self._calculate_confidence(metrics)
            
            # ✅ Explanation generation
            explanation = self._generate_explanation(combined_score, metrics, symbol)
            
            # ✅ Final output
            return {
                "score": combined_score,
                "signal": signal,
                "confidence": confidence,
                "components": metrics,
                "explain": explanation,
                "timestamp": time.time(),
                "module": self.module_name,
                "symbol": symbol
            }
            
        except Exception as e:
            logger.error(f"Error in aggregate_output for {symbol}: {e}")
            return self.utils.create_fallback_output(self.module_name, str(e))

    async def generate_report(self) -> Dict[str, Any]:
        """
        Modül durum raporu oluştur - Tam Async
        BaseAnalysisModule abstract method implementation
        """
        try:
            health_status = await self.health_check()
            performance_metrics = self.get_performance_metrics()
            
            return {
                "module": self.module_name,
                "version": self.version,
                "status": "healthy",
                "timestamp": time.time(),
                "health_check": health_status,
                "performance": performance_metrics,
                "config": {
                    "rolling_window": self.config.get("rolling_window"),
                    "zscore_threshold": self.config.get("zscore_threshold"),
                    "dataframe_type": self.config.get("dataframe_type", "auto")
                },
                "dependencies": self.dependencies
            }
        except Exception as e:
            logger.error(f"Error in generate_report: {e}")
            return {
                "module": self.module_name,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    # ✅ ASYNC DATA PROCESSING METHODS
    async def _fetch_all_data(self, symbol: str) -> Dict[str, Any]:
        """Tüm gerekli verileri async olarak çek"""
        if not self.data_provider:
            raise RuntimeError("Data provider not available")
        
        tasks = {
            "klines": self.data_provider.get_klines(
                symbol, 
                self.config.get("klines_interval", "1h"),
                self.config.get("klines_limit", 500)
            ),
            "oi_hist": self.data_provider.get_open_interest_hist(
                symbol,
                self.config.get("oi_hist_period", "1h"),
                self.config.get("oi_hist_limit", 500)
            ),
            "funding": self.data_provider.get_funding_rate(
                symbol,
                self.config.get("funding_limit", 500)
            )
        }
        
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            data_dict = {}
            
            for key, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Data fetch failed for {key}: {result}")
                    data_dict[key] = None
                else:
                    data_dict[key] = result
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            return {}

    async def _process_data(self, data_dict: Dict[str, Any], dataframe_type: str = "auto") -> Dict[str, Any]:
        """Veriyi işle ve normalize et"""
        processed = {}
        
        for key, data in data_dict.items():
            if data is None:
                processed[key] = None
                continue
                
            if dataframe_type == "polars" and _HAVE_POLARS:
                processed[key] = self._convert_to_polars(data)
            else:
                processed[key] = self._convert_to_pandas(data)
                
        return processed

    def _convert_to_pandas(self, data: Any) -> pd.DataFrame:
        """Veriyi pandas DataFrame'e dönüştür"""
        if isinstance(data, pd.DataFrame):
            return data
        elif _HAVE_POLARS and isinstance(data, pl.DataFrame):
            return data.to_pandas()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return pd.DataFrame()

    def _convert_to_polars(self, data: Any) -> pl.DataFrame:
        """Veriyi polars DataFrame'e dönüştür"""
        if not _HAVE_POLARS:
            return self._convert_to_pandas(data)
            
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        elif isinstance(data, list):
            return pl.DataFrame(data)
        elif isinstance(data, dict):
            return pl.DataFrame([data])
        else:
            return pl.DataFrame()

    # ✅ FEATURE COMPUTATION METHODS
    async def _compute_features(self, processed_data: Dict[str, Any], dataframe_type: str) -> Dict[str, Any]:
        """Özellikleri hesapla"""
        features = {}
        
        # Price-based features
        if processed_data.get("klines") is not None:
            close_series = self._extract_close_series(processed_data["klines"], dataframe_type)
            
            if len(close_series) > 0:
                features["returns"] = self._calculate_returns(close_series, dataframe_type)
                features["volatility"] = self._calculate_volatility(features["returns"], dataframe_type)
                features["zscore"] = self._calculate_zscore(close_series, dataframe_type)
                features["cumulative_returns"] = self._calculate_cumulative_returns(close_series, dataframe_type)
        
        # OI-based features
        if processed_data.get("oi_hist") is not None:
            oi_series = self._extract_oi_series(processed_data["oi_hist"], dataframe_type)
            if len(oi_series) > 0:
                features["oi_change"] = self._calculate_oi_change(oi_series, dataframe_type)
        
        return features

    def _extract_close_series(self, data: Any, dataframe_type: str) -> Any:
        """Close serisini çıkar"""
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(data, pl.DataFrame):
            if "close" in data.columns:
                return data["close"]
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                return data["close"]
        return []

    def _extract_oi_series(self, data: Any, dataframe_type: str) -> Any:
        """Open Interest serisini çıkar"""
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(data, pl.DataFrame):
            if "openInterest" in data.columns:
                return data["openInterest"]
            elif "value" in data.columns:
                return data["value"]
        elif isinstance(data, pd.DataFrame):
            if "openInterest" in data.columns:
                return data["openInterest"]
            elif "value" in data.columns:
                return data["value"]
        return []

    def _calculate_returns(self, close_series: Any, dataframe_type: str) -> Any:
        """Getiri hesapla"""
        window = self.config.get("rolling_window", 50)
        
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(close_series, pl.Series):
            returns = close_series.diff() / close_series.shift(1)
            return returns.fill_nan(0).fill_null(0)
        else:
            # Pandas fallback
            close_pd = close_series if isinstance(close_series, pd.Series) else pd.Series(close_series)
            returns = close_pd.pct_change().fillna(0)
            return returns

    def _calculate_volatility(self, returns: Any, dataframe_type: str) -> Any:
        """Volatilite hesapla"""
        window = self.config.get("rolling_window", 50)
        
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(returns, pl.Series):
            return returns.rolling_std(window_size=window).fill_nan(0).fill_null(0)
        else:
            # Pandas fallback
            returns_pd = returns if isinstance(returns, pd.Series) else pd.Series(returns)
            volatility = returns_pd.rolling(window=window, min_periods=1).std().fillna(0)
            return volatility

    def _calculate_zscore(self, series: Any, dataframe_type: str) -> Any:
        """Z-score hesapla"""
        window = self.config.get("rolling_window", 50)
        
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(series, pl.Series):
            rolling_mean = series.rolling_mean(window_size=window)
            rolling_std = series.rolling_std(window_size=window)
            zscore = (series - rolling_mean) / rolling_std
            return zscore.fill_nan(0).fill_null(0).abs()
        else:
            # Pandas fallback
            series_pd = series if isinstance(series, pd.Series) else pd.Series(series)
            rolling_mean = series_pd.rolling(window=window, min_periods=1).mean()
            rolling_std = series_pd.rolling(window=window, min_periods=1).std().replace(0, np.nan)
            zscore = ((series_pd - rolling_mean) / rolling_std).abs().fillna(0)
            return zscore

    def _calculate_cumulative_returns(self, close_series: Any, dataframe_type: str) -> Any:
        """Kümülatif getiri hesapla"""
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(close_series, pl.Series):
            log_returns = (close_series / close_series.shift(1)).log().fill_nan(0).fill_null(0)
            cumulative_returns = log_returns.cum_sum()
            return cumulative_returns
        else:
            # Pandas fallback
            close_pd = close_series if isinstance(close_series, pd.Series) else pd.Series(close_series)
            log_returns = np.log(close_pd / close_pd.shift(1)).fillna(0)
            cumulative_returns = log_returns.cumsum()
            return cumulative_returns

    def _calculate_oi_change(self, oi_series: Any, dataframe_type: str) -> Any:
        """Open Interest değişimini hesapla"""
        if dataframe_type == "polars" and _HAVE_POLARS and isinstance(oi_series, pl.Series):
            return oi_series.pct_change().fill_nan(0).fill_null(0)
        else:
            # Pandas fallback
            oi_pd = oi_series if isinstance(oi_series, pd.Series) else pd.Series(oi_series)
            return oi_pd.pct_change().fillna(0)

    # ✅ ANOMALY DETECTION METHODS
    async def _compute_anomaly_scores(self, features: Dict[str, Any], dataframe_type: str) -> Dict[str, float]:
        """Anomali skorlarını hesapla"""
        scores = {}
        
        try:
            # CUSUM Score
            if "returns" in features:
                returns_data = self._get_last_values(features["returns"], dataframe_type, 100)
                scores["cusum"] = self._compute_cusum_score(returns_data)
            
            # Isolation Forest Score
            if "cumulative_returns" in features:
                cumulative_data = self._get_last_values(features["cumulative_returns"], dataframe_type, 100)
                scores["iso_forest"] = self._compute_isolation_forest_score(cumulative_data)
            
            # Z-Score Anomaly
            if "zscore" in features:
                zscore_data = self._get_last_values(features["zscore"], dataframe_type, 50)
                scores["zscore_anomaly"] = self._compute_zscore_anomaly(zscore_data)
            
            # Cumulative Return Deviation
            if "cumulative_returns" in features:
                cumret_data = self._get_last_values(features["cumulative_returns"], dataframe_type, 50)
                scores["cumret_deviation"] = self._compute_cumret_deviation(cumret_data)
            
            # Spectral Residual Score
            if "returns" in features:
                returns_data = self._get_last_values(features["returns"], dataframe_type, 100)
                scores["spectral_residual"] = self._compute_spectral_residual_score(returns_data)
                
        except Exception as e:
            logger.error(f"Error computing anomaly scores: {e}")
            # Fallback scores
            scores = {
                "cusum": 0.5,
                "iso_forest": 0.5, 
                "zscore_anomaly": 0.5,
                "cumret_deviation": 0.5,
                "spectral_residual": 0.5
            }
        
        return scores

    def _get_last_values(self, series: Any, dataframe_type: str, n: int) -> np.ndarray:
        """Seriden son n değeri al"""
        try:
            if dataframe_type == "polars" and _HAVE_POLARS and isinstance(series, pl.Series):
                values = series.tail(n).to_numpy()
            elif isinstance(series, pd.Series):
                values = series.tail(n).values
            else:
                values = np.array(series[-n:]) if hasattr(series, '__getitem__') else np.array([0.5])
            
            # Clean NaN values
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
            return values
            
        except Exception as e:
            logger.warning(f"Error getting last values: {e}")
            return np.array([0.5])

    def _compute_cusum_score(self, data: np.ndarray) -> float:
        """CUSUM change detection score"""
        if len(data) < 10:
            return 0.5
            
        try:
            h = self.config.get("cusum_h", 5.0)
            k = self.config.get("cusum_k", 0.5)
            
            s_pos = 0.0
            s_neg = 0.0
            change_points = 0
            
            mu = np.median(data)
            
            for x in data:
                diff = x - mu - k
                s_pos = max(0.0, s_pos + diff)
                
                diff_neg = -(x - mu) - k
                s_neg = max(0.0, s_neg + diff_neg)
                
                if s_pos > h or s_neg > h:
                    change_points += 1
                    s_pos = 0.0
                    s_neg = 0.0
            
            score = min(1.0, change_points / (len(data) / 10.0))
            return float(score)
            
        except Exception as e:
            logger.warning(f"CUSUM computation failed: {e}")
            return 0.5

    def _compute_isolation_forest_score(self, data: np.ndarray) -> float:
        """Isolation Forest anomaly score"""
        if len(data) < 20 or not _HAVE_SKLEARN:
            return 0.5
            
        try:
            from sklearn.ensemble import IsolationForest
            
            X = data.reshape(-1, 1)
            clf = IsolationForest(
                n_estimators=self.config.get("iso_n_estimators", 100),
                contamination=self.config.get("iso_contamination", 0.01),
                random_state=42
            )
            clf.fit(X)
            scores = -clf.decision_function(X)
            
            # Normalize to [0, 1]
            if len(scores) > 0:
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    normalized = (scores - min_score) / (max_score - min_score)
                else:
                    normalized = np.zeros_like(scores)
                return float(np.max(normalized))
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            return 0.5

    def _compute_zscore_anomaly(self, zscores: np.ndarray) -> float:
        """Z-score based anomaly detection"""
        if len(zscores) == 0:
            return 0.5
            
        threshold = self.config.get("zscore_threshold", 3.0)
        max_zscore = np.max(np.abs(zscores))
        
        if max_zscore > threshold:
            anomaly_score = min(1.0, (max_zscore - threshold) / threshold)
        else:
            anomaly_score = 0.0
            
        return float(anomaly_score)

    def _compute_cumret_deviation(self, cumulative_returns: np.ndarray) -> float:
        """Cumulative return deviation score"""
        if len(cumulative_returns) < 10:
            return 0.5
            
        try:
            rolling_mean = np.convolve(cumulative_returns, np.ones(10)/10, mode='valid')
            if len(rolling_mean) == 0:
                return 0.5
                
            last_cumret = cumulative_returns[-1]
            mean_cumret = rolling_mean[-1]
            std_cumret = np.std(cumulative_returns[-10:]) or 1.0
            
            deviation = abs(last_cumret - mean_cumret) / std_cumret
            score = min(1.0, deviation / 2.0)  # Normalize
            return float(score)
            
        except Exception as e:
            logger.warning(f"Cumulative return deviation failed: {e}")
            return 0.5

    def _compute_spectral_residual_score(self, data: np.ndarray) -> float:
        """Spectral residual anomaly score"""
        if len(data) < 10:
            return 0.5
            
        try:
            if _HAVE_SCIPY:
                # Simple spectral analysis
                from scipy import signal
                detrended = signal.detrend(data)
                residuals = np.abs(detrended)
            else:
                # Fallback: second difference
                diff2 = np.abs(np.diff(data, n=2))
                residuals = np.zeros_like(data)
                if len(diff2) > 0:
                    residuals[2:] = diff2
            
            if len(residuals) > 0:
                max_residual = np.max(residuals)
                if max_residual > 0:
                    normalized = residuals / max_residual
                    return float(np.max(normalized))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Spectral residual failed: {e}")
            return 0.5

    # ✅ SCORE AGGREGATION METHODS
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """Metrikleri birleştirerek final skoru hesapla"""
        weights = self.config.get("combine_weights", {})
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, score in metrics.items():
            weight = weights.get(metric_name, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 0.5
        
        # Normalize to [0, 1]
        normalized_score = max(0.0, min(1.0, raw_score))
        return round(normalized_score, 6)

    def _determine_signal(self, score: float, metrics: Dict[str, float]) -> str:
        """Skora göre sinyal belirle"""
        if score >= 0.7:
            # Find dominant component
            dominant_metric = max(metrics.items(), key=lambda x: x[1])
            if dominant_metric[0] in ["cusum", "cumret_deviation"]:
                return "regime_change"
            else:
                return "anomaly_high"
        elif score >= 0.5:
            return "anomaly_medium"
        elif score >= 0.3:
            return "anomaly_low"
        else:
            return "normal"

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Güven skoru hesapla"""
        scores = list(metrics.values())
        if not scores:
            return 0.0
        
        # Higher confidence when scores are consistent
        variance = np.var(scores) if len(scores) > 1 else 0.0
        mean_score = np.mean(scores)
        confidence = mean_score * (1.0 - variance)
        return round(max(0.0, min(1.0, confidence)), 4)

    def _generate_explanation(self, score: float, metrics: Dict[str, float], symbol: str) -> str:
        """Açıklama metni oluştur"""
        if score >= 0.7:
            dominant = max(metrics.items(), key=lambda x: x[1])
            return f"Strong {dominant[0]} detected for {symbol} (score: {score:.3f})"
        elif score >= 0.5:
            return f"Moderate anomaly signals for {symbol} (score: {score:.3f})"
        elif score >= 0.3:
            return f"Minor anomaly detected for {symbol} (score: {score:.3f})"
        else:
            return f"No significant anomalies detected for {symbol} (score: {score:.3f})"

    def _create_fallback_metrics(self, reason: str) -> Dict[str, float]:
        """Fallback metrikler oluştur"""
        logger.warning(f"Using fallback metrics: {reason}")
        return {
            "cusum": 0.5,
            "iso_forest": 0.5,
            "zscore_anomaly": 0.5,
            "cumret_deviation": 0.5,
            "spectral_residual": 0.5,
            "fallback_reason": reason
        }

# ✅ BACKWARD COMPATIBILITY
@BaseAnalysisModule.legacy_compatible
class LegacyCompatibleRegimeAnomalyModule(RegimeAnomalyModule):
    """Eski sistemlerle uyumluluk için"""
    pass

# ✅ MODULE REGISTRATION
async def run(symbol: str, priority: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Legacy run function for module registration
    AnalysisCore ve AnalysisRouter ile uyumlu
    """
    module = RegimeAnomalyModule(config=config)
    return await module.execute_analysis(symbol, priority)

# ✅ TEST FUNCTION
async def test_module():
    """Modül test fonksiyonu"""
    module = RegimeAnomalyModule()
    
    # Test health check
    health = await module.health_check()
    print("Health Check:", health)
    
    # Test metrics computation
    metrics = await module.compute_metrics("BTCUSDT")
    print("Metrics:", metrics)
    
    # Test report generation
    report = await module.generate_report()
    print("Report:", report)

if __name__ == "__main__":
    asyncio.run(test_module())
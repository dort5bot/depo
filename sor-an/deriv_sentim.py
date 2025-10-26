# analysis/deriv_sentim.py
"""
analysis/deriv_sentim.py
Version: 2.0.0 - Tam Async & Analysis Helpers Uyumlu
Derivatives & Sentiment Analysis Module
Futures piyasası pozisyon verilerine dayalı trader sentiment analizi

Metrikler:
- Temel: Funding Rate, Open Interest, Long/Short Ratio
- Gelişmiş: OI Change Rate, Funding Rate Skew, Volume Imbalance Index
- Profesyonel: Liquidation Heatmap, OI Delta Divergence, Volatility Skew

Tam Async, Multi-User uyumlu, Analysis Helpers entegrasyonlu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
import logging
import time

from analysis.analysis_base_module import BaseAnalysisModule, legacy_compatible
from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers, AnalysisUtilities
from utils.binance_api.binance_a import BinanceAggregator
from analysis.config.c_deriv import DerivSentimentConfig, CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SentimentComponents:
    """Sentiment skor bileşenleri"""
    funding_rate: float
    open_interest: float
    long_short_ratio: float
    oi_change_rate: float
    funding_skew: float
    volume_imbalance: float
    liquidation_heat: float
    oi_delta_divergence: float
    volatility_skew: float


@legacy_compatible
class DerivativesSentimentModule(BaseAnalysisModule):
    """
    Futures pozisyon verilerine dayalı sentiment analiz modülü
    Tam Async, Multi-User uyumlu, Analysis Helpers entegrasyonlu
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # ✅ CONFIG YÜKLEME
        self.cfg = DerivSentimentConfig(**(config or CONFIG))
        self.module_name = "derivatives_sentiment"
        self.version = "2.0.0"
        self.dependencies = ["binance_api"]
        
        # ✅ ANALYSIS HELPERS INTEGRATION
        self.helpers = AnalysisHelpers()
        self.utils = AnalysisUtilities()
        
        # ✅ WEIGHTS VE THRESHOLDS
        self.weights = self.cfg.weights
        self.thresholds = {
            "bullish": getattr(self.cfg.thresholds, "bullish", 0.6),
            "bearish": getattr(self.cfg.thresholds, "bearish", 0.4),
            "extreme_bull": getattr(self.cfg.thresholds, "extreme_bull", 0.8),
            "extreme_bear": getattr(self.cfg.thresholds, "extreme_bear", 0.2)
        }
        
        # Initialize Binance client
        self.binance = BinanceAggregator()
        
        # Cache için
        self._cache = {}
        self._cache_ttl = getattr(self.cfg.parameters, "cache_ttl", 300)
        
        # Normalize weights
        self.normalized_weights = self.utils.normalize_weights(self.cfg.weights)
        
        logger.info(f"DerivativesSentimentModule initialized: {self.module_name} v{self.version}")

    async def initialize(self):
        """Initialize module resources"""
        await self.binance.initialize()
        logger.info("DerivativesSentimentModule initialized successfully")

    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        ✅ TAM ASYNC & ANALYSIS_HELPERS UYUMLU ANA METOT
        Tüm sentiment metriklerini hesapla ve standart AnalysisOutput döndür
        """
        start_time = self.helpers.get_timestamp()
        
        try:
            # Verileri paralel olarak getir
            tasks = [
                self._get_funding_data(symbol),
                self._get_open_interest_data(symbol),
                self._get_long_short_data(symbol),
                self._get_liquidation_data(symbol),
                self._get_taker_ratio_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Hata kontrolü
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in data fetch task {i}: {result}")
                    return self._create_fallback_output(f"Data fetch error: {result}")

            funding_data, oi_data, ls_data, liq_data, taker_data = results
            
            # Metrikleri hesapla
            components = await self._calculate_components(
                symbol, funding_data, oi_data, ls_data, liq_data, taker_data
            )
            
            # Sentiment skoru oluştur
            score, signal, explanation, calculated_components = await self._compute_sentiment_score(components)
            
            # Confidence hesapla
            confidence = self._calculate_confidence(components, calculated_components)
            
            # ✅ STANDART OUTPUT ŞABLONU - AnalysisOutput modeli kullan
            output_data = {
                "score": self._normalize_score(score),
                "signal": signal,
                "confidence": confidence,
                "components": calculated_components,
                "explain": explanation,
                "timestamp": self.helpers.get_timestamp(),
                "module": self.module_name,
                "metadata": {
                    "symbol": symbol,
                    "priority": priority,
                    "calculation_time": self.helpers.get_timestamp() - start_time,
                    "data_sources": len([d for d in [funding_data, oi_data, ls_data, liq_data, taker_data] if not d.empty])
                }
            }
            
            # ✅ ANALYSISOUTPUT VALIDATION
            try:
                validated_output = AnalysisOutput(**output_data)
                output_dict = validated_output.dict()
            except ValidationError as e:
                logger.warning(f"Output validation failed: {e}, using fallback")
                return self._create_fallback_output(f"Validation error: {e}")
            
            self._record_execution(self.helpers.get_timestamp() - start_time, True)
            return output_dict
            
        except Exception as e:
            logger.error(f"Compute metrics failed for {symbol}: {e}")
            self._record_execution(self.helpers.get_timestamp() - start_time, False)
            return self._create_fallback_output(str(e))
    
    async def _calculate_components(self, symbol: str, funding_data: pd.DataFrame, 
                                  oi_data: pd.DataFrame, ls_data: pd.DataFrame,
                                  liq_data: pd.DataFrame, taker_data: pd.DataFrame) -> SentimentComponents:
        """Tüm sentiment bileşenlerini hesapla"""
        
        # Temel metrikler
        funding_rate = await self._calculate_funding_sentiment(funding_data)
        open_interest = await self._calculate_oi_sentiment(oi_data)
        long_short_ratio = await self._calculate_ls_sentiment(ls_data)
        
        # Gelişmiş metrikler
        oi_change_rate = await self._calculate_oi_change_rate(oi_data)
        funding_skew = await self._calculate_funding_skew(funding_data)
        volume_imbalance = await self._calculate_volume_imbalance(taker_data)
        
        # Profesyonel metrikler
        liquidation_heat = await self._calculate_liquidation_heat(liq_data)
        oi_delta_divergence = await self._calculate_oi_delta_divergence(oi_data, funding_data)
        volatility_skew = await self._calculate_volatility_skew(symbol, oi_data)
        
        return SentimentComponents(
            funding_rate=funding_rate,
            open_interest=open_interest,
            long_short_ratio=long_short_ratio,
            oi_change_rate=oi_change_rate,
            funding_skew=funding_skew,
            volume_imbalance=volume_imbalance,
            liquidation_heat=liquidation_heat,
            oi_delta_divergence=oi_delta_divergence,
            volatility_skew=volatility_skew
        )
    
    async def _compute_sentiment_score(self, components: SentimentComponents) -> Tuple[float, str, str, Dict[str, float]]:
        """
        Bileşenleri ağırlıklandırarak sentiment skoru oluştur
        
        Returns:
            Tuple: (score, signal, explanation, components_dict)
        """
        # Bileşenleri dictionary'ye çevir
        components_dict = {
            "funding_rate": components.funding_rate,
            "open_interest": components.open_interest,
            "long_short_ratio": components.long_short_ratio,
            "oi_change_rate": components.oi_change_rate,
            "funding_skew": components.funding_skew,
            "volume_imbalance": components.volume_imbalance,
            "liquidation_heat": components.liquidation_heat,
            "oi_delta_divergence": components.oi_delta_divergence,
            "volatility_skew": components.volatility_skew
        }
        
        # Ağırlıklı ortalama hesapla (AnalysisUtilities kullanarak)
        weighted_score = self.utils.calculate_weighted_average(components_dict, self.normalized_weights)
        
        # Normalizasyon (-1 ile 1 arası) ve 0-1 aralığına dönüştürme
        scale_factor = getattr(self.cfg.normalization, "scale_factor", 3.0)
        sentiment_score_raw = np.tanh(weighted_score * scale_factor)
        sentiment_score = (sentiment_score_raw + 1) / 2  # Convert -1..1 to 0..1
        
        # Sinyal belirleme
        signal = self._get_sentiment_signal(sentiment_score)
        
        # Açıklama oluştur
        explanation = self._generate_explanation(sentiment_score, components, signal)
        
        return sentiment_score, signal, explanation, components_dict
    
    def _get_sentiment_signal(self, score: float) -> str:
        """Skora göre sinyal belirle"""
        if score >= self.thresholds["extreme_bull"]:
            return "extreme_bull"
        elif score >= self.thresholds["bullish"]:
            return "bullish"
        elif score <= self.thresholds["extreme_bear"]:
            return "extreme_bear"
        elif score <= self.thresholds["bearish"]:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_confidence(self, components: SentimentComponents, components_dict: Dict[str, float]) -> float:
        """Veri kalitesine ve metrik tutarlılığına dayalı güven skoru hesapla"""
        confidence_factors = []
        
        # Data quality factors
        valid_components = sum(1 for value in components_dict.values() if abs(value) > 0.01)
        if valid_components >= 5:  # At least 5 valid metrics
            confidence_factors.append(0.4)
        
        # Metric consistency
        component_values = list(components_dict.values())
        if len(component_values) >= 3:
            consistency = 1.0 - (np.std(component_values) / 0.5)  # Normalize
            confidence_factors.append(max(0.0, consistency * 0.3))
        
        # Strong signal strength
        extreme_values = sum(1 for value in component_values if abs(value) > 0.5)
        if extreme_values >= 2:
            confidence_factors.append(0.3)
        
        return min(1.0, sum(confidence_factors))
    
    def _generate_explanation(self, score: float, components: SentimentComponents, signal: str) -> str:
        """Sentiment skoru için açıklama oluştur"""
        
        explanations = []
        
        # Funding rate analizi
        if abs(components.funding_rate) > 0.3:
            direction = "pozitif" if components.funding_rate > 0 else "negatif"
            explanations.append(f"Funding rate {direction} bölgede")
        
        # Open Interest
        if components.open_interest > 0.2:
            explanations.append("Open Interest artışı")
        elif components.open_interest < -0.2:
            explanations.append("Open Interest düşüşü")
        
        # Long/Short Ratio
        if components.long_short_ratio > 0.25:
            explanations.append("Long pozisyonlar hakim")
        elif components.long_short_ratio < -0.25:
            explanations.append("Short pozisyonlar hakim")
        
        # Liquidation heat
        if components.liquidation_heat > 0.3:
            explanations.append("Yüksek likidasyon riski")
        
        if not explanations:
            explanations.append("Piyasa dengeli seyirde")
        
        signal_display = signal.upper().replace('_', ' ')
        return f"Derivatives Sentiment {signal_display} - " + ". ".join(explanations)
    
    # ✅ EKSİK METODLAR - ANALYSIS HELPERS UYUMLU
    def _normalize_score(self, score: float) -> float:
        """Skoru normalize et"""
        return self.utils.normalize_score(score)
    
    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Ağırlıklı skor hesapla"""
        return self.utils.calculate_weighted_average(scores, weights)
    
    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """Output validasyonu"""
        return self.utils.validate_output(output)
    
    def _create_output_template(self) -> Dict[str, Any]:
        """Standart output şablonu oluştur"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": "",
            "timestamp": self.helpers.get_timestamp(),
            "module": self.module_name
        }
    
    def _create_fallback_output(self, reason: str) -> Dict[str, Any]:
        """Fallback output oluştur"""
        return self.utils.create_fallback_output(self.module_name, reason)
    
    def _record_execution(self, duration: float, success: bool = True):
        """Performans metriklerini kaydet"""
        self.helpers.update_performance_metrics(f"{self.module_name}_duration", duration)
        self.helpers.update_performance_metrics(f"{self.module_name}_success", 1.0 if success else 0.0)
    
    # METRİK HESAPLAMA FONKSİYONLARI
    
    async def _calculate_funding_sentiment(self, funding_data: pd.DataFrame) -> float:
        """Funding rate sentiment hesapla"""
        if funding_data.empty:
            return 0.0
        
        current_funding = funding_data['fundingRate'].iloc[-1]
        avg_funding = funding_data['fundingRate'].mean()
        
        # Normalize edilmiş funding sentiment
        funding_sentiment = np.tanh((current_funding - avg_funding) * 1000)
        return float(funding_sentiment)
    
    async def _calculate_oi_sentiment(self, oi_data: pd.DataFrame) -> float:
        """Open Interest sentiment hesapla"""
        if len(oi_data) < 2:
            return 0.0
        
        current_oi = oi_data['sumOpenInterest'].iloc[-1]
        prev_oi = oi_data['sumOpenInterest'].iloc[-2]
        
        # OI değişim oranı
        oi_change = (current_oi - prev_oi) / prev_oi if prev_oi != 0 else 0
        return float(np.tanh(oi_change * 10))
    
    async def _calculate_ls_sentiment(self, ls_data: pd.DataFrame) -> float:
        """Long/Short Ratio sentiment hesapla"""
        if ls_data.empty:
            return 0.0
        
        current_ratio = ls_data['longShortRatio'].iloc[-1]
        
        # 1.0 nötr seviye, üstü long hakimiyeti
        ls_sentiment = np.tanh((current_ratio - 1.0) * 2)
        return float(ls_sentiment)
    
    async def _calculate_oi_change_rate(self, oi_data: pd.DataFrame) -> float:
        """OI değişim hızı (momentum)"""
        if len(oi_data) < 5:
            return 0.0
        
        oi_series = oi_data['sumOpenInterest']
        returns = oi_series.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        # Son 5 periyodun momentumu
        momentum = returns.rolling(5).mean().iloc[-1]
        return float(np.tanh(momentum * 100))
    
    async def _calculate_funding_skew(self, funding_data: pd.DataFrame) -> float:
        """Funding rate skew (dağılım çarpıklığı)"""
        if len(funding_data) < 10:
            return 0.0
        
        funding_rates = funding_data['fundingRate']
        skew = funding_rates.skew()
        return float(np.tanh(skew))
    
    async def _calculate_volume_imbalance(self, taker_data: pd.DataFrame) -> float:
        """Volume imbalance index"""
        if taker_data.empty:
            return 0.0
        
        buy_vol = taker_data['buyVol'].iloc[-1]
        sell_vol = taker_data['sellVol'].iloc[-1]
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            return 0.0
        
        imbalance = (buy_vol - sell_vol) / total_vol
        return float(imbalance)
    
    async def _calculate_liquidation_heat(self, liq_data: pd.DataFrame) -> float:
        """Liquidation heatmap metriği"""
        if liq_data.empty:
            return 0.0
        
        # Son 12 saatlik likidasyon toplamı
        liquidation_window = getattr(self.cfg.parameters, "liquidation_window", 12)
        recent_liq = liq_data.head(liquidation_window)
        total_liq = recent_liq['executedQty'].sum()
        
        # Normalize edilmiş liquidation heat
        heat = np.tanh(total_liq / 1e6)  # 1M USDT için normalize
        return float(heat)
    
    async def _calculate_oi_delta_divergence(self, oi_data: pd.DataFrame, 
                                           funding_data: pd.DataFrame) -> float:
        """OI delta divergence (OI vs Funding divergence)"""
        if len(oi_data) < 5 or len(funding_data) < 5:
            return 0.0
        
        # OI momentum
        oi_momentum = oi_data['sumOpenInterest'].pct_change(3).iloc[-1]
        
        # Funding momentum
        funding_momentum = funding_data['fundingRate'].diff(3).iloc[-1]
        
        # Divergence (zıt yönlü hareket)
        divergence = oi_momentum * funding_momentum * -100  # Negatif correlation beklenir
        
        return float(np.tanh(divergence))
    
    async def _calculate_volatility_skew(self, symbol: str, oi_data: pd.DataFrame) -> float:
        """Volatility skew (OI dağılım volatilitesi)"""
        volatility_period = getattr(self.cfg.parameters, "volatility_period", 20)
        if len(oi_data) < volatility_period:
            return 0.0
        
        oi_returns = oi_data['sumOpenInterest'].pct_change().dropna()
        
        if len(oi_returns) < 10:
            return 0.0
        
        # OI volatilitesi (realized vol)
        volatility = oi_returns.std() * np.sqrt(365 * 24)  # Yıllıklaştırılmış
        
        # Normalize edilmiş skew
        skew = np.tanh(volatility * 10)
        return float(skew)
    
    # DATA FETCH FONKSİYONLARI
    
    async def _get_funding_data(self, symbol: str) -> pd.DataFrame:
        """Funding rate verilerini getir"""
        cache_key = f"funding_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_funding_rate(symbol=symbol, limit=50)
            df = pd.DataFrame(data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df.set_index('fundingTime', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch funding data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_open_interest_data(self, symbol: str) -> pd.DataFrame:
        """Open Interest verilerini getir"""
        cache_key = f"oi_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_open_interest_hist(
                symbol=symbol, 
                period='5m', 
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch OI data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_long_short_data(self, symbol: str) -> pd.DataFrame:
        """Long/Short Ratio verilerini getir"""
        cache_key = f"ls_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_long_short_ratio(
                symbol=symbol,
                period='5m',
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch LS ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_liquidation_data(self, symbol: str) -> pd.DataFrame:
        """Liquidation verilerini getir"""
        cache_key = f"liq_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_liquidation_orders(
                symbol=symbol,
                limit=100
            )
            df = pd.DataFrame(data)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                df.sort_index(ascending=False, inplace=True)  # En yeni başta
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch liquidation data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_taker_ratio_data(self, symbol: str) -> pd.DataFrame:
        """Taker buy/sell volume verilerini getir"""
        cache_key = f"taker_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            data = await self.binance.get_taker_long_short_ratio(
                symbol=symbol,
                period='5m',
                limit=100
            )
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch taker ratio for {symbol}: {e}")
            return pd.DataFrame()

    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """✅ ANALYSIS_HELPERS UYUMLU AGGREGATE"""
        return {
            "symbol": symbol,
            "aggregated_score": self._normalize_score(np.mean(list(metrics.values()))),
            "component_scores": metrics,
            "timestamp": self.helpers.get_timestamp(),
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
            "timestamp": self.helpers.get_timestamp(),
            "report_type": "derivatives_sentiment_report",
            "cache_size": len(self._cache)
        }

    async def cleanup(self):
        """Kaynakları temizle"""
        await self.binance.close()
        self._cache.clear()
        logger.info("DerivativesSentimentModule cleanup completed")

# ✅ UYUMLU FACTORY FUNCTION
def create_module(config: Dict[str, Any] = None) -> DerivativesSentimentModule:
    """Factory function for creating DerivativesSentimentModule instances"""
    return DerivativesSentimentModule(config)


# Demo için main fonksiyonu
if __name__ == "__main__":
    async def demo():
        module = DerivativesSentimentModule()
        await module.initialize()
        result = await module.compute_metrics("BTCUSDT")
        
        print("=== Derivatives Sentiment Module Demo ===")
        print(f"Module: {result.get('module', 'N/A')}")
        print(f"Score: {result.get('score', 0):.4f}")
        print(f"Signal: {result.get('signal', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        print(f"Explanation: {result.get('explain', 'N/A')}")
        print("Components:")
        for k, v in result.get('components', {}).items():
            print(f"  {k}: {v:.4f}")
        
        await module.cleanup()

    asyncio.run(demo())
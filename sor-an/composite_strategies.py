# analysis/composite/composite_strategies.py
"""
Bileşik Skor Stratejileri, Strategy Pattern ile Skorlar
Her bileşik skor için özel strateji sınıfları için orkestrasyon katmanı

analysis/analysis_helpers.py uyumlu
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from abc import ABC, abstractmethod

# Analysis helpers import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from analysis_helpers import AnalysisHelpers, AnalysisOutput

logger = logging.getLogger(__name__)

class BaseCompositeStrategy(ABC):
    """Bileşik skor stratejileri için base sınıf"""
    
    def __init__(self):
        self.config = {}
        self.weights = {}
        self.thresholds = {}
        self.helpers = AnalysisHelpers()
    
    def configure(self, config: Dict):
        """Stratejiyi config ile yapılandır"""
        self.config = config
        self.weights = self.helpers.normalize_weights(config.get('weights', {}))
        self.thresholds = config.get('thresholds', {})
    
    @property
    @abstractmethod
    def required_modules(self) -> List[str]:
        """Bu strateji için gerekli modül listesi"""
        pass
    
    @abstractmethod
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Bileşik skoru hesapla"""
        pass
    
    def _extract_scores(self, module_results: Dict) -> Dict[str, float]:
        """Modül sonuçlarından skorları çıkar"""
        scores = {}
        for module_name in self.required_modules:
            if module_results.get(module_name) and 'score' in module_results[module_name]:
                raw_score = module_results[module_name]['score']
                scores[module_name] = self.helpers.normalize_score(raw_score)
            else:
                scores[module_name] = 0.5  # Neutral fallback
                logger.warning(f"Missing score for {module_name}, using neutral 0.5")
        
        return scores
    
    def _calculate_confidence(self, module_results: Dict) -> float:
        """Skor güvenilirliğini hesapla"""
        valid_scores = 0
        total_weight = 0
        
        for module_name, weight in self.weights.items():
            if (module_results.get(module_name) and 
                module_results[module_name].get('score') is not None):
                valid_scores += weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return self.helpers.normalize_score(valid_scores / total_weight)
    
    def _get_signal(self, score: float, signal_type: str = "trend") -> str:
        """Skora göre sinyal belirle"""
        if signal_type == "trend":
            return self._get_trend_signal(score)
        elif signal_type == "risk":
            return self._get_risk_signal(score)
        elif signal_type == "buy_opportunity":
            return self._get_buy_opportunity_signal(score)
        else:
            return self._get_generic_signal(score)
    
    def _get_trend_signal(self, score: float) -> str:
        """Trend sinyali"""
        if score >= self.thresholds.get('strong_bullish', 0.7):
            return "strong_bullish"
        elif score >= self.thresholds.get('weak_bullish', 0.55):
            return "weak_bullish" 
        elif score <= self.thresholds.get('weak_bearish', 0.3):
            return "weak_bearish"
        elif score <= self.thresholds.get('strong_bearish', 0.1):
            return "strong_bearish"
        else:
            return "neutral"
    
    def _get_risk_signal(self, score: float) -> str:
        """Risk sinyali (ters skalada)"""
        if score >= 0.7:
            return "high_risk"
        elif score >= 0.5:
            return "medium_risk"
        else:
            return "low_risk"
    
    def _get_buy_opportunity_signal(self, score: float) -> str:
        """Alım fırsatı sinyali"""
        if score >= self.thresholds.get('strong_buy', 0.7):
            return "excellent_opportunity"
        elif score >= self.thresholds.get('good_buy', 0.6):
            return "good_opportunity"
        elif score >= self.thresholds.get('fair_buy', 0.55):
            return "fair_opportunity"
        elif score <= self.thresholds.get('poor_buy', 0.4):
            return "poor_opportunity"
        else:
            return "neutral_opportunity"
    
    def _get_generic_signal(self, score: float) -> str:
        """Genel sinyal"""
        if score >= 0.6:
            return "bullish"
        elif score <= 0.4:
            return "bearish"
        else:
            return "neutral"
    
    def _validate_composite_output(self, output: Dict[str, Any]) -> bool:
        """Bileşik skor çıktısını validate et"""
        try:
            # Temel required alanlar
            required = ['score', 'signal', 'confidence', 'components', 'timestamp']
            if not all(key in output for key in required):
                return False
            
            # Score validation
            if not 0 <= output['score'] <= 1:
                logger.warning(f"Invalid score: {output['score']}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False
    
    def _create_composite_output(self, score: float, signal: str, confidence: float, 
                               components: Dict, analysis: Dict = None,
                               interpretation: Dict = None) -> Dict[str, Any]:
        """Standart bileşik skor çıktısı oluştur"""
        base_output = {
            'score': self.helpers.normalize_score(score),
            'signal': signal,
            'confidence': self.helpers.normalize_score(confidence),
            'components': components,
            'timestamp': self.helpers.get_timestamp(),
            'composite_type': self.__class__.__name__
        }
        
        # Opsiyonel analiz alanları
        if analysis:
            base_output['analysis'] = analysis
        if interpretation:
            base_output['interpretation'] = interpretation
            
        return base_output
    
    def _get_neutral_module_result(self, module_name: str) -> Dict[str, Any]:
        """Nötr modül sonucu oluştur"""
        return self.helpers.create_fallback_output(module_name, "neutral_fallback")
    
    def _get_error_fallback(self) -> Dict[str, Any]:
        """Hata durumu için fallback response"""
        return self.helpers.create_fallback_output(
            self.__class__.__name__, 
            "composite_calculation_error"
        )


    def _create_composite_output(self, score: float, signal: str, confidence: float, 
                               components: Dict, analysis: Dict = None,
                               interpretation: Dict = None) -> Dict[str, Any]:
        """Standart bileşik skor çıktısı oluştur - AnalysisOutput uyumlu"""
        base_output = {
            'score': self.helpers.normalize_score(score),
            'signal': signal,
            'confidence': self.helpers.normalize_score(confidence),
            'components': self._simplify_components(components),  # 🔹 Yeni metod
            'explain': f"Composite score for {self.__class__.__name__}",  # 🔹 Gerekli alan
            'timestamp': self.helpers.get_timestamp(),
            'module': f"composite_{self.__class__.__name__.lower()}",  # 🔹 Gerekli alan
            'composite_type': self.__class__.__name__
        }
        
        # Opsiyonel analiz alanları
        if analysis:
            base_output['analysis'] = analysis
        if interpretation:
            base_output['interpretation'] = interpretation
            
        return base_output
    
    def _simplify_components(self, components: Dict) -> Dict[str, float]:
        """Components'ı AnalysisOutput formatına uygun basitleştir"""
        simplified = {}
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict) and 'score' in comp_data:
                simplified[comp_name] = comp_data['score']
            else:
                simplified[comp_name] = float(comp_data)
        return simplified
    
    def _get_error_fallback(self) -> Dict[str, Any]:
        """Hata durumu için fallback response - AnalysisOutput uyumlu"""
        fallback = self.helpers.create_fallback_output(
            f"composite_{self.__class__.__name__.lower()}", 
            "composite_calculation_error"
        )
        # Composite-specific alanları ekle
        fallback.update({
            'composite_type': self.__class__.__name__,
            'components': {}  # Boş components garantile
        })
        return fallback




class TrendStrengthStrategy(BaseCompositeStrategy):
    """
    Trend Strength Score Stratejisi
    A (Trend) %35 + B (Regime) %25 + C (Sentiment) %20 + D (Order Flow) %20
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "regime_anomal", "deriv_sentim", "order_micros"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Trend Strength Score hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)
            
            # Ağırlıklı ortalama hesapla
            component_scores = {}
            for module_name in self.required_modules:
                if module_name in scores:
                    component_scores[module_name] = scores[module_name]
            
            final_score = self.helpers.calculate_weights(component_scores, self.weights)
            
            # Güvenilirlik hesapla
            confidence = self._calculate_confidence(module_results)
            
            # Sinyal belirle
            signal = self._get_signal(final_score, "trend")
            
            # Trend yönü ve gücü analizi
            trend_analysis = self._analyze_trend_components(component_scores)
            
            # Component detayları
            component_details = {}
            for module_name, score in component_scores.items():
                weight = self.weights.get(module_name, 0.25)
                component_details[module_name] = {
                    'score': score,
                    'weight': weight,
                    'contribution': score * weight
                }
            
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=component_details,
                analysis=trend_analysis,
                interpretation=self._interpret_trend_strength(final_score, signal)
            )
            
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
                
            return output
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return self._get_error_fallback()
    
    def _analyze_trend_components(self, components: Dict[str, float]) -> Dict[str, Any]:
        """Trend bileşenlerini detaylı analiz et"""
        bullish_components = 0
        total_components = len(components)
        
        for comp_score in components.values():
            if comp_score > 0.6:
                bullish_components += 1
            elif comp_score < 0.4:
                bullish_components -= 1
        
        trend_alignment = bullish_components / total_components if total_components > 0 else 0
        
        # Dominant component
        dominant_component = max(components.items(), key=lambda x: x[1]) if components else (None, 0)
        
        return {
            'bullish_components': bullish_components,
            'total_components': total_components,
            'trend_alignment': trend_alignment,
            'dominant_component': dominant_component[0],
            'dominant_score': dominant_component[1] if dominant_component[0] else 0
        }
    
    def _interpret_trend_strength(self, score: float, signal: str) -> Dict[str, str]:
        """Trend skorunu yorumla"""
        interpretations = {
            'strong_bullish': {
                'summary': 'Çok güçlü yükseliş trendi',
                'action': 'Trend takip stratejileri uygulanabilir',
                'risk': 'Düşük - trend net'
            },
            'weak_bullish': {
                'summary': 'Zayıf yükseliş eğilimi', 
                'action': 'Doğrulama beklenmeli',
                'risk': 'Orta - trend zayıf'
            },
            'neutral': {
                'summary': 'Belirsiz trend',
                'action': 'Yan bant stratejileri uygun',
                'risk': 'Yüksek - yön belirsiz'
            },
            'weak_bearish': {
                'summary': 'Zayıf düşüş eğilimi',
                'action': 'Korunma pozisyonları',
                'risk': 'Orta - trend zayıf'
            },
            'strong_bearish': {
                'summary': 'Güçlü düşüş trendi',
                'action': 'Kısa pozisyon veya korunma',
                'risk': 'Düşük - trend net'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Belirsiz trend durumu',
            'action': 'Temkinli olun',
            'risk': 'Yüksek'
        })


class RiskExposureStrategy(BaseCompositeStrategy):
    """Risk Exposure Score Stratejisi"""
    
    @property
    def required_modules(self) -> List[str]:
        return ["regime_anomal", "onchain", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Risk Exposure Score hesapla"""
        try:
            scores = self._extract_scores(module_results)
            
            # Risk skoru - yüksek = kötü
            component_scores = {}
            for module_name in self.required_modules:
                if module_name in scores:
                    component_scores[module_name] = scores[module_name]
            
            final_score = self.helpers.calculate_weights(component_scores, self.weights)
            confidence = self._calculate_confidence(module_results)
            signal = self._get_signal(final_score, "risk")
            
            # Component detayları
            component_details = {}
            for module_name, score in component_scores.items():
                weight = self.weights.get(module_name, 0.33)
                component_details[module_name] = {
                    'score': score,
                    'weight': weight,
                    'contribution': score * weight
                }
            
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=component_details,
                analysis=self._analyze_risk_factors(module_results),
                interpretation=self._interpret_risk_exposure(final_score, signal)
            )
            
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
                
            return output
            
        except Exception as e:
            logger.error(f"Risk exposure calculation failed: {e}")
            return self._get_error_fallback()
    
    def _analyze_risk_factors(self, module_results: Dict) -> Dict[str, Any]:
        """Risk faktörlerini detaylı analiz et"""
        regime_data = module_results.get("regime_anomal", {})
        onchain_data = module_results.get("onchain", {})
        risk_data = module_results.get("risk_expos", {})
        
        risk_factors = []
        
        # Volatilite riski
        if regime_data.get('signal') in ['high_volatility', 'regime_change']:
            risk_factors.append("high_volatility")
        
        # On-chain riskler
        if onchain_data.get('metrics', {}).get('network_health', 0) < 0.4:
            risk_factors.append("network_weakness")
        
        # Risk exposure detayları
        if risk_data.get('signal') == 'high_risk':
            risk_factors.append("elevated_exposure")
        
        # Risk konsantrasyonu
        risk_scores = [
            regime_data.get('score', 0.5),
            onchain_data.get('score', 0.5),
            risk_data.get('score', 0.5)
        ]
        risk_concentration = np.std(risk_scores)
        
        return {
            'risk_factors': risk_factors,
            'risk_concentration': risk_concentration,
            'volatility_risk': regime_data.get('score', 0.5),
            'network_risk': onchain_data.get('score', 0.5),
            'exposure_risk': risk_data.get('score', 0.5),
            'overall_risk_level': len(risk_factors) / 3.0
        }
    
    def _interpret_risk_exposure(self, score: float, signal: str) -> Dict[str, str]:
        """Risk skorunu yorumla"""
        interpretations = {
            'high_risk': {
                'summary': 'Yüksek risk seviyesi - Dikkatli olunmalı',
                'action': 'Pozisyon boyutlarını küçültün, stop-loss kullanın',
                'warning': 'Büyük kayıp riski yüksek'
            },
            'medium_risk': {
                'summary': 'Orta seviye risk - Standart önlemler yeterli',
                'action': 'Normal risk yönetimi uygulayın',
                'warning': 'Piyasa koşullarını yakından takip edin'
            },
            'low_risk': {
                'summary': 'Düşük risk seviyesi - Nispeten güvenli',
                'action': 'Standart işlem stratejileri uygulanabilir',
                'warning': 'Ani değişimlere karşı hazırlıklı olun'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Risk seviyesi belirsiz',
            'action': 'Temkinli davranın',
            'warning': 'Ek doğrulama gerekli'
        })



# 🔴 BuyOpportunityStrategy
class BuyOpportunityStrategy(BaseCompositeStrategy):
    """Buy Opportunity Score Stratejisi (Bileşik skor yapısına tam uyumlu)"""
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "deriv_sentim", "order_micros", "corr_lead"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Buy Opportunity Score hesapla"""
        try:
            # 1️⃣ Gerekli skorları çek
            scores = self._extract_scores(module_results)

            trend_score = scores.get("trend_moment", 0.5)
            sentiment_score = scores.get("deriv_sentim", 0.5)
            order_flow_score = scores.get("order_micros", 0.5)
            correlation_score = scores.get("corr_lead", 0.5)
            
            # 2️⃣ Ağırlıklı final skor hesapla
            final_score = (
                trend_score * self.weights.get("trend_moment", 0.30) +
                sentiment_score * self.weights.get("deriv_sentim", 0.25) +
                order_flow_score * self.weights.get("order_micros", 0.25) +
                correlation_score * self.weights.get("corr_lead", 0.20)
            )
            
            # 3️⃣ Güvenilirlik ve sinyal hesapla
            confidence = self._calculate_confidence(module_results)
            signal = self._get_signal(final_score, "buy_opportunity")
            
            # 4️⃣ Analiz ve yorum üret
            analysis = self._analyze_opportunity_quality(
                trend_score, sentiment_score, order_flow_score, correlation_score
            )
            interpretation = self._generate_entry_recommendation(final_score, signal, analysis)
            
            # 5️⃣ Component detayları
            components = {
                name: {
                    'score': scores.get(name, 0.5),
                    'weight': self.weights.get(name, 0.25),
                    'contribution': scores.get(name, 0.5) * self.weights.get(name, 0.25)
                }
                for name in self.required_modules
            }
            
            # 6️⃣ Standart composite output oluştur
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=components,
                analysis=analysis,
                interpretation=interpretation
            )
            
            # 7️⃣ Validasyon
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
            
            return output
        
        except Exception as e:
            logger.error(f"BuyOpportunityStrategy calculation failed: {e}")
            return self._get_error_fallback()

    # 🔹 Analiz Fonksiyonları
    def _analyze_opportunity_quality(
        self, trend_score: float, sentiment_score: float,
        order_flow_score: float, correlation_score: float
    ) -> Dict[str, Any]:
        """Alım fırsatı kalitesini analiz et"""
        factors = {
            'trend_alignment': trend_score > 0.6,
            'sentiment_support': sentiment_score > 0.5,
            'order_flow_positive': order_flow_score > 0.5,
            'correlation_support': correlation_score > 0.5
        }
        
        positive_factors = sum(factors.values())
        opportunity_strength = positive_factors / len(factors)
        
        # Zamanlama analizi
        timing_score = (trend_score + order_flow_score) / 2.0
        if timing_score > 0.7:
            timing = "excellent_timing"
        elif timing_score > 0.6:
            timing = "good_timing"
        elif timing_score > 0.5:
            timing = "fair_timing"
        else:
            timing = "poor_timing"
        
        # Risk/Reward hesapla
        reward_potential = (trend_score + sentiment_score) / 2.0
        risk_level = 1.0 - ((order_flow_score + correlation_score) / 2.0)
        risk_reward_ratio = reward_potential / (risk_level + 0.1)
        
        return {
            'positive_factors': positive_factors,
            'total_factors': len(factors),
            'opportunity_strength': opportunity_strength,
            'timing_quality': timing,
            'reward_potential': reward_potential,
            'risk_level': risk_level,
            'risk_reward_ratio': risk_reward_ratio,
            'consistency_score': 1.0 - np.std(
                [trend_score, sentiment_score, order_flow_score, correlation_score]
            )
        }

    def _generate_entry_recommendation(
        self, score: float, signal: str, opportunity_quality: Dict
    ) -> Dict[str, Any]:
        """Giriş önerisi oluştur"""
        recommendations = {
            'excellent_opportunity': {
                'summary': 'Mükemmel alım fırsatı',
                'action': 'AGGRESSIVE_BUY',
                'allocation': '70-80%',
                'entry_strategy': 'Immediate entry, scale in on dips',
                'stop_loss': '3-5% below entry',
                'targets': '10-15% profit, trail stop after 8%'
            },
            'good_opportunity': {
                'summary': 'İyi alım fırsatı',
                'action': 'MODERATE_BUY',
                'allocation': '50-60%',
                'entry_strategy': 'Scale entry over 2-3 periods',
                'stop_loss': '5-7% below entry',
                'targets': '8-12% profit'
            },
            'fair_opportunity': {
                'summary': 'Orta düzey fırsat',
                'action': 'CAUTIOUS_BUY',
                'allocation': '30-40%',
                'entry_strategy': 'Wait for pullback, limit orders',
                'stop_loss': '7-10% below entry',
                'targets': '6-10% profit'
            },
            'poor_opportunity': {
                'summary': 'Zayıf fırsat',
                'action': 'AVOID_BUY',
                'allocation': '0-20%',
                'entry_strategy': 'Wait for better setup',
                'stop_loss': 'N/A',
                'targets': 'N/A'
            },
            'neutral_opportunity': {
                'summary': 'Belirsiz fırsat',
                'action': 'MONITOR',
                'allocation': '0%',
                'entry_strategy': 'Wait for confirmation',
                'stop_loss': 'N/A',
                'targets': 'N/A'
            }
        }
        
        base_rec = recommendations.get(signal, recommendations['neutral_opportunity'])
        
        rr_ratio = opportunity_quality.get('risk_reward_ratio', 1.0)
        if rr_ratio > 2.0:
            rr_quality = "excellent_rr"
        elif rr_ratio > 1.5:
            rr_quality = "good_rr"
        elif rr_ratio > 1.0:
            rr_quality = "fair_rr"
        else:
            rr_quality = "poor_rr"
        
        return {
            **base_rec,
            'risk_adjusted_quality': rr_quality,
            'risk_reward_ratio': round(rr_ratio, 3),
            'confidence_level': round(score, 3),
            'timing_quality': opportunity_quality.get('timing_quality', 'fair_timing')
        }



# 🔵 5. Liquidity Pressure Index - LiquidityPressureStrategy
class LiquidityPressureStrategy(BaseCompositeStrategy):
    """
    Liquidity Pressure Index Stratejisi
    D (Order Flow) %60 + H (Micro Alpha) %40
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["order_micros", "microalpha"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Liquidity Pressure Index hesapla"""
        try:
            # 1️⃣ Modül skorlarını al
            scores = self._extract_scores(module_results)
            order_flow_score = scores.get("order_micros", 0.5)
            micro_alpha_score = scores.get("microalpha", 0.5)
            
            # 2️⃣ Ağırlıklı ortalama hesapla
            final_score = (order_flow_score * 0.6) + (micro_alpha_score * 0.4)
            
            # 3️⃣ Likidite baskı yönü
            pressure_direction = self._calculate_pressure_direction(
                order_flow_score, micro_alpha_score, module_results
            )
            
            # 4️⃣ Güvenilirlik ve sinyal
            confidence = self._calculate_confidence(module_results)
            signal = self._get_liquidity_signal(final_score, pressure_direction)
            
            # 5️⃣ Analiz ve yorum
            analysis = self._analyze_liquidity_components(
                module_results, order_flow_score, micro_alpha_score
            )
            interpretation = self._interpret_liquidity_pressure(
                final_score, signal, pressure_direction
            )
            
            # 6️⃣ Component detayları
            components = {
                'order_micros': {
                    'score': order_flow_score,
                    'weight': 0.6,
                    'contribution': order_flow_score * 0.6
                },
                'microalpha': {
                    'score': micro_alpha_score,
                    'weight': 0.4,
                    'contribution': micro_alpha_score * 0.4
                }
            }
            
            # 7️⃣ Standart composite output oluştur
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=components,
                analysis={
                    **analysis,
                    'pressure_direction': pressure_direction
                },
                interpretation=interpretation
            )
            
            # 8️⃣ Validasyon
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
            
            return output
        
        except Exception as e:
            logger.error(f"Liquidity pressure calculation failed: {e}")
            return self._get_error_fallback()
    
    # 🔹 Yardımcı Metodlar
    def _calculate_pressure_direction(self, order_flow_score: float, 
                                      micro_alpha_score: float, 
                                      module_results: Dict) -> str:
        """Likidite baskı yönünü belirle"""
        order_flow_data = module_results.get("order_micros", {})
        micro_alpha_data = module_results.get("microalpha", {})
        
        buy_pressure, sell_pressure = 0, 0
        
        # Order Flow sinyalinden yön
        if order_flow_data.get('signal') == 'buy_pressure':
            buy_pressure += 1
        elif order_flow_data.get('signal') == 'sell_pressure':
            sell_pressure += 1
        
        # Micro Alpha sinyalinden yön
        if micro_alpha_data.get('signal') == 'bullish':
            buy_pressure += 1
        elif micro_alpha_data.get('signal') == 'bearish':
            sell_pressure += 1
        
        # Skor farklarından yön
        if order_flow_score > 0.6:
            buy_pressure += 1
        elif order_flow_score < 0.4:
            sell_pressure += 1
        if micro_alpha_score > 0.6:
            buy_pressure += 1
        elif micro_alpha_score < 0.4:
            sell_pressure += 1
        
        if buy_pressure > sell_pressure:
            return "buying_pressure"
        elif sell_pressure > buy_pressure:
            return "selling_pressure"
        return "balanced"
    
    def _get_liquidity_signal(self, score: float, direction: str) -> str:
        """Likidite sinyali belirle"""
        if score >= 0.7:
            if direction == "buying_pressure":
                return "strong_buying_pressure"
            elif direction == "selling_pressure":
                return "strong_selling_pressure"
            return "high_liquidity_volatility"
        elif score >= 0.6:
            if direction == "buying_pressure":
                return "moderate_buying_pressure"
            elif direction == "selling_pressure":
                return "moderate_selling_pressure"
            return "elevated_liquidity"
        elif score <= 0.3:
            return "low_liquidity"
        return "normal_liquidity"
    
    def _analyze_liquidity_components(self, module_results: Dict, 
                                      order_flow_score: float, 
                                      micro_alpha_score: float) -> Dict[str, Any]:
        """Likidite bileşenlerini analiz et"""
        order_flow = module_results.get("order_micros", {})
        micro_alpha = module_results.get("microalpha", {})
        
        ofi = order_flow.get('components', {}).get('orderbook_imbalance', 0)
        cvd = order_flow.get('components', {}).get('cvd', 0)
        alpha_metric = micro_alpha.get('score', 0.5)
        market_impact = micro_alpha.get('metrics', {}).get('market_impact', 0)
        
        return {
            'order_flow_imbalance': ofi,
            'cumulative_volume_delta': cvd,
            'micro_alpha_strength': alpha_metric,
            'market_impact_factor': market_impact,
            'pressure_strength': abs(order_flow_score - 0.5) * 2,  # 0–1 normalize
            'consistency': 1 - abs(order_flow_score - micro_alpha_score)
        }
    
    def _interpret_liquidity_pressure(self, score: float, signal: str, direction: str) -> Dict[str, str]:
        """Likidite baskısını yorumla"""
        interpretations = {
            'strong_buying_pressure': {
                'summary': 'Güçlü alım likiditesi - Fiyat yükselme potansiyeli yüksek',
                'action': 'Long pozisyonlar için uygun ortam',
                'warning': 'Aşırı alım bölgesinde olabilir'
            },
            'moderate_buying_pressure': {
                'summary': 'Orta seviye alım likiditesi',
                'action': 'Kademeli long girişleri değerlendirilebilir',
                'warning': 'Trend devamı için diğer göstergeleri kontrol edin'
            },
            'strong_selling_pressure': {
                'summary': 'Güçlü satım likiditesi - Fiyat düşme potansiyeli yüksek',
                'action': 'Short pozisyonlar veya korunma stratejileri',
                'warning': 'Aşırı satım bölgesinde olabilir'
            },
            'moderate_selling_pressure': {
                'summary': 'Orta seviye satım likiditesi',
                'action': 'Kademeli short girişleri değerlendirilebilir',
                'warning': 'Stop-loss kullanımı önemli'
            },
            'high_liquidity_volatility': {
                'summary': 'Yüksek likidite oynaklığı - Belirsizlik hakim',
                'action': 'Pozisyon boyutlarını küçük tutun',
                'warning': 'Yanlış sinyal riski yüksek'
            },
            'low_liquidity': {
                'summary': 'Düşük likidite - Sıçrama riski yüksek',
                'action': 'İşlem yapmaktan kaçının veya küçük pozisyonlarla çalışın',
                'warning': 'Slippage riski çok yüksek'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Normal likidite koşulları',
            'action': 'Standart işlem stratejileri uygulanabilir',
            'warning': 'Diğer göstergelerle teyit edin'
        })



# ⚪ 7. AnomalyDetectionStrategy
class AnomalyDetectionStrategy(BaseCompositeStrategy):
    """
    Anomaly Detection Alert Score Stratejisi
    J (Regime Change) %40 + C (Sentiment) %30 + G (Risk) %30
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["regime_anomal", "deriv_sentim", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Anomaly Detection Alert Score hesapla"""
        try:
            # 1️⃣ Skorları al
            scores = self._extract_scores(module_results)
            regime_score = scores.get("regime_anomal", 0.5)
            sentiment_score = scores.get("deriv_sentim", 0.5)
            risk_score = scores.get("risk_expos", 0.5)
            
            # 2️⃣ Anomali skorlarını normalize et (0-1)
            regime_anomaly = abs(regime_score - 0.5) * 2
            sentiment_anomaly = abs(sentiment_score - 0.5) * 2
            risk_anomaly = risk_score  # Zaten yüksek = anomali
            
            # 3️⃣ Ağırlıklı toplam
            final_score = (
                regime_anomaly * 0.4 +
                sentiment_anomaly * 0.3 +
                risk_anomaly * 0.3
            )
            
            # 4️⃣ Alt analizler
            anomaly_type = self._detect_anomaly_type(
                regime_score, sentiment_score, risk_score, module_results
            )
            severity = self._calculate_anomaly_severity(final_score, anomaly_type)
            urgency = self._calculate_urgency(final_score, anomaly_type, severity)
            confidence = self._calculate_confidence(module_results)
            signal = self._get_anomaly_signal(final_score)
            
            # 5️⃣ Alt analiz detayları
            analysis = self._analyze_anomaly_pattern(
                regime_score, sentiment_score, risk_score, module_results
            )
            recommendations = self._generate_anomaly_recommendations(
                final_score, anomaly_type, severity, urgency
            )
            
            # 6️⃣ Component detayları
            components = {
                'regime_anomal': {
                    'score': regime_score,
                    'anomaly_score': regime_anomaly,
                    'weight': 0.4,
                    'contribution': regime_anomaly * 0.4
                },
                'deriv_sentim': {
                    'score': sentiment_score,
                    'anomaly_score': sentiment_anomaly,
                    'weight': 0.3,
                    'contribution': sentiment_anomaly * 0.3
                },
                'risk_expos': {
                    'score': risk_score,
                    'anomaly_score': risk_anomaly,
                    'weight': 0.3,
                    'contribution': risk_anomaly * 0.3
                }
            }
            
            # 7️⃣ Composite output oluştur
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=components,
                analysis={
                    **analysis,
                    'anomaly_type': anomaly_type,
                    'severity': severity,
                    'urgency': urgency
                },
                interpretation=recommendations
            )
            
            # 8️⃣ Validasyon
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
            
            return output
        
        except Exception as e:
            logger.error(f"Anomaly detection calculation failed: {e}")
            return self._get_error_fallback()
    
    # 🔹 Yardımcı metodlar (değişmeden korundu)
    def _detect_anomaly_type(self, regime_score: float, sentiment_score: float, 
                             risk_score: float, module_results: Dict) -> str:
        """Anomali tipini tespit et"""
        regime_data = module_results.get("regime_anomal", {})
        sentiment_data = module_results.get("deriv_sentim", {})
        risk_data = module_results.get("risk_expos", {})
        
        if regime_score > 0.7:
            return "regime_change"
        if abs(sentiment_score - 0.5) > 0.3:
            return "extreme_bullish_sentiment" if sentiment_score > 0.5 else "extreme_bearish_sentiment"
        if risk_score > 0.7:
            return "high_risk_environment"
        
        regime_signal = regime_data.get('signal', '')
        if 'anomaly' in regime_signal:
            return "volatility_anomaly"
        
        anomaly_count = 0
        if abs(regime_score - 0.5) > 0.2: anomaly_count += 1
        if abs(sentiment_score - 0.5) > 0.2: anomaly_count += 1
        if risk_score > 0.6: anomaly_count += 1
        
        if anomaly_count >= 2:
            return "combined_anomaly"
        return "no_clear_anomaly"
    
    def _calculate_anomaly_severity(self, score: float, anomaly_type: str) -> str:
        """Anomali şiddetini hesapla"""
        if score >= 0.8: return "critical"
        elif score >= 0.7: return "high"
        elif score >= 0.6: return "medium"
        elif score >= 0.5: return "low"
        return "none"
    
    def _calculate_urgency(self, score: float, anomaly_type: str, severity: str) -> str:
        """Aciliyet seviyesi belirle"""
        if severity == "critical": return "immediate"
        elif severity == "high": return "high"
        elif severity == "medium": return "medium"
        return "low"
    
    def _get_anomaly_signal(self, score: float) -> str:
        """Anomali sinyali belirle"""
        if score >= 0.8: return "critical_anomaly"
        elif score >= 0.7: return "high_anomaly"
        elif score >= 0.6: return "medium_anomaly"
        elif score >= 0.5: return "low_anomaly"
        return "normal"
    
    def _analyze_anomaly_pattern(self, regime_score: float, sentiment_score: float,
                                 risk_score: float, module_results: Dict) -> Dict[str, Any]:
        """Anomali pattern analizi"""
        scores = [regime_score, sentiment_score, risk_score]
        avg_score = sum(scores) / len(scores)
        std_dev = np.std(scores)
        
        anomaly_consensus = sum(1 for s in scores if abs(s - 0.5) > 0.2)
        trend_direction = "bullish" if avg_score > 0.5 else "bearish"
        
        return {
            'average_score': avg_score,
            'volatility': std_dev,
            'anomaly_consensus': anomaly_consensus,
            'trend_direction': trend_direction,
            'deviation_from_normal': abs(avg_score - 0.5) * 2,
            'pattern_strength': min(1.0, std_dev * 2)
        }
    
    def _generate_anomaly_recommendations(self, score: float, anomaly_type: str,
                                          severity: str, urgency: str) -> Dict[str, str]:
        """Anomaliye göre öneriler oluştur"""
        base = {
            'critical': {
                'action': 'TÜM POZİSYONLARI KAPAT - ACİL DURUM',
                'monitoring': '15 dakika aralıklarla takip',
                're_entry': 'Anomali çözülene kadar yeni pozisyon açmayın'
            },
            'high': {
                'action': 'Pozisyon boyutlarını %50 azalt',
                'monitoring': '30 dakika aralıklarla takip',
                're_entry': 'Sinyal netleşene kadar bekleyin'
            },
            'medium': {
                'action': 'Stop-loss seviyelerini sıkılaştır',
                'monitoring': '1 saat aralıklarla takip',
                're_entry': 'Ek doğrulama sinyali bekleyin'
            },
            'low': {
                'action': 'Mevcut stratejiyi sürdür, dikkatli ol',
                'monitoring': 'Normal takip periyodu',
                're_entry': 'Standart kurallara göre işlem yapın'
            }
        }
        return base.get(severity, {
            'action': 'Normal işlem stratejisi',
            'monitoring': 'Rutin takip',
            're_entry': 'Standart kurallar'
        })



# 🔶 8. Market Health Score
class MarketHealthStrategy(BaseCompositeStrategy):
    """
    Market Health Score Stratejisi
    Trend Strength %30 + Risk Exposure %25 + Liquidity Pressure %25 + Macro Sentiment %20
    """

    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "risk_expos", "order_micros", "deriv_sentim"]

    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Market Health Score hesapla"""
        try:
            # Modül skorlarını al ve normalize et
            scores = self._extract_scores(module_results)

            trend_score = scores.get("trend_moment", 0.5)
            risk_score = 1.0 - scores.get("risk_expos", 0.5)  # Risk yüksekse sağlık düşer
            liquidity_score = scores.get("order_micros", 0.5)
            macro_score = scores.get("deriv_sentim", 0.5)

            # Ağırlıklı ortalama hesapla
            final_score = (
                trend_score * 0.30 +
                risk_score * 0.25 +
                liquidity_score * 0.25 +
                macro_score * 0.20
            )

            # Durum belirle
            health_status = self._assess_health_status(final_score)
            signal = self._get_health_signal(final_score)
            confidence = self._calculate_confidence(module_results)
            health_trend = self._calculate_health_trend(module_results)

            # Bileşen katkıları
            components = {
                'trend_strength': {
                    'score': trend_score,
                    'weight': 0.30,
                    'contribution': trend_score * 0.30
                },
                'risk_health': {
                    'score': risk_score,
                    'weight': 0.25,
                    'contribution': risk_score * 0.25
                },
                'liquidity_health': {
                    'score': liquidity_score,
                    'weight': 0.25,
                    'contribution': liquidity_score * 0.25
                },
                'macro_sentiment': {
                    'score': macro_score,
                    'weight': 0.20,
                    'contribution': macro_score * 0.20
                }
            }

            # Derin analizler
            health_analysis = self._analyze_market_health(
                trend_score, risk_score, liquidity_score, macro_score
            )
            market_outlook = self._generate_market_outlook(final_score, health_status)

            # 🔸 Standart çıktı formatını oluştur
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=components,
                analysis={
                    'health_status': health_status,
                    'health_trend': health_trend,
                    **health_analysis
                },
                interpretation=market_outlook
            )

            # 🔸 Çıktı doğrulaması (BaseCompositeStrategy standardı)
            if not self._validate_composite_output(output):
                return self._get_error_fallback()

            return output

        except Exception as e:
            logger.error(f"Market health calculation failed: {e}")
            return self._get_error_fallback()

    # ----------------------------- #
    #       Yardımcı Metodlar
    # ----------------------------- #

    def _assess_health_status(self, score: float) -> str:
        """Piyasa sağlık durumunu değerlendir"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "very_healthy"
        elif score >= 0.6:
            return "healthy"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.4:
            return "weak"
        elif score >= 0.3:
            return "unhealthy"
        else:
            return "critical"

    def _calculate_health_trend(self, module_results: Dict) -> str:
        """Sağlık trendini hesapla"""
        trend_data = module_results.get("trend_moment", {})
        signal = trend_data.get('signal', 'neutral')

        if signal in ['strong_bullish', 'bullish']:
            return "improving"
        elif signal in ['strong_bearish', 'bearish']:
            return "deteriorating"
        else:
            return "stable"

    def _get_health_signal(self, score: float) -> str:
        """Sağlık sinyali üret"""
        if score >= 0.75:
            return "optimal_health"
        elif score >= 0.65:
            return "good_health"
        elif score >= 0.55:
            return "moderate_health"
        elif score >= 0.45:
            return "caution_zone"
        else:
            return "poor_health"

    def _analyze_market_health(self, trend_score: float, risk_score: float,
                               liquidity_score: float, macro_score: float) -> Dict[str, Any]:
        """Bileşen bazlı sağlık analizi"""
        scores = [trend_score, risk_score, liquidity_score, macro_score]
        consistency = 1 - (np.std(scores) / 0.5)  # Normalize tutarlılık (0–1)

        strengths, weaknesses = [], []

        if trend_score > 0.6:
            strengths.append("trend_strength")
        else:
            weaknesses.append("trend_weakness")

        if risk_score > 0.6:
            strengths.append("risk_controlled")
        else:
            weaknesses.append("high_risk_exposure")

        if liquidity_score > 0.6:
            strengths.append("good_liquidity")
        else:
            weaknesses.append("liquidity_stress")

        if macro_score > 0.6:
            strengths.append("positive_sentiment")
        else:
            weaknesses.append("negative_sentiment")

        return {
            'consistency': max(0, min(1, consistency)),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'balance': len(strengths) == len(weaknesses),
            'dominant_health_factor': max(
                ('trend_strength', trend_score),
                ('risk_health', risk_score),
                ('liquidity_health', liquidity_score),
                ('macro_sentiment', macro_score),
                key=lambda x: x[1]
            )[0]
        }

    def _generate_market_outlook(self, score: float, status: str) -> Dict[str, str]:
        """Piyasa görünümü oluştur"""
        outlooks = {
            'excellent': {
                'summary': 'Piyasa koşulları mükemmel – güçlü trend ve düşük risk',
                'outlook': 'Çok olumlu',
                'action': 'Trend yönünde işlem fırsatlarını değerlendirin'
            },
            'very_healthy': {
                'summary': 'Piyasa sağlıklı – momentum yüksek',
                'outlook': 'Olumlu',
                'action': 'Mevcut pozisyonları koruyun'
            },
            'healthy': {
                'summary': 'Dengeli piyasa – fırsatlar var',
                'outlook': 'Nötr-pozitif',
                'action': 'Kısa vadeli long fırsatlarını izleyin'
            },
            'moderate': {
                'summary': 'Karışık piyasa – temkinli olunmalı',
                'outlook': 'Nötr',
                'action': 'Risk yönetimi ön planda olmalı'
            },
            'weak': {
                'summary': 'Piyasa zayıf – volatilite artabilir',
                'outlook': 'Negatif eğilimli',
                'action': 'Kısa vadeli işlemlerde dikkatli olun'
            },
            'unhealthy': {
                'summary': 'Sağlıksız piyasa – yüksek risk ortamı',
                'outlook': 'Olumsuz',
                'action': 'Pozisyonlar azaltılmalı veya hedge edilmeli'
            },
            'critical': {
                'summary': 'Kritik piyasa koşulları – çöküş riski',
                'outlook': 'Acil durum',
                'action': 'Tüm pozisyonlar kapatılmalı'
            }
        }

        return outlooks.get(status, {
            'summary': 'Belirsiz piyasa koşulları',
            'outlook': 'Nötr',
            'action': 'İzleme modunda kalın'
        })


 
# 🔶 9. Swing Trading Signal
class SwingTradingStrategy(BaseCompositeStrategy):
    """
    Swing Trading Signal Stratejisi
    Trend Strength %40 + Buy Opportunity %30 + Risk Exposure %30
    """

    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "deriv_sentim", "risk_expos"]

    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Swing Trading Signal hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)

            # Bileşen skorları
            trend_score = scores.get("trend_moment", 0.5)
            buy_opportunity_score = scores.get("deriv_sentim", 0.5)
            risk_score = 1.0 - scores.get("risk_expos", 0.5)  # Risk tersi = fırsat

            # Ağırlıklı ortalama hesapla
            final_score = (
                trend_score * 0.40 +
                buy_opportunity_score * 0.30 +
                risk_score * 0.30
            )

            # Swing sinyali üret
            swing_signal = self._generate_swing_signal(final_score, trend_score, buy_opportunity_score)

            # Zaman çerçevesi uygunluğu
            timeframe_suitability = self._assess_timeframe_suitability(module_results, trend_score)

            # Pozisyon önerisi
            position_suggestion = self._suggest_position(final_score, swing_signal, risk_score)

            # Güvenilirlik hesapla
            confidence = self._calculate_confidence(module_results)

            # Swing analizi
            swing_analysis = self._analyze_swing_setup(
                trend_score, buy_opportunity_score, risk_score, module_results
            )

            # İşlem planı
            trading_plan = self._generate_trading_plan(final_score, swing_signal)

            # Bileşen katkıları
            components = {
                'trend_strength': {
                    'score': trend_score,
                    'weight': 0.40,
                    'contribution': trend_score * 0.40
                },
                'buy_opportunity': {
                    'score': buy_opportunity_score,
                    'weight': 0.30,
                    'contribution': buy_opportunity_score * 0.30
                },
                'risk_adjusted': {
                    'score': risk_score,
                    'weight': 0.30,
                    'contribution': risk_score * 0.30
                }
            }

            # 🔸 Standart BaseCompositeStrategy formatında çıktı oluştur
            output = self._create_composite_output(
                score=final_score,
                signal=swing_signal,
                confidence=confidence,
                components=components,
                analysis={
                    'timeframe_suitability': timeframe_suitability,
                    'position_suggestion': position_suggestion,
                    **swing_analysis
                },
                interpretation=trading_plan
            )

            # 🔸 Çıktı doğrulaması (standart)
            if not self._validate_composite_output(output):
                return self._get_error_fallback()

            return output

        except Exception as e:
            logger.error(f"Swing trading calculation failed: {e}")
            return self._get_error_fallback()

    # --------------------------------------------------
    # Yardımcı Metotlar
    # --------------------------------------------------

    def _generate_swing_signal(self, overall_score: float, trend_score: float,
                               buy_opportunity_score: float) -> str:
        """Swing trading sinyali üret"""
        if overall_score >= 0.7 and trend_score >= 0.6 and buy_opportunity_score >= 0.6:
            return "strong_buy"
        elif overall_score >= 0.6 and trend_score >= 0.5 and buy_opportunity_score >= 0.5:
            return "buy"
        elif overall_score <= 0.3 and trend_score <= 0.4 and buy_opportunity_score <= 0.4:
            return "strong_sell"
        elif overall_score <= 0.4 and trend_score <= 0.5 and buy_opportunity_score <= 0.5:
            return "sell"
        elif overall_score >= 0.55 and trend_score >= 0.5:
            return "hold_long"
        elif overall_score <= 0.45 and trend_score <= 0.5:
            return "hold_short"
        else:
            return "no_trade"

    def _assess_timeframe_suitability(self, module_results: Dict, trend_strength: float) -> Dict[str, Any]:
        """Zaman çerçevesi uygunluğunu değerlendir"""
        if trend_strength > 0.7:
            timeframe = "1-3_gün"
        elif trend_strength > 0.6:
            timeframe = "3-5_gün"
        elif trend_strength > 0.5:
            timeframe = "5-7_gün"
        else:
            timeframe = "1_gün_altı"

        # Volatilite durumu
        volatility_signal = module_results.get("risk_expos", {}).get('signal', 'neutral')
        if 'high' in volatility_signal:
            volatility_suitability = "high_risk"
        elif 'low' in volatility_signal:
            volatility_suitability = "low_opportunity"
        else:
            volatility_suitability = "optimal"

        return {
            'recommended_timeframe': timeframe,
            'volatility_suitability': volatility_suitability,
            'swing_potential': trend_strength,
            'risk_period': "short_term" if timeframe == "1_gün_altı" else "medium_term"
        }

    def _suggest_position(self, score: float, signal: str, risk_score: float) -> Dict[str, Any]:
        """Pozisyon büyüklüğü ve risk ayarlı tahsis"""
        position_map = {
            'strong_buy': {'size': 'large', 'allocation': 0.7},
            'buy': {'size': 'medium', 'allocation': 0.5},
            'hold_long': {'size': 'small', 'allocation': 0.3},
            'no_trade': {'size': 'none', 'allocation': 0.0},
            'hold_short': {'size': 'small', 'allocation': 0.3},
            'sell': {'size': 'medium', 'allocation': 0.5},
            'strong_sell': {'size': 'large', 'allocation': 0.7}
        }

        base = position_map.get(signal, {'size': 'none', 'allocation': 0.0})
        adjusted_allocation = base['allocation'] * risk_score

        return {
            'action': signal,
            'position_size': base['size'],
            'suggested_allocation': round(adjusted_allocation, 3),
            'risk_adjusted': True,
            'confidence_level': round(score, 3)
        }

    def _analyze_swing_setup(self, trend_score: float, opportunity_score: float,
                             risk_score: float, module_results: Dict) -> Dict[str, Any]:
        """Swing setup analizi"""
        quality_score = (trend_score + opportunity_score + risk_score) / 3
        trend_signal = module_results.get("trend_moment", {}).get('signal', 'neutral')

        if trend_signal in ['strong_bullish', 'bullish']:
            entry_timing = "early_trend"
        elif trend_signal == 'neutral':
            entry_timing = "consolidation"
        else:
            entry_timing = "counter_trend"

        if risk_score > 0.7 and trend_score > 0.6:
            risk_reward = "favorable"
        elif risk_score > 0.5 and trend_score > 0.5:
            risk_reward = "moderate"
        else:
            risk_reward = "unfavorable"

        return {
            'setup_quality': round(quality_score, 4),
            'entry_timing': entry_timing,
            'risk_reward_profile': risk_reward,
            'trend_alignment': "aligned" if trend_score > 0.5 else "counter",
            'opportunity_strength': opportunity_score,
            'overall_rating': round(min(1.0, quality_score * 1.2), 4)
        }

    def _generate_trading_plan(self, score: float, signal: str) -> Dict[str, str]:
        """Swing trading planı oluştur"""
        plans = {
            'strong_buy': {
                'entry': 'Aggressive entry (%70-80 allocation)',
                'stop_loss': '3-5% below entry',
                'take_profit': '8-12% target',
                'management': 'Scale out at 5% and 10%'
            },
            'buy': {
                'entry': 'Standard entry (%50 allocation)',
                'stop_loss': '5-7% below entry',
                'take_profit': '10-15% target',
                'management': 'Partial profit at 8%'
            },
            'sell': {
                'entry': 'Short entry (%50 allocation)',
                'stop_loss': '5-7% above entry',
                'take_profit': '8-12% target',
                'management': 'Tight stops, quick profits'
            },
            'no_trade': {
                'entry': 'No position',
                'stop_loss': 'N/A',
                'take_profit': 'N/A',
                'management': 'Wait for better setup'
            }
        }

        return plans.get(signal, {
            'entry': 'Wait for confirmation',
            'stop_loss': 'Standard 5% stop',
            'take_profit': '8-10% target',
            'management': 'Monitor closely'
        })



# 🔶
# ------------------------------------------------------------------
# Add the following classes at the end of your existing composite_strategies.py
# They implement:
# - BuyScoreStrategy: alınabilirlik puanı
# - CapitalFlowStrategy: coin bazlı net para girişi/çıkışı tespiti
# - MarketCashFlowStrategy: stable->risk akışını tahmin edici bileşik skor
#
# Assumes BaseCompositeStrategy, AnalysisHelpers, logger etc. already defined earlier in this file.

class BuyScoreStrategy(BaseCompositeStrategy):
    """
    Alınabilirlik (BuyScore) Stratejisi
    Trend %35 + (1-Risk) %25 + Likidite %15 + Piyasa Sağlığı %15 + Alım Fırsatı %10
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "risk_expos", "liquidity_pressure", "market_health", "buy_opportunity"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Alınabilirlik skoru hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)
            
            # Bileşen skorları
            trend_score = scores.get("trend_moment", 0.5)
            risk_score = scores.get("risk_expos", 0.5)
            liquidity_score = scores.get("liquidity_pressure", 0.5)
            health_score = scores.get("market_health", 0.5)
            opportunity_score = scores.get("buy_opportunity", 0.5)
            
            # Ağırlıklar
            weights = {
                "trend_moment": 0.35,
                "risk_expos": 0.25,  # Risk ters etki: (1 - risk_score)
                "liquidity_pressure": 0.15,
                "market_health": 0.15,
                "buy_opportunity": 0.10
            }
            
            # Ağırlıklı ortalama hesapla (risk ters etki)
            final_score = (
                trend_score * weights["trend_moment"] +
                (1 - risk_score) * weights["risk_expos"] +
                liquidity_score * weights["liquidity_pressure"] +
                health_score * weights["market_health"] +
                opportunity_score * weights["buy_opportunity"]
            )
            
            # Güvenilirlik ve sinyal
            confidence = self._calculate_confidence(module_results)
            signal = self._get_buy_score_signal(final_score)
            
            # Component detayları
            component_details = {}
            for module_name, score in scores.items():
                if module_name in weights:
                    weight = weights[module_name]
                    # Risk için özel katkı hesaplama
                    if module_name == "risk_expos":
                        contribution = (1 - score) * weight
                        adjusted_score = 1 - score
                    else:
                        contribution = score * weight
                        adjusted_score = score
                    
                    component_details[module_name] = {
                        'score': adjusted_score,
                        'weight': weight,
                        'contribution': contribution
                    }
            
            # Alım analizi
            buy_analysis = self._analyze_buy_opportunity(
                trend_score, risk_score, liquidity_score, health_score, opportunity_score
            )
            
            # Yorum ve öneriler
            interpretation = self._generate_buy_recommendation(final_score, signal, buy_analysis)
            
            output = self._create_composite_output(
                score=final_score,
                signal=signal,
                confidence=confidence,
                components=component_details,
                analysis=buy_analysis,
                interpretation=interpretation
            )
            
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
                
            return output
            
        except Exception as e:
            logger.error(f"BuyScore calculation failed for {symbol}: {e}")
            return self._get_error_fallback()
    
    def _get_buy_score_signal(self, score: float) -> str:
        """Alınabilirlik sinyali belirle"""
        if score >= self.thresholds.get('excellent_buy', 0.75):
            return "excellent_buy"
        elif score >= self.thresholds.get('good_buy', 0.65):
            return "good_buy"
        elif score >= self.thresholds.get('fair_buy', 0.55):
            return "fair_buy"
        elif score <= self.thresholds.get('poor_buy', 0.45):
            return "poor_buy"
        else:
            return "neutral_buy"
    
    def _analyze_buy_opportunity(self, trend: float, risk: float, liquidity: float, 
                               health: float, opportunity: float) -> Dict[str, Any]:
        """Alım fırsatını detaylı analiz et"""
        # Risk-adjusted trend
        risk_adjusted_trend = trend * (1 - risk)
        
        # Likidite kalitesi
        liquidity_quality = "high" if liquidity > 0.7 else "medium" if liquidity > 0.5 else "low"
        
        # Genel uyum
        alignment_score = (trend + (1 - risk) + liquidity + health + opportunity) / 5.0
        
        # Zayıf noktalar
        weaknesses = []
        if trend < 0.5:
            weaknesses.append("weak_trend")
        if risk > 0.6:
            weaknesses.append("high_risk")
        if liquidity < 0.4:
            weaknesses.append("poor_liquidity")
        if health < 0.5:
            weaknesses.append("unhealthy_market")
        
        return {
            'risk_adjusted_trend': risk_adjusted_trend,
            'liquidity_quality': liquidity_quality,
            'overall_alignment': alignment_score,
            'weaknesses': weaknesses,
            'strength_factors': 5 - len(weaknesses),
            'buy_confidence': min(trend, (1 - risk), liquidity, health)  # En zayıf halka
        }
    
    def _generate_buy_recommendation(self, score: float, signal: str, 
                                   analysis: Dict[str, Any]) -> Dict[str, str]:
        """Alım önerisi oluştur"""
        recommendations = {
            'excellent_buy': {
                'summary': 'Mükemmel alım fırsatı - tüm faktörler olumlu',
                'action': 'AGGRESSIVE_BUY',
                'allocation': '70-80%',
                'timing': 'Immediate entry',
                'risk_note': 'Düşük risk, yüksek potansiyel'
            },
            'good_buy': {
                'summary': 'İyi alım fırsatı - güçlü sinyaller',
                'action': 'MODERATE_BUY',
                'allocation': '50-60%',
                'timing': 'Scale in over 1-2 days',
                'risk_note': 'Orta risk, iyi potansiyel'
            },
            'fair_buy': {
                'summary': 'Orta düzey fırsat - dikkatli yaklaşım gerekli',
                'action': 'CAUTIOUS_BUY',
                'allocation': '30-40%',
                'timing': 'Wait for pullbacks',
                'risk_note': 'Dikkatli pozisyon yönetimi'
            },
            'poor_buy': {
                'summary': 'Zayıf alım fırsatı - riskler yüksek',
                'action': 'AVOID_BUY',
                'allocation': '0-20%',
                'timing': 'Wait for better setup',
                'risk_note': 'Yüksek risk, düşük potansiyel'
            },
            'neutral_buy': {
                'summary': 'Belirsiz alım fırsatı',
                'action': 'MONITOR',
                'allocation': '0%',
                'timing': 'Wait for confirmation',
                'risk_note': 'Ek sinyal bekleyin'
            }
        }
        
        base_rec = recommendations.get(signal, recommendations['neutral_buy'])
        
        # Zayıf noktalara göre özelleştirme
        weaknesses = analysis.get('weaknesses', [])
        if weaknesses:
            base_rec['warning'] = f"Dikkat: {', '.join(weaknesses)}"
        
        return base_rec


class CapitalFlowStrategy(BaseCompositeStrategy):
    """
    Coin Bazlı Net Para Akışı Stratejisi
    On-chain %35 + Likidite %30 + Piyasa Sağlığı %20 + Türev Sentiment %10 - Risk %15
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["onchain", "liquidity_pressure", "market_health", "deriv_sentim", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Net para akışı skoru hesapla"""
        try:
            scores = self._extract_scores(module_results)
            
            # Bileşen skorları
            onchain_score = scores.get("onchain", 0.5)
            liquidity_score = scores.get("liquidity_pressure", 0.5)
            health_score = scores.get("market_health", 0.5)
            sentiment_score = scores.get("deriv_sentim", 0.5)
            risk_score = scores.get("risk_expos", 0.5)
            
            # Ağırlıklar (risk negatif etki)
            weights = {
                "onchain": 0.35,
                "liquidity_pressure": 0.30,
                "market_health": 0.20,
                "deriv_sentim": 0.10,
                "risk_expos": -0.15  # Negatif ağırlık
            }
            
            # Ağırlıklı toplam
            final_score = (
                onchain_score * weights["onchain"] +
                liquidity_score * weights["liquidity_pressure"] +
                health_score * weights["market_health"] +
                sentiment_score * weights["deriv_sentim"] +
                risk_score * weights["risk_expos"]  # Risk negatif katkı
            )
            
            # 0-1 aralığına normalize et
            normalized_score = max(0.0, min(1.0, (final_score + 0.15) / 1.0))
            
            confidence = self._calculate_confidence(module_results)
            signal = self._get_flow_signal(normalized_score)
            
            # Component detayları
            component_details = {}
            for module_name, score in scores.items():
                if module_name in weights:
                    weight = weights[module_name]
                    contribution = score * weight
                    
                    component_details[module_name] = {
                        'score': score,
                        'weight': abs(weight),  # Mutlak değer
                        'contribution': contribution,
                        'direction': 'positive' if weight > 0 else 'negative'
                    }
            
            # Akış analizi
            flow_analysis = self._analyze_capital_flow(
                onchain_score, liquidity_score, health_score, sentiment_score, risk_score
            )
            
            # Yorum
            interpretation = self._interpret_capital_flow(normalized_score, signal, flow_analysis)
            
            output = self._create_composite_output(
                score=normalized_score,
                signal=signal,
                confidence=confidence,
                components=component_details,
                analysis=flow_analysis,
                interpretation=interpretation
            )
            
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
                
            return output
            
        except Exception as e:
            logger.error(f"CapitalFlow calculation failed for {symbol}: {e}")
            return self._get_error_fallback()
    
    def _get_flow_signal(self, score: float) -> str:
        """Para akışı sinyali belirle"""
        if score >= 0.7:
            return "strong_inflow"
        elif score >= 0.6:
            return "moderate_inflow"
        elif score >= 0.55:
            return "weak_inflow"
        elif score <= 0.45:
            return "weak_outflow"
        elif score <= 0.4:
            return "moderate_outflow"
        elif score <= 0.3:
            return "strong_outflow"
        else:
            return "balanced_flow"
    
    def _analyze_capital_flow(self, onchain: float, liquidity: float, 
                            health: float, sentiment: float, risk: float) -> Dict[str, Any]:
        """Sermaye akışını detaylı analiz et"""
        # Net akış gücü
        net_flow_strength = (onchain + liquidity + health + sentiment - risk) / 4.0
        
        # Akış yönü göstergeleri
        inflow_indicators = 0
        total_indicators = 4
        
        if onchain > 0.6:
            inflow_indicators += 1
        if liquidity > 0.6:
            inflow_indicators += 1
        if health > 0.6:
            inflow_indicators += 1
        if sentiment > 0.6:
            inflow_indicators += 1
        
        flow_consensus = inflow_indicators / total_indicators
        
        # Risk etkisi
        risk_impact = "high" if risk > 0.7 else "medium" if risk > 0.5 else "low"
        
        return {
            'net_flow_strength': net_flow_strength,
            'inflow_indicators': inflow_indicators,
            'total_indicators': total_indicators,
            'flow_consensus': flow_consensus,
            'risk_impact': risk_impact,
            'momentum_score': (onchain + liquidity) / 2.0
        }
    
    def _interpret_capital_flow(self, score: float, signal: str, 
                              analysis: Dict[str, Any]) -> Dict[str, str]:
        """Sermaye akışını yorumla"""
        interpretations = {
            'strong_inflow': {
                'summary': 'Güçlü para girişi - alıcılar hakim',
                'implication': 'Fiyat artış potansiyeli yüksek',
                'action': 'Long pozisyonlar destekleniyor',
                'timeframe': 'Kısa-Orta vadeli'
            },
            'moderate_inflow': {
                'summary': 'Orta seviye para girişi',
                'implication': 'Olumlu fiyat momentumu',
                'action': 'Kademeli long pozisyonlar',
                'timeframe': 'Kısa vadeli'
            },
            'weak_inflow': {
                'summary': 'Zayıf para girişi',
                'implication': 'Sınırlı yükseliş potansiyeli',
                'action': 'Küçük pozisyonlarla test edin',
                'timeframe': 'Çok kısa vadeli'
            },
            'balanced_flow': {
                'summary': 'Dengeli para akışı',
                'implication': 'Yön belirsiz',
                'action': 'Yan bant stratejileri',
                'timeframe': 'Belirsiz'
            },
            'weak_outflow': {
                'summary': 'Zayıf para çıkışı',
                'implication': 'Hafif düşüş baskısı',
                'action': 'Pozisyonları küçültün',
                'timeframe': 'Kısa vadeli'
            },
            'moderate_outflow': {
                'summary': 'Orta seviye para çıkışı',
                'implication': 'Düşüş riski artıyor',
                'action': 'Korunma stratejileri',
                'timeframe': 'Orta vadeli'
            },
            'strong_outflow': {
                'summary': 'Güçlü para çıkışı - satıcılar hakim',
                'implication': 'Belirgin düşüş potansiyeli',
                'action': 'Short pozisyonlar veya nakit',
                'timeframe': 'Orta-Uzun vadeli'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Para akışı belirsiz',
            'implication': 'Ek veri gerekli',
            'action': 'Temkinli olun',
            'timeframe': 'Belirsiz'
        })


class MarketCashFlowStrategy(BaseCompositeStrategy):
    """
    Piyasa Seviyesinde Nakit Akışı Stratejisi
    Piyasa Sağlığı %30 + Likidite %25 + On-chain %25 + Türev Sentiment %10 - Risk %15
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["market_health", "liquidity_pressure", "onchain", "deriv_sentim", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Piyasa nakit akışı skoru hesapla"""
        try:
            scores = self._extract_scores(module_results)
            
            # Bileşen skorları
            health_score = scores.get("market_health", 0.5)
            liquidity_score = scores.get("liquidity_pressure", 0.5)
            onchain_score = scores.get("onchain", 0.5)
            sentiment_score = scores.get("deriv_sentim", 0.5)
            risk_score = scores.get("risk_expos", 0.5)
            
            # Ağırlıklar
            weights = {
                "market_health": 0.30,
                "liquidity_pressure": 0.25,
                "onchain": 0.25,
                "deriv_sentim": 0.10,
                "risk_expos": -0.15  # Negatif ağırlık
            }
            
            # Ağırlıklı toplam
            final_score = (
                health_score * weights["market_health"] +
                liquidity_score * weights["liquidity_pressure"] +
                onchain_score * weights["onchain"] +
                sentiment_score * weights["deriv_sentim"] +
                risk_score * weights["risk_expos"]
            )
            
            # 0-1 aralığına normalize et
            normalized_score = max(0.0, min(1.0, (final_score + 0.15) / 1.0))
            
            confidence = self._calculate_confidence(module_results)
            signal = self._get_market_flow_signal(normalized_score)
            
            # Component detayları
            component_details = {}
            for module_name, score in scores.items():
                if module_name in weights:
                    weight = weights[module_name]
                    contribution = score * weight
                    
                    component_details[module_name] = {
                        'score': score,
                        'weight': abs(weight),
                        'contribution': contribution,
                        'direction': 'positive' if weight > 0 else 'negative'
                    }
            
            # Piyasa akış analizi
            market_analysis = self._analyze_market_cash_flow(
                health_score, liquidity_score, onchain_score, sentiment_score, risk_score
            )
            
            # Risk-on/risk-off yorumu
            interpretation = self._interpret_market_regime(normalized_score, signal, market_analysis)
            
            output = self._create_composite_output(
                score=normalized_score,
                signal=signal,
                confidence=confidence,
                components=component_details,
                analysis=market_analysis,
                interpretation=interpretation
            )
            
            if not self._validate_composite_output(output):
                return self._get_error_fallback()
                
            return output
            
        except Exception as e:
            logger.error(f"MarketCashFlow calculation failed for {symbol}: {e}")
            return self._get_error_fallback()
    
    def _get_market_flow_signal(self, score: float) -> str:
        """Piyasa akış sinyali belirle"""
        if score >= 0.75:
            return "strong_risk_on"
        elif score >= 0.65:
            return "moderate_risk_on"
        elif score >= 0.55:
            return "weak_risk_on"
        elif score <= 0.45:
            return "weak_risk_off"
        elif score <= 0.35:
            return "moderate_risk_off"
        elif score <= 0.25:
            return "strong_risk_off"
        else:
            return "neutral_regime"
    
    def _analyze_market_cash_flow(self, health: float, liquidity: float, 
                                onchain: float, sentiment: float, risk: float) -> Dict[str, Any]:
        """Piyasa nakit akışını detaylı analiz et"""
        # Risk-on göstergeleri
        risk_on_indicators = 0
        total_indicators = 4
        
        if health > 0.6:
            risk_on_indicators += 1
        if liquidity > 0.6:
            risk_on_indicators += 1
        if onchain > 0.6:
            risk_on_indicators += 1
        if sentiment > 0.6:
            risk_on_indicators += 1
        
        risk_on_strength = risk_on_indicators / total_indicators
        
        # Regime kalitesi
        regime_quality = (health + liquidity + onchain + (1 - risk)) / 4.0
        
        # Stable -> Risk akışı tahmini
        stable_to_risk_flow = max(0.0, (health * 0.3 + liquidity * 0.3 + onchain * 0.2 + sentiment * 0.2 - risk * 0.3))
        
        return {
            'risk_on_indicators': risk_on_indicators,
            'risk_on_strength': risk_on_strength,
            'regime_quality': regime_quality,
            'stable_to_risk_flow': stable_to_risk_flow,
            'market_stability': 1.0 - risk,  # Risk tersi = stabilite
            'momentum_alignment': np.std([health, liquidity, onchain, sentiment]) < 0.2  # Düşük std = yüksek uyum
        }
    
    def _interpret_market_regime(self, score: float, signal: str, 
                               analysis: Dict[str, Any]) -> Dict[str, str]:
        """Piyasa rejimini yorumla"""
        interpretations = {
            'strong_risk_on': {
                'summary': 'Güçlü Risk-On ortam - Stabldan Riske akış',
                'implication': 'Risk varlıklarına yönelim artıyor',
                'action': 'BTC/ETH/Altcoin long pozisyonları',
                'warning': 'Aşırı iyimserlik riskine dikkat'
            },
            'moderate_risk_on': {
                'summary': 'Orta Risk-On ortam - Olumlu akış',
                'implication': 'Risk varlıkları destekleniyor',
                'action': 'Kademeli risk pozisyonları',
                'warning': 'Risk yönetimi önemli'
            },
            'weak_risk_on': {
                'summary': 'Zayıf Risk-On eğilimi',
                'implication': 'Sınırlı risk iştahı',
                'action': 'Küçük risk pozisyonları',
                'warning': 'Trend kırılgan olabilir'
            },
            'neutral_regime': {
                'summary': 'Nötr piyasa - Belirsiz akış',
                'implication': 'Yön belirsiz, karışık sinyaller',
                'action': 'Yan bant veya temkinli stratejiler',
                'warning': 'Volatilite artabilir'
            },
            'weak_risk_off': {
                'summary': 'Zayıf Risk-Off eğilimi',
                'implication': 'Riskten kaçış başlıyor',
                'action': 'Pozisyonları küçültün',
                'warning': 'Korunma gerekebilir'
            },
            'moderate_risk_off': {
                'summary': 'Orta Risk-Off ortam - Riske karşı temkinlilik',
                'implication': 'Stablecoin/NAKDİ tercih artıyor',
                'action': 'Risk azaltma stratejileri',
                'warning': 'Düşüş riski artıyor'
            },
            'strong_risk_off': {
                'summary': 'Güçlü Risk-Off ortam - Panik satışları',
                'implication': 'Güçlü riskten kaçış',
                'action': 'Nakit korunma veya short',
                'warning': 'Yüksek volatilite beklenmeli'
            }
        }
        
        base_interpretation = interpretations.get(signal, {
            'summary': 'Piyasa rejimi belirsiz',
            'implication': 'Ek veri analizi gerekli',
            'action': 'Temkinli davranın',
            'warning': 'Yüksek belirsizlik'
        })
        
        # Akış gücüne göre özelleştirme
        flow_strength = analysis.get('stable_to_risk_flow', 0.5)
        if flow_strength > 0.7:
            base_interpretation['flow_intensity'] = 'very_strong'
        elif flow_strength > 0.5:
            base_interpretation['flow_intensity'] = 'strong'
        else:
            base_interpretation['flow_intensity'] = 'moderate'
        
        return base_interpretation


#1*
def circuit_breaker_fallback(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Strategy failed: {e}")
            instance = args[0]
            return instance._get_error_fallback()
    return wrapper


#2*  Strategy factory function
def create_composite_strategy(strategy_name: str) -> BaseCompositeStrategy:
    """Strateji factory fonksiyonu - GÜNCELLENMİŞ"""
    strategies = {
        'trend_strength': TrendStrengthStrategy,
        'risk_exposure': RiskExposureStrategy,
        'buy_opportunity': BuyOpportunityStrategy,
        'liquidity_pressure': LiquidityPressureStrategy,
        'anomaly_alert': AnomalyDetectionStrategy,
        'market_health': MarketHealthStrategy,
        'swing_trading': SwingTradingStrategy,
        'buy_score': BuyScoreStrategy,           # Yeni
        'capital_flow': CapitalFlowStrategy,     # Yeni
        'market_cash_flow': MarketCashFlowStrategy,  # Yeni
    }
    
    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class()

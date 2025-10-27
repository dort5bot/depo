# analysis/composite/composite_engine.py
# Composite Engine (Merkezi) - Analysis Helpers Uyumlu

"""
Bileşik Skor Motoru - Ana Composite Score Engine
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import numpy as np  # ✅ EKLENDİ

from analysis.analysis_base_module import BaseAnalysisModule
from analysis.analysis_helpers import AnalysisHelpers  # ✅ EKLENDİ

logger = logging.getLogger(__name__)

class CompositeScoreEngine:
    """
    Ana bileşik skor hesaplama motoru
    Tüm bileşik skor stratejilerini yönetir
    """
    
    def __init__(self, aggregator, config_path: Optional[str] = None):
        self.aggregator = aggregator
        self.strategies = {}
        self.config = self._load_config(config_path)
        self._initialize_strategies()

        # ✅ ANALYSIS_HELPERS UYUMLU PERFORMANCE METRİKLERİ
        self.performance_metrics = {
            'calculation_times': [],
            'error_rates': {},
            'cache_efficiency': {'hits': 0, 'misses': 0}
        }
        self.alert_thresholds = {
            'max_calculation_time': 5.0,  # 5 saniye
            'max_error_rate': 0.1,  # %10
            'min_confidence': 0.7
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Config dosyasını yükle - ANALYSIS_HELPERS UYUMLU"""
        if config_path is None:
            config_path = Path(__file__).parent / "composite_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Composite config yüklenemedi: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Varsayılan config"""
        return {
            'composite_scores': {
                'trend_strength': {
                    'description': 'Trend yönü ve momentum gücü',
                    'modules': ['trend_moment', 'regime_anomal', 'deriv_sentim', 'order_micros'],
                    'weights': {'trend_moment': 0.35, 'regime_anomal': 0.25, 
                               'deriv_sentim': 0.20, 'order_micros': 0.20},
                    'thresholds': {
                        'strong_bullish': 0.7,
                        'weak_bullish': 0.55, 
                        'neutral': 0.45,
                        'weak_bearish': 0.3,
                        'strong_bearish': 0.0
                    }
                },
                'risk_exposure': {
                    'description': 'Risk maruziyeti skoru',
                    'modules': ['regime_anomal', 'onchain', 'risk_expos'],
                    'weights': {'regime_anomal': 0.4, 'onchain': 0.3, 'risk_expos': 0.3},
                    'thresholds': {
                        'high_risk': 0.7,
                        'medium_risk': 0.5,
                        'low_risk': 0.3
                    }
                },
                'buy_opportunity': {
                    'description': 'Alım fırsatı skoru',
                    'modules': ['trend_moment', 'deriv_sentim', 'order_micros', 'corr_lead'],
                    'weights': {'trend_moment': 0.30, 'deriv_sentim': 0.25, 
                               'order_micros': 0.25, 'corr_lead': 0.20},
                    'thresholds': {
                        'strong_buy': 0.7,
                        'good_buy': 0.6,
                        'fair_buy': 0.55,
                        'poor_buy': 0.4
                    }
                },
                'liquidity_pressure': {
                    'description': 'Likidite baskı indeksi',
                    'modules': ['order_micros', 'microalpha'],
                    'weights': {'order_micros': 0.6, 'microalpha': 0.4},
                    'thresholds': {
                        'strong_buying_pressure': 0.7,
                        'moderate_buying_pressure': 0.6,
                        'low_liquidity': 0.3
                    }
                },
                'anomaly_detection': {
                    'description': 'Anomali tespit skoru',
                    'modules': ['regime_anomal', 'deriv_sentim', 'risk_expos'],
                    'weights': {'regime_anomal': 0.4, 'deriv_sentim': 0.3, 'risk_expos': 0.3},
                    'thresholds': {
                        'critical_anomaly': 0.8,
                        'high_anomaly': 0.7,
                        'medium_anomaly': 0.6,
                        'low_anomaly': 0.5
                    }
                },
                'market_health': {
                    'description': 'Piyasa sağlık skoru',
                    'modules': ['trend_moment', 'risk_expos', 'order_micros', 'deriv_sentim'],
                    'weights': {'trend_moment': 0.30, 'risk_expos': 0.25, 
                               'order_micros': 0.25, 'deriv_sentim': 0.20},
                    'thresholds': {
                        'optimal_health': 0.75,
                        'good_health': 0.65,
                        'moderate_health': 0.55,
                        'caution_zone': 0.45,
                        'poor_health': 0.35
                    }
                },
                'swing_trading': {
                    'description': 'Swing trading sinyali',
                    'modules': ['trend_moment', 'deriv_sentim', 'risk_expos'],
                    'weights': {'trend_moment': 0.40, 'deriv_sentim': 0.30, 'risk_expos': 0.30},
                    'thresholds': {
                        'strong_buy': 0.7,
                        'buy': 0.6,
                        'strong_sell': 0.3,
                        'sell': 0.4
                    }
                },
                'buy_score': {
                    'description': 'Alınabilirlik skoru',
                    'modules': ['trend_moment', 'risk_expos', 'liquidity_pressure', 'market_health', 'buy_opportunity'],
                    'weights': {'trend_moment': 0.35, 'risk_expos': 0.25, 
                               'liquidity_pressure': 0.15, 'market_health': 0.15, 'buy_opportunity': 0.10},
                    'thresholds': {
                        'excellent_buy': 0.75,
                        'good_buy': 0.65,
                        'fair_buy': 0.55,
                        'poor_buy': 0.45
                    }
                },
                'capital_flow': {
                    'description': 'Coin bazlı net para akışı',
                    'modules': ['onchain', 'liquidity_pressure', 'market_health', 'deriv_sentim', 'risk_expos'],
                    'weights': {'onchain': 0.35, 'liquidity_pressure': 0.30, 
                               'market_health': 0.20, 'deriv_sentim': 0.10, 'risk_expos': -0.15},
                    'thresholds': {
                        'strong_inflow': 0.7,
                        'moderate_inflow': 0.6,
                        'weak_inflow': 0.55,
                        'weak_outflow': 0.45,
                        'moderate_outflow': 0.4,
                        'strong_outflow': 0.3
                    }
                },
                'market_cash_flow': {
                    'description': 'Piyasa seviyesinde nakit akışı',
                    'modules': ['market_health', 'liquidity_pressure', 'onchain', 'deriv_sentim', 'risk_expos'],
                    'weights': {'market_health': 0.30, 'liquidity_pressure': 0.25, 
                               'onchain': 0.25, 'deriv_sentim': 0.10, 'risk_expos': -0.15},
                    'thresholds': {
                        'strong_risk_on': 0.75,
                        'moderate_risk_on': 0.65,
                        'weak_risk_on': 0.55,
                        'weak_risk_off': 0.45,
                        'moderate_risk_off': 0.35,
                        'strong_risk_off': 0.25
                    }
                }
            }
        }

    def _initialize_strategies(self):
        """Stratejileri başlat - TÜM STRATEJİLER EKLENDİ"""
        from .composite_strategies import (
            TrendStrengthStrategy,
            RiskExposureStrategy,
            BuyOpportunityStrategy,
            LiquidityPressureStrategy,
            AnomalyDetectionStrategy,  
            MarketHealthStrategy,
            SwingTradingStrategy,
            BuyScoreStrategy,           # ✅ YENİ EKLENDİ
            CapitalFlowStrategy,        # ✅ YENİ EKLENDİ
            MarketCashFlowStrategy      # ✅ YENİ EKLENDİ
        )
        
        strategy_map = {
            'trend_strength': TrendStrengthStrategy(),
            'risk_exposure': RiskExposureStrategy(),
            'buy_opportunity': BuyOpportunityStrategy(),
            'liquidity_pressure': LiquidityPressureStrategy(),
            'anomaly_detection': AnomalyDetectionStrategy(),
            'market_health': MarketHealthStrategy(),
            'swing_trading': SwingTradingStrategy(),
            'buy_score': BuyScoreStrategy(),           # ✅ YENİ EKLENDİ
            'capital_flow': CapitalFlowStrategy(),     # ✅ YENİ EKLENDİ
            'market_cash_flow': MarketCashFlowStrategy()  # ✅ YENİ EKLENDİ
        }
        
        # Config'te tanımlı stratejileri yükle
        for score_name, config in self.config.get('composite_scores', {}).items():
            if score_name in strategy_map:
                strategy_map[score_name].configure(config)
                self.strategies[score_name] = strategy_map[score_name]
                logger.info(f"Composite strategy loaded: {score_name}")

    async def calculate_composite_scores(self, symbol: str) -> Dict[str, Any]:
        """
        Tüm bileşik skorları hesapla
        """
        # ✅ ANALYSIS_HELPERS UYUMLU ZAMAN YÖNETİMİ
        start_time = AnalysisHelpers.get_timestamp()
        
        try:
            # Tüm modül sonuçlarını paralel al
            module_results = await self._gather_module_results_optimized(symbol)
            
            # Bileşik skorları hesapla
            composite_scores = {}
            for score_name, strategy in self.strategies.items():
                try:
                    composite_scores[score_name] = await strategy.calculate(
                        module_results, symbol
                    )
                    
                    # ✅ ANALYSIS_HELPERS UYUMLU PERFORMANCE TRACKING
                    calculation_time = AnalysisHelpers.get_timestamp() - start_time
                    self._update_performance_metrics(symbol, calculation_time, success=True)
                    self._check_alert_conditions(symbol, calculation_time, composite_scores)
                            
                except Exception as e:
                    logger.error(f"Error calculating {score_name}: {e}")
                    composite_scores[score_name] = self._get_error_score(score_name)
                    self._update_performance_metrics(symbol, 0, success=False)
                    self._send_alert(f"CRITICAL: Composite calculation failed for {symbol}: {e}")
            
            # ✅ ANALYSIS_HELPERS UYUMLU OUTPUT FORMATI
            return {
                'symbol': symbol,
                'composite_scores': composite_scores,
                'timestamp': AnalysisHelpers.get_timestamp(),  # ✅ Tutarlı zaman
                'metadata': {
                    'strategies_used': list(self.strategies.keys()),
                    'modules_analyzed': list(module_results.keys()),
                    'calculation_time': AnalysisHelpers.get_timestamp() - start_time
                }
            }
            
        except Exception as e:
            logger.error(f"Composite score calculation failed for {symbol}: {e}")
            return self._get_fallback_response(symbol)

    async def _gather_module_results_optimized(self, symbol: str) -> Dict[str, Any]:
        """Paralel module result toplama - optimize edilmiş"""
        required_modules = set()
        for strategy in self.strategies.values():
            required_modules.update(strategy.required_modules)
        
        tasks = {}
        for module_name in required_modules:
            tasks[module_name] = self.aggregator.get_module_analysis(module_name, symbol)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        module_results = {}
        for i, (module_name, task) in enumerate(tasks.items()):
            if not isinstance(results[i], Exception):
                module_results[module_name] = results[i]
            else:
                logger.warning(f"Module {module_name} failed: {results[i]}")
                module_results[module_name] = self._get_neutral_module_result(module_name)
        
        return module_results

    def _update_performance_metrics(self, symbol: str, calculation_time: float, success: bool):
        """Performance metriklerini güncelle - ANALYSIS_HELPERS UYUMLU"""
        # ✅ MERKEZİ HELPER KULLANIMI
        AnalysisHelpers.update_performance_metrics(
            self.performance_metrics, 
            'calculation_times', 
            calculation_time
        )
        
        # Error rate tracking
        if symbol not in self.performance_metrics['error_rates']:
            self.performance_metrics['error_rates'][symbol] = {'success': 0, 'failure': 0}
        
        if success:
            self.performance_metrics['error_rates'][symbol]['success'] += 1
        else:
            self.performance_metrics['error_rates'][symbol]['failure'] += 1

    def _check_alert_conditions(self, symbol: str, calculation_time: float, composite_scores: Dict):
        """Alert koşullarını kontrol et"""
        # Calculation time alert
        if calculation_time > self.alert_thresholds['max_calculation_time']:
            self._send_alert(f"PERFORMANCE: Slow calculation for {symbol}: {AnalysisHelpers.format_duration(calculation_time)}")
        
        # Confidence alert
        for score_name, score_data in composite_scores.items():
            if score_data.get('confidence', 1.0) < self.alert_thresholds['min_confidence']:
                self._send_alert(f"CONFIDENCE: Low confidence for {symbol}.{score_name}: {score_data['confidence']:.2f}")

    def _send_alert(self, message: str):
        """Alert gönder"""
        logger.warning(f"ALERT: {message}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Performance istatistiklerini getir - ANALYSIS_HELPERS UYUMLU"""
        calculation_times = self.performance_metrics['calculation_times']
        if not calculation_times:
            return {}
        
        total_requests = sum(
            rates['success'] + rates['failure'] 
            for rates in self.performance_metrics['error_rates'].values()
        )
        
        total_success = sum(rates['success'] for rates in self.performance_metrics['error_rates'].values())
        error_rate = (total_requests - total_success) / total_requests if total_requests > 0 else 0
        
        return {
            'avg_calculation_time': np.mean(calculation_times) if calculation_times else 0,
            'max_calculation_time': max(calculation_times) if calculation_times else 0,
            'min_calculation_time': min(calculation_times) if calculation_times else 0,
            'total_requests': total_requests,
            'success_rate': 1 - error_rate,
            'error_rate': error_rate,
            'cache_efficiency': self.performance_metrics['cache_efficiency'],
            'timestamp': AnalysisHelpers.get_timestamp()  # ✅ Tutarlı zaman
        }

    def _get_neutral_module_result(self, module_name: str) -> Dict[str, Any]:
        """✅ EKSİK METOT - ANALYSIS_HELPERS UYUMLU"""
        return AnalysisHelpers.create_fallback_output(
            module_name, 
            "Module analysis failed - using neutral result"
        )

    def _get_error_score(self, score_name: str) -> Dict:
        """Hata durumu için default skor - ANALYSIS_HELPERS UYUMLU"""
        return AnalysisHelpers.create_fallback_output(
            f"composite_{score_name}",
            f"Composite score calculation failed for {score_name}"
        )

    def _get_fallback_response(self, symbol: str) -> Dict:
        """Fallback response - ANALYSIS_HELPERS UYUMLU"""
        return {
            'symbol': symbol,
            'composite_scores': {},
            'timestamp': AnalysisHelpers.get_timestamp(),  # ✅ Tutarlı zaman
            'error': 'Composite calculation failed',
            'metadata': {'fallback': True}
        }

    async def calculate_single_score(self, score_name: str, symbol: str) -> Optional[Dict]:
        """Tekil bileşik skor hesapla"""
        if score_name not in self.strategies:
            logger.error(f"Unknown composite score: {score_name}")
            return None
        
        module_results = await self._gather_module_results_optimized(symbol)
        return await self.strategies[score_name].calculate(module_results, symbol)
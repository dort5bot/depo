# analysis/composite/composite_optimizer.py

"""
Bileşik Skor Optimizasyon Motoru
Ağırlıkların backtest ile optimizasyonu
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

# AnalysisHelpers import
from analysis.analysis_helpers import AnalysisOutput, AnalysisHelpers

logger = logging.getLogger(__name__)

class CompositeOptimizer:
    """Bileşik skor ağırlık optimizasyonu"""
    
    def __init__(self, composite_engine):
        self.engine = composite_engine
        self.helpers = AnalysisHelpers()
    
    async def optimize_weights(self, historical_data: pd.DataFrame, 
                             target_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Ağırlıkları historical data ile optimize et
        
        Args:
            historical_data: Tarihsel fiyat ve sinyal verisi
            target_metric: Optimize edilecek metrik (sharpe_ratio, win_rate, etc.)
        """
        try:
            # Input validation
            if historical_data.empty:
                return self._create_fallback_result("Empty historical data")
            
            # Grid search veya genetic algorithm ile optimizasyon
            best_weights = None
            best_score = -np.inf
            
            # Weight kombinasyonlarını test et
            weight_combinations = self._generate_weight_combinations()
            
            if not weight_combinations:
                return self._create_fallback_result("No weight combinations generated")
            
            for weights in weight_combinations:
                score = await self._evaluate_weight_set(weights, historical_data, target_metric)
                if score > best_score:
                    best_score = score
                    best_weights = weights
            
            # Normalize best weights
            if best_weights:
                best_weights = self.helpers.normalize_weights(best_weights)
            
            return {
                'optimized_weights': best_weights,
                'best_score': best_helpers.normalize_score(best_score),
                'target_metric': target_metric,
                'timestamp': self.helpers.get_timestamp(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return self._create_fallback_result(f"Optimization failed: {str(e)}")
    
    def _generate_weight_combinations(self) -> List[Dict]:
        """Ağırlık kombinasyonları üret"""
        try:
            # Basit grid search için örnek implementasyon
            combinations = []
            strategies = ['momentum', 'mean_reversion', 'volatility', 'sentiment']
            
            # Örnek: 0.1'lik adımlarla ağırlık kombinasyonları
            for w1 in np.arange(0.1, 1.0, 0.3):
                for w2 in np.arange(0.1, 1.0, 0.3):
                    for w3 in np.arange(0.1, 1.0, 0.3):
                        for w4 in np.arange(0.1, 1.0, 0.3):
                            total = w1 + w2 + w3 + w4
                            if total > 0:
                                weights = {
                                    strategies[0]: w1/total,
                                    strategies[1]: w2/total,
                                    strategies[2]: w3/total,
                                    strategies[3]: w4/total
                                }
                                combinations.append(weights)
            
            return combinations[:10]  # Limit for performance
            
        except Exception as e:
            logger.error(f"Weight generation error: {str(e)}")
            return []
    
    async def _evaluate_weight_set(self, weights: Dict, data: pd.DataFrame, 
                                 target_metric: str) -> float:
        """Ağırlık setini değerlendir"""
        try:
            # Backtest simulation
            # Basit implementasyon - gerçek backtest logic buraya gelecek
            
            if target_metric == "sharpe_ratio":
                # Örnek Sharpe ratio hesaplama
                returns = data.pct_change().dropna()
                if returns.empty:
                    return 0.0
                
                excess_returns = returns - 0.02/252  # Risk-free rate assumption
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                return float(sharpe.iloc[0]) if not sharpe.empty else 0.0
                
            elif target_metric == "win_rate":
                # Örnek win rate hesaplama
                returns = data.pct_change().dropna()
                win_rate = (returns > 0).mean()
                return float(win_rate.iloc[0]) if not win_rate.empty else 0.0
                
            else:
                logger.warning(f"Unknown target metric: {target_metric}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Weight evaluation error: {str(e)}")
            return 0.0
    
    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """Fallback optimizasyon sonucu oluştur"""
        return {
            'optimized_weights': {},
            'best_score': 0.0,
            'target_metric': 'unknown',
            'timestamp': self.helpers.get_timestamp(),
            'status': 'error',
            'error_reason': reason
        }
    
    def validate_optimization_result(self, result: Dict[str, Any]) -> bool:
        """Optimizasyon sonucunu validate et"""
        required_keys = {'optimized_weights', 'best_score', 'target_metric', 'timestamp', 'status'}
        return all(key in result for key in required_keys)
# analysis/port_alloc.py
"""
Portfolio Optimization & Allocation Module - Analysis Helpers Uyumlu
Version: 1.1.0
Black-Litterman, HRP, Risk Parity optimizasyonları ile dinamik portföy ayırma

⚠️ Küçük Not:
port_alloc.py'de compute_metrics metodu symbols: List[str] parametresi alıyor (tek sembol yerine liste). Bu analysis_core.py'nin batch processing modunda sorunsuz çalışacak, ancak tekil sembol analizi için adaptasyon gerekebilir.

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput
from analysis.analysis_base_module import BaseAnalysisModule

from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Portföy metriklerini tutan data class"""
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    conditional_var: float
    max_drawdown: float
    volatility: float
    expected_return: float
    correlation_matrix: pd.DataFrame

@dataclass
class AllocationResult:
    """Portföy ayırma sonucu"""
    weights: Dict[str, float]
    metrics: PortfolioMetrics
    optimization_method: str
    score: float
    components: Dict[str, float]
    signal: str
    explain: Dict[str, Any]
    confidence: float

class PortfolioAllocationModule(BaseAnalysisModule):
    """
    Portfolio Optimization & Allocation Module - Analysis Helpers Uyumlu
    Dinamik portföy optimizasyonu ve varlık ayırma
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # ✅ ANALYSIS_HELPERS INTEGRATION
        self.helpers = AnalysisHelpers
        self.module_name = "portfolio_allocation"
        self.version = "1.1.0"
        
        # Load configuration - ANALYSIS_HELPERS UYUMLU
        if config is None:
            from analysis.config.cm_loader import config_manager
            config_obj = config_manager.get_config("portalloc")
            if config_obj:
                self.config_dict = config_obj.to_flat_dict()
            else:
                self.config_dict = self._get_default_config()
        else:
            self.config_dict = config
            
        self.weights = self.config_dict.get("weights", {})
        self.parameters = self.config_dict.get("parameters", {})
        self.thresholds = self.config_dict.get("thresholds", {})
        
        self.binance_agg = BinanceAggregator()
        self.parallel_executor = ThreadPoolExecutor(
            max_workers=self.parameters.get("parallel_processing", {}).get("max_workers", 4)
        )
        
        # Metrik dependency graph
        self.dependencies = {
            "returns": [],
            "volatility": ["returns"],
            "correlation": ["returns"],
            "covariance": ["returns"],
            "sharpe": ["returns", "volatility"],
            "sortino": ["returns", "volatility"],
            "var": ["returns", "volatility"]
        }
        
        logger.info(f"PortfolioAllocationModule initialized with {len(self.weights)} scoring components")

    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback config oluştur"""
        logger.warning("Using default config for PortfolioAllocationModule")
        return {
            "weights": {
                "sharpe": 0.30,
                "sortino": 0.25,
                "var": 0.20,
                "drawdown": 0.15,
                "volatility": 0.10
            },
            "thresholds": {
                "optimal_allocation": 0.7,
                "moderate_allocation": 0.4,
                "suboptimal_allocation": 0.2
            },
            "parameters": {
                "optimization_methods": {
                    "black_litterman": {"enabled": True, "tau": 0.05},
                    "hierarchical_risk_parity": {"enabled": True},
                    "risk_parity": {"enabled": True}
                },
                "data": {"lookback_period": 252, "min_data_points": 100},
                "parallel_processing": {"max_workers": 4}
            }
        }
    
    @cache_result(ttl=300)  # 5 dakika cache
    async def get_historical_prices(self, symbols: List[str], lookback: int = 252) -> pd.DataFrame:
        """Semboller için geçmiş fiyat verilerini getir"""
        try:
            price_data = {}
            
            for symbol in symbols:
                # Binance API'den kline verisi
                klines_data = await self.binance_agg.get_klines(
                    symbol=symbol,
                    interval='1d',
                    limit=lookback
                )
                
                if klines_data and len(klines_data) > 0:
                    closes = [float(k[4]) for k in klines_data]  # Close price
                    price_data[symbol] = closes
            
            return pd.DataFrame(price_data)
            
        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            raise
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Log getirileri hesapla"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> PortfolioMetrics:
        """Portföy metriklerini hesapla"""
        
        # Portföy getirisi ve volatilitesi
        portfolio_returns = returns.dot(weights)
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        expected_return = np.mean(portfolio_returns) * 252
        
        # Sharpe Ratio
        risk_free_rate = self.parameters.get("metrics", {}).get("sharpe_ratio", {}).get("risk_free_rate", 0.02)
        sharpe = (expected_return - risk_free_rate) / portfolio_volatility
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # VaR (Parametrik)
        var_confidence = self.parameters.get("metrics", {}).get("var", {}).get("confidence_level", 0.95)
        var = self._calculate_var(portfolio_returns, var_confidence)
        
        # Conditional VaR (Expected Shortfall)
        cvar = self._calculate_conditional_var(portfolio_returns, var_confidence)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Correlation Matrix
        correlation_matrix = returns.corr()
        
        return PortfolioMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            var_95=var,
            conditional_var=cvar,
            max_drawdown=max_drawdown,
            volatility=portfolio_volatility,
            expected_return=expected_return,
            correlation_matrix=correlation_matrix
        )
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Value at Risk hesapla"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence: float) -> float:
        """Conditional VaR (Expected Shortfall) hesapla"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def black_litterman_optimization(self, returns: pd.DataFrame, market_caps: Dict[str, float] = None) -> np.ndarray:
        """Black-Litterman model ile optimizasyon"""
        try:
            # Equilibrium returns (CAPM)
            cov_matrix = returns.cov() * 252
            market_weights = self._calculate_market_weights(market_caps, returns.columns)
            
            # Implied equilibrium returns
            tau = self.parameters.get("optimization_methods", {}).get("black_litterman", {}).get("tau", 0.05)
            risk_aversion = self.parameters.get("optimization_methods", {}).get("black_litterman", {}).get("risk_aversion", 2.5)
            
            pi = risk_aversion * cov_matrix.dot(market_weights)  # Implied returns
            
            # Views (burada basit trend views kullanıyoruz)
            P, Q = self._generate_views(returns)
            omega = self._generate_confidence_matrix(P, cov_matrix, tau)
            
            # Black-Litterman formula
            pi_bl = self._calculate_black_litterman_returns(
                pi, cov_matrix, tau, P, Q, omega
            )
            
            # Optimize weights
            weights = self._mean_variance_optimization(pi_bl, cov_matrix)
            return weights
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity optimizasyonu"""
        try:
            cov_matrix = returns.cov() * 252
            corr_matrix = returns.corr()
            
            # Distance matrix
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)
            
            # Hierarchical clustering
            linkage_method = self.parameters.get("optimization_methods", {}).get("hierarchical_risk_parity", {}).get("linkage_method", "ward")
            Z = linkage(squareform(distance_matrix.values), method=linkage_method)
            
            # HRP allocation
            weights = self._hrp_allocation(cov_matrix.values, Z)
            return weights
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk Parity optimizasyonu"""
        try:
            cov_matrix = returns.cov() * 252
            
            def risk_parity_objective(weights, cov_matrix):
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_risk
                
                # Equal risk contribution objective
                target_risk = portfolio_risk / len(weights)
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
            ]
            
            # Bounds
            max_alloc = self.parameters.get("constraints", {}).get("max_allocation_per_asset", 0.3)
            bounds = [(0, max_alloc) for _ in range(len(returns.columns))]
            
            # Initial guess (equal weight)
            x0 = np.array([1.0 / len(returns.columns)] * len(returns.columns))
            
            # Optimization
            result = minimize(
                risk_parity_objective,
                x0,
                args=(cov_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.parameters.get("optimization_methods", {}).get("risk_parity", {}).get("max_iter", 1000)}
            )
            
            return result.x if result.success else x0
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def _hrp_allocation(self, cov_matrix: np.ndarray, linkage_matrix: np.ndarray) -> np.ndarray:
        """HRP allocation implementation"""
        num_assets = cov_matrix.shape[0]
        weights = np.ones(num_assets)
        
        # Quasi-diagonalization
        clusters = self._quasi_diagonalize(linkage_matrix)
        
        # Recursive bisection
        weights = self._recursive_bisection(weights, clusters, cov_matrix)
        
        return weights / np.sum(weights)  # Normalize
    
    def _quasi_diagonalize(self, linkage_matrix: np.ndarray) -> List:
        """Quasi-diagonalization for HRP"""
        return list(range(linkage_matrix.shape[0] + 1))
    
    def _recursive_bisection(self, weights: np.ndarray, clusters: List, cov_matrix: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP"""
        num_assets = len(weights)
        return np.ones(num_assets) / num_assets
    
    def _calculate_market_weights(self, market_caps: Dict, symbols: List[str]) -> np.ndarray:
        """Piyasa ağırlıklarını hesapla"""
        if market_caps:
            total_cap = sum(market_caps.values())
            return np.array([market_caps.get(sym, 0) / total_cap for sym in symbols])
        else:
            return np.ones(len(symbols)) / len(symbols)
    
    def _generate_views(self, returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Basit trend-based views oluştur"""
        recent_returns = returns.tail(20).mean()
        
        P = np.eye(len(returns.columns))
        Q = recent_returns.values * 0.1
        
        return P, Q
    
    def _generate_confidence_matrix(self, P: np.ndarray, cov_matrix: pd.DataFrame, tau: float) -> np.ndarray:
        """View confidence matrix oluştur"""
        return np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))
    
    def _calculate_black_litterman_returns(self, pi: np.ndarray, cov_matrix: pd.DataFrame, 
                                         tau: float, P: np.ndarray, Q: np.ndarray, 
                                         omega: np.ndarray) -> np.ndarray:
        """Black-Litterman expected returns hesapla"""
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        first_term = np.linalg.inv(tau_sigma_inv + P.T @ np.linalg.inv(omega) @ P)
        second_term = tau_sigma_inv @ pi + P.T @ np.linalg.inv(omega) @ Q
        
        return first_term @ second_term
    
    def _mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Mean-Variance optimizasyonu"""
        def objective(weights):
            portfolio_return = weights.T @ expected_returns
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            return -portfolio_return / portfolio_risk  # Maximize Sharpe
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        max_alloc = self.parameters.get("constraints", {}).get("max_allocation_per_asset", 0.3)
        bounds = [(0, max_alloc) for _ in range(len(expected_returns))]
        
        x0 = np.ones(len(expected_returns)) / len(expected_returns)
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    def _equal_weight_allocation(self, num_assets: int) -> np.ndarray:
        """Eşit ağırlıklı portföy"""
        return np.ones(num_assets) / num_assets
    
    async def compute_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Portföy metriklerini hesapla - ANALYSIS_HELPERS UYUMLU"""
        try:
            # Historical prices
            lookback = self.parameters.get("data", {}).get("lookback_period", 252)
            min_data_points = self.parameters.get("data", {}).get("min_data_points", 100)
            
            prices = await self.get_historical_prices(symbols, lookback)
            
            if prices.empty or len(prices) < min_data_points:
                raise ValueError("Insufficient price data for portfolio analysis")
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            
            # Apply different optimization methods
            allocation_results = {}
            
            optimization_methods = self.parameters.get("optimization_methods", {})
            
            if optimization_methods.get("black_litterman", {}).get("enabled", True):
                bl_weights = self.black_litterman_optimization(returns)
                allocation_results["black_litterman"] = bl_weights
            
            if optimization_methods.get("hierarchical_risk_parity", {}).get("enabled", True):
                hrp_weights = self.hierarchical_risk_parity(returns)
                allocation_results["hierarchical_risk_parity"] = hrp_weights
            
            if optimization_methods.get("risk_parity", {}).get("enabled", True):
                rp_weights = self.risk_parity_optimization(returns)
                allocation_results["risk_parity"] = rp_weights
            
            # Calculate metrics for each method
            results = {}
            for method, weights in allocation_results.items():
                metrics = self.calculate_portfolio_metrics(returns, weights)
                score, components, signal, explain, confidence = self._generate_signal(metrics)
                
                results[method] = AllocationResult(
                    weights=dict(zip(symbols, weights)),
                    metrics=metrics,
                    optimization_method=method,
                    score=score,
                    components=components,
                    signal=signal,
                    explain=explain,
                    confidence=confidence
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing portfolio metrics: {e}")
            raise
    
    def _generate_signal(self, metrics: PortfolioMetrics) -> Tuple[float, Dict, str, Dict, float]:
        """Portföy sinyali ve skor oluştur"""
        
        # Component scores
        components = {
            "sharpe": max(0, metrics.sharpe_ratio / 2.0),
            "sortino": max(0, metrics.sortino_ratio / 2.0),
            "var": 1 - min(1, abs(metrics.var_95) / 0.1),
            "drawdown": 1 - min(1, abs(metrics.max_drawdown)),
            "volatility": 1 - min(1, metrics.volatility / 0.4)
        }
        
        # ✅ ANALYSIS_HELPERS ILE AGIRLIKLI ORTALAMA
        if self.weights and self.helpers.validate_score_dict(components):
            score = self.helpers.calculate_weights(components, self.weights)
        else:
            score = np.mean(list(components.values())) if components else 0.5
        
        # ✅ NORMALIZE SCORE
        score = self.helpers.normalize_score(score)
        
        # Signal generation
        optimal_threshold = self.thresholds.get("optimal_allocation", 0.7)
        moderate_threshold = self.thresholds.get("moderate_allocation", 0.4)
        
        if score > optimal_threshold:
            signal = "optimal_allocation"
        elif score > moderate_threshold:
            signal = "moderate_allocation"
        else:
            signal = "suboptimal_allocation"
        
        # Confidence calculation
        confidence = self._calculate_portfolio_confidence(metrics, components)
        
        # ✅ ANALYSIS_HELPERS UYUMLU EXPLAIN
        explain = {
            "summary": f"Portfolio analysis indicates {signal.replace('_', ' ')}",
            "confidence": confidence,
            "key_metrics": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility,
                "expected_return": metrics.expected_return
            },
            "interpretation": self._interpret_portfolio_score(score, signal),
            "recommendation": self._generate_allocation_recommendation(signal)
        }
        
        return score, components, signal, explain, confidence
    
    def _calculate_portfolio_confidence(self, metrics: PortfolioMetrics, components: Dict[str, float]) -> float:
        """Portföy analizi için confidence skoru hesapla"""
        if not components:
            return 0.0
        
        # Consistency of component scores
        component_variance = np.var(list(components.values()))
        consistency_score = 1.0 - min(1.0, component_variance / 0.25)
        
        # Metric quality indicators
        quality_indicators = [
            1.0 if metrics.sharpe_ratio > 0 else 0.5,  # Positive Sharpe
            1.0 if abs(metrics.max_drawdown) < 0.5 else 0.5,  # Reasonable drawdown
            1.0 if metrics.volatility < 0.5 else 0.5,  # Manageable volatility
        ]
        
        quality_score = np.mean(quality_indicators)
        
        return float((consistency_score + quality_score) / 2.0)
    
    def _interpret_portfolio_score(self, score: float, signal: str) -> str:
        """Portföy skoru için yorum oluştur"""
        if score >= 0.8:
            return "Excellent portfolio characteristics with strong risk-adjusted returns"
        elif score >= 0.6:
            return "Good portfolio construction with favorable risk-reward balance"
        elif score >= 0.4:
            return "Moderate portfolio quality with acceptable risk metrics"
        else:
            return "Suboptimal allocation requiring review and rebalancing"
    
    def _generate_allocation_recommendation(self, signal: str) -> str:
        """Portföy ayırma önerisi oluştur"""
        if signal == "optimal_allocation":
            return "Maintain current allocation with periodic rebalancing"
        elif signal == "moderate_allocation":
            return "Consider minor adjustments to improve risk-adjusted returns"
        else:
            return "Significant reallocation recommended to optimize portfolio structure"
    
    async def aggregate_output(self, results: Dict) -> Dict[str, Any]:
        """Sonuçları aggregate et - ANALYSIS_HELPERS UYUMLU"""
        if not results:
            return self.helpers.create_fallback_output(self.module_name, "No optimization results available")
        
        best_method = max(results.keys(), key=lambda x: results[x].score)
        best_result = results[best_method]
        
        # ✅ ANALYSIS_HELPERS UYUMLU OUTPUT FORMATI
        output = {
            "score": best_result.score,
            "signal": best_result.signal,
            "confidence": best_result.confidence,
            "components": best_result.components,
            "explain": best_result.explain,
            "timestamp": self.helpers.get_timestamp(),
            "module": self.module_name,
            "weights": best_result.weights,
            "optimization_method": best_result.optimization_method,
            "metrics": {
                "sharpe_ratio": best_result.metrics.sharpe_ratio,
                "sortino_ratio": best_result.metrics.sortino_ratio,
                "var_95": best_result.metrics.var_95,
                "expected_return": best_result.metrics.expected_return,
                "volatility": best_result.metrics.volatility,
                "max_drawdown": best_result.metrics.max_drawdown
            },
            "all_methods": {
                method: {
                    "score": result.score,
                    "weights": result.weights
                } for method, result in results.items()
            }
        }
        
        # ✅ OUTPUT VALIDATION
        if not self.helpers.validate_output(output):
            logger.warning("Portfolio output validation failed")
            return self.helpers.create_fallback_output(self.module_name, "Output validation failed")
        
        return output
    
    async def run(self, symbols: List[str], priority: str = "normal") -> Dict[str, Any]:
        """
        Main execution method - ANALYSIS_HELPERS UYUMLU
        """
        try:
            results = await self.compute_metrics(symbols)
            return await self.aggregate_output(results)
        except Exception as e:
            logger.error(f"Error in portfolio allocation: {e}")
            return self.helpers.create_fallback_output(self.module_name, str(e))
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return module metadata"""
        return {
            "module_name": self.module_name,
            "version": self.version,
            "description": "Portfolio optimization and asset allocation",
            "optimization_methods": list(self.parameters.get("optimization_methods", {}).keys()),
            "metrics": list(self.weights.keys()),
            "parallel_mode": "batch",
            "lifecycle": "development",
            "analysis_helpers_compatible": True,
            "supports_multiple_assets": True
        }

# Factory pattern için
class PortfolioAllocationFactory:
    """Portfolio Allocation factory sınıfı"""
    
    @classmethod
    def create_module(cls, config: Dict = None) -> PortfolioAllocationModule:
        """Portfolio allocation modülü oluştur"""
        return PortfolioAllocationModule(config)

# FastAPI router için yardımcı fonksiyon
async def allocate_portfolio(symbols: List[str], config: Dict = None) -> Dict:
    """Portföy ayırma endpoint'i için yardımcı fonksiyon"""
    module = PortfolioAllocationFactory.create_module(config)
    return await module.run(symbols)
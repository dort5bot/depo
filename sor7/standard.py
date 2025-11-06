# 📁 metrics/standard.py
"""
MAPS Framework - Metric Standardization Interface
Author: ysf-bot-framework
Version: 2025.1

YAML Standartlar:
Element	Standard	Kullanım Yeri
Teknik İsimler	snake_case	YAML keys, formula references, module names
Business İsimler	PascalCase	UI display, raporlar, dokümantasyon
Dosya İsimleri	snake_case.yaml	Tüm YAML dosyaları
Kategori/Type	snake_case	Metric categorization
Referanslar	snake_case	Cross-references between YAMLs
"""

from typing import Union, Dict, Any, Optional, Tuple, Callable, List
import re
import os
import yaml

import pandas as pd
import numpy as np

# Lazy import for optional dependencies
_pl = None

def get_polars():
    global _pl
    if _pl is None:
        try:
            import polars as _pl
        except ImportError:
            _pl = None
    return _pl


# ==================== CENTRALIZED NAMING STANDARDS ====================

class NamingStandard:
    """Merkezi isimlendirme standartları"""
    
    _business_names = None
    
    @classmethod
    def _load_business_names(cls) -> Dict[str, str]:
        """Business isimleri YAML'dan yükle"""
        if cls._business_names is not None:
            return cls._business_names
            
        try:
            # 1. Önce external YAML'dan dene
            yaml_path = "config/business_names.yaml"
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    cls._business_names = yaml.safe_load(f)
                    return cls._business_names
                    
            # 2. Yoksa embedded YAML kullan
            cls._business_names = cls._load_embedded_yaml()
            
        except Exception as e:
            print(f"Business names YAML yüklenemedi: {e}")
            # 3. Fallback olarak embedded dict
            cls._business_names = cls._get_fallback_names()
            
        return cls._business_names
            
    @staticmethod
    def _load_embedded_yaml() -> Dict[str, str]:
        """Embedded YAML string'den yükle"""
        yaml_content = """
        # BUSINESS NAMES MAPPING - Merkezi Yönetim
        # technical_name: "Business Display Name"
        
        # === CLASSICAL METRICS ===
        simple_ema: "Exponential Moving Average"
        sma: "Simple Moving Average"
        macd: "Moving Average Convergence Divergence"
        rsi: "Relative Strength Index" 
        atr: "Average True Range"
        bollinger_bands: "Bollinger Bands"
        historical_volatility: "Historical Volatility"
        
        # === SENTIMENT METRICS ===
        funding_rate: "Funding Rate Analysis"
        open_interest: "Open Interest Trend"
        long_short_ratio: "Long Short Ratio"
        oi_change_rate: "Open Interest Change Rate"
        liquidation_heat: "Liquidation Heat Map"
        funding_rate_trend: "Funding Rate Trend"
        funding_premium: "Funding Premium Indicator"
        oi_momentum: "Open Interest Momentum"
        long_short_imbalance: "Long Short Imbalance"
        
        # === VOLATILITY METRICS ===
        garch_1_1: "GARCH(1,1) Volatility"
        hurst_exponent: "Hurst Exponent"
        entropy_index: "Entropy Index"
        variance_ratio_test: "Variance Ratio Test"
        range_expansion_index: "Range Expansion Index"
        
        # === COMPOSITE METRICS ===
        trend_composite: "Trend Composite Score"
        momentum_composite: "Momentum Composite Score" 
        volatility_composite: "Volatility Composite Score"
        risk_composite: "Risk Composite Score"
        market_regime: "Market Regime Detection"
        market_stress: "Market Stress Indicator"
        
        # === ON-CHAIN METRICS ===
        etf_net_flow: "ETF Net Flow"
        exchange_netflow: "Exchange Netflow"
        stablecoin_flow: "Stablecoin Flow"
        net_realized_pl: "Net Realized P/L"
        realized_cap: "Realized Cap"
        nupl: "Net Unrealized Profit/Loss"
        exchange_whale_ratio: "Exchange Whale Ratio"
        mvrv_zscore: "MVRV Z-Score"
        sopr: "Spent Output Profit Ratio"
        etf_flow_composite: "ETF Flow Composite"
        
        # === MICROSTRUCTURE METRICS ===
        ofi: "Order Flow Imbalance"
        cvd: "Cumulative Volume Delta"
        microprice_deviation: "Microprice Deviation"
        market_impact: "Market Impact"
        depth_elasticity: "Depth Elasticity"
        taker_dominance_ratio: "Taker Dominance Ratio"
        liquidity_density: "Liquidity Density"
        
        # === ADVANCED MATH METRICS ===
        kalman_filter_trend: "Kalman Filter Trend"
        wavelet_transform: "Wavelet Transform"
        hilbert_transform_slope: "Hilbert Transform Slope"
        hilbert_transform_amplitude: "Hilbert Transform Amplitude"
        fractal_dimension_index_fdi: "Fractal Dimension Index"
        shannon_entropy: "Shannon Entropy"
        permutation_entropy: "Permutation Entropy"
        sample_entropy: "Sample Entropy"
        """
        
        return yaml.safe_load(yaml_content)
    
    @staticmethod
    def _get_fallback_names() -> Dict[str, str]:
        """Fallback mapping"""
        return {
            "simple_ema": "Exponential Moving Average",
            "sma": "Simple Moving Average",
            "macd": "Moving Average Convergence Divergence",
            "rsi": "Relative Strength Index",
            "atr": "Average True Range",
        }
    
    @staticmethod
    def to_snake_case(name: str) -> str:
        """Convert any string to snake_case"""
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        return name.lower()
    
    @classmethod
    def to_business_name(cls, technical_name: str) -> str:
        """Convert technical name to business name"""
        business_names = cls._load_business_names()
        return business_names.get(technical_name, technical_name.replace('_', ' ').title())

    @staticmethod
    def validate_technical_name(name: str) -> bool:
        """Validate snake_case format"""
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))
    
    @staticmethod
    def validate_file_name(file_path: str) -> bool:
        """Validate snake_case file naming"""
        filename = os.path.basename(file_path)
        return bool(re.match(r'^[a-z][a-z0-9_]*\.(py|yaml|yml)$', filename))


# ==================== YAML STANDARD INTEGRATION ====================

class YAMLStandard:
    """YAML output standardization"""
    
    @staticmethod
    def create_metric_yaml(
        technical_name: str,
        category: str,
        description: str,
        formula: str = "",
        input_type: str = "price_series",
        output_type: str = "float_series", 
        parameters: Dict[str, Any] = None,
        min_periods: int = 1
    ) -> Dict[str, Any]:
        """Standard YAML structure for basic metrics"""
        return {
            "metrics": {
                technical_name: {
                    "category": category,
                    "description": description,
                    "formula": formula,
                    "input": input_type,
                    "output": output_type,
                    "parameters": parameters or {},
                    "min_periods": min_periods
                }
            }
        }
  
    @staticmethod
    def create_composite_yaml(
        composite_name: str,
        category: str,
        description: str,
        formula: str,
        input_type: str = "multi_modal_data",
        output_type: str = "float_score",
        parameters: Dict[str, Any] = None,
        interpretation: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Standard YAML structure for composite metrics"""
        composite_data = {
            "category": category,
            "description": description,
            "formula": formula,
            "input": input_type,
            "output": output_type,
            "parameters": parameters or {}
        }
        
        if interpretation:
            composite_data["interpretation"] = interpretation
        if metadata:
            composite_data["metadata"] = metadata
            
        return {
            "composite_metrics": {
                composite_name: composite_data
            }
        }

    @staticmethod
    def generate_module_registry(
        module_name: str,
        data_model: str = "pandas",
        compute_intensity: str = "medium", 
        metrics: Dict[str, List[str]] = None,
        endpoints: List[str] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate module configuration for module_registry.yaml"""
        return {
            "modules": {
                module_name: {
                    "multi_user": True,
                    "data_source": "utils/binance_api/binance_a.BinanceAggregator",
                    "api_type": "public",
                    "endpoints": endpoints or ["klines"],
                    "job_type": "metric_calculation",
                    "parallel_mode": "async",
                    "data_model": data_model,
                    "compute_intensity": compute_intensity,
                    "metrics": metrics or {},
                    "config": config or {}
                }
            }
        }
        
    @staticmethod
    def generate_from_function(func: Callable, **kwargs) -> Dict[str, Any]:
        """Auto-generate YAML from function metadata"""
        technical_name = func.__name__
        
        formula = func.__doc__ or "See function implementation"
        if isinstance(formula, str):
            formula = formula.strip().split('\n')[0]
        
        return YAMLStandard.create_metric_yaml(
            technical_name=technical_name,
            formula=formula,
            **kwargs
        )
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: str):
        """Save YAML with naming validation"""
        if not NamingStandard.validate_file_name(file_path):
            raise ValueError(f"Invalid file name: {file_path}. Must be snake_case.")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    @staticmethod
    def batch_generate_metrics(metrics_config: Dict[str, Dict], output_dir: str):
        """Batch generate metric YAML files from configuration"""
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        for metric_name, config in metrics_config.items():
            yaml_data = YAMLStandard.create_metric_yaml(
                technical_name=metric_name,
                **config
            )
            
            file_path = os.path.join(output_dir, f"{metric_name}.yaml")
            YAMLStandard.save_yaml(yaml_data, file_path)
            generated_files.append(file_path)
        
        return generated_files
    
    @staticmethod 
    def batch_generate_composites(composites_config: Dict[str, Dict], output_file: str):
        """Batch generate composite metrics into single module_registry.yaml"""
        all_composites = {}
        
        for composite_name, config in composites_config.items():
            yaml_data = YAMLStandard.create_composite_yaml(
                composite_name=composite_name,
                **config
            )
            all_composites.update(yaml_data["composite_metrics"])
        
        final_yaml = {"composite_metrics": all_composites}
        YAMLStandard.save_yaml(final_yaml, output_file)
        return output_file


# ==================== ENHANCED METRIC STANDARD ====================

class MetricStandard:
    """Enhanced with comprehensive YAML standards"""
    
    @staticmethod
    def _convert_data(data, target_type: str):
        """Unified data conversion method"""
        if target_type == "pandas":
            return MetricStandard._as_pandas(data)
        elif target_type == "numpy":
            return MetricStandard._as_numpy(data)
        elif target_type == "polars":
            return MetricStandard._as_polars(data)
        return data
    
    @staticmethod
    def _as_pandas(data):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        elif isinstance(data, np.ndarray):
            return pd.Series(data)
        elif isinstance(data, list):
            return pd.Series(data)
        
        pl = get_polars()
        if pl and isinstance(data, pl.Series):
            return data.to_pandas()
        return data
    
    @staticmethod
    def _as_numpy(data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data.to_numpy()
        elif isinstance(data, list):
            return np.array(data)
        
        pl = get_polars()
        if pl and isinstance(data, pl.Series):
            return data.to_numpy()
        return data
    
    @staticmethod
    def _as_polars(data):
        pl = get_polars()
        if not pl:
            raise ImportError("Polars not available")
            
        if isinstance(data, pl.Series):
            return data
        elif isinstance(data, pd.Series):
            return pl.from_pandas(data)
        elif isinstance(data, (np.ndarray, list)):
            return pl.Series(data)
        return data

    @staticmethod
    def _ffill_series(series):
        """Enhanced forward fill with type checking"""
        if series is None:
            return series
            
        if isinstance(series, dict):
            return series
            
        if hasattr(series, 'ffill'):  # Pandas
            return series.ffill().bfill()
        elif hasattr(series, 'fill_null'):  # Polars
            return series.fill_null(strategy="forward").fill_null(strategy="backward")
        elif isinstance(series, np.ndarray):
            try:
                mask = np.isnan(series)
                indices = np.where(~mask, np.arange(len(series)), 0)
                np.maximum.accumulate(indices, out=indices)
                return series[indices]
            except (TypeError, ValueError):
                return series
        else:
            return series

    @staticmethod
    def standardize_input(
        data: Union[pd.Series, pd.DataFrame, np.ndarray, list, Dict[str, Any]],
        expected_type: str = "pandas",
        fillna: bool = True,
        min_periods: int = 1
    ) -> Tuple[Any, str]:
        """Standardize input data to target type"""
        if isinstance(data, dict):
            standardized = {k: MetricStandard._convert_data(v, expected_type) for k, v in data.items()}
        else:
            standardized = MetricStandard._convert_data(data, expected_type)
        
        if fillna:
            standardized = MetricStandard._fill_missing(standardized)
            
        return standardized, expected_type
    
    @staticmethod
    def _fill_missing(data):
        if isinstance(data, dict):
            return {k: MetricStandard._ffill_series(v) for k, v in data.items()}
        else:
            return MetricStandard._ffill_series(data)
    
    @staticmethod
    def standardize_output(
        result: Any,
        return_type: str = "pandas",
        input_data: Any = None
    ) -> Any:
        """Convert result to requested output type"""
        if return_type == "native":
            return result
            
        if isinstance(result, dict):
            return {k: MetricStandard._convert_data(v, return_type) for k, v in result.items()}
        else:
            return MetricStandard._convert_data(result, return_type)
    
    @staticmethod
    def validate_input(data: Any, min_periods: int = 1) -> bool:
        """Validate input data meets requirements"""
        if data is None:
            return False
            
        if isinstance(data, dict):
            return all(len(v) >= min_periods for v in data.values() if v is not None)
        else:
            return len(data) >= min_periods

    @staticmethod
    def create_decorated_metric(metric_func: Callable, 
                              input_type: str = "pandas", 
                              output_type: str = "pandas",
                              min_periods: int = 1, 
                              fillna: bool = True) -> Callable:
        """Runtime'da metric fonksiyonuna decorator uygular"""
        @metric_standard(
            input_type=input_type,
            output_type=output_type, 
            min_periods=min_periods,
            fillna=fillna
        )
        def decorated_metric(data, *args, **kwargs):
            return metric_func(data, *args, **kwargs)
            
        return decorated_metric
    
    @staticmethod
    def batch_standardize_metrics(metrics_dict: Dict[str, Callable], 
                                 module_config: Dict[str, Any]) -> Dict[str, Callable]:
        """Bir modülün tüm metriklerini toplu standardize eder"""
        metric_config = module_config.get('metric_config', {})
        
        standardized_metrics = {}
        for metric_name, metric_func in metrics_dict.items():
            specific_config = module_config.get('metrics', {}).get(metric_name, {})
            config = {**metric_config, **specific_config}
            
            standardized_metrics[metric_name] = MetricStandard.create_decorated_metric(
                metric_func, **config
            )
        
        return standardized_metrics

    @staticmethod
    def generate_metric_yaml(metric_func: Callable, **kwargs) -> Dict[str, Any]:
        """Generate YAML for a metric function"""
        return YAMLStandard.generate_from_function(metric_func, **kwargs)
    
    @staticmethod
    def validate_metric_name(name: str) -> Tuple[bool, str]:
        """Validate metric name and suggest corrections"""
        if NamingStandard.validate_technical_name(name):
            return True, "Valid snake_case"
        else:
            suggested = NamingStandard.to_snake_case(name)
            return False, f"Invalid. Suggested: {suggested}"

    @staticmethod
    def generate_composite_yaml(composite_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate composite metric YAML"""
        return YAMLStandard.create_composite_yaml(**composite_config)
    
    @staticmethod
    def generate_module_config(module_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate module registry configuration"""
        return YAMLStandard.generate_module_registry(**module_config)
    
    @staticmethod
    def validate_composite_name(name: str) -> Tuple[bool, str]:
        """Validate composite metric naming convention"""
        if not name.endswith('_composite'):
            return False, "Composite metrics must end with '_composite'"
        return MetricStandard.validate_metric_name(name)


# ==================== DECORATOR FOR EASY USAGE ====================

def metric_standard(
    input_type: str = "pandas",
    output_type: str = "pandas", 
    min_periods: int = 1,
    fillna: bool = True
):
    """Decorator to automatically standardize metric inputs/outputs"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = args[0] if args else kwargs.get('data')
            
            if data is None:
                raise ValueError("No data provided to metric")
            
            standardized_data, actual_type = MetricStandard.standardize_input(
                data, input_type, fillna, min_periods
            )
            
            if not MetricStandard.validate_input(standardized_data, min_periods):
                return MetricStandard.standardize_output(
                    np.nan, output_type, standardized_data
                )
            
            if args:
                args = (standardized_data,) + args[1:]
            else:
                kwargs['data'] = standardized_data
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                return MetricStandard.standardize_output(
                    np.nan, output_type, standardized_data
                )
            
            return MetricStandard.standardize_output(result, output_type, standardized_data)
        
        return wrapper
    return decorator


# ==================== GLOBAL INSTANCE FOR BACKWARDS COMPATIBILITY ====================

standardizer = MetricStandard()

# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # 1. BASIC METRIC GENERATION
    basic_metric_config = {
        "technical_name": "historical_volatility",
        "category": "volatility", 
        "description": "Rolling standard deviation of log returns",
        "formula": "std(log_returns) * √annualization_factor",
        "input": "price_series",
        "output": "volatility_series",
        "parameters": {
            "window": 30,
            "annualize": True
        },
        "min_periods": 30
    }

    basic_yaml = YAMLStandard.create_metric_yaml(**basic_metric_config)

    # 2. COMPOSITE METRIC GENERATION  
    composite_config = {
        "composite_name": "trend_composite",
        "category": "trend_composite", 
        "description": "Combines multiple trend indicators",
        "formula": "0.3*EMA + 0.3*MACD + 0.2*RSI + 0.2*ADX",
        "input": "pandas_dataframe",
        "output": "float_score",
        "interpretation": [
            "-1.0 to -0.6: Strong Bearish",
            "-0.6 to -0.3: Weak Bearish", 
            "-0.3 to 0.3: Neutral",
            "0.3 to 0.6: Weak Bullish",
            "0.6 to 1.0: Strong Bullish"
        ]
    }

    composite_yaml = MetricStandard.generate_composite_yaml(composite_config)

    print("✓ Standardization module optimized and ready")
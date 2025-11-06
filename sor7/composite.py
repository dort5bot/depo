# -*- coding: utf-8 -*-
"""
metrics/composite.py
- Composite scoring engine with enhanced YAML integration.

Features:
- Full compatibility with module_registry.yaml new structure
- composite_modules and composite_metrics support
- depends_on dependency resolution
- performance_profiles integration
- handler_mapping support
- metric_standardization auto-mapping
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import asyncio
import importlib
import logging
import inspect
import ast
import operator as op
from functools import lru_cache
import re

import numpy as np
import pandas as pd
import os
import yaml

try:
    import polars as pl
except Exception:
    pl = None

from .standard import MetricStandard

logger = logging.getLogger("metrics.composite")
standardizer = MetricStandard()

# -------------------------
# Enhanced Safe Evaluator
# -------------------------
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.BitXor: op.xor,
    ast.Lt: op.lt,
    ast.Gt: op.gt,
    ast.LtE: op.le,
    ast.GtE: op.ge,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.And: lambda a, b: np.logical_and(a, b),
    ast.Or: lambda a, b: np.logical_or(a, b),
}

_ALLOWED_FUNCTIONS = {
    "np": np,
    "tanh": np.tanh,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "nan": lambda: np.nan,
    "mean": lambda x: np.nanmean(x),
    "min": lambda x: np.nanmin(x),
    "max": lambda x: np.nanmax(x),
    "sum": lambda x: np.nansum(x),
    "std": lambda x: np.nanstd(x),
    "clip": np.clip,
}

class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, names: Dict[str, Any], functions: Dict[str, Callable]):
        self.names = names
        self.functions = functions

    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        if hasattr(self, method):
            return getattr(self, method)(node)
        else:
            raise ValueError(f"Unsupported AST node: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Operator {op_type} not allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError("Unary operator not allowed")

    def visit_Num(self, node: ast.Num):
        return node.n

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            return self.names[node.id]
        if node.id in self.functions:
            return self.functions[node.id]
        raise NameError(f"Name '{node.id}' is not defined in formula context")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.functions:
                raise NameError(f"Function '{func_name}' not allowed in formula")
            func = self.functions[func_name]
            args = [self.visit(a) for a in node.args]
            return func(*args)
        else:
            raise ValueError("Only simple function calls allowed")

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        results = []
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_type = type(op_node)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError("Comparison operator not allowed")
            results.append(_ALLOWED_OPERATORS[op_type](left, right))
            left = right
        if all(isinstance(r, (bool, np.bool_)) for r in results):
            return results[-1]
        return np.logical_and.reduce(results)

    def visit_BoolOp(self, node: ast.BoolOp):
        vals = [self.visit(v) for v in node.values]
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            result = vals[0]
            for v in vals[1:]:
                result = _ALLOWED_OPERATORS[op_type](result, v)
            return result
        raise ValueError("Boolean operator not allowed")

    def generic_visit(self, node):
        raise ValueError(f"Unsupported node type: {node.__class__.__name__}")

def safe_eval(expr: str, names: Dict[str, Any], functions: Dict[str, Callable]):
    """Safely evaluate arithmetic expression with enhanced safety."""
    parsed = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator(names=names, functions=functions)
    return evaluator.visit(parsed)

# -------------------------
# Enhanced YAML Loader
# -------------------------
@lru_cache(maxsize=2)
def _load_full_yaml_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load complete module_registry.yaml configuration."""
    if path is None:
        candidate = os.path.join(os.getcwd(), "analysis", "schemas", "module_registry.yaml")
        if not os.path.exists(candidate):
            candidate = os.path.join(os.getcwd(), "schemas", "module_registry.yaml")
    else:
        candidate = path

    if not os.path.exists(candidate):
        logger.warning(f"YAML config not found at {candidate}")
        return {}

    with open(candidate, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        return loaded or {}

# -------------------------
# Enhanced CompositeEngine
# -------------------------
class CompositeEngine:
    def __init__(self, base_package: str = "analysis.metrics", yaml_config_path: Optional[str] = None):
        self.base_package = base_package
        self._metric_cache: Dict[str, Callable] = {}
        
        # Load complete YAML configuration
        self._full_config = _load_full_yaml_config(yaml_config_path)
        self._composite_modules = self._full_config.get("composite_modules", {})
        self._composite_metrics = self._full_config.get("composite_metrics", {})
        self._performance_profiles = self._full_config.get("performance_profiles", {})
        self._handler_mapping = self._full_config.get("handler_mapping", {})
        self._metric_standardization = self._full_config.get("metric_standardization", {})
        self._modules_config = self._full_config.get("modules", {})
        
        logger.info(f"Loaded {len(self._composite_modules)} composite modules, "
                   f"{len(self._composite_metrics)} composite metrics")

    # -------------------------
    # Core Resolution Methods
    # -------------------------
    def _resolve_metric_function(self, metric_name: str) -> Optional[Callable]:
        """Resolve metric function with enhanced module support."""
        if metric_name in self._metric_cache:
            return self._metric_cache[metric_name]

        modules = [
            "classical", "advanced", "sentiment", "volatility",
            "microstructure", "onchain", "standard"
        ]

        variants = [metric_name, metric_name.lower(), metric_name.upper()]

        for mod in modules:
            try:
                module = importlib.import_module(f"{self.base_package}.{mod}")
            except Exception:
                continue
                
            for v in variants:
                if hasattr(module, v):
                    func = getattr(module, v)
                    if inspect.iscoroutinefunction(func) or callable(func):
                        self._metric_cache[metric_name] = func
                        return func

        self._metric_cache[metric_name] = None
        return None

    def _get_performance_profile(self, intensity: str = "medium") -> Dict[str, Any]:
        """Get performance profile for compute intensity."""
        return self._performance_profiles.get(intensity, {
            "max_workers": 10,
            "timeout": 30,
            "execution_strategy": "async"
        })

    def _get_module_metrics(self, module_name: str) -> List[str]:
        """Extract all metrics from a module configuration."""
        if module_name not in self._modules_config:
            return []
            
        metrics = []
        for category, metric_list in self._modules_config[module_name].get("metrics", {}).items():
            metrics.extend(metric_list)
        return metrics


    # 📁 metrics/composite.py - EKLE
    def _prepare_metric_data(self, data: Any, metric_name: str) -> Any:
        """Prepare data for specific metric types"""
        if metric_name in ['GARCH_1_1', 'Hurst_Exponent', 'Entropy_Index']:
            # Volatility metrics need pure numpy arrays
            if hasattr(data, 'values'):
                return data.values
            elif isinstance(data, (list, tuple)):
                return np.array(data, dtype=float)
            return data
        
        elif metric_name in ['OFI', 'CVD', 'Microprice_Deviation']:
            # Microstructure metrics need dict values converted
            if isinstance(data, dict):
                return {k: self._extract_numpy_values(v) for k, v in data.items()}
        
        return data

    def _extract_numpy_values(self, data: Any) -> np.ndarray:
        """Extract numpy values from various data types"""
        if hasattr(data, 'values'):
            return data.values
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array([data], dtype=float)
            
    # -------------------------
    # Enhanced Execution Methods
    # -------------------------

    # --- call_callable_async metodu
    def _safe_data_conversion(self, data: Any) -> np.ndarray:
        """Güvenli veri dönüşümü"""
        try:
            if hasattr(data, 'values'):
                return data.values
            elif isinstance(data, (list, tuple)):
                return np.array(data, dtype=float)
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array([data], dtype=float)
        except Exception:
            return np.array([], dtype=float)
 
    async def _call_callable_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute callable with enhanced data preparation"""
        
        # Prepare args and kwargs
        prepared_args = []
        for arg in args:
            if hasattr(arg, 'dtype') and arg.dtype == object:
                prepared_args.append(self._safe_data_conversion(arg))
            else:
                prepared_args.append(arg)
        
        prepared_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, 'dtype') and value.dtype == object:
                prepared_kwargs[key] = self._safe_data_conversion(value)
            else:
                prepared_kwargs[key] = value
        
        # ÖNEMLİ: Asenkron fonksiyon kontrolü ve doğru await
        if asyncio.iscoroutinefunction(func):
            return await func(*prepared_args, **prepared_kwargs)
        elif inspect.iscoroutinefunction(func):
            # Async fonksiyon ama doğru şekilde çağır
            result = func(*prepared_args, **prepared_kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        
        # Senkron fonksiyonlar için thread executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: func(*prepared_args, **prepared_kwargs)
        )

    

    def _parse_worker_count(self, worker_spec: Any) -> int:
        """Parse worker count specification (e.g., 'cpu_count * 2')."""
        if isinstance(worker_spec, int):
            return worker_spec
        if isinstance(worker_spec, str):
            if worker_spec == "cpu_count":
                return os.cpu_count() or 4
            if "*" in worker_spec:
                parts = worker_spec.split("*")
                if len(parts) == 2 and "cpu_count" in parts[0]:
                    multiplier = float(parts[1].strip())
                    return int((os.cpu_count() or 4) * multiplier)
        return 10  # default


    async def compute_metrics(
        self,
        metrics: List[str],
        data: Any,
        intensity: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Compute multiple metrics with enhanced error handling"""
        profile = self._get_performance_profile(intensity)
        max_concurrent = self._parse_worker_count(profile.get("max_workers", 10))
        timeout = profile.get("timeout", 30.0)
        
        results: Dict[str, Any] = {}
        sem = asyncio.Semaphore(max_concurrent)

        async def _compute_single(metric: str):
            async with sem:
                try:
                    func = self._resolve_metric_function(metric)
                    if func is None:
                        logger.warning(f"Metric function not found: {metric}")
                        results[metric] = np.nan
                        return

                    logger.debug(f"Computing metric: {metric}")
                    
                    # Fonksiyon tipini kontrol et
                    if asyncio.iscoroutinefunction(func):
                        logger.debug(f"Metric {metric} is async coroutine")
                    else:
                        logger.debug(f"Metric {metric} is sync function")
                    
                    res = await asyncio.wait_for(
                        self._call_callable_async(func, data, **kwargs),
                        timeout=timeout
                    )
                    results[metric] = res
                    logger.debug(f"Metric {metric} completed successfully")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Metric {metric} timeout after {timeout}s")
                    results[metric] = np.nan
                except Exception as e:
                    logger.error(f"Metric {metric} failed: {str(e)}")
                    logger.exception(f"Detailed error for {metric}:")
                    results[metric] = np.nan

        await asyncio.gather(*[_compute_single(m) for m in metrics])
        return results
        

    # -------------------------
    # Dependency Resolution
    # -------------------------
    async def _resolve_module_dependencies(self, module_name: str, data: Any) -> Dict[str, Any]:
        """Resolve all dependencies for a composite module."""
        if module_name not in self._composite_modules:
            return {}

        module_config = self._composite_modules[module_name]
        dependencies = module_config.get("depends_on", [])
        
        all_results = {}
        
        for dep in dependencies:
            # Get dependency module metrics
            dep_metrics = self._get_module_metrics(dep)
            if dep_metrics:
                dep_results = await self.compute_metrics(
                    dep_metrics, 
                    data,
                    intensity=self._modules_config.get(dep, {}).get("compute_intensity", "medium")
                )
                all_results.update(dep_results)
                
        return all_results

    # -------------------------
    # Composite Calculation
    # -------------------------
    async def compute_composite_module(self, module_name: str, data: Any, **kwargs) -> Any:
        """Compute composite module with full dependency resolution."""
        if module_name not in self._composite_modules:
            raise KeyError(f"Composite module '{module_name}' not found")

        module_config = self._composite_modules[module_name]
        
        # 1. Resolve dependencies
        dependency_results = await self._resolve_module_dependencies(module_name, data)
        
        # 2. Compute module's own metrics
        module_metrics = []
        for category, metrics in module_config.get("metrics", {}).items():
            module_metrics.extend(metrics)
            
        module_results = await self.compute_metrics(
            module_metrics,
            data,
            intensity=module_config.get("compute_intensity", "medium")
        )
        
        # 3. Merge all results
        all_results = {**dependency_results, **module_results}
        
        # 4. Apply composite formula or weighted average
        composite_config = module_config.get("composite_config", {})
        score_components = composite_config.get("score_components", [])
        weights = composite_config.get("weights", [])
        
        # Try formula first
        formula = composite_config.get("formula")
        if formula:
            return await self._evaluate_composite_formula(formula, all_results, data)
        
        # Fallback to weighted average
        if score_components and weights and len(score_components) == len(weights):
            return self._calculate_weighted_composite(score_components, weights, all_results)
        
        # Final fallback: average of all module metrics
        return self._calculate_simple_average(module_metrics, all_results)

    async def _evaluate_composite_formula(self, formula: str, variables: Dict[str, Any], data: Any) -> float:
        """Evaluate composite formula with safe execution."""
        try:
            # Prepare functions with normalize
            functions = dict(_ALLOWED_FUNCTIONS)
            functions["normalize"] = self._normalize_metric
            
            result = safe_eval(formula, names=variables, functions=functions)
            return self._extract_final_score(result)
        except Exception as e:
            logger.exception(f"Formula evaluation failed: {e}")
            return np.nan

    def _calculate_weighted_composite(self, components: List[str], weights: List[float], 
                                    results: Dict[str, Any]) -> float:
        """Calculate weighted composite score."""
        weighted_scores = []
        valid_weights = []
        
        for component, weight in zip(components, weights):
            if component in results:
                score = self._extract_score(results[component])
                if not np.isnan(score):
                    weighted_scores.append(score * weight)
                    valid_weights.append(weight)
        
        if weighted_scores and valid_weights:
            final_score = sum(weighted_scores) / sum(valid_weights)
            return float(np.clip(final_score, -1.0, 1.0))
        
        return np.nan

    def _calculate_simple_average(self, metrics: List[str], results: Dict[str, Any]) -> float:
        """Calculate simple average of metrics."""
        scores = []
        for metric in metrics:
            if metric in results:
                score = self._extract_score(results[metric])
                if not np.isnan(score):
                    scores.append(score)
        
        if scores:
            avg_score = np.nanmean(scores)
            return float(np.clip(avg_score, -1.0, 1.0))
        
        return np.nan

    # -------------------------
    # Score Extraction & Normalization
    # -------------------------
    def _extract_score(self, value: Any) -> float:
        """Extract float score from various data types."""
        if value is None:
            return np.nan
        if np.isscalar(value):
            return float(value)
        if isinstance(value, (pd.Series, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        if hasattr(value, 'iloc') and len(value) > 0:  # DataFrame-like
            return float(value.iloc[-1, 0] if hasattr(value, 'shape') and value.shape[1] > 0 else value.iloc[-1])
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _extract_final_score(self, result: Any) -> float:
        """Extract and normalize final composite score."""
        score = self._extract_score(result)
        return float(np.clip(score, -1.0, 1.0))

    def _normalize_metric(self, value: Any, metric_name: str = None) -> Any:
        """Normalize metric to [-1, 1] range."""
        score = self._extract_score(value)
        if np.isnan(score):
            return np.nan
            
        # Enhanced normalization logic
        if metric_name:
            key = metric_name.upper()
            if "RSI" in key or "ADX" in key:
                return (score - 50.0) / 50.0
            if "VOL" in key or "ATR" in key:
                return np.tanh(score / 100.0)  # Scale volatility
                
        # Default normalization
        return np.tanh(score / 10.0)  # General scaling

    # -------------------------
    # Main Entry Point
    # -------------------------
    async def run(self, target: str, data: Any, **kwargs) -> Any:
        """
        Enhanced unified entry point.
        
        Args:
            target: Can be composite_module name, handler name, or metric name
            data: Input data for computation
        """
        # 1. Check if target is a composite module
        if target in self._composite_modules:
            return await self.compute_composite_module(target, data, **kwargs)
        
        # 2. Check handler mapping
        if target in self._handler_mapping:
            mapped_target = self._handler_mapping[target]
            if mapped_target and mapped_target in self._composite_modules:
                return await self.compute_composite_module(mapped_target, data, **kwargs)
        
        # 3. Check if target is a single metric
        func = self._resolve_metric_function(target)
        if func:
            return await self._call_callable_async(func, data, **kwargs)
        
        # 4. Check if target is a composite metric
        if target in self._composite_metrics:
            # Implement composite metric calculation if needed
            logger.warning(f"Direct composite_metrics not fully implemented: {target}")
            return np.nan
        
        raise ValueError(f"Unknown target: {target}")

    # -------------------------
    # Batch Operations
    # -------------------------
    async def run_batch(self, targets: List[str], data: Any, **kwargs) -> Dict[str, Any]:
        """Run multiple targets in batch."""
        results = {}
        
        async def _process_target(target: str):
            try:
                result = await self.run(target, data, **kwargs)
                results[target] = result
            except Exception as e:
                logger.error(f"Failed to process {target}: {e}")
                results[target] = np.nan
        
        await asyncio.gather(*[_process_target(target) for target in targets])
        return results

    # -------------------------
    # Utility Methods
    # -------------------------
    def get_available_composites(self) -> List[str]:
        """Get list of available composite modules."""
        return list(self._composite_modules.keys())
    
    def get_available_handlers(self) -> List[str]:
        """Get list of available handlers."""
        return list(self._handler_mapping.keys())
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get detailed information about a composite module."""
        if module_name in self._composite_modules:
            return self._composite_modules[module_name]
        return {}
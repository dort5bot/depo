# core.py
# Single-file production core engine — FULL final version
# Generated: 30.11.2025 v1

from __future__ import annotations
import importlib
import asyncio
import types
import logging
import math
import gc
import time
import ast
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import pandas as pd

# polars optional
try:
    import polars as pl
except Exception:
    pl = None

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
logger = logging.getLogger("analysis.core")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# -------------------------
# COMPOSITES / MACROS / METRICS TEST CONFIG
# (temporary — for development & debugging; move to analysis/composites.py later)
# -------------------------
METRICS = {# Bu yapı uzun vadede en iyi tasarıdır,
    "classical": [
        "ema", "sma", "macd", "rsi", "adx", "stochastic_oscillator", 
        "roc", "atr", "bollinger_bands", "value_at_risk",
        "conditional_value_at_risk", "max_drawdown", "oi_growth_rate",
        "oi_price_correlation", "spearman_corr", "cross_correlation", "futures_roc"
    ]
}
# Single test composite (keeps in core for rapid debug)
COMPOSITES = {
    "trend_momentum_composite": {
        "depends": ["ema", "macd", "rsi", "adx", "roc", "stochastic_oscillator"],
        "formula": "0.25*ema + 0.25*macd + 0.20*rsi + 0.10*adx + 0.10*roc + 0.10*stochastic_oscillator",
    }
}
# Single test macro
MACROS = {
    "core_macro": {
        "depends": ["trend_momentum_composite"],   # simple for test
        "formula": "0.5*trend_momentum_composite",
        "output": "core_score",
    }
}

# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------
class MetricExecutionError(Exception):
    pass

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def safe_nan():
    return float("nan")


def is_dataframe_like(x: Any) -> bool:
    return isinstance(x, pd.DataFrame) or (pl is not None and isinstance(x, pl.DataFrame))


def _trim_rows(df: Any, max_rows: int) -> Any:
    try:
        if isinstance(df, pd.DataFrame) and len(df) > max_rows:
            return df.iloc[-max_rows:].copy()
        if pl is not None and isinstance(df, pl.DataFrame) and df.height > max_rows:
            return df.tail(max_rows)
    except Exception:
        pass
    return df

# ------------------------------------------------------------
# Data adapter: convert raw input to requested model
# ------------------------------------------------------------

def adapt_data_model(data: Any, target: str) -> Any:
    target = (target or "pandas").lower()
    if target not in ("pandas", "polars", "numpy"):
        raise TypeError(f"Unsupported data model target: {target}")

    # pandas target
    if target == "pandas":
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if pl is not None and isinstance(data, pl.DataFrame):
            return data.to_pandas()
        if isinstance(data, (np.ndarray, list)):
            return pd.DataFrame(data)

    # polars target
    if target == "polars":
        if pl is None:
            raise TypeError("polars requested but library not installed")
        if isinstance(data, pl.DataFrame):
            return data.clone()
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data.copy())
        if isinstance(data, (np.ndarray, list)):
            return pl.DataFrame(data)

    # numpy target
    if target == "numpy":
        if isinstance(data, np.ndarray):
            return data.copy()
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(copy=True)
        if pl is not None and isinstance(data, pl.DataFrame):
            return data.to_numpy()

    raise TypeError(f"Cannot adapt data type {type(data)} to {target}")

# ------------------------------------------------------------
# Column groups resolver & preprocessor
# ------------------------------------------------------------

def resolve_required_columns(module: types.ModuleType, metric_name: str) -> List[str]:
    # Try modern API
    cfg = {}
    groups = {}
    try:
        cfg = getattr(module, "get_module_config")()
    except Exception:
        cfg = getattr(module, "_MODULE_CONFIG", {}) or {}

    try:
        groups = getattr(module, "get_column_groups")()
    except Exception:
        groups = getattr(module, "COLUMN_GROUPS", {}) or {}

    required_groups = cfg.get("required_groups", {}) if isinstance(cfg, dict) else {}
    group_name = required_groups.get(metric_name)
    if group_name is None:
        # legacy support
        rc = cfg.get("required_columns") if isinstance(cfg, dict) else None
        if rc and isinstance(rc, dict) and metric_name in rc:
            return list(rc[metric_name])
        return []

    cols = groups.get(group_name)
    if cols is None:
        raise MetricExecutionError(f"Module declares required group '{group_name}' but group not found")
    return list(cols)


def select_and_validate(df: Union[pd.DataFrame, "pl.DataFrame"], required_cols: List[str]) -> Tuple[Optional[Union[pd.DataFrame, "pl.DataFrame"]], List[str]]:
    if df is None:
        return None, list(required_cols)
    try:
        cols = list(df.columns)
    except Exception:
        cols = []
    present = [c for c in required_cols if c in cols]
    missing = [c for c in required_cols if c not in present]
    if len(present) == 0:
        return None, missing
    if isinstance(df, pd.DataFrame):
        return df.loc[:, present].copy(), missing
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.select(present), missing
    # as last resort, try numpy->pandas
    try:
        arr = np.asarray(df)
        return pd.DataFrame(arr, columns=present), missing
    except Exception:
        raise MetricExecutionError("Unable to select required columns from input data")

# ------------------------------------------------------------
# Execution helpers (sync/async)
# ------------------------------------------------------------

def _is_coroutine_callable(fn: Any) -> bool:
    return asyncio.iscoroutinefunction(fn)


def _exec_sync(func, data, params: Dict[str, Any]):
    return func(data, **params)


async def _exec_async(func, data, params: Dict[str, Any]):
    return await func(data, **params)


def safe_execute(func, execution_type: str, data, params: Dict[str, Any]):
    try:
        if execution_type == "async" or _is_coroutine_callable(func):
            # ensure running in event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                coro = _exec_async(func, data, params)
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                return fut.result()
            else:
                return asyncio.run(_exec_async(func, data, params))
        else:
            return _exec_sync(func, data, params)
    except Exception as e:
        raise MetricExecutionError(str(e)) from e

# ------------------------------------------------------------
# Scoring utils
# ------------------------------------------------------------

def score_minmax(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        return (value - lo) / (hi - lo) if hi != lo else float("nan")
    except Exception:
        return float("nan")


def score_zscore(value: float, mean: float, std: float) -> float:
    try:
        if std == 0:
            return float("nan")
        return (value - mean) / std
    except Exception:
        return float("nan")


def score_bounded(value: float, lower: float, upper: float, invert: bool = False) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        v = max(min(value, upper), lower)
        scaled = (v - lower) / (upper - lower) if upper != lower else float("nan")
        return 1.0 - scaled if invert else scaled
    except Exception:
        return float("nan")


def apply_score_profile(raw_value: Any, profile: Optional[dict]) -> float:
    if raw_value is None:
        return float("nan")
    if not profile:
        return float("nan") if isinstance(raw_value, float) and math.isnan(raw_value) else raw_value
    method = profile.get("method")
    params = profile.get("params", {})
    try:
        if method == "minmax":
            return score_minmax(float(raw_value), params.get("lo", 0.0), params.get("hi", 1.0))
        if method == "zscore":
            return score_zscore(float(raw_value), params.get("mean", 0.0), params.get("std", 1.0))
        if method == "bounded":
            return score_bounded(float(raw_value), params.get("lower", 0.0), params.get("upper", 1.0), params.get("invert", False))
        # fallback: return raw
        return float(raw_value) if isinstance(raw_value, (int, float)) else raw_value
    except Exception:
        return float("nan")

# ------------------------------------------------------------
# Module loader
# ------------------------------------------------------------

def load_metric_module(module_or_path: Union[str, types.ModuleType]) -> types.ModuleType:
    if isinstance(module_or_path, types.ModuleType):
        return module_or_path
    if isinstance(module_or_path, str):
        try:
            return importlib.import_module(module_or_path)
        except Exception as e:
            raise MetricExecutionError(f"Failed to import metric module '{module_or_path}': {e}") from e
    raise MetricExecutionError("module_or_path must be module or import path string")

# ------------------------------------------------------------
# Safe formula parser (AST-based) - supports + - * / and names
# ------------------------------------------------------------
#, ast.LParen, ast.RParen
ALLOWED_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Add,
    ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name,
    ast.Call, ast.Constant, ast.Mod, ast.FloorDiv
}

class FormulaEvaluator(ast.NodeVisitor):
    def __init__(self, context: Dict[str, float]):
        self.context = context

    def visit(self, node):
        if type(node) not in ALLOWED_AST_NODES:
            raise MetricExecutionError(f"Forbidden expression element: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            try:
                return left / right
            except Exception:
                return float("nan")
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise MetricExecutionError("Unsupported binary operator")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise MetricExecutionError("Unsupported unary operator")

    def visit_Name(self, node: ast.Name):
        return float(self.context.get(node.id, float("nan")))

    def visit_Constant(self, node: ast.Constant):
        return float(node.value)

    def visit_Num(self, node: ast.Num):
        return float(node.n)


def evaluate_formula_safe(formula: str, context: Dict[str, float]) -> float:
    if formula is None:
        return float("nan")
    try:
        tree = ast.parse(formula, mode="eval")
        ev = FormulaEvaluator(context)
        val = ev.visit(tree)
        if isinstance(val, (int, float)):
            return float(val)
        return float("nan")
    except MetricExecutionError:
        raise
    except Exception as e:
        logger.exception("Formula parse/eval failed")
        return float("nan")

# ------------------------------------------------------------
# Core metric runner (single metric)
# ------------------------------------------------------------

def run_metric(module_or_path: Union[str, types.ModuleType], metric_name: str, raw_data: Any, params: Optional[Dict[str, Any]] = None, *, max_rows: int = 100_000, enforce_real_data: bool = True) -> Dict[str, Any]:
    params = params or {}
    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    module = load_metric_module(module_or_path)

    try:
        module_cfg = getattr(module, "get_module_config")()
    except Exception:
        module_cfg = getattr(module, "_MODULE_CONFIG", {}) or {}

    data_model = module_cfg.get("data_model", "pandas")
    execution_type = module_cfg.get("execution_type", "sync")

    # required cols
    try:
        required_cols = resolve_required_columns(module, metric_name)
    except Exception as e:
        logger.exception("Resolve required columns failed")
        return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": [], "warnings": [str(e)], "debug": debug}

    # limit raw_data rows to protect memory
    try:
        if is_dataframe_like(raw_data) and max_rows is not None:
            raw_data = _trim_rows(raw_data, max_rows)
    except Exception:
        pass

    # adapt data
    try:
        adapted = adapt_data_model(raw_data, data_model)
    except Exception as e:
        msg = f"Data adapter error: {e}"
        logger.warning(msg)
        return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": required_cols, "warnings": [msg], "debug": debug}

    # select & validate
    selected, missing = select_and_validate(adapted, required_cols)
    if missing:
        msg = f"Missing columns for {metric_name}: {missing}"
        logger.warning(msg)
        warnings.append(msg)

    if selected is None:
        msg = f"No available data for metric {metric_name}; returning NaN (no synthetic fallback)."
        logger.info(msg)
        warnings.append(msg)
        _cleanup_refs(adapted)
        return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": missing, "warnings": warnings, "debug": debug}

    # check non-empty
    try:
        if isinstance(selected, pd.DataFrame):
            if selected.dropna(how="all").shape[0] == 0:
                msg = f"Selected columns for metric '{metric_name}' contain no non-NaN values"
                logger.info(msg)
                warnings.append(msg)
                _cleanup_refs(adapted, selected)
                return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": missing, "warnings": warnings, "debug": debug}
        elif pl is not None and isinstance(selected, pl.DataFrame):
            any_non_null = False
            for c in selected.columns:
                s = selected[c]
                if s.filter(pl.col(c).is_not_null()).height > 0:
                    any_non_null = True
                    break
            if not any_non_null:
                msg = f"Selected columns for metric '{metric_name}' contain no non-null values"
                logger.info(msg)
                warnings.append(msg)
                _cleanup_refs(adapted, selected)
                return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": missing, "warnings": warnings, "debug": debug}
    except Exception:
        warnings.append("Could not fully validate emptiness of selected data; proceeding")

    # get function
    try:
        func = getattr(module, "get_function")(metric_name)
        if func is None:
            raise AttributeError("function not found in module registry")
    except Exception as e:
        msg = f"Metric function lookup failed: {e}"
        logger.exception(msg)
        _cleanup_refs(adapted, selected)
        return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": missing, "warnings": [msg], "debug": debug}

    # execute safely
    raw_result = None
    try:
        raw_result = safe_execute(func, execution_type, selected, params)
    except Exception as e:
        msg = f"Metric execution failed: {e}"
        logger.exception(msg)
        warnings.append(msg)
        _cleanup_refs(adapted, selected)
        return {"ok": False, "value": safe_nan(), "score": safe_nan(), "missing_columns": missing, "warnings": warnings, "debug": debug}

    # apply score profile if provided
    score = float("nan")
    try:
        score_profile = (module_cfg.get("score_profile") or {})
        metric_profile = score_profile.get(metric_name) if isinstance(score_profile, dict) else None
        if metric_profile is None:
            metric_profile = getattr(module, "_SCORE_PROFILE", {}).get(metric_name, None)
        if metric_profile:
            score = apply_score_profile(raw_result, metric_profile)
        else:
            score = float(raw_result) if isinstance(raw_result, (int, float)) else float("nan")
    except Exception:
        warnings.append("Scoring failed; returning raw value")
        score = float("nan")

    _cleanup_refs(adapted, selected)

    return {"ok": True, "value": raw_result, "score": score, "missing_columns": missing, "warnings": warnings, "debug": debug}

# ------------------------------------------------------------
# cleanup
# ------------------------------------------------------------

def _cleanup_refs(*objs):
    try:
        for o in objs:
            try:
                del o
            except Exception:
                pass
    finally:
        gc.collect()

# ------------------------------------------------------------
# Composite & Macro engines
# ------------------------------------------------------------

def run_composite_def(comp_name: str, comp_def: dict, metric_results: Dict[str, Dict[str, Any]]) -> float:
    ctx: Dict[str, float] = {}
    for dep in comp_def.get("depends", []):
        mr = metric_results.get(dep)
        val = None
        if mr:
            val = mr.get("score")
        ctx[dep] = float(val) if isinstance(val, (int, float)) else float("nan")
    return evaluate_formula_safe(comp_def.get("formula"), ctx)


def run_macro_def(macro_def: dict, composite_results: Dict[str, float]) -> float:
    ctx: Dict[str, float] = {}
    for dep in macro_def.get("depends", []):
        ctx[dep] = float(composite_results.get(dep, float("nan")))
    return evaluate_formula_safe(macro_def.get("formula"), ctx)

# ------------------------------------------------------------
# Master pipeline (synchronous interface)
# ------------------------------------------------------------

def run_full_pipeline(symbol: str, raw_df: Any, requested_scores: List[str], *, METRICS: Dict[str, Tuple[str, str]], COMPOSITES: Dict[str, dict], MACROS: Dict[str, dict], binance_aggregator: Optional[Any] = None, max_rows: int = 100_000) -> Dict[str, Any]:
    """
    symbol: e.g. 'BTCUSDT'
    raw_df: raw data (pandas/polars/numpy) for price/time series. If None, core may call binance_aggregator when provided.
    requested_scores: list of composite names to compute (e.g. ['trend_momentum_composite'])
    METRICS: mapping from metric short name -> (module.import.path, metric_function_name)
    COMPOSITES / MACROS: dicts as described in spec
    Returns dict with metrics, composites, macros
    """
    # If raw_df not provided and binance_aggregator given, try to fetch OHLCV by default
    if raw_df is None and binance_aggregator is not None:
        # Attempt simple fetch pattern; callers can override
        try:
            # This code assumes BinanceAggregator has an async get_public_data method
            async def _fetch():
                return await binance_aggregator.get_public_data("klines", symbol=symbol, interval="1h", limit=500)
            raw_df = asyncio.run(_fetch())
        except Exception as e:
            logger.warning(f"Failed to fetch data from BinanceAggregator: {e}")
            raw_df = None

    # run required metrics
    metric_results: Dict[str, Dict[str, Any]] = {}

    # Determine all dependent metrics for requested composites
    needed_metrics: Set[str] = set()
    for comp in requested_scores:
        comp_def = COMPOSITES.get(comp)
        if not comp_def:
            logger.warning(f"Requested composite not found: {comp}")
            continue
        for dep in comp_def.get("depends", []):
            needed_metrics.add(dep)

    # Expand if composites depend on composites (rare) - currently assume direct metric deps

    # Execute metrics
    for m in sorted(needed_metrics):
        try:
            module_path, fn_name = METRICS[m]
        except Exception:
            logger.error(f"METRICS mapping missing for metric: {m}")
            metric_results[m] = {"ok": False, "value": safe_nan(), "score": safe_nan(), "warnings": ["mapping missing"]}
            continue
        try:
            res = run_metric(module_path, fn_name, raw_df, {}, max_rows=max_rows)
            metric_results[m] = res
        except Exception as e:
            logger.exception(f"Metric {m} failed: {e}")
            metric_results[m] = {"ok": False, "value": safe_nan(), "score": safe_nan(), "warnings": [str(e)]}

    # Compute composites
    composite_results: Dict[str, float] = {}
    for comp_name, comp_def in COMPOSITES.items():
        try:
            val = run_composite_def(comp_name, comp_def, metric_results)
            composite_results[comp_name] = val
        except Exception as e:
            logger.exception(f"Composite {comp_name} eval failed: {e}")
            composite_results[comp_name] = float("nan")

    # Compute macros
    macro_results: Dict[str, float] = {}
    for macro_name, macro_def in MACROS.items():
        try:
            val = run_macro_def(macro_def, composite_results)
            macro_results[macro_name] = val
        except Exception as e:
            logger.exception(f"Macro {macro_name} eval failed: {e}")
            macro_results[macro_name] = float("nan")

    _cleanup_refs(raw_df)
    return {
        "metrics": metric_results,
        "composites": composite_results,
        "macros": macro_results,
    }

# ------------------------------------------------------------
# Convenience async wrapper for handler-level integration
# ------------------------------------------------------------

async def run_full_pipeline_async(symbol: str, raw_df: Any, requested_scores: List[str], *, METRICS: Dict[str, Tuple[str, str]], COMPOSITES: Dict[str, dict], MACROS: Dict[str, dict], binance_aggregator: Optional[Any] = None, max_rows: int = 100_000) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_full_pipeline, symbol, raw_df, requested_scores, METRICS, COMPOSITES, MACROS, binance_aggregator, max_rows)

# ------------------------------------------------------------
# End of core.py
# ------------------------------------------------------------

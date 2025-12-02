# analysis/a_core.py
# Single-file production core engine - Simplified Version
# Generated: 02/12

from __future__ import annotations
import logging
import math
import ast
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

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
# COMPOSITES / MACROS maps
# -------------------------
COMPOSITES = {
    "trend": {
        "depends": ["ema", "macd", "rsi", "adx", "roc", "stochastic_oscillator"],
        "formula": "0.25*ema + 0.25*macd + 0.20*rsi + 0.10*adx + 0.10*roc + 0.10*stochastic_oscillator",
    },
    "vol": {
        "depends": ["atr", "historical_volatility", "garch_1_1", "hurst_exponent"],
        "formula": "0.33*atr + 0.28*historical_volatility + 0.22*garch_1_1 + 0.17*hurst_exponent",
    },
}

MACROS = {
    "core": {
        "depends": ["trend", "vol"],
        "formula": "0.5*trend + 0.5*vol"
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

# ------------------------------------------------------------
# Safe formula parser (AST-based) - supports + - * / and names
# ------------------------------------------------------------
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
# Score to Metric Resolver
# ------------------------------------------------------------
def resolve_scores_to_metrics(
    requested_scores: List[str], 
    COMPOSITES: Dict[str, dict] = None,
    MACROS: Dict[str, dict] = None
) -> Dict[str, List[str]]:
    """
    İstenen skorlardan gerekli metrikleri çözer.
    
    Args:
        requested_scores: ["trend", "vol", "core"]
        COMPOSITES: Composite tanımları
        MACROS: Macro tanımları
    
    Returns:
        {"trend": ["ema", "macd", ...], "vol": ["atr", ...]}
    """
    COMPOSITES = COMPOSITES or {}
    MACROS = MACROS or {}
    
    score_to_metrics = {}
    
    for score_name in requested_scores:
        metrics = []
        
        # Önce composite'leri kontrol et
        if score_name in COMPOSITES:
            metrics.extend(COMPOSITES[score_name].get("depends", []))
        
        # Sonra macro'ları kontrol et
        elif score_name in MACROS:
            # Macro'nun depends'i composite adları içerir
            for dep in MACROS[score_name].get("depends", []):
                if dep in COMPOSITES:
                    metrics.extend(COMPOSITES[dep].get("depends", []))
        
        # Benzersiz metrikleri al ve sırala (debug için daha okunaklı)
        score_to_metrics[score_name] = sorted(list(set(metrics)))
    
    return score_to_metrics

# ------------------------------------------------------------
# Metric Calculation from Definitions
# ------------------------------------------------------------
async def calculate_metrics(data: pd.DataFrame, metric_defs: Dict) -> Dict[str, Any]:
    """
    Metrik tanımlarını kullanarak hesaplama yap
    
    Args:
        data: Ham veri DataFrame
        metric_defs: MetricResolver'dan alınan metrik tanımları
    
    Returns:
        Hesaplanan metrik değerleri
    """
    results = {}
    
    for metric_name, def_info in metric_defs.items():
        if def_info is None:
            results[metric_name] = float("nan")
            continue
        
        try:
            # 1. Gerekli kolonları kontrol et
            required_cols = def_info.get("columns", [])
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {metric_name}: {missing_cols}")
                results[metric_name] = float("nan")
                continue
            
            # 2. Sadece gerekli kolonları seç
            subset = data[required_cols].copy()
            
            # 3. Fonksiyonu çalıştır
            func = def_info.get("function")
            if func is None:
                logger.warning(f"No function for {metric_name}")
                results[metric_name] = float("nan")
                continue
                
            params = def_info.get("params", {})
            
            # 4. Fonksiyonu çağır
            result = func(subset, **params)
            
            # 5. Sonucu sakla
            results[metric_name] = result
            
        except Exception as e:
            logger.error(f"Failed to calculate {metric_name}: {e}")
            results[metric_name] = float("nan")
    
    return results

# ------------------------------------------------------------
# Composite Calculation
# ------------------------------------------------------------
def calculate_composite_scores(
    metric_results: Dict[str, Any], 
    COMPOSITES: Dict[str, dict]
) -> Dict[str, float]:
    """
    Metrik sonuçlarını composite formüllerle birleştirir.
    
    Args:
        metric_results: Hesaplanan metrik değerleri
        COMPOSITES: Composite tanımları
    
    Returns:
        {"trend": 0.75, "vol": 0.60}
    """
    composite_scores = {}
    
    for comp_name, comp_def in COMPOSITES.items():
        try:
            # Metric değerlerini al
            ctx = {}
            for dep in comp_def.get("depends", []):
                value = metric_results.get(dep, float("nan"))
                try:
                    ctx[dep] = float(value) if value is not None else float("nan")
                except (TypeError, ValueError):
                    ctx[dep] = float("nan")
            
            # Formülü hesapla
            formula = comp_def.get("formula", "")
            if not formula:
                logger.warning(f"Composite {comp_name} has no formula")
                score = float("nan")
            else:
                score = evaluate_formula_safe(formula, ctx)
            
            composite_scores[comp_name] = score
            
        except Exception as e:
            logger.warning(f"Composite {comp_name} calculation failed: {e}")
            composite_scores[comp_name] = float("nan")
    
    return composite_scores

# ------------------------------------------------------------
# Macro Calculation
# ------------------------------------------------------------
def calculate_macro_scores(
    composite_results: Dict[str, float],
    MACROS: Dict[str, dict]
) -> Dict[str, float]:
    """
    Composite sonuçlarını macro formüllerle birleştirir.
    
    Args:
        composite_results: {"trend": 0.75, "vol": 0.60}
        MACROS: Macro tanımları
    
    Returns:
        {"core": 0.675}
    """
    macro_scores = {}
    
    for macro_name, macro_def in MACROS.items():
        try:
            # Composite değerlerini al
            ctx = {}
            for dep in macro_def.get("depends", []):
                ctx[dep] = composite_results.get(dep, float("nan"))
            
            # Formülü hesapla
            formula = macro_def.get("formula", "")
            if not formula:
                logger.warning(f"Macro {macro_name} has no formula")
                score = float("nan")
            else:
                score = evaluate_formula_safe(formula, ctx)
            
            macro_scores[macro_name] = score
            
        except Exception as e:
            logger.warning(f"Macro {macro_name} calculation failed: {e}")
            macro_scores[macro_name] = float("nan")
    
    return macro_scores

# ------------------------------------------------------------
# Master Pipeline
# ------------------------------------------------------------
async def run_full_pipeline(
    symbol: str, 
    requested_scores: List[str], 
    raw_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Ana pipeline akışı:
    1. Score'lardan metrik isimlerini bul
    2. MetricResolver'dan metrik TANIMLARINI al
    3. Ham veriden metrikleri hesapla
    4. Composite/Macro hesapla
    
    Args:
        symbol: Sembol adı (örn: "BTCUSDT")
        requested_scores: İstenen skorlar (örn: ["trend", "vol", "core"])
        raw_data: Ham veri DataFrame
    
    Returns:
        Hesaplanan skorlar
    """
    logger.info(f"Starting pipeline for {symbol}")
    
    # 1. Score'lardan metrik isimlerini bul
    metric_map = resolve_scores_to_metrics(requested_scores, COMPOSITES, MACROS)
    all_metrics = []
    for metrics in metric_map.values():
        all_metrics.extend(metrics)
    all_metrics = list(set(all_metrics))
    
    logger.info(f"Required metrics: {all_metrics}")
    
    # 2. MetricResolver'dan metrik TANIMLARINI al
    # NOT: MetricResolver import'u burada yapılıyor, dışarıda olabilir
    try:
        from analysis.metricresolver import MetricResolver
        # Burada METRICS tanımını almak gerekebilir
        # Örnek: METRICS = {...} veya MetricResolver'ın kendi içinden
        metric_resolver = MetricResolver()  # Parametreler gerekiyorsa
        
        metric_defs = {}
        for metric_name in all_metrics:
            try:
                # MetricResolver'dan TANIM al (veri değil!)
                def_info = metric_resolver.resolve_metric_definition(metric_name)
                metric_defs[metric_name] = def_info
            except Exception as e:
                logger.error(f"Failed to resolve {metric_name}: {e}")
                metric_defs[metric_name] = None
                
    except ImportError as e:
        logger.error(f"Cannot import MetricResolver: {e}")
        return {"error": f"MetricResolver not available: {e}"}
    except Exception as e:
        logger.error(f"Failed to initialize MetricResolver: {e}")
        return {"error": f"MetricResolver init failed: {e}"}
    
    # 3. Ham veriden metrikleri hesapla
    metric_results = await calculate_metrics(raw_data, metric_defs)
    
    # 4. Composite hesapla (eğer istenen skorlar arasında varsa)
    composite_scores = {}
    composite_scores_to_calc = [s for s in requested_scores if s in COMPOSITES]
    if composite_scores_to_calc:
        composite_scores = calculate_composite_scores(metric_results, {
            k: v for k, v in COMPOSITES.items() if k in composite_scores_to_calc
        })
    
    # 5. Macro hesapla (eğer istenen skorlar arasında varsa)
    macro_scores = {}
    macro_scores_to_calc = [s for s in requested_scores if s in MACROS]
    if macro_scores_to_calc:
        macro_scores = calculate_macro_scores(composite_scores, {
            k: v for k, v in MACROS.items() if k in macro_scores_to_calc
        })
    
    # 6. Tüm sonuçları birleştir
    final_scores = {}
    final_scores.update(metric_results)
    final_scores.update(composite_scores)
    final_scores.update(macro_scores)
    
    # Sadece istenen skorları filtrele
    result = {score: final_scores.get(score, float("nan")) for score in requested_scores}
    
    logger.info(f"Pipeline completed for {symbol}. Scores: {result}")
    
    return {
        "symbol": symbol,
        "scores": result,
        "metrics_calculated": len(metric_results),
        "composites_calculated": len(composite_scores),
        "macros_calculated": len(macro_scores)
    }

# ------------------------------------------------------------
# Sync wrapper for async pipeline
# ------------------------------------------------------------
def run_full_pipeline_sync(
    symbol: str, 
    requested_scores: List[str], 
    raw_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Senkron wrapper for pipeline execution.
    
    Args:
        symbol: Sembol adı
        requested_scores: İstenen skorlar
        raw_data: Ham veri DataFrame
    
    Returns:
        Hesaplanan skorlar
    """
    import asyncio
    
    try:
        # Mevcut event loop'u kontrol et
        loop = asyncio.get_running_loop()
        # Event loop varsa, async fonksiyonu çalıştır
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                run_full_pipeline(symbol, requested_scores, raw_data)
            )
            return future.result()
            
    except RuntimeError:
        # Event loop yoksa, yeni oluştur
        return asyncio.run(run_full_pipeline(symbol, requested_scores, raw_data))
    except Exception as e:
        logger.error(f"Pipeline sync execution failed: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------
# End of core.py
# ------------------------------------------------------------
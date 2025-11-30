
EKLENEN DOSYAYI BU ŞABLON DOSYAYA GÖRE TAM DÖNÜŞÜMÜNÜ SAĞLA
TAM KODU VER

"""
analysis/metrics/[module_name].py
date: 30.11.2025 19:25
Enhanced standard template
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "ohlc": ["open", "high", "low", "close"],
    "close_only": ["close"],
    "ohlcv": ["open", "high", "low", "close", "volume"],
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "technical",

    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "abc_metric": "ohlc",
        "klm_metric": "close_only",
    },

    # opsiyonel
    "score_profile": {
        "abc_metric": {
            "method": "minmax",
            "range": [-1, 1],
            "direction": "positive"
        },
        "klm_metric": {
            "method": "zscore",
            "direction": "negative"
        }
    }
}


# ==================== PURE FUNCTIONS ====================
def abc_metric(data, **params):
    # data: only open/high/low/close
    return ...

def klm_metric(data, **params):
    # data: only close
    return ...

# ==================== REGISTRY ====================
_METRICS = {
    "abc_metric": abc_metric,
    "klm_metric": klm_metric,
}

def get_metrics() -> List[str]:
    return list(_METRICS.keys())

def get_function(metric_name: str):
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    return _MODULE_CONFIG.copy()

def get_column_groups() -> Dict[str, List[str]]:
    return COLUMN_GROUPS.copy()

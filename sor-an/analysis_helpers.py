# analysis/analysis_helpers.py
"""
Merkezi Analysis Helper Sınıfı
Tüm analiz modülleri için ortak fonksiyonlar
Async yapı
"""
# analysis/analysis_helpers.py

import os
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from pydantic import BaseModel, ValidationError
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisOutput(BaseModel):
    """Standart analiz çıktı şeması"""
    score: float
    signal: str
    confidence: Optional[float] = 0.0
    components: Dict[str, float]
    explain: str
    timestamp: float
    module: str
    
    class Config:
        extra = "forbid"


# ✅ YENİ: UTILITY FUNCTIONS CLASS'I
class AnalysisUtilities:
    """
    Saf utility fonksiyonları - stateless
    Matematiksel, validation, normalization işlemleri
    """
    
    @staticmethod
    def validate_output(output: Dict[str, Any]) -> bool:
        """Output formatını validate et"""
        required_keys = {'score', 'signal', 'components', 'timestamp', 'module'}
        return all(key in output for key in required_keys)
    
    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Skoru 0-1 aralığına normalize et"""
        return max(min_val, min(max_val, score))
    
    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Ağırlıkları normalize et (toplam 1.0 yap)"""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0/len(weights) for k in weights}
        return {k: v/total for k, v in weights.items()}
    

    @staticmethod
    def calculate_weighted_average(
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """Ağırlıklı ortalama hesapla"""
        # ✅ Basit ve etkili validation
        if not scores or not weights:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for name, score in scores.items():
            weight = weights.get(name, 0.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        raw_score = total_score / total_weight
        return AnalysisUtilities.normalize_score(raw_score)
        
    @staticmethod
    def create_fallback_output(module_name: str, reason: str = "Error") -> Dict[str, Any]:
        """Standart fallback output oluştur"""
        return {
            "score": 0.5,
            "signal": "neutral",
            "confidence": 0.0,
            "components": {},
            "explain": f"Fallback: {reason}",
            "timestamp": time.time(),
            "module": module_name,
            "fallback": True
        }
    
    @staticmethod
    def validate_score_dict(scores: Dict[str, float]) -> bool:
        """Score dict formatını kontrol et"""
        if not isinstance(scores, dict):
            return False
        return all(isinstance(k, str) and isinstance(v, (int, float)) 
                  for k, v in scores.items())
    
    @staticmethod
    def exponential_smoothing(values: List[float], alpha: float = 0.3) -> float:
        """Üstel düzeltme ile smoothing"""
        if not values:
            return 0.0
        
        result = values[0]
        for value in values[1:]:
            result = alpha * value + (1 - alpha) * result
        return result
    
    @staticmethod
    def calculate_confidence(scores: List[float], min_samples: int = 3) -> float:
        """Skorların tutarlılığına göre confidence hesapla"""
        if len(scores) < min_samples:
            return len(scores) / min_samples * 0.5
        
        if len(scores) == 1:
            return 0.3
            
        variance = np.var(scores) if len(scores) > 1 else 0.0
        consistency = 1.0 - min(variance, 1.0)
        return consistency * 0.8 + 0.2


# ✅ MEVCUT AnalysisHelpers Class'ı (Güncellenmiş)
class AnalysisHelpers:
    """Stateful helper sınıfı - performance tracking, config loading vb."""
    
    def __init__(self):
        self.performance_metrics: Dict[str, List] = {}
        self.utils = AnalysisUtilities()  # ✅ Utility fonksiyonlarına erişim
    
    # === STATIC METHODS ===
    @staticmethod
    def get_module_key(module_file: str) -> str:
        return os.path.splitext(os.path.basename(module_file))[0].lower()
    
    @staticmethod
    def get_circuit_breaker_key(module_file: str) -> str:
        return f"cb_{AnalysisHelpers.get_module_key(module_file)}"
    
    @staticmethod
    def get_module_instance_key(module_file: str) -> str:
        base_key = AnalysisHelpers.get_module_key(module_file)
        return f"{base_key}_{hashlib.md5(module_file.encode()).hexdigest()[:8]}"
    
    @staticmethod
    def resolve_analysis_path(relative_path: str) -> Path:
        base_path = Path(__file__).parent
        resolved = (base_path / relative_path).resolve()
        
        if not str(resolved).startswith(str(base_path)):
            raise PermissionError(f"Unauthorized path access: {resolved}")
        return resolved
    
    @staticmethod
    def get_timestamp() -> float:
        return time.time()
    
    @staticmethod
    def get_iso_timestamp() -> str:
        return datetime.now().isoformat()
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            return f"{seconds/60:.1f}m"
    
    # === INSTANCE METHODS ===
    def update_performance_metrics(self, key: str, value: float, max_history: int = 1000):
        """Performance metriklerini güncelle"""
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []
        
        self.performance_metrics[key].append(value)
        
        if len(self.performance_metrics[key]) > max_history:
            self.performance_metrics[key] = self.performance_metrics[key][-max_history//2:]
    
    def load_config_safe(self, module_name: str, default_config: Dict) -> Dict:
        """Güvenli config yükleme"""
        try:
            config_module = f"analysis.config.c_{module_name}"
            module = __import__(config_module, fromlist=['CONFIG'])
            return getattr(module, 'CONFIG', default_config)
        except ImportError:
            logger.warning(f"Config not found for {module_name}, using defaults")
            return default_config
    
    def calculate_weights(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Instance versiyonu - utility class'ını kullanır"""
        return self.utils.calculate_weighted_average(scores, weights)  # ✅ Utility class'ından


# Kullanım kolaylığı için instance'lar
analysis_helpers = AnalysisHelpers()
utility_functions = AnalysisUtilities()  # ✅ Yeni utility instance'ı
# analysis/config/cm_loader.py
"""
Merkezi config yükleyici ve yönetici - Analysis Helpers Uyumlu
"""

import importlib
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# ✅ ANALYSIS_HELPERS IMPORT
from analysis.analysis_helpers import AnalysisHelpers

# ✅ YEREL IMPORT
from .cm_base import BaseModuleConfig

logger = logging.getLogger(__name__)

class ConfigManager:
    """Config yöneticisi - singleton pattern - Analysis Helpers Uyumlu"""
    
    _instance = None
    _configs: Dict[str, BaseModuleConfig] = {}
    _config_metadata: Dict[str, Dict[str, Any]] = {}  # ✅ Config metadata tracking
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configs:
            self._load_all_configs()
    
    def _load_all_configs(self):
        """Tüm modül config'lerini yükle - ANALYSIS_HELPERS UYUMLU"""
        # ✅ ANALYSIS_HELPERS UYUMLU MODÜL LİSTESİ
        modules_to_load = [
            'trend', 'volatility', 'sentiment', 'corr', 'deriv', 'micro',
            'onchain', 'order', 'portalloc', 'risk'
        ]
        
        successful_loads = 0
        failed_loads = 0
        
        for module_name in modules_to_load:
            try:
                config = self._load_single_config(module_name)
                if config:
                    self._configs[module_name] = config
                    successful_loads += 1
                    logger.info(f"✅ Config loaded: {module_name}")
                else:
                    failed_loads += 1
                    logger.warning(f"⚠️ Config load failed: {module_name}")
                    
            except Exception as e:
                failed_loads += 1
                logger.error(f"❌ Config load error for {module_name}: {e}")
        
        # ✅ PERFORMANCE TRACKING
        AnalysisHelpers.update_performance_metrics(
            self._config_metadata,
            'load_stats',
            successful_loads,
            max_history=100
        )
        
        logger.info(f"Config loading completed: {successful_loads} successful, {failed_loads} failed")
    
    def _load_single_config(self, module_name: str) -> Optional[BaseModuleConfig]:
        """Tekil config yükle - ANALYSIS_HELPERS UYUMLU"""
        load_start_time = AnalysisHelpers.get_timestamp()
        
        try:
            # ✅ GÜVENLİ MODÜL IMPORT
            config_module_name = f"analysis.config.c_{module_name}"
            module = importlib.import_module(config_module_name)
            
            # ✅ CONFIG OBJESİNİ AL
            config_obj = getattr(module, "CONFIG", None)
            if config_obj is None:
                logger.warning(f"CONFIG not found in {config_module_name}")
                return None
            
            # ✅ CONFIG VALIDATION
            if not self._validate_config(config_obj, module_name):
                logger.warning(f"Config validation failed for {module_name}")
                return None
            
            # ✅ METADATA KAYDET
            self._config_metadata[module_name] = {
                'load_time': AnalysisHelpers.get_timestamp() - load_start_time,
                'load_timestamp': AnalysisHelpers.get_timestamp(),
                'version': getattr(config_obj, 'version', 'unknown'),
                'module_name': getattr(config_obj, 'module_name', module_name)
            }
            
            return config_obj
            
        except ImportError as e:
            logger.warning(f"Config module not found: {module_name} - {e}")
            return self._get_fallback_config(module_name)
        except AttributeError as e:
            logger.error(f"Config attribute error for {module_name}: {e}")
            return self._get_fallback_config(module_name)
        except Exception as e:
            logger.error(f"Unexpected error loading config {module_name}: {e}")
            return self._get_fallback_config(module_name)
    
    def _validate_config(self, config_obj: Any, module_name: str) -> bool:
        """Config objesini validate et - ANALYSIS_HELPERS UYUMLU"""
        # ✅ BASE MODULE CONFIG KONTROLÜ
        if not isinstance(config_obj, BaseModuleConfig):
            logger.warning(f"Config for {module_name} is not BaseModuleConfig instance")
            return False
        
        # ✅ GEREKLİ ALANLAR KONTROLÜ
        required_fields = ['module_name', 'version']
        for field in required_fields:
            if not hasattr(config_obj, field):
                logger.warning(f"Config for {module_name} missing required field: {field}")
                return False
        
        # ✅ VERSION VALIDATION
        version = getattr(config_obj, 'version', '')
        if not version or not isinstance(version, str):
            logger.warning(f"Invalid version in config for {module_name}: {version}")
            return False
        
        return True
    
    def _get_fallback_config(self, module_name: str) -> Optional[BaseModuleConfig]:
        """Fallback config oluştur - ANALYSIS_HELPERS UYUMLU"""
        try:
            logger.info(f"Creating fallback config for {module_name}")
            
            # ✅ ANALYSIS_HELPERS UYUMLU FALLBACK CONFIG
            fallback_config = BaseModuleConfig(
                module_name=module_name,
                file=f"analysis_{module_name}.py",
                config=f"c_{module_name}.py",
                command=f"/api/analysis/{module_name}",
                api_type="public",
                job_type="batch",
                parallel_mode="async",
                output_type="score",
                objective=f"Fallback analysis for {module_name}",
                maintainer="system",
                description=f"Fallback configuration for {module_name}",
                version="1.0.0-fallback"
            )
            
            # ✅ METADATA KAYDET
            self._config_metadata[f"fallback_{module_name}"] = {
                'load_time': 0.0,
                'load_timestamp': AnalysisHelpers.get_timestamp(),
                'version': 'fallback',
                'module_name': module_name,
                'is_fallback': True
            }
            
            return fallback_config
            
        except Exception as e:
            logger.error(f"Failed to create fallback config for {module_name}: {e}")
            return None
    
    def get_config(self, module_name: str) -> Optional[BaseModuleConfig]:
        """Modül config'ini getir - ANALYSIS_HELPERS UYUMLU"""
        config = self._configs.get(module_name)
        
        if config is None:
            logger.warning(f"Config not found for {module_name}, attempting reload")
            # ✅ OTOMATİK RELOAD DENEMESİ
            reloaded_config = self._load_single_config(module_name)
            if reloaded_config:
                self._configs[module_name] = reloaded_config
                return reloaded_config
            else:
                logger.error(f"Config reload failed for {module_name}")
                return None
        
        return config
    
    def get_all_configs(self) -> Dict[str, BaseModuleConfig]:
        """Tüm config'leri getir - ANALYSIS_HELPERS UYUMLU"""
        return self._configs.copy()
    
    def get_config_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Config metadata'sını getir"""
        return self._config_metadata.copy()
    
    def update_config(self, module_name: str, **updates):
        """Config güncelleme (validation ile) - ANALYSIS_HELPERS UYUMLU"""
        if module_name not in self._configs:
            logger.warning(f"Cannot update non-existent config: {module_name}")
            return
        
        try:
            current_config = self._configs[module_name]
            
            # ✅ MEVCUT CONFIG'İ DICT'E ÇEVİR
            if hasattr(current_config, 'dict'):
                config_dict = current_config.dict()
            else:
                config_dict = current_config.__dict__.copy()
            
            # ✅ UPDATELERİ UYGULA
            config_dict.update(updates)
            
            # ✅ YENİ INSTANCE OLUŞTUR (VALIDATION İÇİN)
            config_class = type(current_config)
            updated_config = config_class(**config_dict)
            
            # ✅ VALIDATION
            if self._validate_config(updated_config, module_name):
                self._configs[module_name] = updated_config
                
                # ✅ METADATA GÜNCELLE
                if module_name in self._config_metadata:
                    self._config_metadata[module_name].update({
                        'last_update': AnalysisHelpers.get_timestamp(),
                        'update_count': self._config_metadata[module_name].get('update_count', 0) + 1
                    })
                
                logger.info(f"Config updated successfully: {module_name}")
            else:
                logger.error(f"Config validation failed after update for {module_name}")
                
        except Exception as e:
            logger.error(f"Config update failed for {module_name}: {e}")
    
    def reload_config(self, module_name: str) -> bool:
        """Config'i yeniden yükle - ANALYSIS_HELPERS UYUMLU"""
        try:
            logger.info(f"Reloading config: {module_name}")
            
            # ✅ MODÜLÜ RELOAD ET
            config_module_name = f"analysis.config.c_{module_name}"
            if config_module_name in importlib.sys.modules:
                importlib.reload(importlib.sys.modules[config_module_name])
            
            # ✅ YENİDEN YÜKLE
            new_config = self._load_single_config(module_name)
            if new_config and self._validate_config(new_config, module_name):
                self._configs[module_name] = new_config
                logger.info(f"Config reloaded successfully: {module_name}")
                return True
            else:
                logger.error(f"Config reload validation failed: {module_name}")
                return False
                
        except Exception as e:
            logger.error(f"Config reload failed for {module_name}: {e}")
            return False
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Config istatistiklerini getir - ANALYSIS_HELPERS UYUMLU"""
        total_configs = len(self._configs)
        fallback_configs = sum(1 for meta in self._config_metadata.values() 
                              if meta.get('is_fallback', False))
        
        load_times = [meta.get('load_time', 0) for meta in self._config_metadata.values()]
        
        return {
            "total_configs": total_configs,
            "fallback_configs": fallback_configs,
            "successful_configs": total_configs - fallback_configs,
            "avg_load_time": np.mean(load_times) if load_times else 0,
            "max_load_time": max(load_times) if load_times else 0,
            "min_load_time": min(load_times) if load_times else 0,
            "timestamp": AnalysisHelpers.get_timestamp()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Config manager health check - ANALYSIS_HELPERS UYUMLU"""
        stats = self.get_config_stats()
        
        health_status = "healthy"
        if stats["fallback_configs"] > stats["total_configs"] * 0.5:  # %50'den fazla fallback
            health_status = "degraded"
        elif stats["total_configs"] == 0:
            health_status = "unhealthy"
        
        return {
            "service": "config_manager",
            "status": health_status,
            "stats": stats,
            "timestamp": AnalysisHelpers.get_timestamp(),
            "config_modules_loaded": list(self._configs.keys())
        }

# ✅ GLOBAL INSTANCE WITH ERROR HANDLING
try:
    config_manager = ConfigManager()
    logger.info("ConfigManager initialized successfully")
except Exception as e:
    logger.error(f"ConfigManager initialization failed: {e}")
    # ✅ FALLBACK EMPTY MANAGER
    class FallbackConfigManager:
        def get_config(self, module_name: str):
            return None
        def get_all_configs(self):
            return {}
        def health_check(self):
            return {"status": "unhealthy", "error": "Initialization failed"}
    
    config_manager = FallbackConfigManager()
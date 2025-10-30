# analysis/module_loader.py
# -*- coding: utf-8 -*-

import importlib
import threading
import yaml
from pathlib import Path
from types import ModuleType
from typing import Dict, Any, List, Optional

from analysis.metric_engine import MetricEngine

LOCK = threading.Lock()

class ModuleLoader:
    """
    Hybrid ModuleLoader:
    - Eğer modules/{file}.py varsa import eder
    - Yoksa MetricEngine üzerinden YAML-only modda çalışır
    """

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.modules: Dict[str, Any] = {}      # module instance or virtual executor
        self.configs: Dict[str, Any] = {}
        self._load_registry()

    def _load_registry(self):
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Module registry not found: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.registry = yaml.safe_load(f)
        if "modules" not in self.registry:
            raise ValueError("Invalid module registry: 'modules' key missing")

    def load_module(self, name: str):
        with LOCK:
            mod_info = next((m for m in self.registry["modules"] if m["name"] == name), None)
            if not mod_info:
                raise ValueError(f"Module '{name}' not found in registry")

            if name in self.modules:
                return self.modules[name]

            py_path = Path(__file__).resolve().parent.parent / "modules" / mod_info["file"]

            # 1️⃣ Eğer fiziksel .py dosyası varsa normal import yap
            if py_path.exists():
                try:
                    module = importlib.import_module(f"modules.{mod_info['file'].replace('.py','')}")
                except ModuleNotFoundError as e:
                    raise ImportError(f"Cannot import module {mod_info['file']}: {e}")

                config_class_name = mod_info.get("config_class")
                if config_class_name:
                    config_class = getattr(module, config_class_name, None)
                    if config_class is None:
                        raise AttributeError(f"Config class {config_class_name} not found in {mod_info['file']}")
                    config_file = mod_info.get("config_file")
                    config = config_class(config_file) if config_file != "inline" else config_class()
                else:
                    config = None

                self.modules[name] = module
                self.configs[name] = config
                return module

            # 2️⃣ Eğer .py yoksa, MetricEngine temelli “virtual module” oluştur
            else:
                engine = MetricEngine()

                class VirtualModule:
                    """Dynamic YAML-only virtual module"""
                    def __init__(self, module_name: str, metrics: Dict[str, Any]):
                        self.module_name = module_name
                        self.metrics = metrics
                        self.engine = engine

                    async def run(self, *args, **kwargs):
                        results = {}
                        for metric_group, metric_list in self.metrics.items():
                            for metric_name in metric_list:
                                result = await self.engine.compute_async(
                                    self.module_name,
                                    metric_name,
                                    func=lambda *a, **kw: 0.0,  # placeholder or dynamic resolver
                                    *args,
                                    use_last_valid=True,
                                    default=0.0,
                                    **kwargs
                                )
                                results[metric_name] = result
                        return results

                vmod = VirtualModule(name, mod_info.get("metrics", {}))
                self.modules[name] = vmod
                return vmod

    def get_module_config(self, name: str) -> Optional[Any]:
        with LOCK:
            return self.configs.get(name)

    def list_modules(self) -> List[str]:
        return [m["name"] for m in self.registry["modules"]]

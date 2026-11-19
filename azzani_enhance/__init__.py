"""AZZANI Image Enhancements — Open-source hybrid AI & float32 photometric image enhancer."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.0"
__license__ = "MIT"
__author__ = "Azzani (7AZZANI)"

_LAZY_EXPORTS = {
    "ImageDiagnostics": (".analyzer", "ImageDiagnostics"),
    "analyze_image": (".analyzer", "analyze_image"),
    "LoadedImage": (".image_io", "LoadedImage"),
    "load_image": (".image_io", "load_image"),
    "save_image": (".image_io", "save_image"),
    "MODEL_REGISTRY": (".models", "MODEL_REGISTRY"),
    "ModelSpec": (".models", "ModelSpec"),
    "ensure_model": (".models", "ensure_model"),
    "EnhancementConfig": (".pipeline", "EnhancementConfig"),
    "EnhancementResult": (".pipeline", "EnhancementResult"),
    "PreparedEnhancementLayers": (".pipeline", "PreparedEnhancementLayers"),
    "compose_enhancement_result": (".pipeline", "compose_enhancement_result"),
    "enhance_file": (".pipeline", "enhance_file"),
    "enhance_loaded_image": (".pipeline", "enhance_loaded_image"),
    "prepare_enhancement_layers": (".pipeline", "prepare_enhancement_layers"),
    "EnginePool": (".inference", "EnginePool"),
    "WatchEvent": (".automation", "WatchEvent"),
    "WatchFolderRunner": (".automation", "WatchFolderRunner"),
    "GenerativeConfig": (".generative", "GenerativeConfig"),
    "refine_generative_detail": (".generative", "refine_generative_detail"),
    "VideoConfig": (".video", "VideoConfig"),
    "VideoResult": (".video", "VideoResult"),
    "enhance_video_file": (".video", "enhance_video_file"),
    "apply_profile_to_config": (".profiles", "apply_profile_to_config"),
    "list_profiles": (".profiles", "list_profiles"),
    "write_sample_profiles": (".profiles", "write_sample_profiles"),
    "AsyncJobManager": (".server", "AsyncJobManager"),
    "dispatch_webhook": (".server", "dispatch_webhook"),
    "launch_api_server": (".server", "launch_api_server"),
    "compute_gmsd": (".benchmark", "compute_gmsd"),
    "compute_perceptual_score": (".benchmark", "compute_perceptual_score"),
    "run_benchmark_suite": (".benchmark", "run_benchmark_suite"),
    "HardwareReport": (".hardware", "HardwareReport"),
    "probe_system_hardware": (".hardware", "probe_system_hardware"),
}

__all__ = [
    "__version__",
    "__license__",
    "__author__",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

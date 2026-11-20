"""End-to-end hybrid AI + photometric image enhancement orchestrator."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .analyzer import ImageDiagnostics, analyze_image
from .color_tone import (
    ToneDetailConfig,
    apply_photometric_finish,
    compute_focus_bokeh_mask,
    self_critic_refinement,
)
from .faces import blend_face_layer, compute_face_restoration_layer, detect_faces_yunet, restore_faces
from .generative import GenerativeConfig, refine_generative_detail
from .image_io import LoadedImage, bleed_alpha_edges, load_image, save_image, upscale_alpha
from .inference import (
    EnginePool,
    RuntimeConfig,
    SpandrelEngine,
    _get_free_device_gb,
    create_inference_engine,
    resolve_device,
)
from .memmap_utils import create_managed_memmap, extract_resized_strip, is_memmap_array, stride_preview
from .models import MODEL_REGISTRY, ensure_model


PRESET_MODEL_MAP = {
    "fast": "photo-fast",
    "pro": "photo-pro",
    "photo-pro": "photo-pro",
    "ultra": "photo-ultra",
    "photo-ultra": "photo-ultra",
    "hq": "photo-hq",
    "photo-hq": "photo-hq",
    "portrait": "portrait-4x",
    "anime": "anime-4x",
    "anime-2x": "anime-2x",
    "anime-4x": "anime-4x",
    "deblur": "deblur-1x",
    "restore": "dejpeg-1x",
    "photometric": None,
}


@dataclass
class EnhancementConfig:
    """User-configurable pipeline parameters for AZZANI Image Enhancements."""

    preset: str = "auto"  # auto | fast | pro | ultra | hq | portrait | anime | anime-2x | restore | photometric
    model: Optional[str] = None  # Custom model key or local .safetensors/.pth path overriding preset
    scale: Optional[float] = None  # None = smart auto or model native scale
    width: Optional[int] = None
    height: Optional[int] = None
    max_dim: Optional[int] = None
    max_megapixels: float = 48.0  # Safety memory ceiling (0 = unlimited)

    # Pre-restoration 1x passes
    dejpeg: str = "auto"  # auto | on | off
    denoise: str = "auto"  # auto | on | off
    deblur: str = "auto"  # auto | on | off
    exposure_ai: str = "off"  # auto | on | off

    # Face restoration & SFace identity lock
    face_restore: str = "auto"  # auto | on | off
    face_fidelity: float = 0.78
    identity_guard: float = 0.65

    # Photometric finishing & Phase 1 self-critic controls
    color_fix: float = 0.85
    white_balance: str = "auto"  # auto | on | off
    white_balance_strength: Optional[float] = None
    tone_strength: Optional[float] = None
    detail_strength: Optional[float] = None
    natural_grain: Optional[float] = None
    ca_suppress: float = 0.45
    bokeh_guard: bool = True
    halo_guard: float = 0.75

    # Inference runtime & out-of-core memmap ceiling
    backend: str = "auto"  # "auto" | "pytorch" | "compile" | "onnx" | "directml" | "tensorrt"
    device: str = "auto"
    precision: str = "auto"
    tile_size: int = 0
    tile_overlap: int = 32
    tta: int = 1
    force_reprobe: bool = False
    memmap_threshold_mp: float = 64.0

    # Hybrid generative micro-texture refiner (Stage D, 0.0 = strictly faithful)
    creativity: float = 0.0
    generative_backend: str = "auto"  # "auto" | "diffusion" | "procedural"
    generative_model: Optional[str] = None

    # Output options
    bit_depth: int = 8  # 8 or 16 (for PNG/TIFF)
    jpeg_quality: int = 98
    webp_lossless: bool = False
    preserve_metadata: bool = True
    dither: bool = True
    models_dir: Optional[Path] = None
    quiet: bool = False


@dataclass
class EnhancementResult:
    """Structured result and diagnostics returned after enhancing an image."""

    output_rgb: np.ndarray
    output_alpha: Optional[np.ndarray]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    diagnostics_before: ImageDiagnostics
    diagnostics_after: ImageDiagnostics
    stages_applied: List[str] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    device_used: str = "cpu"
    precision_used: str = "fp32"
    tile_size_used: int = 0
    faces_restored: int = 0
    elapsed_seconds: float = 0.0
    comparison_path: Optional[Path] = None
    quality_confidence: float = 100.0
    critic_metrics: Dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict:
        return {
            "input_size": list(self.input_size),
            "output_size": list(self.output_size),
            "stages_applied": self.stages_applied,
            "models_used": self.models_used,
            "device_used": self.device_used,
            "precision_used": self.precision_used,
            "tile_size_used": self.tile_size_used,
            "faces_restored": self.faces_restored,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "quality_confidence": round(self.quality_confidence, 1),
            "critic_metrics": self.critic_metrics,
            "before": self.diagnostics_before.to_dict(),
            "after": self.diagnostics_after.to_dict(),
        }


@dataclass
class PreparedEnhancementLayers:
    """Intermediate neural/face/focus layers cached in RAM for <50ms non-destructive retuning."""

    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    diag_before: ImageDiagnostics
    reference_clean_resized: np.ndarray
    sr_raw_rgb: np.ndarray
    output_alpha: Optional[np.ndarray]
    face_restored_canvas: Optional[np.ndarray]
    face_mask: Optional[np.ndarray]
    focus_mask: np.ndarray
    primary_model: Optional[str]
    neural_stages: List[str]
    models_used: List[str]
    device_used: str
    precision_used: str
    tile_size_used: int
    faces_detected: int
    elapsed_neural: float
    identity_stats: Dict[str, Any] = field(default_factory=dict)
    generative_backend_used: Optional[str] = None


def _compute_target_dimensions(
    src_w: int,
    src_h: int,
    cfg: EnhancementConfig,
    default_scale: float,
) -> tuple[int, int]:
    """Resolve exact (target_w, target_h) honoring scale, width, height, max_dim, and MP cap."""
    if cfg.width and cfg.height:
        tw, th = int(cfg.width), int(cfg.height)
    elif cfg.width:
        tw = int(cfg.width)
        th = max(1, int(round(src_h * (tw / src_w))))
    elif cfg.height:
        th = int(cfg.height)
        tw = max(1, int(round(src_w * (th / src_h))))
    else:
        scale = float(cfg.scale) if cfg.scale is not None else float(default_scale)
        tw = max(1, int(round(src_w * scale)))
        th = max(1, int(round(src_h * scale)))

    if cfg.max_dim and max(tw, th) > cfg.max_dim:
        ratio = float(cfg.max_dim) / float(max(tw, th))
        tw = max(1, int(round(tw * ratio)))
        th = max(1, int(round(th * ratio)))

    if cfg.max_megapixels > 0 and (tw * th) / 1_000_000.0 > cfg.max_megapixels:
        mp_ratio = math.sqrt((cfg.max_megapixels * 1_000_000.0) / float(tw * th))
        tw = max(1, int(round(tw * mp_ratio)))
        th = max(1, int(round(th * mp_ratio)))

    return tw, th


def _resize_float32(
    rgb: np.ndarray,
    target_hw: tuple[int, int],
    *,
    memmap_threshold_mp: float = 64.0,
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """High-precision resize in float32, streaming strip-by-strip into ManagedMemmap above threshold."""
    th, tw = target_hw
    sh, sw = rgb.shape[:2]
    if (sh, sw) == (th, tw):
        return rgb
    if interpolation is None:
        interp = cv2.INTER_AREA if (th < sh or tw < sw) else cv2.INTER_LANCZOS4
    else:
        interp = interpolation

    target_mp = (th * tw) / 1_000_000.0
    use_memmap = is_memmap_array(rgb) or (memmap_threshold_mp > 0.0 and target_mp >= float(memmap_threshold_mp))
    if not use_memmap:
        resized = cv2.resize(rgb, (tw, th), interpolation=interp)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    channels = rgb.shape[2] if rgb.ndim == 3 else 1
    out_shape: tuple[int, ...] = (th, tw, channels) if rgb.ndim == 3 else (th, tw)
    out_mm = create_managed_memmap(out_shape, dtype=np.float32, prefix="azzani_res_")
    x_map = (np.arange(tw, dtype=np.float32) + 0.5) * (sw / float(tw)) - 0.5
    strip_h = 512 if tw < 10000 else 256
    for y0 in range(0, th, strip_h):
        y1 = min(th, y0 + strip_h)
        strip = extract_resized_strip(rgb, th, tw, y0, y1, interpolation=interp, x_map=x_map)
        out_mm[y0:y1] = np.clip(strip, 0.0, 1.0)
    return out_mm


def detect_hardware_tier(requested_device: str = "auto") -> str:
    """Classify runtime hardware into 'cpu', 'gpu_compact' (<7GB free), or 'gpu_high' (>=7GB CUDA)."""
    try:
        dev = resolve_device(requested_device)
    except Exception:
        return "cpu"
    if dev.type == "cpu":
        return "cpu"
    free_gb = _get_free_device_gb(dev)
    if dev.type == "cuda" and free_gb >= 6.8:
        return "gpu_high"
    return "gpu_compact"


def _run_model_step(
    rgb: np.ndarray,
    model_key_or_path: str | Path,
    cfg: EnhancementConfig,
    stage_label: str,
    status_cb: Optional[Callable[[str], None]] = None,
    engine_pool: Optional[EnginePool] = None,
) -> tuple[np.ndarray, int, str, str, int]:
    """Download/load a Spandrel model, run tiled inference, and reuse pool or release VRAM cleanly."""
    if status_cb:
        status_cb(f"Loading model: {model_key_or_path} ({stage_label})")
    model_path, spec = ensure_model(
        model_key_or_path,
        models_dir=cfg.models_dir,
        quiet=cfg.quiet,
    )
    runtime = RuntimeConfig(
        backend=cfg.backend,
        device=cfg.device,
        precision=cfg.precision,
        tile_size=cfg.tile_size,
        tile_overlap=cfg.tile_overlap,
        tta=cfg.tta,
        tile_complexity=spec.tile_complexity if spec else "medium",
        force_reprobe=cfg.force_reprobe,
        memmap_threshold_mp=cfg.memmap_threshold_mp,
    )
    val_range = spec.value_range if spec else "0_1"
    if engine_pool is not None:
        engine = engine_pool.acquire(model_path, runtime=runtime, value_range=val_range)
    else:
        engine = create_inference_engine(model_path, runtime=runtime, value_range=val_range)
    try:
        if status_cb:
            status_cb(f"Running {stage_label} on {engine.device} [{engine.precision_note}]")
        out_rgb, tile_used = engine.enhance(rgb)
        return (
            out_rgb,
            engine.scale,
            str(engine.device),
            engine.precision_note,
            tile_used,
        )
    finally:
        if engine_pool is None:
            engine.close()


# Minimum agreement (dB) between heavily low-passed input and output of a neural pass. Legitimate restoration
# and SR keep coarse structure and colour essentially unchanged (typically > 30 dB); unstable models fed
# out-of-distribution inputs (e.g. NAFNet deblur on noisy JPEG thumbnails) collapse far below this.
MIN_PASS_LOWFREQ_PSNR_DB = 22.0
# Exposure correction shifts brightness on purpose, so only coarse structural agreement is checked for it.
MIN_PASS_STRUCTURE_CORR = 0.85

# GFPGAN renders the face box at roughly 230px inside its 512x512 canvas, so faces below this source size gain
# real resolution from restoration; degraded sources (blur/noise/JPEG) also benefit up to the larger limit.
AUTO_FACE_RESTORE_MAX_SRC_PX = 256.0
AUTO_FACE_RESTORE_MAX_DEGRADED_SRC_PX = 768.0

# Primary models that already synthesize abundant high-frequency detail; post-sharpening only adds crunch.
_DETAIL_RICH_MODELS = {"photo-pro", "photo-ultra", "photo-hq", "portrait-4x", "anime-4x"}


def _lowpass_preview(x: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Bounded-memory, heavily low-passed preview used to compare coarse structure/colour."""
    big = is_memmap_array(x) or (x.shape[0] * x.shape[1]) > 16_000_000
    src = (
        stride_preview(x, max_side=4 * max(size_wh))
        if big
        else np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    )
    small = cv2.resize(src, size_wh, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), sigmaX=1.0)


def neural_pass_is_sane(
    before: np.ndarray,
    after: np.ndarray,
    *,
    tone_changing: bool = False,
) -> tuple[bool, float]:
    """Detect catastrophic model failures (NaN, blow-ups, garbage) by comparing coarse input vs output.

    Returns ``(ok, score)`` where score is the low-frequency PSNR in dB (or the luma structure correlation
    for passes that legitimately change tone).
    """
    h, w = before.shape[:2]
    s = min(1.0, 256.0 / float(max(h, w)))
    size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
    pb = _lowpass_preview(before, size)
    pa = _lowpass_preview(after, size)
    if not np.all(np.isfinite(pa)):
        return False, 0.0
    if tone_changing:
        yb = pb.mean(axis=2).ravel()
        ya = pa.mean(axis=2).ravel()
        if float(np.std(yb)) < 1e-4 or float(np.std(ya)) < 1e-4:
            return True, 1.0
        corr = float(np.corrcoef(yb, ya)[0, 1])
        return corr >= MIN_PASS_STRUCTURE_CORR, corr
    mse = float(np.mean((pa - pb) ** 2))
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    return psnr >= MIN_PASS_LOWFREQ_PSNR_DB, psnr


def _deblur_is_safe(diag: ImageDiagnostics) -> bool:
    """Deblurring acts like an inverse filter: on noisy or blocky inputs it amplifies artifacts."""
    return (
        diag.recommended_deblur
        and not diag.is_illustration_likely
        and diag.noise_severity < 0.45
        and diag.jpeg_blockiness < 0.6
    )


def _select_faces_to_restore(
    detected: list,
    diag: ImageDiagnostics,
    up_factor: float,
) -> list:
    """Pick faces that genuinely benefit from GFPGAN in auto mode (small and/or degraded source faces)."""
    degraded = (
        diag.blur_severity > 0.25
        or diag.noise_severity > 0.45
        or diag.jpeg_blockiness > 0.45
    )
    limit = AUTO_FACE_RESTORE_MAX_DEGRADED_SRC_PX if degraded else AUTO_FACE_RESTORE_MAX_SRC_PX
    up = max(float(up_factor), 1e-3)
    return [f for f in detected if min(f.bbox[2], f.bbox[3]) / up < limit]


def _resolve_tone_config(
    diag: ImageDiagnostics,
    cfg: EnhancementConfig,
    primary_model: Optional[str],
) -> ToneDetailConfig:
    """Resolve photometric & artifact-guard configuration for a given image and model choice."""
    if cfg.white_balance == "off":
        wb_strength = 0.0
    elif cfg.white_balance == "on":
        wb_strength = cfg.white_balance_strength if cfg.white_balance_strength is not None else 0.50
    else:
        # Auto preserves the original white balance: gray-world balancing cannot tell a colour cast from
        # intentional scene colour (sunsets, foliage, underwater, tungsten interiors) and measurably
        # degraded natural photos. An explicit strength (CLI/UI) or white_balance="on" still applies it.
        wb_strength = cfg.white_balance_strength if cfg.white_balance_strength is not None else 0.0

    if cfg.tone_strength is not None:
        tone_str = float(cfg.tone_strength)
    else:
        exp_need = abs(diag.recommended_exposure_boost)
        if primary_model is None:
            # Pure photometric preset: the user asked for a tonal enhancement, keep a visible base amount.
            dr_need = max(0.0, (62.0 - diag.dynamic_range) / 50.0)
            clip_need = max(0.0, (diag.shadow_clip_pct + diag.highlight_clip_pct - 4.0) / 15.0)
            tone_str = float(np.clip(0.25 + 0.55 * max(dr_need, exp_need, clip_need), 0.10, 0.72))
        else:
            # Neural presets preserve the photographer's tonality and only repair measurable defects:
            # genuinely flat/hazy histograms and whole-image underexposure. Clipped shadows/highlights are
            # not repairable by tone mapping (lifting crushed blacks just turns them milky), and low-key
            # images are dark by intent, so neither triggers a tone change here.
            dr_need = max(0.0, (55.0 - diag.dynamic_range) / 45.0)
            tone_str = float(np.clip(0.6 * max(dr_need, exp_need), 0.0, 0.6))

    if cfg.detail_strength is not None:
        detail_str = float(cfg.detail_strength)
    else:
        if primary_model is None:
            detail_str = 0.20 if diag.is_illustration_likely else 0.36
        elif primary_model in _DETAIL_RICH_MODELS:
            # These models already synthesize more high-frequency energy than real photos contain.
            detail_str = 0.03 if diag.is_illustration_likely else 0.04
        else:
            # Lighter models (2x SPAN, 1x restoration) stay slightly soft; a touch of guided detail helps.
            detail_str = 0.06 if diag.is_illustration_likely else 0.12

    if cfg.natural_grain is not None:
        grain_str = float(cfg.natural_grain)
    else:
        grain_str = 0.0

    return ToneDetailConfig(
        color_fix_strength=float(cfg.color_fix) if primary_model else 0.0,
        color_fix_luma_strength=0.65 if primary_model else 0.0,
        white_balance_strength=float(wb_strength),
        tone_strength=tone_str,
        detail_strength=detail_str,
        natural_grain=grain_str,
        ca_suppress=float(cfg.ca_suppress) if not diag.is_illustration_likely else 0.0,
        bokeh_guard=bool(cfg.bokeh_guard) and not diag.is_illustration_likely,
        halo_guard=float(cfg.halo_guard) if primary_model else 0.0,
    )


def _resolve_plan(
    diag: ImageDiagnostics,
    cfg: EnhancementConfig,
) -> tuple[Optional[str], float, bool, bool, bool, bool, ToneDetailConfig]:
    """Determine primary model, default scale, pre-passes, and photometric parameters."""
    preset = cfg.preset.lower().strip()
    if preset not in PRESET_MODEL_MAP and preset != "auto":
        valid = ", ".join(["auto"] + sorted(PRESET_MODEL_MAP.keys()))
        raise ValueError(f"Unknown preset '{cfg.preset}'. Choose from: {valid}")

    # 1. Pick primary SR/restoration model.
    if cfg.model:
        primary_model: Optional[str] = cfg.model
        if cfg.model in MODEL_REGISTRY:
            default_scale = float(MODEL_REGISTRY[cfg.model].scale)
        else:
            default_scale = 4.0
    elif preset == "auto":
        max_side = max(diag.width, diag.height)
        hw_tier = detect_hardware_tier(cfg.device)
        if diag.is_illustration_likely:
            if hw_tier == "cpu":
                primary_model = "anime-2x"
            else:
                primary_model = "anime-2x" if max_side > 1400 else "anime-4x"
        else:
            if hw_tier == "cpu":
                primary_model = "photo-fast"
            elif hw_tier == "gpu_high" and max_side <= 1280:
                primary_model = "photo-ultra"
            else:
                primary_model = "photo-pro"

        if cfg.scale is not None:
            default_scale = float(cfg.scale)
        elif hw_tier == "cpu":
            if max_side <= 1400:
                default_scale = 2.0
            else:
                default_scale = 1.0
                if diag.jpeg_blockiness >= 0.35:
                    primary_model = "dejpeg-1x"
                elif diag.noise_severity >= 0.48:
                    primary_model = "denoise-1x"
                elif _deblur_is_safe(diag):
                    primary_model = "deblur-1x"
                else:
                    primary_model = None
        elif max_side <= 1280:
            default_scale = 4.0
        elif max_side <= 2200:
            default_scale = 2.0
        else:
            default_scale = 1.0
            # Already >2200px: avoid 4x neural bloat unless user requested upscale.
            if diag.jpeg_blockiness >= 0.30:
                primary_model = "dejpeg-1x"
            elif diag.noise_severity >= 0.42:
                primary_model = "denoise-1x"
            elif _deblur_is_safe(diag):
                primary_model = "deblur-1x"
            elif diag.blur_severity > 0.55:
                primary_model = "photo-fast"
            else:
                primary_model = None  # Clean high-res photo -> pure photometric finishing!
    else:
        primary_model = PRESET_MODEL_MAP[preset]
        if primary_model and primary_model in MODEL_REGISTRY:
            default_scale = float(MODEL_REGISTRY[primary_model].scale)
        else:
            default_scale = 1.0

    # 2. Decide optional 1x pre-restoration passes.
    handles_degradations = primary_model in {"photo-fast", "photo-pro", "photo-ultra", "portrait-4x"}
    if cfg.denoise == "on":
        run_denoise = True
    elif cfg.denoise == "off":
        run_denoise = False
    else:
        noise_thresh = 0.72 if handles_degradations else 0.45
        run_denoise = (
            diag.noise_severity >= noise_thresh
            and primary_model not in {"denoise-1x", None}
        )

    if cfg.dejpeg == "on":
        run_dejpeg = True
    elif cfg.dejpeg == "off":
        run_dejpeg = False
    else:
        run_dejpeg = (
            not handles_degradations
            and diag.jpeg_blockiness >= 0.45
            and primary_model not in {"dejpeg-1x", None}
        )

    deblur_mode = cfg.deblur.lower().strip()
    if deblur_mode == "on":
        run_deblur = True
    elif deblur_mode == "off":
        run_deblur = False
    else:
        run_deblur = _deblur_is_safe(diag) and primary_model not in {"deblur-1x", None}

    if cfg.exposure_ai == "on":
        run_exp_ai = True
    elif cfg.exposure_ai == "off":
        run_exp_ai = False
    else:
        run_exp_ai = abs(diag.recommended_exposure_boost) >= 0.45

    tone_cfg = _resolve_tone_config(diag, cfg, primary_model)
    return primary_model, default_scale, run_denoise, run_dejpeg, run_deblur, run_exp_ai, tone_cfg


def prepare_enhancement_layers(
    loaded: LoadedImage,
    cfg: Optional[EnhancementConfig] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    *,
    engine_pool: Optional[EnginePool] = None,
) -> PreparedEnhancementLayers:
    """Run neural SR, resizing, alpha reconstruction, face detection/GFPGAN & focus mapping once."""
    t_start = time.perf_counter()
    cfg = cfg or EnhancementConfig()

    src_h, src_w = loaded.rgb.shape[:2]
    work_rgb = loaded.rgb
    if loaded.alpha is not None:
        work_rgb = bleed_alpha_edges(work_rgb, loaded.alpha)

    diag_before = analyze_image(work_rgb)
    primary_model, default_scale, run_denoise, run_dejpeg, run_deblur, run_exp_ai, _ = _resolve_plan(
        diag_before, cfg
    )
    target_w, target_h = _compute_target_dimensions(src_w, src_h, cfg, default_scale)

    stages: List[str] = []
    models_used: List[str] = []
    device_used = "cpu"
    precision_used = "fp32"
    tile_used = 0
    faces_detected = 0

    current = work_rgb
    reference_clean = work_rgb
    runtime_info = {"device": device_used, "precision": precision_used, "tile": tile_used}

    def _guarded_step(
        src: np.ndarray,
        model_key: str,
        label: str,
        *,
        tone_changing: bool = False,
    ) -> tuple[np.ndarray, Optional[int]]:
        """Run one neural pass; discard it (keeping ``src``) if its output diverges catastrophically."""
        out, model_scale, dev, prec, tile = _run_model_step(
            src, model_key, cfg, label, status_cb, engine_pool=engine_pool
        )
        ok, score = neural_pass_is_sane(src, out, tone_changing=tone_changing)
        if not ok:
            stages.append(f"rejected:{model_key}")
            if status_cb:
                status_cb(f"Discarded {label}: output diverged from input (score={score:.2f}); keeping previous stage")
            return src, None
        runtime_info.update(device=dev, precision=prec, tile=tile)
        models_used.append(str(model_key))
        return out, model_scale

    # Stage A: Optional 1x denoising, JPEG artifact pre-cleaning, directional deblurring & exposure rescue.
    if run_denoise:
        current, ran = _guarded_step(current, "denoise-1x", "1x neural denoise")
        if ran is not None:
            stages.append("denoise-1x")
            reference_clean = current

    if run_dejpeg:
        current, ran = _guarded_step(current, "dejpeg-1x", "1x neural de-JPEG")
        if ran is not None:
            stages.append("dejpeg-1x")
            reference_clean = current

    if run_deblur:
        current, ran = _guarded_step(current, "deblur-1x", "1x directional motion deblur")
        if ran is not None:
            stages.append("deblur-1x")
            reference_clean = current

    if run_exp_ai:
        current, ran = _guarded_step(current, "exposure-1x", "1x neural exposure", tone_changing=True)
        if ran is not None:
            stages.append("exposure-1x")

    # Stage B: Super-resolution or primary neural restoration.
    req_scale_x = target_w / float(src_w)
    req_scale_y = target_h / float(src_h)
    req_scale = max(req_scale_x, req_scale_y)

    if primary_model is not None:
        current, model_scale = _guarded_step(
            current, primary_model, f"neural enhancement ({primary_model})"
        )
        if model_scale is None:
            # Model diverged on this input: fall back to clean resampling + photometric finishing.
            primary_model = None
        else:
            stages.append(f"model:{primary_model}")
            # Support >4x (e.g. 8x) via a second clean 2x pass when needed.
            if req_scale > model_scale * 1.25:
                extra_model = "anime-2x" if diag_before.is_illustration_likely else "photo-fast"
                current, extra_scale = _guarded_step(
                    current, extra_model, f"2nd-stage upscale ({extra_model})"
                )
                if extra_scale is not None:
                    stages.append(f"model:{extra_model}")

    device_used = str(runtime_info["device"])
    precision_used = str(runtime_info["precision"])
    tile_used = int(runtime_info["tile"])

    # Exact target resampling (supersampled downscale when e.g. 4x model -> 2x target).
    if current.shape[:2] != (target_h, target_w):
        current = _resize_float32(
            current, (target_h, target_w), memmap_threshold_mp=cfg.memmap_threshold_mp
        )
        stages.append(f"resample:{target_w}x{target_h}")

    ref_resized = _resize_float32(
        reference_clean,
        (target_h, target_w),
        memmap_threshold_mp=cfg.memmap_threshold_mp,
        interpolation=cv2.INTER_CUBIC,
    )

    # Stage C: Face detection & GFPGAN full-fidelity cache (enables instant <5ms fidelity retuning).
    face_canvas: Optional[np.ndarray] = None
    face_mask: Optional[np.ndarray] = None
    id_stats: Dict[str, Any] = {}
    face_mode = cfg.face_restore.lower().strip()
    if face_mode in {"on", "auto"} and not diag_before.is_illustration_likely:
        if status_cb:
            status_cb("Checking for faces (YuNet)...")
        yunet_path, _ = ensure_model("face-detector-yunet", models_dir=cfg.models_dir, quiet=cfg.quiet)
        detected = detect_faces_yunet(current, yunet_path)
        if detected and face_mode == "auto":
            detected = _select_faces_to_restore(detected, diag_before, target_w / float(src_w))
        if detected:
            if status_cb:
                status_cb(f"Restoring {len(detected)} face(s) via GFPGANv1.4...")
            face_canvas, face_mask, faces_detected = compute_face_restoration_layer(
                current,
                device=cfg.device,
                precision=cfg.precision,
                models_dir=cfg.models_dir,
                quiet=cfg.quiet,
                pre_detected_faces=detected,
                engine_pool=engine_pool,
                force_reprobe=cfg.force_reprobe,
                identity_guard=cfg.identity_guard,
                identity_stats_out=id_stats,
                # Crop faces from the cleaned, resampled input: GFPGAN restores it more naturally than the
                # generic SR output, whose painterly hair/skin textures it would otherwise inherit.
                restore_source=ref_resized if primary_model is not None else None,
            )
            if faces_detected > 0:
                stages.append(f"face-restore:{faces_detected}")
                models_used.append("face-gfpgan")
                if "face_identity_cosine" in id_stats:
                    models_used.append("face-recognizer-sface")
                if id_stats.get("faces_guarded", 0) > 0:
                    stages.append("face-identity-guard")

    out_alpha = None
    if loaded.alpha is not None:
        out_alpha = upscale_alpha(loaded.alpha, (target_h, target_w))
        stages.append("alpha-reconstruct")

    effective_photo_mp = min(24.0, cfg.memmap_threshold_mp) if cfg.memmap_threshold_mp > 0.0 else 1e9
    focus_mask = compute_focus_bokeh_mask(
        ref_resized, face_mask=face_mask, memmap_threshold_mp=effective_photo_mp
    )

    # Stage D: Optional hybrid generative micro-texture refinement (--creativity > 0), strictly face/sky-gated.
    gen_used: Optional[str] = None
    if float(cfg.creativity) > 0.0:
        gen_cfg = GenerativeConfig(
            creativity=float(cfg.creativity),
            backend=cfg.generative_backend,
            model_id=cfg.generative_model or "stabilityai/sd-turbo",
        )
        current, gen_used = refine_generative_detail(
            current,
            ref_resized,
            gen_cfg,
            focus_mask=focus_mask,
            face_mask=face_mask,
            device=cfg.device,
            status_cb=status_cb,
        )
        if gen_used != "none":
            stages.append(f"generative:{gen_used}:{float(cfg.creativity):.2f}")

    sr_out = current if is_memmap_array(current) else np.clip(current, 0.0, 1.0).astype(np.float32)

    return PreparedEnhancementLayers(
        input_size=(src_w, src_h),
        output_size=(target_w, target_h),
        diag_before=diag_before,
        reference_clean_resized=ref_resized,
        sr_raw_rgb=sr_out,
        output_alpha=out_alpha,
        face_restored_canvas=face_canvas if faces_detected > 0 else None,
        face_mask=face_mask,
        focus_mask=focus_mask,
        primary_model=primary_model,
        neural_stages=stages,
        models_used=models_used,
        device_used=device_used,
        precision_used=precision_used,
        tile_size_used=tile_used,
        faces_detected=faces_detected,
        elapsed_neural=time.perf_counter() - t_start,
        identity_stats=id_stats,
        generative_backend_used=gen_used,
    )


def compose_enhancement_result(
    layers: PreparedEnhancementLayers,
    cfg: Optional[EnhancementConfig] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    *,
    include_neural_time: bool = True,
) -> EnhancementResult:
    """Apply fast (<50ms) face fidelity blending, artifact self-critic & float32 photometrics."""
    t_comp = time.perf_counter()
    cfg = cfg or EnhancementConfig()
    tone_cfg = _resolve_tone_config(layers.diag_before, cfg, layers.primary_model)

    stages = list(layers.neural_stages)
    current = layers.sr_raw_rgb
    faces_applied = 0

    if (
        layers.face_restored_canvas is not None
        and layers.faces_detected > 0
        and cfg.face_restore.lower().strip() != "off"
        and cfg.face_fidelity > 0.0
    ):
        current = blend_face_layer(current, layers.face_restored_canvas, fidelity=cfg.face_fidelity)
        faces_applied = layers.faces_detected

    # Closed-loop artifact self-critic: clamp post-SR overshoot halos & measure color drift.
    effective_photo_mp = min(24.0, cfg.memmap_threshold_mp) if cfg.memmap_threshold_mp > 0.0 else 1e9
    critic_metrics: Dict[str, Any] = {}
    if layers.primary_model is not None or cfg.halo_guard > 0.0:
        clamp_amt = float(tone_cfg.halo_guard if layers.primary_model is not None else 0.0)
        in_w, in_h = layers.input_size
        out_w, out_h = layers.output_size
        upscale = max(out_w / float(max(1, in_w)), out_h / float(max(1, in_h)))
        current, critic_metrics = self_critic_refinement(
            current,
            layers.reference_clean_resized,
            clamp_strength=clamp_amt,
            focus_mask=layers.focus_mask if tone_cfg.bokeh_guard else None,
            memmap_threshold_mp=effective_photo_mp,
            scale_hint=upscale,
            protect_mask=layers.face_mask if faces_applied > 0 else None,
        )
        if critic_metrics.get("raw_halo_index", 0.0) > 0.08 and clamp_amt > 0.0:
            stages.append("self-critic:halo-clamp")
            # Auto-moderate excess sharpness if raw model already exhibited edge overshoot.
            tone_cfg.detail_strength = min(tone_cfg.detail_strength, 0.14)
    else:
        f_prev = (
            stride_preview(layers.focus_mask, max_side=512, clip_unit=False)
            if is_memmap_array(layers.focus_mask)
            else layers.focus_mask
        )
        critic_metrics = {
            "raw_halo_index": 0.0,
            "halo_ringing_index": 0.0,
            "color_drift_delta_e": 0.0,
            "focus_coverage_pct": round(float(np.mean(f_prev > 0.65) * 100.0), 1),
            "confidence_score": 99.0,
        }

    if layers.identity_stats:
        critic_metrics.update(layers.identity_stats)

    # Float32 photometric finishing (CA suppress, wavelet color fix, WB, Retinex tone, bokeh-guarded detail).
    if status_cb:
        status_cb("Applying float32 photometric tone, color & detail finishing...")
    current = apply_photometric_finish(
        current,
        layers.reference_clean_resized,
        tone_cfg,
        exposure_hint=layers.diag_before.recommended_exposure_boost,
        focus_mask=layers.focus_mask if tone_cfg.bokeh_guard else None,
        face_mask=layers.face_mask,
        memmap_threshold_mp=effective_photo_mp,
    )
    stages.append("photometric-finish")

    diag_after = analyze_image(current)
    comp_elapsed = time.perf_counter() - t_comp
    total_elapsed = (layers.elapsed_neural + comp_elapsed) if include_neural_time else comp_elapsed

    return EnhancementResult(
        output_rgb=current,
        output_alpha=layers.output_alpha,
        input_size=layers.input_size,
        output_size=layers.output_size,
        diagnostics_before=layers.diag_before,
        diagnostics_after=diag_after,
        stages_applied=stages,
        models_used=list(layers.models_used),
        device_used=layers.device_used,
        precision_used=layers.precision_used,
        tile_size_used=layers.tile_size_used,
        faces_restored=faces_applied,
        elapsed_seconds=total_elapsed,
        quality_confidence=float(critic_metrics.get("confidence_score", 98.0)),
        critic_metrics=critic_metrics,
    )


def enhance_loaded_image(
    loaded: LoadedImage,
    cfg: Optional[EnhancementConfig] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    *,
    engine_pool: Optional[EnginePool] = None,
) -> EnhancementResult:
    """Run the complete AZZANI Image Enhancements pipeline on an already-loaded image."""
    layers = prepare_enhancement_layers(
        loaded,
        cfg=cfg,
        status_cb=status_cb,
        engine_pool=engine_pool,
    )
    return compose_enhancement_result(layers, cfg=cfg, status_cb=status_cb)


def create_comparison_image(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Create a side-by-side Before vs After visual comparison image with crisp headers."""
    ah, aw = after_rgb.shape[:2]
    # Keep comparison sheet practical for instant viewing (max 1920px per half).
    max_half_w = 1920
    if aw > max_half_w:
        scale = max_half_w / float(aw)
        panel_w = max_half_w
        panel_h = max(1, int(round(ah * scale)))
        src_after = (
            stride_preview(after_rgb, max_side=max_half_w * 2)
            if (is_memmap_array(after_rgb) or (ah * aw) > 12_000_000)
            else after_rgb
        )
        after_panel = cv2.resize(src_after, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    else:
        panel_w, panel_h = aw, ah
        after_panel = np.asarray(after_rgb, dtype=np.float32)

    src_before = (
        stride_preview(before_rgb, max_side=max_half_w * 2)
        if (is_memmap_array(before_rgb) or (before_rgb.shape[0] * before_rgb.shape[1]) > 12_000_000)
        else before_rgb
    )
    before_panel = cv2.resize(src_before, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
    divider = np.full((panel_h, 6, 3), 0.12, dtype=np.float32)
    combined = np.concatenate([before_panel, divider, after_panel], axis=1)

    u8 = np.rint(np.clip(combined, 0.0, 1.0) * 255.0).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(1.3, panel_w / 700.0))
    thickness = max(1, int(round(font_scale * 2)))
    for label, x_pos in [("BEFORE (Original)", 18), ("AFTER (AZZANI Enhanced)", panel_w + 24)]:
        cv2.putText(u8, label, (x_pos + 2, 38), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(u8, label, (x_pos, 36), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    out_f32 = u8.astype(np.float32) / 255.0
    return save_image(output_path, out_f32, dither=False, jpeg_quality=96)


def enhance_file(
    input_path: str | Path,
    output_path: str | Path,
    cfg: Optional[EnhancementConfig] = None,
    *,
    comparison_path: Optional[str | Path] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    engine_pool: Optional[EnginePool] = None,
) -> EnhancementResult:
    """Load an image file, enhance it, and write the color-managed result to disk."""
    cfg = cfg or EnhancementConfig()
    loaded = load_image(input_path)
    result = enhance_loaded_image(loaded, cfg=cfg, status_cb=status_cb, engine_pool=engine_pool)

    out_bit_depth = cfg.bit_depth
    if out_bit_depth == 0:
        out_bit_depth = loaded.source_bit_depth

    save_image(
        output_path,
        result.output_rgb,
        alpha=result.output_alpha,
        bit_depth=out_bit_depth,
        jpeg_quality=cfg.jpeg_quality,
        webp_lossless=cfg.webp_lossless,
        dither=cfg.dither,
        exif_bytes=loaded.exif_bytes if cfg.preserve_metadata else None,
        icc_profile=loaded.icc_profile if cfg.preserve_metadata else None,
    )

    if comparison_path is not None:
        result.comparison_path = create_comparison_image(
            loaded.rgb,
            result.output_rgb,
            comparison_path,
        )

    return result

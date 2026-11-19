"""Hybrid beat-Magnific generative tile refiner: optional 4-step Diffusion & procedural texture synthesis."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .color_tone import (
    _apply_wavelet_low_diff_strip,
    _compute_wavelet_low_diff_small,
    wavelet_color_fix,
)
from .inference import (
    _build_feather_mask,
    _compute_tile_starts,
    empty_device_cache,
    resolve_device,
)
from .memmap_utils import (
    create_managed_memmap,
    extract_resized_strip,
    is_memmap_array,
    stride_preview,
)

GEN_BACKEND_CHOICES = ("auto", "diffusion", "procedural")

# Cache loaded diffusers pipelines cleanly so interactive retries avoid re-loading multi-GB checkpoints.
_DIFFUSION_PIPE_CACHE: Dict[Tuple[str, str], Any] = {}


@dataclass
class GenerativeConfig:
    """Configuration for post-SR generative texture refinement (--creativity)."""

    creativity: float = 0.0  # [0.0..1.0], 0.0 = strictly faithful non-generative restoration
    backend: str = "auto"  # "auto" | "diffusion" | "procedural"
    model_id: str = "stabilityai/sd-turbo"
    prompt: str = "crisp natural photographic micro-texture, realistic surface fibers, sharp optical detail"
    tile_size: int = 512
    tile_overlap: int = 64
    seed: int = 1337
    face_protect: float = 0.90  # [0..1] suppresses generative hallucination inside detected faces


def is_diffusers_available() -> bool:
    """Return True when diffusers and transformers extras are installed."""
    return (
        importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def _compute_spatial_gate_block(
    rgb_blk: np.ndarray,
    *,
    focus_blk: Optional[np.ndarray],
    face_blk: Optional[np.ndarray],
    face_protect: float,
) -> np.ndarray:
    """Compute [0, 1] synthesis weight: strong on textured subject areas, muted on skies & faces."""
    y = (
        0.2126 * rgb_blk[:, :, 0]
        + 0.7152 * rgb_blk[:, :, 1]
        + 0.0722 * rgb_blk[:, :, 2]
    ).astype(np.float32)

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    grad_smooth = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigmaX=2.2)
    tex_gate = np.clip((grad_smooth - 0.008) / 0.032, 0.0, 1.0)

    # Protect deep blacks and clipped whites from synthetic contrast pumping.
    tonal_safe = np.clip(4.2 * y * (1.0 - y) + 0.15, 0.0, 1.0)
    gate = tex_gate * tonal_safe

    if focus_blk is not None:
        f2d = focus_blk[:, :, 0] if focus_blk.ndim == 3 else focus_blk
        focus_w = np.clip((f2d - 0.20) / 0.80, 0.15, 1.0)
        gate = gate * focus_w

    if face_blk is not None and face_protect > 0.0:
        fm2d = face_blk[:, :, 0] if face_blk.ndim == 3 else face_blk
        gate = gate * np.clip(1.0 - face_protect * np.clip(fm2d, 0.0, 1.0), 0.0, 1.0)

    return np.clip(gate, 0.0, 1.0)[:, :, None].astype(np.float32)


def _procedural_synthesize_block(
    rgb_blk: np.ndarray,
    *,
    creativity: float,
    focus_blk: Optional[np.ndarray],
    face_blk: Optional[np.ndarray],
    face_protect: float,
    seed: int,
    y_offset: int = 0,
) -> np.ndarray:
    """Structure-guided multi-octave procedural micro-detail synthesis (no external model needed)."""
    c = float(np.clip(creativity, 0.0, 1.0))
    if c <= 0.0:
        return rgb_blk

    img = np.clip(rgb_blk, 0.0, 1.0).astype(np.float32)
    gate = _compute_spatial_gate_block(
        img, focus_blk=focus_blk, face_blk=face_blk, face_protect=face_protect
    )

    y = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.float32)
    b1 = cv2.GaussianBlur(y, (0, 0), sigmaX=0.75)
    b2 = cv2.GaussianBlur(y, (0, 0), sigmaX=1.55)
    b3 = cv2.GaussianBlur(y, (0, 0), sigmaX=3.10)

    # Multi-scale Difference-of-Gaussians captures natural micro-contrast harmonics.
    octave_fine = b1 - b2
    octave_mid = b2 - b3

    # Coherent anisotropic local micro-weave modulated along local structure rather than raw white noise.
    rng = np.random.default_rng(int(seed) + int(y_offset) * 31)
    h, w = y.shape
    coarse_noise = rng.normal(0.0, 1.0, size=(max(4, h // 2), max(4, w // 2))).astype(np.float32)
    tex_field = cv2.resize(coarse_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    tex_field = tex_field - cv2.GaussianBlur(tex_field, (0, 0), sigmaX=2.0)

    gx = cv2.Sobel(b2, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(b2, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    local_energy = np.clip(np.hypot(gx, gy) * 6.5, 0.0, 1.0)

    # Edge-preserving structural micro-texture harmonic with hard halo clamp.
    raw_delta = (
        0.58 * octave_fine
        + 0.34 * octave_mid
        + 0.018 * local_energy * tex_field
    )[:, :, None]
    safe_delta = np.tanh(raw_delta / 0.055) * 0.055

    gain = (0.45 + 1.15 * c) * gate * c
    # Apply predominantly on luminance while letting 35% track RGB ratios so colors stay clean.
    return np.clip(img + gain * safe_delta, 0.0, 1.0).astype(np.float32)


def _refine_procedural(
    sr_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    cfg: GenerativeConfig,
    *,
    focus_mask: Optional[np.ndarray],
    face_mask: Optional[np.ndarray],
) -> np.ndarray:
    c = float(np.clip(cfg.creativity, 0.0, 1.0))
    h, w = sr_rgb.shape[:2]
    use_memmap = is_memmap_array(sr_rgb) or (h * w) >= 24_000_000

    if not use_memmap:
        f_use = (
            cv2.resize(focus_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            if (focus_mask is not None and focus_mask.shape[:2] != (h, w))
            else focus_mask
        )
        fm_use = (
            cv2.resize(face_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            if (face_mask is not None and face_mask.shape[:2] != (h, w))
            else face_mask
        )
        syn = _procedural_synthesize_block(
            sr_rgb,
            creativity=c,
            focus_blk=f_use,
            face_blk=fm_use,
            face_protect=cfg.face_protect,
            seed=cfg.seed,
        )
        # Re-anchor low-frequency CIE LAB color & brightness so creative texture never drifts tint.
        return wavelet_color_fix(syn, ref_rgb, chroma_strength=0.90, luma_strength=0.70)

    sr_small = stride_preview(sr_rgb, max_side=512)
    ref_small = stride_preview(ref_rgb, max_side=512)
    low_diff_small = _compute_wavelet_low_diff_small(sr_small, ref_small, full_h=h, full_w=w)
    low_x_map = (np.arange(w, dtype=np.float32) + 0.5) * (low_diff_small.shape[1] / float(w)) - 0.5

    out_mm = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azz_gen_")
    strip_h = 512 if w < 10000 else 256
    pad = 24
    for y0 in range(0, h, strip_h):
        y1 = min(h, y0 + strip_h)
        py0 = max(0, y0 - pad)
        py1 = min(h, y1 + pad)

        blk = np.clip(np.asarray(sr_rgb[py0:py1], dtype=np.float32), 0.0, 1.0)
        f_blk = (
            extract_resized_strip(focus_mask, h, w, py0, py1, interpolation=cv2.INTER_LINEAR)
            if focus_mask is not None
            else None
        )
        fm_blk = (
            extract_resized_strip(face_mask, h, w, py0, py1, interpolation=cv2.INTER_LINEAR)
            if face_mask is not None
            else None
        )
        syn_blk = _procedural_synthesize_block(
            blk,
            creativity=c,
            focus_blk=f_blk,
            face_blk=fm_blk,
            face_protect=cfg.face_protect,
            seed=cfg.seed,
            y_offset=py0,
        )
        low_blk = extract_resized_strip(
            low_diff_small, h, w, py0, py1, interpolation=cv2.INTER_LINEAR, x_map=low_x_map
        )
        anchored = _apply_wavelet_low_diff_strip(
            syn_blk, low_blk, chroma_strength=0.90, luma_strength=0.70
        )
        iy0 = y0 - py0
        iy1 = iy0 + (y1 - y0)
        out_mm[y0:y1] = np.clip(anchored[iy0:iy1], 0.0, 1.0)
    return out_mm


def _load_diffusion_pipeline(model_id: str, device_str: str = "auto") -> Tuple[Any, Any]:
    """Lazily load and cache an Apache-2.0 distilled Img2Img pipeline from diffusers."""
    import torch
    from diffusers import AutoPipelineForImage2Image  # type: ignore

    dev = resolve_device(device_str)
    key = (str(model_id), str(dev))
    if key in _DIFFUSION_PIPE_CACHE:
        return _DIFFUSION_PIPE_CACHE[key], dev

    dtype = torch.float16 if dev.type in {"cuda", "mps"} else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    if dev.type == "cuda":
        try:
            _, total_b = torch.cuda.mem_get_info(dev)
            if total_b < 5.5 * (1024 ** 3) and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(gpu_id=dev.index or 0)
            else:
                pipe = pipe.to(dev)
        except Exception:
            pipe = pipe.to(dev)
    else:
        pipe = pipe.to(dev)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    _DIFFUSION_PIPE_CACHE[key] = pipe
    return pipe, dev


def _refine_diffusion(
    sr_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    cfg: GenerativeConfig,
    *,
    focus_mask: Optional[np.ndarray],
    face_mask: Optional[np.ndarray],
    device: str = "auto",
) -> np.ndarray:
    """Run overlap-tiled 4-step latent diffusion refinement with face, focus & LAB color locks."""
    if not is_diffusers_available():
        raise RuntimeError(
            "Generative diffusion backend requires optional extras. "
            "Install with: pip install -e \".[generative]\""
        )

    import torch
    from PIL import Image

    pipe, dev = _load_diffusion_pipeline(cfg.model_id, device_str=device)
    c = float(np.clip(cfg.creativity, 0.0, 1.0))
    # Map creativity [0..1] onto mild img2img denoising strength so underlying SR edges anchor geometry.
    img2img_strength = float(np.clip(0.14 + 0.42 * c, 0.12, 0.56))
    num_steps = 4
    guidance = 0.0 if "turbo" in cfg.model_id.lower() else 1.2

    h, w = sr_rgb.shape[:2]
    tile = max(256, min(int(cfg.tile_size), max(h, w, 256)))
    tile = ((tile + 63) // 64) * 64
    ov = max(32, min(int(cfg.tile_overlap), tile // 3))

    ys = _compute_tile_starts(h, min(h, tile), ov)
    xs = _compute_tile_starts(w, min(w, tile), ov)

    use_memmap = is_memmap_array(sr_rgb) or (h * w) >= 24_000_000
    accum = (
        create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azz_gacc_")
        if use_memmap
        else np.zeros((h, w, 3), dtype=np.float32)
    )
    weight = (
        create_managed_memmap((h, w, 1), dtype=np.float32, prefix="azz_gwgt_")
        if use_memmap
        else np.zeros((h, w, 1), dtype=np.float32)
    )

    gen_device = dev if dev.type in {"cuda", "cpu"} else torch.device("cpu")

    try:
        for y0 in ys:
            y1 = min(h, y0 + min(h, tile))
            for x0 in xs:
                x1 = min(w, x0 + min(w, tile))
                patch_f = np.clip(np.asarray(sr_rgb[y0:y1, x0:x1], dtype=np.float32), 0.0, 1.0)
                ph, pw = patch_f.shape[:2]
                # Ensure tile size divisible by 64 for latent VAE downsampling.
                pad_h = ((ph + 63) // 64) * 64
                pad_w = ((pw + 63) // 64) * 64
                if (pad_h, pad_w) != (ph, pw):
                    patch_pad = cv2.copyMakeBorder(
                        patch_f, 0, pad_h - ph, 0, pad_w - pw, borderType=cv2.BORDER_REFLECT_101
                    )
                else:
                    patch_pad = patch_f

                pil_in = Image.fromarray(np.rint(patch_pad * 255.0).astype(np.uint8), mode="RGB")
                generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed) + y0 * 13 + x0)
                out_pil = pipe(
                    prompt=cfg.prompt,
                    image=pil_in,
                    strength=img2img_strength,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
                out_pad = np.asarray(out_pil, dtype=np.float32) / 255.0
                out_tile = np.clip(out_pad[:ph, :pw], 0.0, 1.0)

                f_sub = (
                    np.asarray(focus_mask[y0:y1, x0:x1], dtype=np.float32)
                    if (focus_mask is not None and focus_mask.shape[:2] == (h, w))
                    else None
                )
                fm_sub = (
                    np.asarray(face_mask[y0:y1, x0:x1], dtype=np.float32)
                    if (face_mask is not None and face_mask.shape[:2] == (h, w))
                    else None
                )
                gate = _compute_spatial_gate_block(
                    patch_f,
                    focus_blk=f_sub,
                    face_blk=fm_sub,
                    face_protect=cfg.face_protect,
                )
                blend_w = np.clip((0.25 + 0.75 * c) * gate, 0.0, 1.0)
                mixed_tile = patch_f * (1.0 - blend_w) + out_tile * blend_w

                fmask = _build_feather_mask(
                    ph,
                    pw,
                    ov,
                    fade_top=(y0 > 0),
                    fade_bottom=(y1 < h),
                    fade_left=(x0 > 0),
                    fade_right=(x1 < w),
                )[:, :, None]
                accum[y0:y1, x0:x1] += mixed_tile * fmask
                weight[y0:y1, x0:x1] += fmask
    finally:
        empty_device_cache(dev)

    if use_memmap:
        sr_small = stride_preview(sr_rgb, max_side=512)
        ref_small = stride_preview(ref_rgb, max_side=512)
        low_diff_small = _compute_wavelet_low_diff_small(sr_small, ref_small, full_h=h, full_w=w)
        low_x_map = (np.arange(w, dtype=np.float32) + 0.5) * (low_diff_small.shape[1] / float(w)) - 0.5
        out_mm = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azz_gdout_")
        strip_h = 512 if w < 10000 else 256
        for r0 in range(0, h, strip_h):
            r1 = min(h, r0 + strip_h)
            safe_w = np.maximum(np.asarray(weight[r0:r1], dtype=np.float32), 1e-6)
            blended = np.clip(np.asarray(accum[r0:r1], dtype=np.float32) / safe_w, 0.0, 1.0)
            low_blk = extract_resized_strip(
                low_diff_small, h, w, r0, r1, interpolation=cv2.INTER_LINEAR, x_map=low_x_map
            )
            out_mm[r0:r1] = _apply_wavelet_low_diff_strip(
                blended, low_blk, chroma_strength=0.92, luma_strength=0.72
            )
        del accum, weight
        return out_mm

    blended_full = np.clip(accum / np.maximum(weight, 1e-6), 0.0, 1.0)
    return wavelet_color_fix(blended_full, ref_rgb, chroma_strength=0.92, luma_strength=0.72)


def refine_generative_detail(
    sr_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    cfg: GenerativeConfig,
    *,
    focus_mask: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    device: str = "auto",
    status_cb: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, str]:
    """Run hybrid post-SR generative tile refiner when creativity > 0. Returns (out_rgb, backend_used)."""
    c = float(np.clip(cfg.creativity, 0.0, 1.0))
    if c <= 0.0:
        return sr_rgb, "none"

    req = cfg.backend.lower().strip()
    if req not in GEN_BACKEND_CHOICES:
        raise ValueError(f"Unknown generative backend '{cfg.backend}'. Choose from: {', '.join(GEN_BACKEND_CHOICES)}")

    if req == "diffusion" or (req == "auto" and is_diffusers_available()):
        if status_cb:
            status_cb(f"Running 4-step generative diffusion tile refiner ({cfg.model_id}, creativity={c:.2f})...")
        return (
            _refine_diffusion(
                sr_rgb,
                ref_rgb,
                cfg,
                focus_mask=focus_mask,
                face_mask=face_mask,
                device=device,
            ),
            "diffusion",
        )

    if status_cb:
        status_cb(f"Running structure-guided procedural micro-texture synthesis (creativity={c:.2f})...")
    return (
        _refine_procedural(
            sr_rgb,
            ref_rgb,
            cfg,
            focus_mask=focus_mask,
            face_mask=face_mask,
        ),
        "procedural",
    )

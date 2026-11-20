"""Spandrel PyTorch inference engine with Turing FP16 self-check, Hann tiling, OOM backoff & TTA."""

from __future__ import annotations

import gc
import os
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor, ModelLoader, ModelTiling

from .hardware import load_precision_cache, save_precision_entry
from .memmap_utils import create_managed_memmap


@dataclass
class RuntimeConfig:
    """Execution hardware, precision, backend, tiling, and ensemble configuration."""

    device: str = "auto"  # "auto" | "cuda" | "cuda:N" | "xpu" | "mps" | "cpu"
    precision: str = "auto"  # "auto" | "fp16" | "fp32" | "bf16"
    tile_size: int = 0  # 0 = auto from VRAM and model complexity
    tile_overlap: int = 32
    tta: int = 1  # 1 (off), 2 (flip), 4 (flips+rot180), 8 (full dihedral)
    tile_complexity: str = "medium"
    force_reprobe: bool = False
    memmap_threshold_mp: float = 64.0  # Switch canvas accumulation to disk memmap >= N MP
    backend: str = "auto"  # "auto" | "pytorch" | "compile" | "onnx" | "directml" | "tensorrt"


def empty_device_cache(device: torch.device) -> None:
    """Release cached allocator memory cleanly across CUDA, XPU, and Apple MPS."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _select_best_cuda_device() -> torch.device:
    """Pick the CUDA/ROCm GPU with the most available VRAM and highest compute capability."""
    count = torch.cuda.device_count()
    if count <= 1:
        return torch.device("cuda:0")
    best_idx = 0
    best_score = (-1, -1, -1)
    for idx in range(count):
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            props = torch.cuda.get_device_properties(idx)
            score = (int(free_b // (64 * 1024 * 1024)), props.major * 10 + props.minor, int(total_b))
            if score > best_score:
                best_score = score
                best_idx = idx
        except Exception:
            continue
    return torch.device(f"cuda:{best_idx}")


def resolve_device(requested: str = "auto") -> torch.device:
    """Select best available compute device across multi-GPU CUDA/ROCm, Intel XPU, Apple MPS, and CPU."""
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return _select_best_cuda_device()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--device cuda) but no CUDA GPU is available to PyTorch.")
        return _select_best_cuda_device()
    if req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{req}' requested but CUDA is unavailable to PyTorch.")
        return torch.device(req)
    if req.startswith("xpu"):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(f"Intel XPU device '{req}' requested but XPU backend is unavailable.")
        return torch.device(req if ":" in req else "xpu:0")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested (--device mps) but MPS backend is unavailable.")
        return torch.device("mps")
    return torch.device(req)


def configure_torch_backends(device: torch.device) -> None:
    """Apply safe numerical stability and throughput flags for the resolved device."""
    if device.type == "cuda":
        is_rocm = bool(getattr(torch.version, "hip", None))
        if not is_rocm:
            # Crucial fix for GTX 1650/1650 Ti/1660 (Turing sm_75 without Tensor Cores).
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    elif device.type == "cpu" and not os.environ.get("OMP_NUM_THREADS"):
        # Leave explicit user OMP setting intact; otherwise use balanced physical thread cap.
        cores = os.cpu_count() or 4
        try:
            torch.set_num_threads(max(1, min(cores, 8)))
        except Exception:
            pass


def _safe_reflect_pad(tensor: torch.Tensor, pad_r: int, pad_b: int) -> torch.Tensor:
    """Pad (B, C, H, W) on right/bottom safely even when pad exceeds tensor dimension."""
    cur = tensor
    rem_r = max(0, int(pad_r))
    rem_b = max(0, int(pad_b))
    while rem_r > 0 or rem_b > 0:
        h, w = cur.shape[-2], cur.shape[-1]
        step_r = min(rem_r, max(0, w - 1))
        step_b = min(rem_b, max(0, h - 1))
        if step_r == 0 and rem_r > 0:
            cur = F.pad(cur, (0, rem_r, 0, 0), mode="replicate")
            rem_r = 0
            continue
        if step_b == 0 and rem_b > 0:
            cur = F.pad(cur, (0, 0, 0, rem_b), mode="replicate")
            rem_b = 0
            continue
        cur = F.pad(cur, (0, step_r, 0, step_b), mode="reflect")
        rem_r -= step_r
        rem_b -= step_b
    return cur


def _pad_to_size_requirements(
    tensor: torch.Tensor,
    descriptor: ImageModelDescriptor,
) -> tuple[torch.Tensor, int, int]:
    """Pad input tensor (1, C, H, W) to satisfy Spandrel model's SizeRequirements."""
    _, _, orig_h, orig_w = tensor.shape
    reqs = descriptor.size_requirements
    min_dim = max(1, int(getattr(reqs, "minimum", 1) or 1))
    mult = max(1, int(getattr(reqs, "multiple_of", 1) or 1))
    square = bool(getattr(reqs, "square", False))

    target_h = max(orig_h, min_dim)
    target_w = max(orig_w, min_dim)
    if mult > 1:
        target_h = ((target_h + mult - 1) // mult) * mult
        target_w = ((target_w + mult - 1) // mult) * mult
    if square:
        side = max(target_h, target_w)
        target_h = side
        target_w = side

    pad_b = target_h - orig_h
    pad_r = target_w - orig_w
    if pad_b == 0 and pad_r == 0:
        return tensor, orig_h, orig_w
    return _safe_reflect_pad(tensor, pad_r=pad_r, pad_b=pad_b), orig_h, orig_w


def _precision_cache_key(
    device: torch.device,
    model_id: str,
    value_range: str,
) -> Optional[str]:
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        backend_ver = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None) or "none"
        return (
            f"{props.name}|cc_{props.major}{props.minor}|{backend_ver}|"
            f"torch_{torch.__version__}|{model_id}|{value_range}"
        )
    except Exception:
        return None


def _is_oom_error(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def _first_tensor(out: Any) -> torch.Tensor:
    """Return the image tensor from raw module outputs that may be ``(image, aux, ...)`` tuples (e.g. GFPGAN)."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _estimate_descriptor_fp32_bytes(descriptor: ImageModelDescriptor) -> int:
    try:
        return int(sum(p.numel() * 4 for p in descriptor.model.parameters()))
    except Exception:
        return 64 * 1024 * 1024


def _resolve_best_cuda_or_cpu(requested_device: str, free_gb: float, min_bytes: int) -> bool:
    """Return True when device=='auto' should transparently fall back to CPU due to tight free VRAM."""
    if requested_device.lower().strip() != "auto":
        return False
    return (free_gb * (1024 ** 3)) < float(max(96 * 1024 * 1024, min_bytes))


def _probe_safe_dtype(
    descriptor: ImageModelDescriptor,
    device: torch.device,
    requested_precision: str,
    value_range: str = "0_1",
    model_id: str = "",
    force_reprobe: bool = False,
) -> tuple[torch.dtype, str]:
    """Choose execution dtype with cached/empirical CUDA self-test to prevent FP16 NaN/black bugs."""
    req = requested_precision.lower().strip()
    if device.type == "cpu":
        if req == "bf16":
            return torch.bfloat16, "bf16 (cpu)"
        return torch.float32, "fp32 (cpu)"

    if req == "fp32":
        return torch.float32, "fp32 (requested)"

    if req == "bf16":
        if device.type == "cuda" and getattr(descriptor, "supports_bfloat16", False):
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return torch.bfloat16, "bf16"
        return torch.float32, "fp32 (bf16 unsupported on GPU)"

    # Auto or explicit fp16: check descriptor & persistent cache first.
    supports_half = bool(getattr(descriptor, "supports_half", False))
    if not supports_half:
        return torch.float32, "fp32 (model requires fp32)"

    if device.type == "mps":
        return (torch.float16, "fp16 (mps)") if req == "fp16" else (torch.float32, "fp32 (mps)")

    if device.type == "xpu":
        return (torch.float16, "fp16 (xpu)") if req == "fp16" else (torch.float32, "fp32 (xpu)")

    cache_key = _precision_cache_key(device, model_id, value_range) if (req == "auto" and model_id) else None
    if cache_key and not force_reprobe:
        cached = load_precision_cache().get(cache_key)
        if isinstance(cached, dict):
            dt_name = cached.get("dtype")
            note = cached.get("note", "")
            if dt_name == "fp16":
                return torch.float16, f"{note} [cached]".strip()
            if dt_name == "fp32":
                return torch.float32, f"{note} [cached]".strip()

    # On CUDA (especially Turing sm_75 like GTX 1650 Ti), verify FP16 doesn't yield NaN/black/drift.
    reqs = descriptor.size_requirements
    min_dim = max(32, int(getattr(reqs, "minimum", 32) or 32))
    mult = max(1, int(getattr(reqs, "multiple_of", 1) or 1))
    side = min(96, ((min_dim + mult - 1) // mult) * mult)
    if side < min_dim:
        side = ((min_dim + mult - 1) // mult) * mult

    c_in = int(descriptor.input_channels)
    gen = torch.Generator(device="cpu").manual_seed(7)
    probe_cpu = torch.rand((1, c_in, side, side), generator=gen, dtype=torch.float32)
    if value_range == "-1_1":
        probe_cpu = probe_cpu * 2.0 - 1.0

    try:
        descriptor.model.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            out_fp32 = descriptor(probe_cpu.to(device=device, dtype=torch.float32)).float().cpu()

        descriptor.model.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            out_fp16 = descriptor(probe_cpu.to(device=device, dtype=torch.float16))

        if not torch.isfinite(out_fp16).all():
            raise ValueError("FP16 probe produced NaN/Inf")

        out_fp16_cpu = out_fp16.float().cpu()
        if float(out_fp16_cpu.std()) < 1e-5 and float(out_fp32.std()) > 1e-3:
            raise ValueError("FP16 probe collapsed to flat/black output")

        mse = float(torch.mean((out_fp32 - out_fp16_cpu) ** 2))
        if mse > 2e-2:
            raise ValueError(f"FP16 probe diverged from FP32 (mse={mse:.4f})")

        note = "fp16 (verified)"
        if cache_key:
            save_precision_entry(cache_key, "fp16", note)
        return torch.float16, note
    except Exception as exc:
        if _is_oom_error(exc):
            empty_device_cache(device)
            raise
        # Cleanly fall back to rock-solid FP32 when FP16 is unstable on this hardware/architecture.
        descriptor.model.to(device=device, dtype=torch.float32)
        empty_device_cache(device)
        note = f"fp32 (safe fallback: {exc})"
        if cache_key:
            save_precision_entry(cache_key, "fp32", note)
        return torch.float32, note


def _get_free_device_gb(device: torch.device) -> float:
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return free_bytes / (1024 ** 3)
    if device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            free_bytes, _ = torch.xpu.mem_get_info(device)
            return free_bytes / (1024 ** 3)
        except Exception:
            return 2.5
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            rec = float(torch.mps.recommended_max_memory())
            cur = float(torch.mps.current_allocated_memory())
            return max(1.0, (rec - cur) / (1024 ** 3))
        except Exception:
            return 3.5
    return 0.0


def _recommend_tile_size(
    descriptor: ImageModelDescriptor,
    device: torch.device,
    dtype: torch.dtype,
    tile_complexity: str,
) -> int:
    """Compute a safe starting tile size from free VRAM/RAM and architecture heaviness."""
    if getattr(descriptor, "tiling", None) == ModelTiling.INTERNAL:
        return 0

    if device.type == "cpu":
        return 384 if tile_complexity != "heavy" else 256

    free_gb = _get_free_device_gb(device)
    fp_factor = 1.4 if dtype == torch.float16 else 1.0

    if tile_complexity == "light":
        base = 640 if free_gb >= 3.0 else (448 if free_gb >= 1.8 else 320)
    elif tile_complexity == "heavy":
        # DAT2 / HAT attention maps scale quadratically; 288px fits ~1.5GB on 3.6GB free VRAM.
        if free_gb >= 7.0:
            base = 448
        elif free_gb >= 3.2:
            base = 288
        elif free_gb >= 2.0:
            base = 192
        else:
            base = 128
    else:
        # RealPLKSR / SwinIR medium memory footprint (~210MB at 384px in FP32).
        if free_gb >= 7.0:
            base = 640
        elif free_gb >= 3.2:
            base = 512
        elif free_gb >= 2.0:
            base = 320
        else:
            base = 224

    tile = int(base * (fp_factor if tile_complexity != "heavy" else 1.0))
    mult = max(1, int(getattr(descriptor.size_requirements, "multiple_of", 1) or 1))
    min_dim = max(64, int(getattr(descriptor.size_requirements, "minimum", 64) or 64))
    tile = max(min_dim, (tile // mult) * mult)
    return tile


def _build_feather_mask(
    tile_h: int,
    tile_w: int,
    overlap: int,
    *,
    fade_top: bool,
    fade_bottom: bool,
    fade_left: bool,
    fade_right: bool,
) -> np.ndarray:
    """Create a separable 2D raised-cosine feather mask so overlapping tiles blend with zero seams."""
    wy = np.ones(tile_h, dtype=np.float32)
    wx = np.ones(tile_w, dtype=np.float32)

    if overlap > 0:
        ov_y = min(overlap, tile_h // 2)
        if ov_y > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(ov_y, dtype=np.float32) + 0.5) / ov_y))
            ramp = np.clip(ramp, 1e-3, 1.0)
            if fade_top:
                wy[:ov_y] *= ramp
            if fade_bottom:
                wy[-ov_y:] *= ramp[::-1]

        ov_x = min(overlap, tile_w // 2)
        if ov_x > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(ov_x, dtype=np.float32) + 0.5) / ov_x))
            ramp = np.clip(ramp, 1e-3, 1.0)
            if fade_left:
                wx[:ov_x] *= ramp
            if fade_right:
                wx[-ov_x:] *= ramp[::-1]

    return wy[:, None] * wx[None, :]


def _compute_tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(16, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def apply_tta_forward(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply D4 dihedral symmetry transform k in 0..7 to (B, C, H, W)."""
    if k == 0:
        return x
    if k == 1:
        return torch.flip(x, dims=[-1])
    if k == 2:
        return torch.flip(x, dims=[-2])
    if k == 3:
        return torch.rot90(x, k=2, dims=[-2, -1])
    if k == 4:
        return torch.rot90(x, k=1, dims=[-2, -1])
    if k == 5:
        return torch.rot90(x, k=3, dims=[-2, -1])
    if k == 6:
        return x.transpose(-2, -1).contiguous()
    if k == 7:
        return torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1])
    raise ValueError(f"Invalid TTA index {k}; expected 0..7")


def apply_tta_inverse(y: torch.Tensor, k: int) -> torch.Tensor:
    """Exact algebraic inverse of apply_tta_forward(x, k)."""
    if k == 0:
        return y
    if k == 1:
        return torch.flip(y, dims=[-1])
    if k == 2:
        return torch.flip(y, dims=[-2])
    if k == 3:
        return torch.rot90(y, k=2, dims=[-2, -1])
    if k == 4:
        return torch.rot90(y, k=3, dims=[-2, -1])
    if k == 5:
        return torch.rot90(y, k=1, dims=[-2, -1])
    if k == 6:
        return y.transpose(-2, -1).contiguous()
    if k == 7:
        return torch.flip(torch.rot90(y, k=3, dims=[-2, -1]), dims=[-1])
    raise ValueError(f"Invalid TTA index {k}; expected 0..7")


def execute_tiled_patch_fn(
    rgb_f32: np.ndarray,
    *,
    scale: int,
    forward_patch_fn: Callable[[np.ndarray], np.ndarray],
    tile_size: int,
    overlap: int,
    mult: int = 1,
    min_dim: int = 32,
    memmap_threshold_mp: float = 64.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Shared Hann-feathered tiled inference & ManagedMemmap streaming accumulator."""
    h, w, _ = rgb_f32.shape
    out_h, out_w = h * scale, w * scale

    out_mp = (out_h * out_w) / 1_000_000.0
    mm_thresh = float(memmap_threshold_mp)
    use_memmap = mm_thresh > 0.0 and out_mp >= mm_thresh

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        if progress_cb:
            progress_cb(0, 1)
        res = forward_patch_fn(np.asarray(rgb_f32, dtype=np.float32))
        if progress_cb:
            progress_cb(1, 1)
        if use_memmap:
            mm = create_managed_memmap((out_h, out_w, 3), dtype=np.float32, prefix="azz_out_")
            mm[:] = res
            mm.flush()
            return mm
        return res

    tile_h = min(h, max(min_dim, (tile_size // mult) * mult))
    tile_w = min(w, max(min_dim, (tile_size // mult) * mult))
    ov = max(8, min(overlap, min(tile_h, tile_w) // 3))

    ys = _compute_tile_starts(h, tile_h, ov)
    xs = _compute_tile_starts(w, tile_w, ov)
    total_tiles = len(ys) * len(xs)

    use_fp16_accum = (not use_memmap) and ((out_h * out_w) > 48_000_000)
    if use_memmap:
        accum_out = create_managed_memmap((out_h, out_w, 3), dtype=np.float32, prefix="azz_acc_")
        accum_w = create_managed_memmap((out_h, out_w, 1), dtype=np.float32, prefix="azz_wgt_")
    else:
        acc_dtype = np.float16 if use_fp16_accum else np.float32
        accum_out = np.zeros((out_h, out_w, 3), dtype=acc_dtype)
        accum_w = np.zeros((out_h, out_w, 1), dtype=np.float32)

    idx = 0
    for y0 in ys:
        y1 = min(h, y0 + tile_h)
        for x0 in xs:
            x1 = min(w, x0 + tile_w)
            patch = np.asarray(rgb_f32[y0:y1, x0:x1], dtype=np.float32)
            out_patch = forward_patch_fn(patch)

            oy0, oy1 = y0 * scale, y1 * scale
            ox0, ox1 = x0 * scale, x1 * scale
            mask = _build_feather_mask(
                oy1 - oy0,
                ox1 - ox0,
                ov * scale,
                fade_top=(y0 > 0),
                fade_bottom=(y1 < h),
                fade_left=(x0 > 0),
                fade_right=(x1 < w),
            )[:, :, None]

            if use_fp16_accum:
                cur = accum_out[oy0:oy1, ox0:ox1].astype(np.float32)
                accum_out[oy0:oy1, ox0:ox1] = (cur + out_patch * mask).astype(np.float16)
            else:
                accum_out[oy0:oy1, ox0:ox1] += out_patch * mask
            accum_w[oy0:oy1, ox0:ox1] += mask

            idx += 1
            if progress_cb:
                progress_cb(idx, total_tiles)

    if use_memmap:
        result_mm = create_managed_memmap((out_h, out_w, 3), dtype=np.float32, prefix="azz_out_")
        strip_rows = 512
        for r0 in range(0, out_h, strip_rows):
            r1 = min(out_h, r0 + strip_rows)
            safe_w = np.maximum(np.asarray(accum_w[r0:r1], dtype=np.float32), 1e-6)
            strip = np.asarray(accum_out[r0:r1], dtype=np.float32) / safe_w
            result_mm[r0:r1] = np.clip(strip, 0.0, 1.0)
        result_mm.flush()
        del accum_out, accum_w
        return result_mm

    safe_w = np.maximum(accum_w, 1e-6)
    if use_fp16_accum:
        result = accum_out.astype(np.float32) / safe_w
    else:
        result = accum_out / safe_w
    return np.clip(result, 0.0, 1.0).astype(np.float32)


class SpandrelEngine:
    """Loads and executes any Spandrel-compatible restoration/SR model safely on GPU or CPU."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        runtime: Optional[RuntimeConfig] = None,
        value_range: str = "0_1",
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime or RuntimeConfig()
        self.value_range = value_range

        self.device = resolve_device(self.runtime.device)
        configure_torch_backends(self.device)

        loader = ModelLoader(device=torch.device("cpu"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.meshgrid.*", category=UserWarning)
            loaded = loader.load_from_file(self.model_path)
        if not isinstance(loaded, ImageModelDescriptor):
            raise TypeError(
                f"Model '{self.model_path.name}' is not a supported ImageModelDescriptor."
            )
        self.descriptor: ImageModelDescriptor = loaded
        self.descriptor.eval()

        self._exec_callable: Any = self.descriptor
        self._compiled = False
        # Spandrel's ImageModelDescriptor.__call__ hard-clamps every output to [0, 1]. Models that natively
        # operate in [-1, 1] (GFPGAN) would lose their entire negative half -> every value below mid-grey
        # collapses to 0.5 after de-normalization (washed-out faces, bleached eyes/hair). Call the raw module.
        self._bypass_descriptor_clamp = self.value_range == "-1_1"

        est_bytes = _estimate_descriptor_fp32_bytes(self.descriptor)
        req_dev = self.runtime.device.lower().strip()
        if (
            self.device.type in {"cuda", "xpu"}
            and _resolve_best_cuda_or_cpu(req_dev, _get_free_device_gb(self.device), int(est_bytes * 0.65))
        ):
            self._switch_to_cpu_fallback("low free GPU VRAM")
        else:
            try:
                self.dtype, self.precision_note = _probe_safe_dtype(
                    self.descriptor,
                    self.device,
                    self.runtime.precision,
                    value_range=self.value_range,
                    model_id=self.model_path.name,
                    force_reprobe=bool(self.runtime.force_reprobe),
                )
                self.descriptor.model.to(device=self.device, dtype=self.dtype)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if req_dev == "auto" and self.device.type != "cpu" and _is_oom_error(err):
                    self._switch_to_cpu_fallback("GPU VRAM full")
                else:
                    raise

        if self.runtime.backend.lower().strip() == "compile" and hasattr(torch, "compile"):
            try:
                self._exec_callable = torch.compile(
                    self.descriptor.model,
                    mode="reduce-overhead",
                    dynamic=True,
                )
                self._compiled = True
                self.precision_note = f"{self.precision_note} +compile"
            except Exception:
                self._exec_callable = self.descriptor

        self.scale = int(self.descriptor.scale)
        self.input_channels = int(self.descriptor.input_channels)
        self.output_channels = int(self.descriptor.output_channels)

    def _switch_to_cpu_fallback(self, reason: str) -> None:
        old_dev = self.device
        self._compiled = False
        self._exec_callable = self.descriptor
        try:
            self.descriptor.model.to("cpu", dtype=torch.float32)
        except Exception:
            pass
        empty_device_cache(old_dev)
        gc.collect()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.precision_note = f"fp32 (CPU fallback: {reason})"
        configure_torch_backends(self.device)

    def close(self) -> None:
        """Release model weights from GPU VRAM cleanly."""
        self._exec_callable = None
        try:
            self.descriptor.model.to("cpu")
        except Exception:
            pass
        empty_device_cache(self.device)
        gc.collect()

    def _invoke(self, fn: Any, t_dev: torch.Tensor) -> torch.Tensor:
        """Run one forward pass, bypassing Spandrel's [0, 1] output clamp for [-1, 1] models."""
        if fn is self.descriptor and self._bypass_descriptor_clamp:
            call_fn = getattr(self.descriptor, "_call_fn", None)
            model = self.descriptor.model
            raw = call_fn(model, t_dev) if callable(call_fn) else model(t_dev)
            return _first_tensor(raw)
        return _first_tensor(fn(t_dev))

    def _forward_single_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        """Run model (+ optional TTA) on a float32 HxWx3 [0, 1] RGB patch."""
        arr = np.ascontiguousarray(patch_rgb.transpose(2, 0, 1))[None, ...]
        if self.input_channels == 1:
            arr = (
                0.2126 * arr[:, 0:1] + 0.7152 * arr[:, 1:2] + 0.0722 * arr[:, 2:3]
            )
        t_in = torch.from_numpy(arr)
        if self.value_range == "-1_1":
            t_in = t_in * 2.0 - 1.0

        tta_count = int(self.runtime.tta)
        if tta_count not in (1, 2, 4, 8):
            raise ValueError(f"tta must be 1, 2, 4, or 8 (got {tta_count})")

        accum: Optional[torch.Tensor] = None
        for k in range(tta_count):
            t_aug = apply_tta_forward(t_in, k)
            t_pad, unpad_h, unpad_w = _pad_to_size_requirements(t_aug, self.descriptor)
            t_dev = t_pad.to(device=self.device, dtype=self.dtype, non_blocking=True)
            with torch.inference_mode():
                try:
                    fn = self._exec_callable if self._exec_callable is not None else self.descriptor
                    out_dev = self._invoke(fn, t_dev)
                except Exception as err:
                    # Seamless eager fallback if torch.compile encounters dynamic graph breaks on OOM/edge shapes.
                    if self._compiled and "out of memory" not in str(err).lower():
                        self._compiled = False
                        self._exec_callable = self.descriptor
                        out_dev = self._invoke(self.descriptor, t_dev)
                    else:
                        raise
            out_cpu = out_dev.float().cpu()
            out_crop = out_cpu[:, :, : unpad_h * self.scale, : unpad_w * self.scale]
            out_aligned = apply_tta_inverse(out_crop, k)
            accum = out_aligned if accum is None else (accum + out_aligned)

        assert accum is not None
        out_avg = accum / float(tta_count)
        if self.value_range == "-1_1":
            out_avg = (out_avg + 1.0) * 0.5
        out_avg = torch.clamp(out_avg, 0.0, 1.0)

        out_np = out_avg.squeeze(0).numpy().transpose(1, 2, 0)
        if out_np.shape[2] == 1:
            out_np = np.repeat(out_np, 3, axis=2)
        return np.ascontiguousarray(out_np[:, :, :3], dtype=np.float32)

    def _run_tiled(
        self,
        rgb_f32: np.ndarray,
        tile_size: int,
        overlap: int,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        mult = max(1, int(getattr(self.descriptor.size_requirements, "multiple_of", 1) or 1))
        min_dim = max(32, int(getattr(self.descriptor.size_requirements, "minimum", 32) or 32))
        return execute_tiled_patch_fn(
            rgb_f32,
            scale=self.scale,
            forward_patch_fn=self._forward_single_patch,
            tile_size=tile_size,
            overlap=overlap,
            mult=mult,
            min_dim=min_dim,
            memmap_threshold_mp=float(self.runtime.memmap_threshold_mp),
            progress_cb=progress_cb,
        )

    def enhance(
        self,
        rgb_f32: np.ndarray,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[np.ndarray, int]:
        """Run inference with automatic VRAM OOM tile backoff. Returns (output_rgb, tile_used)."""
        tile_size = (
            int(self.runtime.tile_size)
            if self.runtime.tile_size > 0
            else _recommend_tile_size(
                self.descriptor,
                self.device,
                self.dtype,
                self.runtime.tile_complexity,
            )
        )

        # If model discourages tiling and image is compact, try whole image first before tiling.
        if (
            self.runtime.tile_size <= 0
            and getattr(self.descriptor, "tiling", None) == ModelTiling.DISCOURAGED
            and max(rgb_f32.shape[:2]) <= 512
        ):
            tile_size = 0

        min_tile = max(64, int(getattr(self.descriptor.size_requirements, "minimum", 64) or 64))
        attempt_tile = tile_size
        while True:
            try:
                out = self._run_tiled(
                    rgb_f32,
                    tile_size=attempt_tile,
                    overlap=int(self.runtime.tile_overlap),
                    progress_cb=progress_cb,
                )
                return out, attempt_tile
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if not _is_oom_error(err) or self.device.type not in {"cuda", "mps", "xpu"}:
                    raise
                gc.collect()
                empty_device_cache(self.device)

                if attempt_tile == 0:
                    attempt_tile = min(384, max(rgb_f32.shape[:2]))
                else:
                    next_tile = max(min_tile, (attempt_tile // 2 // 16) * 16)
                    if next_tile >= attempt_tile:
                        if self.runtime.device.lower().strip() == "auto":
                            self._switch_to_cpu_fallback("GPU OOM at minimum tile")
                            attempt_tile = max(256, min_tile)
                            continue
                        raise RuntimeError(
                            f"{self.device.type.upper()} out of memory even at minimum tile size {attempt_tile}."
                        ) from err
                    attempt_tile = next_tile


def create_inference_engine(
    model_path: str | Path,
    *,
    runtime: Optional[RuntimeConfig] = None,
    value_range: str = "0_1",
) -> object:
    """Instantiate the optimal execution engine (PyTorch, torch.compile, ONNX, DirectML, or TensorRT)."""
    from .backends import OnnxEngine, detect_onnx_providers, export_spandrel_to_onnx_cached

    rt = runtime or RuntimeConfig()
    mpath = Path(model_path)
    req_backend = rt.backend.lower().strip()

    if mpath.suffix.lower() == ".onnx":
        return OnnxEngine(mpath, runtime=rt, value_range=value_range)

    # Auto-engage DirectML DX12 on Windows AMD/Intel GPUs when PyTorch would fall back to CPU.
    use_onnx_backend = req_backend in {"onnx", "directml", "tensorrt"}
    if (
        not use_onnx_backend
        and req_backend == "auto"
        and resolve_device(rt.device).type == "cpu"
        and "DmlExecutionProvider" in detect_onnx_providers()
    ):
        try:
            import importlib.util

            if importlib.util.find_spec("onnx") is not None:
                use_onnx_backend = True
                rt = RuntimeConfig(**{**rt.__dict__, "backend": "directml"})
        except Exception:
            pass

    if use_onnx_backend:
        loader = ModelLoader(device=torch.device("cpu"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.meshgrid.*", category=UserWarning)
            descriptor = loader.load_from_file(mpath)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"Model '{mpath.name}' is not a supported ImageModelDescriptor.")
        descriptor.eval()
        onnx_path = export_spandrel_to_onnx_cached(descriptor, mpath)
        scale_hint = int(descriptor.scale)
        del descriptor
        return OnnxEngine(onnx_path, runtime=rt, value_range=value_range, scale_hint=scale_hint)

    return SpandrelEngine(mpath, runtime=rt, value_range=value_range)


class EnginePool:
    """Thread-safe warm GPU model pool across PyTorch, Compiled, ONNX, DirectML & TensorRT engines."""

    def __init__(self, max_models: int = 2) -> None:
        self.max_models = max(1, int(max_models))
        self.lock = threading.RLock()
        self._pool: "OrderedDict[tuple[str, str, str, str, str], object]" = OrderedDict()

    def acquire(
        self,
        model_path: str | Path,
        *,
        runtime: Optional[RuntimeConfig] = None,
        value_range: str = "0_1",
    ) -> Any:
        """Return a hot cached inference engine (evicting LRU cleanly when VRAM is tight)."""
        rt = runtime or RuntimeConfig()
        dev = resolve_device(rt.device)
        resolved = str(Path(model_path).resolve())
        key = (
            resolved,
            str(dev),
            rt.precision.lower().strip(),
            value_range,
            rt.backend.lower().strip(),
        )

        with self.lock:
            if key in self._pool and not rt.force_reprobe:
                eng = self._pool.pop(key)
                stale_cpu_fallback = (
                    getattr(eng, "device", None) is not None
                    and eng.device.type == "cpu"  # type: ignore[attr-defined]
                    and dev.type in {"cuda", "xpu"}
                    and _get_free_device_gb(dev) >= 1.25
                )
                if stale_cpu_fallback:
                    eng.close()  # type: ignore[attr-defined]
                else:
                    eng.runtime = rt  # type: ignore[attr-defined]
                    self._pool[key] = eng
                    return eng

            # Evict oldest models if cache limit reached or free GPU memory is tight (< 1.15 GB).
            while self._pool and (
                len(self._pool) >= self.max_models
                or (dev.type in {"cuda", "xpu"} and _get_free_device_gb(dev) < 1.15)
            ):
                _, evicted = self._pool.popitem(last=False)
                evicted.close()  # type: ignore[attr-defined]
                empty_device_cache(dev)

            if key in self._pool:
                self._pool.pop(key).close()  # type: ignore[attr-defined]

            eng = create_inference_engine(model_path, runtime=rt, value_range=value_range)
            self._pool[key] = eng
            return eng

    def clear(self) -> None:
        """Unload all cached models and reclaim GPU/system memory cleanly."""
        with self.lock:
            while self._pool:
                _, eng = self._pool.popitem(last=False)
                eng.close()  # type: ignore[attr-defined]
            gc.collect()


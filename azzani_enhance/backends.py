"""Multi-vendor accelerated inference backends: ONNX Runtime, DirectML DX12, TensorRT & torch.compile."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hardware import get_cache_root

BACKEND_CHOICES = ("auto", "pytorch", "compile", "onnx", "directml", "tensorrt")


def detect_onnx_providers() -> List[str]:
    """Return available ONNX Runtime execution providers, or empty list if onnxruntime is absent."""
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers())
    except Exception:
        return []


def select_onnx_providers(
    backend: str = "auto",
    device_req: str = "auto",
    *,
    cache_dir: Optional[Path] = None,
    available_providers: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Build prioritized ONNX Runtime Execution Provider list with safe tuning options."""
    avail = list(available_providers) if available_providers is not None else detect_onnx_providers()
    avail_set = set(avail)
    req_b = backend.lower().strip()
    req_d = device_req.lower().strip()

    trt_cache = (cache_dir or (get_cache_root() / "trt_cache")).resolve()
    trt_opts: Dict[str, Any] = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(trt_cache),
    }

    if req_d == "cpu":
        if "CPUExecutionProvider" in avail_set or not avail:
            return ["CPUExecutionProvider"]
        return avail

    if req_b == "tensorrt":
        if "TensorrtExecutionProvider" not in avail_set:
            raise RuntimeError(
                "Requested --backend tensorrt, but onnxruntime TensorrtExecutionProvider is not available. "
                "Install TensorRT + onnxruntime-gpu, or use '--backend pytorch' / '--backend compile'."
            )
        trt_cache.mkdir(parents=True, exist_ok=True)
        providers: List[Any] = [("TensorrtExecutionProvider", trt_opts)]
        if "CUDAExecutionProvider" in avail_set:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    if req_b == "directml":
        if "DmlExecutionProvider" not in avail_set:
            raise RuntimeError(
                "Requested --backend directml, but DmlExecutionProvider is unavailable. "
                "Run 'pip install -e \".[directml]\"' on Windows to enable DirectML DX12 GPU acceleration."
            )
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    # Automatic / general ONNX EP ranking across NVIDIA, AMD, Intel, and Apple silicon.
    ordered: List[Any] = []
    if req_d.startswith("cuda") and "CUDAExecutionProvider" in avail_set:
        ordered.append("CUDAExecutionProvider")
    elif req_d == "mps" and "CoreMLExecutionProvider" in avail_set:
        ordered.append("CoreMLExecutionProvider")
    elif req_d.startswith("xpu") and "OpenVINOExecutionProvider" in avail_set:
        ordered.append("OpenVINOExecutionProvider")
    else:
        for candidate in (
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
        ):
            if candidate in avail_set:
                ordered.append(candidate)

    if "CPUExecutionProvider" in avail_set or not ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def _onnx_cache_path(source_path: Path, opset: int, cache_dir: Optional[Path] = None) -> Path:
    resolved = source_path.resolve()
    try:
        st = resolved.stat()
        sig = f"{resolved}|{st.st_size}|{st.st_mtime_ns}|opset={opset}"
    except OSError:
        sig = f"{resolved}|opset={opset}"
    digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
    root = (cache_dir or (get_cache_root() / "onnx")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{resolved.stem}_{digest}.onnx"


def export_spandrel_to_onnx_cached(
    descriptor: Any,
    source_path: str | Path,
    *,
    cache_dir: Optional[Path] = None,
    opset: int = 17,
) -> Path:
    """Export any Spandrel ImageModelDescriptor to cached dynamic-shape ONNX graph."""
    src = Path(source_path)
    out_onnx = _onnx_cache_path(src, opset=opset, cache_dir=cache_dir)
    if out_onnx.is_file() and out_onnx.stat().st_size > 0:
        return out_onnx

    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "Converting a .safetensors/.pth model to ONNX/DirectML/TensorRT requires the 'onnx' package. "
            "Install it with: pip install -e \".[onnx]\" (or \".[directml]\" on Windows)."
        )

    import torch

    model_cpu = descriptor.model.to("cpu").float().eval()
    in_ch = max(1, int(getattr(descriptor, "input_channels", 3) or 3))
    sample = torch.zeros((1, in_ch, 64, 64), dtype=torch.float32)

    tmp_path = out_onnx.with_name(f".{out_onnx.name}.{os.getpid()}.tmp")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model_cpu,
                sample,
                str(tmp_path),
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "out_height", 3: "out_width"},
                },
            )
        tmp_path.replace(out_onnx)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_onnx


def apply_tta_forward_np(x: np.ndarray, k: int) -> np.ndarray:
    """Apply D4 dihedral transform k in 0..7 to (B, C, H, W) float32 NumPy tensor."""
    if k == 0:
        return x
    if k == 1:
        return np.ascontiguousarray(np.flip(x, axis=-1))
    if k == 2:
        return np.ascontiguousarray(np.flip(x, axis=-2))
    if k == 3:
        return np.ascontiguousarray(np.rot90(x, k=2, axes=(-2, -1)))
    if k == 4:
        return np.ascontiguousarray(np.rot90(x, k=1, axes=(-2, -1)))
    if k == 5:
        return np.ascontiguousarray(np.rot90(x, k=3, axes=(-2, -1)))
    if k == 6:
        return np.ascontiguousarray(np.swapaxes(x, -2, -1))
    if k == 7:
        return np.ascontiguousarray(np.rot90(np.flip(x, axis=-1), k=1, axes=(-2, -1)))
    raise ValueError(f"Invalid TTA index {k}; expected 0..7")


def apply_tta_inverse_np(y: np.ndarray, k: int) -> np.ndarray:
    """Exact algebraic inverse of apply_tta_forward_np(x, k)."""
    if k == 0:
        return y
    if k == 1:
        return np.ascontiguousarray(np.flip(y, axis=-1))
    if k == 2:
        return np.ascontiguousarray(np.flip(y, axis=-2))
    if k == 3:
        return np.ascontiguousarray(np.rot90(y, k=2, axes=(-2, -1)))
    if k == 4:
        return np.ascontiguousarray(np.rot90(y, k=3, axes=(-2, -1)))
    if k == 5:
        return np.ascontiguousarray(np.rot90(y, k=1, axes=(-2, -1)))
    if k == 6:
        return np.ascontiguousarray(np.swapaxes(y, -2, -1))
    if k == 7:
        return np.ascontiguousarray(np.flip(np.rot90(y, k=3, axes=(-2, -1)), axis=-1))
    raise ValueError(f"Invalid TTA index {k}; expected 0..7")


def _pad_to_multiple_np(arr: np.ndarray, mult: int = 16) -> Tuple[np.ndarray, int, int]:
    _, _, h, w = arr.shape
    target_h = max(32, ((h + mult - 1) // mult) * mult)
    target_w = max(32, ((w + mult - 1) // mult) * mult)
    pad_b = target_h - h
    pad_r = target_w - w
    if pad_b == 0 and pad_r == 0:
        return arr, h, w
    mode = "reflect" if (h > 1 and w > 1 and pad_b < h and pad_r < w) else "edge"
    padded = np.pad(arr, ((0, 0), (0, 0), (0, pad_b), (0, pad_r)), mode=mode)
    return np.ascontiguousarray(padded, dtype=np.float32), h, w


class _PseudoDevice:
    """Lightweight device descriptor compatible with SpandrelEngine.device."""

    def __init__(self, device_type: str, label: str) -> None:
        self.type = device_type
        self._label = label

    def __str__(self) -> str:
        return self._label


class OnnxEngine:
    """ONNX Runtime restoration/SR engine supporting CUDA, DirectML DX12, TensorRT, CoreML & CPU."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        runtime: Optional[Any] = None,
        value_range: str = "0_1",
        scale_hint: Optional[int] = None,
    ) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install -e \".[onnx]\" (or \".[directml]\" on Windows AMD/Intel)."
            )
        import onnxruntime as ort  # type: ignore

        from .inference import RuntimeConfig

        self.model_path = Path(onnx_path)
        self.runtime = runtime or RuntimeConfig()
        self.value_range = value_range

        providers = select_onnx_providers(self.runtime.backend, self.runtime.device)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)

        active = list(self.session.get_providers())
        top_ep = active[0] if active else "CPUExecutionProvider"
        ep_to_dev = {
            "TensorrtExecutionProvider": ("cuda", "tensorrt"),
            "CUDAExecutionProvider": ("cuda", "onnx-cuda"),
            "DmlExecutionProvider": ("dml", "directml"),
            "ROCMExecutionProvider": ("cuda", "onnx-rocm"),
            "CoreMLExecutionProvider": ("mps", "onnx-coreml"),
            "OpenVINOExecutionProvider": ("xpu", "onnx-openvino"),
            "CPUExecutionProvider": ("cpu", "onnx-cpu"),
        }
        dev_type, label = ep_to_dev.get(top_ep, ("cpu", f"onnx:{top_ep}"))
        self.device = _PseudoDevice(dev_type, label)
        self.precision_note = f"{label} ({top_ep})"

        in_meta = self.session.get_inputs()[0]
        out_meta = self.session.get_outputs()[0]
        self.input_name = in_meta.name
        self.output_name = out_meta.name

        in_shape = list(in_meta.shape)
        self.input_channels = int(in_shape[1]) if (len(in_shape) >= 4 and isinstance(in_shape[1], int)) else 3
        self.output_channels = 3
        self.scale = int(scale_hint) if scale_hint else self._probe_scale()

    def _probe_scale(self) -> int:
        dummy = np.zeros((1, self.input_channels, 32, 32), dtype=np.float32)
        out = self.session.run([self.output_name], {self.input_name: dummy})[0]
        if out.ndim >= 4 and out.shape[2] >= 32:
            self.output_channels = int(out.shape[1])
            return max(1, int(round(out.shape[2] / 32.0)))
        return 1

    def close(self) -> None:
        self.session = None
        gc.collect()

    def _forward_single_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(patch_rgb.transpose(2, 0, 1))[None, ...].astype(np.float32)
        if self.input_channels == 1:
            arr = 0.2126 * arr[:, 0:1] + 0.7152 * arr[:, 1:2] + 0.0722 * arr[:, 2:3]
        if self.value_range == "-1_1":
            arr = arr * 2.0 - 1.0

        tta_count = int(self.runtime.tta)
        if tta_count not in (1, 2, 4, 8):
            raise ValueError(f"tta must be 1, 2, 4, or 8 (got {tta_count})")

        accum: Optional[np.ndarray] = None
        for k in range(tta_count):
            t_aug = apply_tta_forward_np(arr, k)
            t_pad, uh, uw = _pad_to_multiple_np(t_aug, mult=16)
            out_raw = self.session.run([self.output_name], {self.input_name: t_pad})[0]
            out_crop = out_raw[:, :, : uh * self.scale, : uw * self.scale]
            out_aligned = apply_tta_inverse_np(out_crop, k)
            accum = out_aligned if accum is None else (accum + out_aligned)

        assert accum is not None
        out_avg = accum / float(tta_count)
        if self.value_range == "-1_1":
            out_avg = (out_avg + 1.0) * 0.5
        out_avg = np.clip(out_avg, 0.0, 1.0)

        out_np = out_avg.squeeze(0).transpose(1, 2, 0)
        if out_np.shape[2] == 1:
            out_np = np.repeat(out_np, 3, axis=2)
        return np.ascontiguousarray(out_np[:, :, :3], dtype=np.float32)

    def enhance(
        self,
        rgb_f32: np.ndarray,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[np.ndarray, int]:
        """Run tiled ONNX inference with automatic EP OOM backoff."""
        from .inference import execute_tiled_patch_fn

        tile_size = int(self.runtime.tile_size) if self.runtime.tile_size > 0 else (
            0 if self.scale <= 1 and max(rgb_f32.shape[:2]) <= 512 else 384
        )
        attempt_tile = tile_size
        while True:
            try:
                out = execute_tiled_patch_fn(
                    rgb_f32,
                    scale=self.scale,
                    forward_patch_fn=self._forward_single_patch,
                    tile_size=attempt_tile,
                    overlap=int(self.runtime.tile_overlap),
                    mult=16,
                    min_dim=64,
                    memmap_threshold_mp=float(self.runtime.memmap_threshold_mp),
                    progress_cb=progress_cb,
                )
                return out, attempt_tile
            except Exception as err:
                msg = str(err).lower()
                if "out of memory" not in msg and "alloc" not in msg:
                    raise
                if attempt_tile == 0:
                    attempt_tile = min(256, max(rgb_f32.shape[:2]))
                else:
                    next_tile = max(64, (attempt_tile // 2 // 16) * 16)
                    if next_tile >= attempt_tile:
                        raise
                    attempt_tile = next_tile

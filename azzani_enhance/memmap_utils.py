"""Out-of-core disk-backed memmap helpers for gigantic >64 MP to 500+ MP prints."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import weakref
from typing import Any, Tuple

import numpy as np


def _is_tmpfs_linux(path: Path) -> bool:
    """Return True if path resides on a RAM-backed tmpfs mount on Linux."""
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return False
    resolved = str(path.resolve())
    best_mount = ""
    best_fstype = ""
    try:
        for line in mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mnt, fstype = parts[1], parts[2]
                if resolved == mnt or resolved.startswith(mnt.rstrip("/") + "/") or mnt == "/":
                    if len(mnt) > len(best_mount):
                        best_mount = mnt
                        best_fstype = fstype
    except Exception:
        return False
    return best_fstype == "tmpfs"


def select_memmap_dir() -> Path:
    """Choose a real-disk directory (avoiding RAM-backed /tmp tmpfs) for out-of-core buffers."""
    env_dir = os.environ.get("AZZANI_MEMMAP_DIR")
    if env_dir:
        target = Path(env_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    sys_tmp = Path(tempfile.gettempdir())
    var_tmp = Path("/var/tmp")
    if var_tmp.is_dir() and os.access(var_tmp, os.W_OK):
        try:
            sys_is_tmpfs = _is_tmpfs_linux(sys_tmp)
            var_is_tmpfs = _is_tmpfs_linux(var_tmp)
            if sys_is_tmpfs and not var_is_tmpfs:
                return var_tmp
            if shutil.disk_usage(var_tmp).free > shutil.disk_usage(sys_tmp).free:
                return var_tmp
        except Exception:
            pass
    return sys_tmp


class ManagedMemmap(np.memmap):
    """NumPy memmap subclass whose temporary backing folder auto-cleans upon garbage collection."""

    def __array_finalize__(self, obj: Any) -> None:
        super().__array_finalize__(obj)
        self._memmap_finalizer = getattr(obj, "_memmap_finalizer", None)


def create_managed_memmap(
    shape: Tuple[int, ...],
    dtype: Any = np.float32,
    *,
    prefix: str = "azzani_mm_",
) -> ManagedMemmap:
    """Allocate a zero-initialized disk-backed memmap array outside tmpfs."""
    base_dir = select_memmap_dir()
    scratch_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir)))
    raw_file = scratch_dir / "buffer.dat"
    arr = ManagedMemmap(raw_file, dtype=dtype, mode="w+", shape=tuple(int(d) for d in shape))
    arr._memmap_finalizer = weakref.finalize(arr, shutil.rmtree, str(scratch_dir), True)  # type: ignore[attr-defined]
    return arr


def is_memmap_array(arr: Any) -> bool:
    """Return True when an array is disk-backed via np.memmap."""
    return isinstance(arr, np.memmap)


def allocate_canvas(
    shape: Tuple[int, ...],
    dtype: Any = np.float32,
    *,
    memmap_threshold_mp: float = 64.0,
    force_memmap: bool = False,
) -> np.ndarray:
    """Allocate a canvas in RAM for normal sizes, or disk-backed memmap above threshold."""
    pixels = int(shape[0]) * int(shape[1]) if len(shape) >= 2 else int(np.prod(shape))
    mp = pixels / 1_000_000.0
    if force_memmap or (memmap_threshold_mp > 0.0 and mp >= memmap_threshold_mp):
        return create_managed_memmap(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def stride_preview(arr: np.ndarray, *, max_side: int = 512, clip_unit: bool = True) -> np.ndarray:
    """Return a bounded-memory float32 strided view copy without loading a huge memmap into RAM."""
    h, w = arr.shape[:2]
    step = max(1, int(max(h, w) // max(16, int(max_side))))
    sub = np.asarray(arr[::step, ::step], dtype=np.float32)
    return np.clip(sub, 0.0, 1.0) if clip_unit else sub


def extract_resized_strip(
    src: np.ndarray,
    dst_h: int,
    dst_w: int,
    dy0: int,
    dy1: int,
    *,
    interpolation: int | None = None,
    x_map: np.ndarray | None = None,
) -> np.ndarray:
    """Compute strip [dy0:dy1, :] of resizing src -> (dst_h, dst_w) without allocating full target."""
    import cv2

    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    sh, sw = src.shape[:2]
    was_singleton = src.ndim == 3 and src.shape[2] == 1
    if sh == dst_h and sw == dst_w:
        strip = np.asarray(src[dy0:dy1], dtype=np.float32)
        if was_singleton and strip.ndim == 2:
            strip = strip[:, :, None]
        return strip

    scale_y = sh / float(dst_h)
    scale_x = sw / float(dst_w)
    y_coords = (np.arange(dy0, dy1, dtype=np.float32) + 0.5) * scale_y - 0.5
    if x_map is None:
        x_coords = (np.arange(dst_w, dtype=np.float32) + 0.5) * scale_x - 0.5
        map_x = np.broadcast_to(x_coords[None, :], (dy1 - dy0, dst_w))
    else:
        map_x = (
            np.broadcast_to(x_map[None, :], (dy1 - dy0, dst_w))
            if x_map.ndim == 1
            else x_map
        )

    pad_src = 4 if interpolation in (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4) else 2
    # If strongly downscaling, coarse-stride the source block slightly so remap avoids aliasing.
    pre_step_y = max(1, int(scale_y // 2)) if scale_y >= 3.0 else 1
    pre_step_x = max(1, int(scale_x // 2)) if scale_x >= 3.0 else 1

    sy0 = max(0, int(np.floor(float(y_coords[0]))) - pad_src * pre_step_y)
    sy1 = min(sh, int(np.ceil(float(y_coords[-1]))) + (pad_src + 1) * pre_step_y)
    # Snap sy0 to pre_step_y multiple if pre-striding so y mapping aligns consistently.
    if pre_step_y > 1:
        sy0 = (sy0 // pre_step_y) * pre_step_y

    src_patch = np.asarray(src[sy0:sy1:pre_step_y, ::pre_step_x], dtype=np.float32)
    y_local = (y_coords - float(sy0)) / float(pre_step_y)
    map_x_local = (map_x / float(pre_step_x)).astype(np.float32, copy=False)
    map_y = np.broadcast_to(y_local[:, None], (dy1 - dy0, dst_w)).astype(np.float32, copy=False)

    interp_remap = cv2.INTER_CUBIC if (interpolation == cv2.INTER_AREA) else interpolation
    remapped = cv2.remap(
        src_patch,
        map_x_local,
        map_y,
        interpolation=interp_remap,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if was_singleton and remapped.ndim == 2:
        remapped = remapped[:, :, None]
    elif not was_singleton and src.ndim == 3 and remapped.ndim == 2:
        remapped = remapped[:, :, None]
    return remapped

"""Temporal-consistent video enhancement with scene-cut resets, Farneback flow & warm EnginePool."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .image_io import LoadedImage
from .inference import EnginePool
from .pipeline import EnhancementConfig, enhance_loaded_image

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".flv"}


def is_video_file(path: Path) -> bool:
    """Return True when file suffix matches a supported video container."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


@dataclass
class VideoConfig:
    """Temporal stability and video container encoding options."""

    temporal_strength: float = 0.35  # [0..0.85] occlusion-gated optical-flow EMA blend
    scene_cut_threshold: float = 0.32  # Hard-cut histogram/luma jump threshold
    max_frames: Optional[int] = None  # Optional frame limit for quick previews
    preserve_audio: bool = True  # Copy source audio tracks when ffmpeg is available
    crf: int = 16  # H.264 CRF when encoding via ffmpeg
    fps: Optional[float] = None  # Override output FPS (None = match source)


@dataclass
class VideoResult:
    """Summary metrics produced after enhancing a video file."""

    output_path: Path
    frames_processed: int
    scene_cuts_detected: int
    fps: float
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    audio_copied: bool
    elapsed_seconds: float
    io_backend: str


def detect_scene_cut(
    prev_rgb: np.ndarray,
    cur_rgb: np.ndarray,
    *,
    threshold: float = 0.32,
) -> tuple[bool, float]:
    """Return (is_scene_cut, delta_score) comparing consecutive frames via histogram & luma MAD."""
    p_s = cv2.resize(np.clip(prev_rgb, 0.0, 1.0), (96, 96), interpolation=cv2.INTER_AREA)
    c_s = cv2.resize(np.clip(cur_rgb, 0.0, 1.0), (96, 96), interpolation=cv2.INTER_AREA)

    py = (0.2126 * p_s[:, :, 0] + 0.7152 * p_s[:, :, 1] + 0.0722 * p_s[:, :, 2]).astype(np.float32)
    cy = (0.2126 * c_s[:, :, 0] + 0.7152 * c_s[:, :, 1] + 0.0722 * c_s[:, :, 2]).astype(np.float32)
    mad_y = float(np.mean(np.abs(cy - py)))

    # Smoothed 16-bin grayscale histogram Bhattacharyya distance (invariant to pans & mild AE flicker).
    kern = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    hist_p = np.convolve(np.histogram(py, bins=16, range=(0.0, 1.001))[0].astype(np.float32), kern, mode="same")
    hist_c = np.convolve(np.histogram(cy, bins=16, range=(0.0, 1.001))[0].astype(np.float32), kern, mode="same")
    hist_p /= max(1e-6, float(hist_p.sum()))
    hist_c /= max(1e-6, float(hist_c.sum()))
    hist_dist = float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(hist_p * hist_c))))))

    score = float(np.clip(0.60 * hist_dist + 0.40 * mad_y, 0.0, 1.0))
    return (score >= float(threshold)), score


def temporal_stabilize_frame(
    cur_enh: np.ndarray,
    prev_enh: np.ndarray,
    prev_in: np.ndarray,
    cur_in: np.ndarray,
    *,
    temporal_strength: float = 0.35,
) -> np.ndarray:
    """Warp prev_enh onto cur_enh using dense Farneback optical flow with occlusion gating."""
    strength = float(np.clip(temporal_strength, 0.0, 0.85))
    if strength <= 0.0:
        return cur_enh
    out_h, out_w = cur_enh.shape[:2]
    if prev_enh.shape[:2] != (out_h, out_w):
        return cur_enh
    if (out_h * out_w) >= 24_000_000:
        return cur_enh

    in_h, in_w = cur_in.shape[:2]
    flow_side = min(384, max(in_h, in_w))
    fw = max(32, int(round(in_w * (flow_side / float(max(in_h, in_w))))))
    fh = max(32, int(round(in_h * (flow_side / float(max(in_h, in_w))))))

    def _to_gray_u8(rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(np.clip(rgb, 0.0, 1.0), (fw, fh), interpolation=cv2.INTER_AREA)
        y = 0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]
        return np.rint(np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Backward optical flow cur -> prev aligns prev_enh directly into cur coordinates via remap.
    flow_s = cv2.calcOpticalFlowFarneback(
        _to_gray_u8(cur_in),
        _to_gray_u8(prev_in),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_out = cv2.resize(flow_s, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    flow_out[:, :, 0] *= float(out_w) / float(fw)
    flow_out[:, :, 1] *= float(out_h) / float(fh)

    grid_x, grid_y = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    map_x = grid_x + flow_out[:, :, 0]
    map_y = grid_y + flow_out[:, :, 1]

    warped_prev = cv2.remap(
        np.clip(prev_enh, 0.0, 1.0).astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    in_bounds = (
        (map_x >= 0.0) & (map_x <= float(out_w - 1))
        & (map_y >= 0.0) & (map_y <= float(out_h - 1))
    ).astype(np.float32)

    cur_y = 0.2126 * cur_enh[:, :, 0] + 0.7152 * cur_enh[:, :, 1] + 0.0722 * cur_enh[:, :, 2]
    wrp_y = 0.2126 * warped_prev[:, :, 0] + 0.7152 * warped_prev[:, :, 1] + 0.0722 * warped_prev[:, :, 2]
    err = cv2.GaussianBlur(np.abs(cur_y - wrp_y).astype(np.float32), (5, 5), sigmaX=1.0)

    # Smooth Gaussian rejection cancels temporal blending wherever occlusion / fast motion diverges.
    occ_keep = np.exp(-((err / 0.055) ** 2)).astype(np.float32)
    alpha = (strength * occ_keep * in_bounds)[:, :, None]

    return np.clip(cur_enh * (1.0 - alpha) + warped_prev * alpha, 0.0, 1.0).astype(np.float32)


def _open_cv2_writer(out_path: Path, fps: float, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1", "MJPG", "XVID"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size_wh)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open OpenCV VideoWriter for '{out_path}'.")


def _try_ffmpeg_audio_mux(
    src_video: Path,
    video_only_path: Path,
    final_output: Path,
    *,
    crf: int,
) -> Tuple[bool, str]:
    """Transcode OpenCV temp video to crisp H.264/AAC MP4 with copied source audio when ffmpeg exists."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.move(str(video_only_path), str(final_output))
        return False, "opencv"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(src_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(int(crf)),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        str(final_output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode == 0 and final_output.is_file() and final_output.stat().st_size > 0:
        video_only_path.unlink(missing_ok=True)
        has_audio = shutil.which("ffprobe") is not None and _probe_has_audio(src_video)
        return has_audio, "ffmpeg"
    # Fallback cleanly if ffmpeg libx264 flags unsupported by container extension (e.g. .avi)
    shutil.move(str(video_only_path), str(final_output))
    return False, "opencv"


def _probe_has_audio(src_video: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(src_video),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.returncode == 0 and "audio" in (proc.stdout or "").lower()


def enhance_video_file(
    input_path: str | Path,
    output_path: str | Path,
    cfg: Optional[EnhancementConfig] = None,
    video_cfg: Optional[VideoConfig] = None,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    engine_pool: Optional[EnginePool] = None,
) -> VideoResult:
    """Enhance a video frame-by-frame with warm EnginePool, scene-cut reset & optical-flow stability."""
    t_start = time.perf_counter()
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cfg = cfg or EnhancementConfig()
    vcfg = video_cfg or VideoConfig()

    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video '{in_p}'.")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if src_fps <= 1.0 or not np.isfinite(src_fps):
        src_fps = 24.0
    out_fps = float(vcfg.fps) if vcfg.fps and vcfg.fps > 0 else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    own_pool = engine_pool is None
    pool = engine_pool or EnginePool(max_models=2)

    writer: Optional[cv2.VideoWriter] = None
    temp_dir = Path(tempfile.mkdtemp(prefix="azzani_vid_"))
    temp_out = temp_dir / f"temp{out_p.suffix or '.mp4'}"

    prev_in: Optional[np.ndarray] = None
    prev_enh: Optional[np.ndarray] = None
    frames_done = 0
    cuts = 0
    in_size = (0, 0)
    out_size = (0, 0)

    try:
        while True:
            if vcfg.max_frames is not None and frames_done >= int(vcfg.max_frames):
                break
            ok, bgr_u8 = cap.read()
            if not ok or bgr_u8 is None:
                break

            rgb_f32 = np.ascontiguousarray(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            ih, iw = rgb_f32.shape[:2]
            in_size = (iw, ih)

            loaded = LoadedImage(
                rgb=rgb_f32,
                alpha=None,
                source_bit_depth=8,
                source_format="VIDEO_FRAME",
                exif_bytes=None,
                icc_profile=None,
                original_size=(iw, ih),
            )

            res = enhance_loaded_image(loaded, cfg=cfg, engine_pool=pool)
            cur_enh = np.asarray(res.output_rgb, dtype=np.float32)
            oh, ow = cur_enh.shape[:2]
            out_size = (ow, oh)

            if prev_in is not None and prev_enh is not None and vcfg.temporal_strength > 0.0:
                is_cut, _ = detect_scene_cut(prev_in, rgb_f32, threshold=vcfg.scene_cut_threshold)
                if is_cut:
                    cuts += 1
                    prev_enh = None
                else:
                    cur_enh = temporal_stabilize_frame(
                        cur_enh,
                        prev_enh,
                        prev_in,
                        rgb_f32,
                        temporal_strength=vcfg.temporal_strength,
                    )

            if writer is None:
                writer = _open_cv2_writer(temp_out, out_fps, (ow, oh))

            out_u8 = cv2.cvtColor(np.rint(np.clip(cur_enh, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(out_u8)

            prev_in = rgb_f32
            prev_enh = cur_enh
            frames_done += 1

            if status_cb:
                denom = str(total_frames) if total_frames > 0 else "?"
                status_cb(f"Video frame {frames_done}/{denom} ({ow}x{oh}, cuts={cuts})")

        if frames_done == 0:
            raise RuntimeError(f"Video '{in_p}' contained zero readable frames.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if own_pool:
            pool.clear()

    if vcfg.preserve_audio and shutil.which("ffmpeg"):
        audio_copied, io_backend = _try_ffmpeg_audio_mux(in_p, temp_out, out_p, crf=vcfg.crf)
    else:
        shutil.move(str(temp_out), str(out_p))
        audio_copied, io_backend = False, "opencv"
    shutil.rmtree(temp_dir, ignore_errors=True)

    return VideoResult(
        output_path=out_p,
        frames_processed=frames_done,
        scene_cuts_detected=cuts,
        fps=out_fps,
        input_size=in_size,
        output_size=out_size,
        audio_copied=audio_copied,
        elapsed_seconds=time.perf_counter() - t_start,
        io_backend=io_backend,
    )

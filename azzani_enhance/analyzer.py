"""No-reference image quality diagnostics and adaptive degradation analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

# Minimum short side (px) for FFT motion-blur analysis; smaller images yield unreliable anisotropy.
MIN_MOTION_ANALYSIS_SIDE = 256


@dataclass
class ImageDiagnostics:
    """Comprehensive mathematical quality and degradation profile for an input image."""

    width: int
    height: int
    megapixels: float
    sharpness_score: float  # [0, 1] higher = crisper
    blur_severity: float  # [0, 1] higher = blurrier
    noise_sigma: float  # Estimated Gaussian/ISO noise sigma on 0..255 scale
    noise_severity: float  # [0, 1] higher = noisier
    jpeg_blockiness: float  # [0, 1] higher = heavier 8x8 compression blocking
    mean_luma: float  # [0, 100] CIE L* mean
    dynamic_range: float  # [0, 100] P99 - P01 in CIE L*
    shadow_clip_pct: float  # % pixels with L* < 3
    highlight_clip_pct: float  # % pixels with L* > 97
    color_cast_strength: float  # [0, 1] neutral-weighted LAB tint deviation
    color_cast_ab: tuple[float, float]  # (mean_a_offset, mean_b_offset)
    is_illustration_likely: bool
    recommended_preset: str
    recommended_dejpeg: bool
    recommended_denoise: bool
    recommended_exposure_boost: float  # >0 lift shadows / brighten, <0 recover brights
    motion_blur_score: float = 0.0  # [0, 1] directional camera-shake / motion streak strength
    motion_blur_angle_deg: float = 0.0  # [0, 180) dominant motion blur orientation angle
    recommended_deblur: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_noise_sigma(gray_255: np.ndarray) -> float:
    """Immerkaer 3x3 high-pass noise variance estimator masked to smooth regions."""
    h, w = gray_255.shape
    if h < 16 or w < 16:
        return 0.0

    # Fast Immerkaer noise operator (sum of squared weights == 1.0).
    kernel = np.array(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        dtype=np.float32,
    ) / 6.0
    residual = cv2.filter2D(gray_255, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)

    # Mask out textured edges AND clipped flat areas (0 or 255) so texture isn't mistaken for noise.
    gx = cv2.Sobel(gray_255, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_255, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    valid = (gray_255 > 4.0) & (gray_255 < 251.0)
    if np.count_nonzero(valid) < 64:
        return 0.0

    grad_valid = grad[valid]
    thresh = float(np.percentile(grad_valid, 45))
    smooth_mask = valid & (grad <= max(thresh, 1e-3))
    samples = np.abs(residual[smooth_mask])
    if samples.size < 32:
        samples = np.abs(residual[valid])

    # Robust MAD estimator for N(0, sigma^2): sigma = median(|r|) / 0.67449
    mad_sigma = float(np.median(samples)) / 0.67449
    return float(np.clip(mad_sigma, 0.0, 60.0))


def _estimate_multiscale_noise_sigma(gray_255: np.ndarray) -> float:
    """Detect both fine sensor noise (1x) and coarse scanned film grain / compression speckles (0.5x)."""
    sig_1x = _estimate_noise_sigma(gray_255)
    h, w = gray_255.shape
    if min(h, w) < 128:
        return sig_1x
    gray_half = cv2.resize(gray_255, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    # 2x2 area averaging halves white noise std; 1.6x catches coarse 2x2 grain without magnifying clean edges.
    sig_05x = _estimate_noise_sigma(gray_half) * 1.6
    return float(max(sig_1x, sig_05x))


def _estimate_jpeg_blockiness(gray_255: np.ndarray) -> float:
    """Detect 8x8 DCT grid discontinuities characteristic of JPEG/WebP block artifacts."""
    h, w = gray_255.shape
    if h < 32 or w < 32:
        return 0.0

    dx = np.abs(gray_255[:, 1:] - gray_255[:, :-1])
    dy = np.abs(gray_255[1:, :] - gray_255[:-1, :])

    # 8-pixel boundaries are at index 7, 15, 23, ...
    x_idx = np.arange(dx.shape[1])
    y_idx = np.arange(dy.shape[0])
    bx_mask = (x_idx % 8) == 7
    by_mask = (y_idx % 8) == 7

    # Compare boundary gradient vs immediate neighbors (+/- 1 pixel) to ignore natural slopes.
    ix_mask = ((x_idx % 8) == 3) | ((x_idx % 8) == 4)
    iy_mask = ((y_idx % 8) == 3) | ((y_idx % 8) == 4)

    bx = float(np.mean(dx[:, bx_mask])) if np.any(bx_mask) else 0.0
    ix = float(np.mean(dx[:, ix_mask])) if np.any(ix_mask) else 0.0
    by = float(np.mean(dy[by_mask, :])) if np.any(by_mask) else 0.0
    iy = float(np.mean(dy[iy_mask, :])) if np.any(iy_mask) else 0.0

    # Small epsilon prevents division instability on synthetic flat blocks.
    ratio_x = (bx - ix) / (ix + 1.5)
    ratio_y = (by - iy) / (iy + 1.5)
    raw = max(0.0, 0.5 * (ratio_x + ratio_y))
    return float(np.clip(raw / 0.35, 0.0, 1.0))


def _estimate_sharpness(gray_255: np.ndarray, noise_sigma: float) -> float:
    """Compute noise-compensated multi-scale edge sharpness score in [0, 1]."""
    h, w = gray_255.shape
    if h < 16 or w < 16:
        return 0.5

    # Mildly suppress pixel noise prior to edge measurement so high ISO noise doesn't fake sharpness.
    if noise_sigma > 1.5:
        sigma_s = float(np.clip(noise_sigma * 0.12, 0.4, 1.5))
        clean = cv2.GaussianBlur(gray_255, (0, 0), sigmaX=sigma_s)
    else:
        clean = gray_255

    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    grad_mag = np.hypot(gx, gy)

    lap = cv2.Laplacian(clean, cv2.CV_32F, ksize=3)
    # Focus on top edge percentile (P90..P99) so blurred subjects against flat skies are caught.
    p95_grad = float(np.percentile(grad_mag, 95))
    p95_lap = float(np.percentile(np.abs(lap), 95))

    # Downsample comparison (blurry images barely change when blurred slightly more).
    reblur = cv2.GaussianBlur(clean, (5, 5), sigmaX=1.2)
    gx_b = cv2.Sobel(reblur, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy_b = cv2.Sobel(reblur, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    grad_b = np.hypot(gx_b, gy_b)
    p95_grad_b = float(np.percentile(grad_b, 95))
    edge_retention_drop = max(0.0, (p95_grad - p95_grad_b) / (p95_grad + 1e-3))

    score_grad = np.clip(p95_grad / 32.0, 0.0, 1.0)
    score_lap = np.clip(p95_lap / 45.0, 0.0, 1.0)
    score_drop = np.clip(edge_retention_drop / 0.42, 0.0, 1.0)

    sharpness = 0.35 * score_grad + 0.35 * score_lap + 0.30 * score_drop
    return float(np.clip(sharpness, 0.0, 1.0))


def _detect_illustration_heuristic(rgb_f32: np.ndarray, gray_255: np.ndarray) -> bool:
    """Heuristic distinguishing flat-shaded anime/illustrations from real-world photography."""
    h, w = gray_255.shape
    if h < 32 or w < 32:
        return False

    # Downsample for speed while preserving gradient statistics.
    scale = min(1.0, 384.0 / max(h, w))
    if scale < 1.0:
        small_gray = cv2.resize(
            gray_255,
            (max(16, int(w * scale)), max(16, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_gray = gray_255

    gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    grad = np.hypot(gx, gy)

    # Illustrations have a high fraction of nearly-zero gradient regions punctuated by crisp ink lines.
    flat_ratio = float(np.mean(grad < 1.2))
    strong_edge_ratio = float(np.mean(grad > 14.0))
    return bool(flat_ratio > 0.62 and strong_edge_ratio > 0.04)


def _estimate_motion_blur(gray_255: np.ndarray, noise_sigma: float = 0.0) -> tuple[float, float]:
    """Detect directional camera-shake/motion streaks via Hann-windowed 2D FFT angular transfer ratio."""
    h, w = gray_255.shape
    # Below ~256px each 10-degree spectral wedge holds only a handful of FFT samples and the 8x8 JPEG grid
    # lands inside the high-frequency ring, so anisotropy estimates are dominated by noise (false positives
    # that previously routed tiny web images into the NAFNet deblur pass).
    if min(h, w) < MIN_MOTION_ANALYSIS_SIDE:
        return 0.0, 0.0

    g = gray_255.astype(np.float32, copy=False)
    if noise_sigma > 3.5:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=0.65)

    side = min(h, w, 384)
    if min(h, w) > side:
        scale = side / float(min(h, w))
        g = cv2.resize(g, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = g.shape

    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    patch = g[y0 : y0 + side, x0 : x0 + side]

    # High-pass pre-filtering strips low-frequency sky/wall ramps that could mimic angular skew.
    hp = patch - cv2.GaussianBlur(patch, (0, 0), sigmaX=6.0)
    win = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    pwr = np.abs(np.fft.fftshift(np.fft.fft2(hp * win)))

    cy, cx = side // 2, side // 2
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    rr = np.hypot(yy - cy, xx - cx)
    rmax = 0.5 * side
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 180.0) % 180.0

    low_ring = (rr >= 0.05 * rmax) & (rr < 0.16 * rmax)
    high_ring = (rr >= 0.22 * rmax) & (rr <= 0.46 * rmax)

    bins = 18
    lows = np.zeros(bins, dtype=np.float32)
    ratios = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        wmask = (ang >= i * 10.0) & (ang < (i + 1) * 10.0)
        lm = low_ring & wmask
        hm = high_ring & wmask
        el = float(np.mean(pwr[lm])) if np.any(lm) else 0.0
        eh = float(np.mean(pwr[hm])) if np.any(hm) else 0.0
        lows[i] = el
        ratios[i] = eh / (el + 1e-3)

    valid = lows > max(0.03 * float(np.max(lows)), 0.20 * float(np.percentile(lows, 75)))
    best_ortho_score = 0.0
    best_angle = 0.0
    valid_pairs = 0
    for i in range(bins // 2):
        j = i + bins // 2
        if valid[i] and valid[j]:
            valid_pairs += 1
            hi = max(float(ratios[i]), float(ratios[j]))
            lo = min(float(ratios[i]), float(ratios[j]))
            score_i = (hi - lo) / (hi + lo + 1e-6)
            if score_i > best_ortho_score:
                best_ortho_score = score_i
                blur_idx = i if ratios[i] < ratios[j] else j
                best_angle = (blur_idx * 10.0 + 5.0) % 180.0

    # Reject synthetic 1D periodic stripes that lack 2D frequency support across orthogonal wedges.
    if valid_pairs < 2:
        return 0.0, 0.0

    mean_hf_ratio = float(np.mean(ratios[valid]))
    soft_gate = float(np.clip((0.72 - mean_hf_ratio) / 0.42, 0.20, 1.0))
    motion_score = float(np.clip((best_ortho_score - 0.24) / 0.36, 0.0, 1.0) * soft_gate)
    return motion_score, best_angle


def analyze_image(rgb_f32: np.ndarray) -> ImageDiagnostics:
    """Run fast, memory-safe quality & degradation analysis on a float32 [0, 1] RGB image."""
    h, w = rgb_f32.shape[:2]
    mp = (h * w) / 1_000_000.0

    # For ultra-high-resolution images (>12 MP or memmap), avoid multi-GB full-res copies:
    # analyze an 8px-aligned native 1:1 crop (up to 2048px) for noise/DCT/FFT and a strided view for LAB tone.
    if (h * w) > 12_000_000 or isinstance(rgb_f32, np.memmap):
        ch = min(h, 2048)
        cw = min(w, 2048)
        y0 = max(0, ((h - ch) // 2 // 8) * 8)
        x0 = max(0, ((w - cw) // 2 // 8) * 8)
        native_c = np.clip(np.asarray(rgb_f32[y0 : y0 + ch, x0 : x0 + cw], dtype=np.float32), 0.0, 1.0)
        step = max(1, int(math.sqrt((h * w) / 2_000_000.0)))
        global_c = np.clip(np.asarray(rgb_f32[::step, ::step], dtype=np.float32), 0.0, 1.0)
    else:
        native_c = np.clip(rgb_f32, 0.0, 1.0)
        global_c = native_c

    gray_255 = (
        0.2126 * native_c[:, :, 0] + 0.7152 * native_c[:, :, 1] + 0.0722 * native_c[:, :, 2]
    ).astype(np.float32) * 255.0

    noise_sigma = _estimate_multiscale_noise_sigma(gray_255)
    noise_severity = float(np.clip((noise_sigma - 1.2) / 14.0, 0.0, 1.0))

    blockiness = _estimate_jpeg_blockiness(gray_255)
    sharpness = _estimate_sharpness(gray_255, noise_sigma)
    blur_severity = float(1.0 - sharpness)
    motion_score, motion_angle = _estimate_motion_blur(gray_255, noise_sigma=noise_sigma)

    # CIE LAB analysis (OpenCV converts float32 RGB [0,1] directly into L*[0,100], a*[-127,127], b*[-127,127]).
    lab = cv2.cvtColor(global_c, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    mean_luma = float(np.mean(l_chan))
    p01 = float(np.percentile(l_chan, 1))
    p99 = float(np.percentile(l_chan, 99))
    dynamic_range = max(0.0, p99 - p01)
    shadow_clip_pct = float(np.mean(l_chan < 3.0) * 100.0)
    highlight_clip_pct = float(np.mean(l_chan > 97.0) * 100.0)

    # Strictly neutral-weighted color cast; green-magenta (a*) shifts are weighted more strongly
    # than warm/cool (b*) so indoor fluorescent/scan casts are caught while pastel skies stay natural.
    chroma = np.hypot(a_chan, b_chan)
    mid_luma = (l_chan > 15.0) & (l_chan < 88.0)
    neutral_weights = np.exp(-((chroma / 14.0) ** 2)) * mid_luma.astype(np.float32)
    w_sum = float(np.sum(neutral_weights))
    if w_sum > 25.0:
        cast_a = float(np.sum(a_chan * neutral_weights) / w_sum)
        cast_b = float(np.sum(b_chan * neutral_weights) / w_sum)
    else:
        cast_a = 0.0
        cast_b = 0.0
    cast_mag = float(np.hypot(1.5 * cast_a, cast_b))
    color_cast_strength = float(np.clip((cast_mag - 4.0) / 14.0, 0.0, 1.0))

    is_illus = _detect_illustration_heuristic(native_c, gray_255)

    # Adaptive routing recommendation.
    recommended_dejpeg = blockiness >= 0.28
    recommended_denoise = noise_severity >= 0.35
    # Spectral anisotropy alone also fires on sharp scenes with strongly oriented content (tree trunks,
    # shelves, facades). Genuine camera shake additionally lowers overall edge acuity, so require both.
    recommended_deblur = bool(motion_score >= 0.45 and blur_severity >= 0.30 and not is_illus)

    # Low mean luminance alone is not underexposure: night skies, low-key still lifes and moody foliage are
    # dark by intent and still reach bright highlights. Only recommend brightening when the whole histogram
    # is compressed into the shadows (no highlight anywhere near the top of the range).
    if mean_luma < 34.0 and p99 < 65.0:
        exposure_boost = float(
            np.clip((34.0 - mean_luma) / 40.0, 0.0, 0.5) * np.clip((65.0 - p99) / 20.0, 0.0, 1.0)
        )
    elif mean_luma > 76.0 and highlight_clip_pct > 4.0:
        exposure_boost = float(-np.clip((mean_luma - 72.0) / 35.0, 0.0, 0.5))
    else:
        exposure_boost = 0.0

    if is_illus:
        preset = "anime"
    elif blur_severity > 0.52 or blockiness > 0.35 or noise_severity > 0.45 or recommended_deblur:
        preset = "photo-pro"
    elif mp >= 2.0 and sharpness >= 0.65:
        preset = "photo-hq"
    else:
        preset = "photo-pro"

    return ImageDiagnostics(
        width=w,
        height=h,
        megapixels=round(mp, 3),
        sharpness_score=round(sharpness, 4),
        blur_severity=round(blur_severity, 4),
        noise_sigma=round(noise_sigma, 3),
        noise_severity=round(noise_severity, 4),
        jpeg_blockiness=round(blockiness, 4),
        mean_luma=round(mean_luma, 2),
        dynamic_range=round(dynamic_range, 2),
        shadow_clip_pct=round(shadow_clip_pct, 2),
        highlight_clip_pct=round(highlight_clip_pct, 2),
        color_cast_strength=round(color_cast_strength, 4),
        color_cast_ab=(round(cast_a, 2), round(cast_b, 2)),
        is_illustration_likely=is_illus,
        recommended_preset=preset,
        recommended_dejpeg=recommended_dejpeg,
        recommended_denoise=recommended_denoise,
        recommended_exposure_boost=round(exposure_boost, 3),
        motion_blur_score=round(motion_score, 4),
        motion_blur_angle_deg=round(motion_angle, 1),
        recommended_deblur=recommended_deblur,
    )

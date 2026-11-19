"""Float32 photometric enhancement engine: guided filtering, wavelet color fix, tone & detail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .memmap_utils import (
    create_managed_memmap,
    extract_resized_strip,
    is_memmap_array,
    stride_preview,
)


@dataclass
class ToneDetailConfig:
    """Parameters governing photometric finishing and post-upscale refinement."""

    color_fix_strength: float = 0.85  # [0, 1] low-frequency chroma preservation vs reference
    color_fix_luma_strength: float = 0.45  # [0, 1] low-frequency luma anchor vs reference
    white_balance_strength: float = 0.40  # [0, 1] highlight-safe neutral balance strength
    tone_strength: float = 0.55  # [0, 1] shadow/highlight + dynamic range + micro-contrast
    detail_strength: float = 0.35  # [0, 1] halo-suppressed guided detail enhancement
    natural_grain: float = 0.0025  # [0, 0.02] subtle anti-plastic midtone texture
    ca_suppress: float = 0.45  # [0, 1] lateral chromatic aberration fringe suppression
    bokeh_guard: bool = True  # Preserve creamy DoF bokeh while focusing detail on in-focus subject
    halo_guard: float = 0.75  # [0, 1] closed-loop post-SR ringing halo suppression strength


def guided_filter_gray(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Exact He et al. edge-preserving Guided Filter in pure float32 OpenCV boxFilter."""
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)

    g = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)

    mean_g = cv2.boxFilter(g, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_gg = cv2.boxFilter(g * g, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    corr_gp = cv2.boxFilter(g * p, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    var_g = np.maximum(corr_gg - mean_g * mean_g, 0.0)
    cov_gp = corr_gp - mean_g * mean_p

    a = cov_gp / (var_g + float(eps))
    b = mean_p - a * mean_g

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    return mean_a * g + mean_b


def guided_filter_rgb(rgb: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Apply luminance-guided edge-preserving smoothing identically across RGB channels."""
    guide = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(
        np.float32
    )
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = guided_filter_gray(guide, rgb[:, :, c], radius=radius, eps=eps)
    return out


def wavelet_color_fix(
    target_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    chroma_strength: float = 0.85,
    luma_strength: float = 0.45,
) -> np.ndarray:
    """Eliminate AI model color/exposure drift while preserving 100% of synthesized high-freq detail.

    Decomposes both target and resized reference in perceptual CIE LAB using a large spatial
    low-pass filter and transfers only the low-frequency color/illumination anchor.
    """
    chroma_strength = float(np.clip(chroma_strength, 0.0, 1.0))
    luma_strength = float(np.clip(luma_strength, 0.0, 1.0))
    if chroma_strength <= 0.0 and luma_strength <= 0.0:
        return target_rgb

    th, tw = target_rgb.shape[:2]
    if reference_rgb.shape[:2] != (th, tw):
        interp = (
            cv2.INTER_CUBIC
            if (reference_rgb.shape[0] < th or reference_rgb.shape[1] < tw)
            else cv2.INTER_AREA
        )
        ref = cv2.resize(reference_rgb, (tw, th), interpolation=interp)
    else:
        ref = reference_rgb

    tgt_lab = cv2.cvtColor(np.clip(target_rgb, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(np.clip(ref, 0.0, 1.0), cv2.COLOR_RGB2LAB)

    sigma = float(np.clip(0.028 * min(th, tw), 8.0, 64.0))
    # Compute low-frequency field at reduced resolution when large for huge speedup with identical math.
    ds = 4 if min(th, tw) >= 512 else (2 if min(th, tw) >= 256 else 1)
    if ds > 1:
        small_hw = (max(8, tw // ds), max(8, th // ds))
        tgt_s = cv2.resize(tgt_lab, small_hw, interpolation=cv2.INTER_AREA)
        ref_s = cv2.resize(ref_lab, small_hw, interpolation=cv2.INTER_AREA)
        diff_s = ref_s - tgt_s
        low_diff_s = cv2.GaussianBlur(diff_s, (0, 0), sigmaX=max(2.0, sigma / ds))
        low_diff = cv2.resize(low_diff_s, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        diff = ref_lab - tgt_lab
        low_diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=sigma)

    corrected = tgt_lab.copy()
    corrected[:, :, 0] = np.clip(tgt_lab[:, :, 0] + luma_strength * low_diff[:, :, 0], 0.0, 100.0)
    corrected[:, :, 1] = np.clip(tgt_lab[:, :, 1] + chroma_strength * low_diff[:, :, 1], -127.0, 127.0)
    corrected[:, :, 2] = np.clip(tgt_lab[:, :, 2] + chroma_strength * low_diff[:, :, 2], -127.0, 127.0)

    out_rgb = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
    return np.clip(out_rgb, 0.0, 1.0)


def _compute_wavelet_low_diff_small(
    target_small: np.ndarray,
    ref_small: np.ndarray,
    *,
    full_h: int,
    full_w: int,
) -> np.ndarray:
    """Compute compact low-frequency CIE LAB difference field between target and reference."""
    th, tw = target_small.shape[:2]
    tgt_s = cv2.cvtColor(np.clip(target_small, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    ref_r = (
        ref_small
        if ref_small.shape[:2] == (th, tw)
        else cv2.resize(np.clip(ref_small, 0.0, 1.0), (tw, th), interpolation=cv2.INTER_AREA)
    )
    ref_s = cv2.cvtColor(np.clip(ref_r, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    sigma = float(np.clip(0.028 * min(full_h, full_w), 8.0, 64.0))
    scale = min(th / float(max(1, full_h)), tw / float(max(1, full_w)))
    sigma_s = max(1.5, sigma * scale)
    return cv2.GaussianBlur(ref_s - tgt_s, (0, 0), sigmaX=sigma_s).astype(np.float32)


def _apply_wavelet_low_diff_strip(
    strip_rgb: np.ndarray,
    low_diff_strip: np.ndarray,
    *,
    chroma_strength: float,
    luma_strength: float,
) -> np.ndarray:
    """Apply precomputed low-frequency CIE LAB anchor to a single RGB strip."""
    tgt_lab = cv2.cvtColor(np.clip(strip_rgb, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    tgt_lab[:, :, 0] = np.clip(tgt_lab[:, :, 0] + luma_strength * low_diff_strip[:, :, 0], 0.0, 100.0)
    tgt_lab[:, :, 1] = np.clip(tgt_lab[:, :, 1] + chroma_strength * low_diff_strip[:, :, 1], -127.0, 127.0)
    tgt_lab[:, :, 2] = np.clip(tgt_lab[:, :, 2] + chroma_strength * low_diff_strip[:, :, 2], -127.0, 127.0)
    return np.clip(cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)


def _compute_wb_gains(rgb_c: np.ndarray, strength: float) -> Optional[np.ndarray]:
    """Compute highlight-safe damped Gray-World RGB gains."""
    max_c = np.max(rgb_c, axis=2)
    min_c = np.min(rgb_c, axis=2)
    sat = (max_c - min_c) / (max_c + 1e-5)

    # Sample neutral-to-moderate midtones so sunsets/flowers/clipped skies don't bias gains.
    mask = (max_c < 0.94) & (min_c > 0.04) & (sat < 0.48)
    if np.count_nonzero(mask) < max(64, int(0.01 * rgb_c.shape[0] * rgb_c.shape[1])):
        mask = (max_c < 0.97) & (min_c > 0.02)
    if np.count_nonzero(mask) < 16:
        return None

    means = np.mean(rgb_c[mask], axis=0)  # (R, G, B)
    target_gray = float(np.mean(means))
    if target_gray < 1e-4:
        return None

    raw_gains = target_gray / (means + 1e-5)
    # Blend toward 1.0 and hard-clamp so atmospheric scenes retain natural warmth/coolness.
    gains = 1.0 + strength * (raw_gains - 1.0)
    return np.clip(gains, 0.86, 1.16).astype(np.float32)


def _apply_wb_gains(rgb_c: np.ndarray, gains: Optional[np.ndarray]) -> np.ndarray:
    """Apply precomputed white balance gains with soft highlight protection."""
    if gains is None:
        return rgb_c
    out = rgb_c * gains[None, None, :]
    # Softly compress any highlight overshoot without harsh channel clipping.
    peak = np.maximum(np.max(out, axis=2, keepdims=True), 1.0)
    exceed = peak > 1.0
    if np.any(exceed):
        out = np.where(exceed, out / (1.0 + 0.35 * (peak - 1.0)), out)
    return np.clip(out, 0.0, 1.0)


def auto_white_balance(rgb: np.ndarray, strength: float = 0.40) -> np.ndarray:
    """Highlight-safe damped Gray-World white balance excluding clipped & hyper-saturated pixels."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb

    rgb_c = np.clip(rgb, 0.0, 1.0)
    gains = _compute_wb_gains(rgb_c, strength)
    return _apply_wb_gains(rgb_c, gains)


def _compute_tone_global_stats(rgb_sample: np.ndarray, full_h: int, full_w: int) -> Tuple[float, float, float, int, int]:
    """Compute global luminance statistics and bounded guided radii for tone mapping."""
    y = (0.2126 * rgb_sample[:, :, 0] + 0.7152 * rgb_sample[:, :, 1] + 0.0722 * rgb_sample[:, :, 2]).astype(
        np.float32
    )
    mean_y = float(np.mean(y))
    p02 = float(np.percentile(y, 2))
    p98 = float(np.percentile(y, 98))
    radius = max(8, min(64, min(full_h, full_w) // 20))
    l_radius = max(6, min(64, min(full_h, full_w) // 48))
    return mean_y, p02, p98, radius, l_radius


def _apply_tone_enhance_with_stats(
    rgb_c: np.ndarray,
    *,
    strength: float,
    exposure_hint: float,
    mean_y: float,
    p02: float,
    p98: float,
    radius: int,
    l_radius: int,
) -> np.ndarray:
    """Apply edge-aware Retinex tone enhancement using precomputed global statistics."""
    y = (0.2126 * rgb_c[:, :, 0] + 0.7152 * rgb_c[:, :, 1] + 0.0722 * rgb_c[:, :, 2]).astype(
        np.float32
    )

    # Edge-aware illumination field prevents dark-halo glow when lifting shadows.
    illum = np.clip(guided_filter_gray(y, y, radius=radius, eps=0.045), 0.0, 1.0)

    # 1. Shadow recovery in dark regions while anchoring true blacks.
    shadow_need = float(np.clip((0.36 - mean_y) * 1.4 + max(0.0, exposure_hint) * 0.5, 0.0, 0.55))
    shadow_boost = strength * shadow_need
    if shadow_boost > 0.01:
        dark_weight = np.clip((0.45 - illum) / 0.45, 0.0, 1.0) ** 1.4
        # Smooth logarithmic shadow lift preserving near-zero black point.
        y_lifted = np.power(np.maximum(y, 1e-5), 1.0 - 0.38 * shadow_boost * dark_weight)
    else:
        y_lifted = y

    # 2. Gentle highlight protection roll-off when bright regions crowd 1.0.
    hi_excess = max(0.0, (p98 - 0.88) / 0.12)
    hi_strength = strength * float(np.clip(hi_excess * 0.5 + max(0.0, -exposure_hint) * 0.4, 0.0, 0.45))
    if hi_strength > 0.01:
        hi_mask = np.clip((illum - 0.68) / 0.32, 0.0, 1.0)
        y_target = y_lifted * (1.0 - 0.18 * hi_strength * hi_mask)
    else:
        y_target = y_lifted

    # 3. Gentle contrast normalization + milky haze black-point anchor + smooth filmic S-curve.
    raw_spread = p98 - p02
    haze_need = max(0.0, (p02 - 0.055) / 0.16)
    if 0.04 < raw_spread < 0.78 or (raw_spread > 0.04 and haze_need > 0.05):
        spread_need = max(0.0, (0.78 - raw_spread) / 0.65)
        stretch_amt = float(np.clip(max(strength * spread_need, 0.55 * haze_need), 0.0, 0.72))
        lo = p02 * 0.85
        hi = p98 + (1.0 - p98) * 0.15
        denom = max(0.20, hi - lo)
        y_stretched = np.clip((y_target - lo) / denom, 0.0, 1.0)
        y_target = (1.0 - stretch_amt) * y_target + stretch_amt * y_stretched

    # Subtle perceptual S-curve: f(x) = x - c * sin(2*pi*x) / (2*pi) -> monotonic when |c| < 1.
    s_amt = 0.16 * strength
    y_scurve = np.clip(y_target - (s_amt / (2.0 * np.pi)) * np.sin(2.0 * np.pi * y_target), 0.0, 1.0)

    # Apply luminance ratio cleanly while protecting saturated colors against hue drift.
    ratio = (y_scurve + 1e-4) / (y + 1e-4)
    ratio = np.clip(ratio, 0.65, 2.1)[:, :, None]
    out = rgb_c * ratio

    # Continuous hue-preserving normalization when any channel exceeds 1.0.
    max_out = np.maximum(np.max(out, axis=2, keepdims=True), 1.0)
    out = out / max_out

    # Mild local micro-contrast in LAB L* via guided base/detail boost (only when strength > 0.25).
    if strength > 0.25:
        lab = cv2.cvtColor(np.clip(out, 0.0, 1.0), cv2.COLOR_RGB2LAB)
        l_norm = lab[:, :, 0] / 100.0
        l_base = guided_filter_gray(l_norm, l_norm, radius=l_radius, eps=0.025)
        l_micro = l_norm - l_base
        l_micro_clamped = np.tanh(l_micro / 0.06) * 0.06
        micro_weight = 0.20 * max(0.0, strength - 0.15)
        lab[:, :, 0] = np.clip((l_norm + micro_weight * l_micro_clamped) * 100.0, 0.0, 100.0)

        sat_boost = 1.0 + 0.04 * strength
        lab[:, :, 1] = np.clip(lab[:, :, 1] * sat_boost, -127.0, 127.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * sat_boost, -127.0, 127.0)
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def adaptive_tone_enhance(
    rgb: np.ndarray,
    *,
    strength: float = 0.55,
    exposure_hint: float = 0.0,
) -> np.ndarray:
    """Edge-aware Retinex shadow/highlight recovery and perceptual contrast enhancement."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb

    rgb_c = np.clip(rgb, 0.0, 1.0)
    h, w = rgb_c.shape[:2]
    mean_y, p02, p98, radius, l_radius = _compute_tone_global_stats(rgb_c, h, w)
    return _apply_tone_enhance_with_stats(
        rgb_c,
        strength=strength,
        exposure_hint=exposure_hint,
        mean_y=mean_y,
        p02=p02,
        p98=p98,
        radius=radius,
        l_radius=l_radius,
    )


def guided_detail_sharpen(
    rgb: np.ndarray,
    strength: float = 0.35,
    *,
    detail_mask: Optional[np.ndarray] = None,
    radius_override: Optional[int] = None,
) -> np.ndarray:
    """Halo-free edge-aware micro-detail enhancement using soft-clamped Guided Filter residuals."""
    strength = float(np.clip(strength, 0.0, 1.5))
    if strength <= 0.0:
        return rgb

    rgb_c = np.clip(rgb, 0.0, 1.0)
    h, w = rgb_c.shape[:2]
    radius = (
        int(radius_override)
        if radius_override is not None
        else max(2, min(6, round(min(h, w) * 0.003)))
    )

    # Smooth base via luminance-guided filter; eps=0.008 preserves major edges and avoids halos.
    base = guided_filter_rgb(rgb_c, radius=radius, eps=0.008)
    detail = rgb_c - base

    # Soft-limit overshoot on strong contrast edges via tanh clamping.
    halo_limit = 0.065
    safe_detail = np.tanh(detail / halo_limit) * halo_limit

    weight = (0.85 * strength) if detail_mask is None else ((0.85 * strength) * detail_mask)
    enhanced = np.clip(rgb_c + weight * safe_detail, 0.0, 1.0)
    return enhanced


def _compute_ca_edge_scale(rgb_c: np.ndarray) -> Optional[float]:
    """Return global 92nd-percentile edge band magnitude for CA gating, or None if image is soft."""
    lab = cv2.cvtColor(rgb_c, cv2.COLOR_RGB2LAB)
    l_norm = (lab[:, :, 0] / 100.0).astype(np.float32)
    gx = cv2.Sobel(l_norm, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(l_norm, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    grad = np.hypot(gx, gy)
    if float(np.percentile(grad, 95)) < 0.04:
        return None
    edge_band = cv2.GaussianBlur(grad, (0, 0), sigmaX=1.35)
    return max(float(np.percentile(edge_band, 92)), 0.05)


def suppress_chromatic_aberration(
    rgb: np.ndarray,
    strength: float = 0.45,
    *,
    edge_scale_override: Optional[float] = None,
) -> np.ndarray:
    """Remove purple/green optical edge fringes using luma-guided local chroma correlation.

    True colored boundaries maintain linear correlation with L*, whereas lateral chromatic
    aberration blooms asymmetrically across steep luminance edges adjacent to neutral surfaces.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb

    rgb_c = np.clip(rgb, 0.0, 1.0)
    h, w = rgb_c.shape[:2]
    if min(h, w) < 16:
        return rgb_c

    lab = cv2.cvtColor(rgb_c, cv2.COLOR_RGB2LAB)
    l_norm = (lab[:, :, 0] / 100.0).astype(np.float32)
    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    gx = cv2.Sobel(l_norm, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(l_norm, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    grad = np.hypot(gx, gy)
    if edge_scale_override is None and float(np.percentile(grad, 95)) < 0.04:
        return rgb_c

    edge_band = cv2.GaussianBlur(grad, (0, 0), sigmaX=1.35)
    edge_scale = (
        float(edge_scale_override)
        if edge_scale_override is not None
        else max(float(np.percentile(edge_band, 92)), 0.05)
    )
    edge_weight = np.clip((edge_band - 0.15 * edge_scale) / (0.85 * edge_scale), 0.0, 1.0)

    # Identify pixels near neutral tones so true solid saturated adjacent blocks remain untouched.
    chroma = np.hypot(a_chan, b_chan)
    local_min_chroma = cv2.erode(chroma, np.ones((7, 7), np.uint8))
    neutral_adj = np.clip((18.0 - local_min_chroma) / 14.0, 0.0, 1.0)

    a_guided = guided_filter_gray(l_norm, a_chan, radius=3, eps=0.002)
    b_guided = guided_filter_gray(l_norm, b_chan, radius=3, eps=0.002)

    blend = (strength * edge_weight * neutral_adj)
    lab[:, :, 1] = np.clip(a_chan * (1.0 - blend) + a_guided * blend, -127.0, 127.0)
    lab[:, :, 2] = np.clip(b_chan * (1.0 - blend) + b_guided * blend, -127.0, 127.0)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)


def compute_focus_bokeh_mask(
    rgb: np.ndarray,
    *,
    face_mask: Optional[np.ndarray] = None,
    memmap_threshold_mp: float = 24.0,
) -> np.ndarray:
    """Compute in-focus subject weight [0.25, 1.0] so bokeh stays creamy and subjects stay crisp."""
    h, w = rgb.shape[:2]
    if min(h, w) < 32:
        return np.ones((h, w, 1), dtype=np.float32)

    use_memmap = (
        is_memmap_array(rgb)
        or (face_mask is not None and is_memmap_array(face_mask))
        or ((h * w) / 1_000_000.0 >= float(memmap_threshold_mp))
    )

    scale = min(1.0, 384.0 / float(max(h, w)))
    if scale < 1.0:
        sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        rgb_src = stride_preview(rgb, max_side=768) if (use_memmap or (h * w) > 2_000_000) else np.clip(rgb, 0.0, 1.0)
        small = cv2.resize(rgb_src, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh = w, h
        small = np.clip(rgb, 0.0, 1.0)

    y = (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]).astype(np.float32)
    lap = np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    energy = 0.55 * lap + 0.45 * np.hypot(gx, gy)

    sigma = float(np.clip(0.032 * min(sw, sh), 3.0, 16.0))
    smooth_e = cv2.GaussianBlur(energy, (0, 0), sigmaX=sigma)
    p20 = float(np.percentile(smooth_e, 20))
    p90 = float(np.percentile(smooth_e, 90))
    if (p90 - p20) < 1e-3:
        focus_s = np.ones((sh, sw), dtype=np.float32)
    else:
        norm = np.clip((smooth_e - p20) / ((p90 - p20) * 0.65 + 1e-5), 0.0, 1.0)
        focus_s = 0.25 + 0.75 * norm

    if not use_memmap:
        focus = cv2.resize(focus_s, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        if face_mask is not None:
            f_use = (
                cv2.resize(face_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
                if face_mask.shape[:2] != (h, w)
                else face_mask
            )
            focus = np.maximum(focus, np.clip(f_use, 0.0, 1.0))
        return np.clip(focus, 0.25, 1.0).astype(np.float32)

    focus_mm = create_managed_memmap((h, w, 1), dtype=np.float32, prefix="azzani_focus_")
    focus_s_3d = focus_s[:, :, None]
    x_map = (np.arange(w, dtype=np.float32) + 0.5) * (sw / float(w)) - 0.5
    f_x_map = (
        (np.arange(w, dtype=np.float32) + 0.5) * (face_mask.shape[1] / float(w)) - 0.5
        if face_mask is not None
        else None
    )
    strip_h = 512 if w < 10000 else 256
    for y0 in range(0, h, strip_h):
        y1 = min(h, y0 + strip_h)
        strip = extract_resized_strip(
            focus_s_3d, h, w, y0, y1, interpolation=cv2.INTER_LINEAR, x_map=x_map
        )
        if face_mask is not None:
            if face_mask.shape[:2] == (h, w):
                f_strip = np.asarray(face_mask[y0:y1], dtype=np.float32)
            else:
                f_strip = extract_resized_strip(
                    face_mask, h, w, y0, y1, interpolation=cv2.INTER_LINEAR, x_map=f_x_map
                )
            if f_strip.ndim == 2:
                f_strip = f_strip[:, :, None]
            strip = np.maximum(strip, np.clip(f_strip, 0.0, 1.0))
        focus_mm[y0:y1] = np.clip(strip, 0.25, 1.0)
    return focus_mm


def _critic_process_block(
    enh_blk: np.ndarray,
    ref_blk: np.ndarray,
    *,
    clamp_strength: float,
    focus_blk: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run halo + flat-sky critic pass on an aligned enhanced/reference block."""
    kernel = np.ones((5, 5), np.uint8)
    loc_max = cv2.dilate(ref_blk, kernel)
    loc_min = cv2.erode(ref_blk, kernel)
    tol = 0.045
    over_hi = np.maximum(0.0, enh_blk - (loc_max + tol))
    over_lo = np.maximum(0.0, (loc_min - tol) - enh_blk)
    raw_err = over_hi + over_lo

    out = enh_blk
    if clamp_strength > 0.0:
        suppress = 0.85 * clamp_strength
        out = out - suppress * over_hi + suppress * over_lo

        ref_y = (0.2126 * ref_blk[:, :, 0] + 0.7152 * ref_blk[:, :, 1] + 0.0722 * ref_blk[:, :, 2]).astype(
            np.float32
        )
        mean_y = cv2.boxFilter(ref_y, cv2.CV_32F, (9, 9), borderType=cv2.BORDER_REFLECT)
        mean_y2 = cv2.boxFilter(ref_y * ref_y, cv2.CV_32F, (9, 9), borderType=cv2.BORDER_REFLECT)
        local_std = np.sqrt(np.maximum(mean_y2 - mean_y * mean_y, 0.0))
        flat_bg = np.clip((0.022 - local_std) / 0.015, 0.0, 1.0)[:, :, None]
        if focus_blk is not None:
            bokeh_soft = np.clip((0.55 - focus_blk) / 0.30, 0.0, 1.0)
            flat_bg = flat_bg * (0.45 + 0.55 * bokeh_soft)
        bg_pull = 0.25 * clamp_strength * flat_bg
        out = np.clip(out * (1.0 - bg_pull) + ref_blk * bg_pull, 0.0, 1.0)

    post_err = np.maximum(0.0, out - (loc_max + tol)) + np.maximum(0.0, (loc_min - tol) - out)
    return np.clip(out, 0.0, 1.0).astype(np.float32), raw_err, post_err


def _compute_delta_e_pair(out_arr: np.ndarray, ref_arr: np.ndarray, h: int, w: int) -> float:
    """Compute perceptual CIE76 delta-E from bounded-memory previews."""
    ds_w, ds_h = max(16, min(256, w)), max(16, min(256, h))
    out_src = stride_preview(out_arr, max_side=512) if ((h * w) > 1_000_000 or is_memmap_array(out_arr)) else np.clip(out_arr, 0.0, 1.0)
    ref_src = stride_preview(ref_arr, max_side=512) if ((ref_arr.shape[0] * ref_arr.shape[1]) > 1_000_000 or is_memmap_array(ref_arr)) else np.clip(ref_arr, 0.0, 1.0)
    out_ds = cv2.GaussianBlur(cv2.resize(out_src, (ds_w, ds_h), interpolation=cv2.INTER_AREA), (0, 0), sigmaX=4.0)
    ref_ds = cv2.GaussianBlur(cv2.resize(ref_src, (ds_w, ds_h), interpolation=cv2.INTER_AREA), (0, 0), sigmaX=4.0)
    lab_out = cv2.cvtColor(np.clip(out_ds, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    lab_ref = cv2.cvtColor(np.clip(ref_ds, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    return float(np.mean(np.sqrt(np.sum((lab_out - lab_ref) ** 2, axis=2))))


def self_critic_refinement(
    enhanced_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    clamp_strength: float = 0.75,
    focus_mask: Optional[np.ndarray] = None,
    memmap_threshold_mp: float = 24.0,
    force_chunked: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate post-SR artifacts (ringing halos, flat-sky worms, color drift) and damp them safely."""
    h, w = enhanced_rgb.shape[:2]
    clamp_strength = float(np.clip(clamp_strength, 0.0, 1.0))

    use_chunked = (
        force_chunked
        or is_memmap_array(enhanced_rgb)
        or is_memmap_array(reference_rgb)
        or (focus_mask is not None and is_memmap_array(focus_mask))
        or ((h * w) / 1_000_000.0 >= float(memmap_threshold_mp))
    )

    if not use_chunked:
        enh = np.clip(enhanced_rgb, 0.0, 1.0).astype(np.float32)
        if reference_rgb.shape[:2] != (h, w):
            ref_up = np.clip(
                cv2.resize(reference_rgb, (w, h), interpolation=cv2.INTER_CUBIC),
                0.0,
                1.0,
            ).astype(np.float32)
        else:
            ref_up = np.clip(reference_rgb, 0.0, 1.0).astype(np.float32)

        f_use = focus_mask
        if f_use is not None and f_use.shape[:2] != (h, w):
            f_use = cv2.resize(f_use.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]

        out, raw_err, post_err = _critic_process_block(
            enh, ref_up, clamp_strength=clamp_strength, focus_blk=f_use
        )
        raw_halo_index = float(np.clip(np.mean(raw_err) / 0.025, 0.0, 1.0))
        post_halo_index = float(np.clip(np.mean(post_err) / 0.025, 0.0, 1.0))
        delta_e = _compute_delta_e_pair(out, ref_up, h, w)
        focus_cov = float(np.mean(f_use > 0.65) * 100.0) if f_use is not None else 100.0
    else:
        out = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azzani_critic_")
        ref_x_map = (np.arange(w, dtype=np.float32) + 0.5) * (reference_rgb.shape[1] / float(w)) - 0.5
        f_x_map = (
            (np.arange(w, dtype=np.float32) + 0.5) * (focus_mask.shape[1] / float(w)) - 0.5
            if focus_mask is not None
            else None
        )
        strip_h = 512 if w < 10000 else 256
        pad = 12
        raw_sum = 0.0
        post_sum = 0.0
        focus_above = 0
        total_vals = float(h * w * 3)

        for y0 in range(0, h, strip_h):
            y1 = min(h, y0 + strip_h)
            py0 = max(0, y0 - pad)
            py1 = min(h, y1 + pad)
            enh_blk = np.clip(np.asarray(enhanced_rgb[py0:py1], dtype=np.float32), 0.0, 1.0)
            ref_blk = np.clip(
                extract_resized_strip(
                    reference_rgb, h, w, py0, py1, interpolation=cv2.INTER_CUBIC, x_map=ref_x_map
                ),
                0.0,
                1.0,
            )
            f_blk = None
            if focus_mask is not None:
                if focus_mask.shape[:2] == (h, w):
                    f_blk = np.asarray(focus_mask[py0:py1], dtype=np.float32)
                else:
                    f_blk = extract_resized_strip(
                        focus_mask, h, w, py0, py1, interpolation=cv2.INTER_LINEAR, x_map=f_x_map
                    )
                if f_blk.ndim == 2:
                    f_blk = f_blk[:, :, None]

            out_blk, raw_blk, post_blk = _critic_process_block(
                enh_blk, ref_blk, clamp_strength=clamp_strength, focus_blk=f_blk
            )
            iy0 = y0 - py0
            iy1 = iy0 + (y1 - y0)
            core_out = out_blk[iy0:iy1]
            out[y0:y1] = core_out
            raw_sum += float(np.sum(raw_blk[iy0:iy1], dtype=np.float64))
            post_sum += float(np.sum(post_blk[iy0:iy1], dtype=np.float64))
            if f_blk is not None:
                focus_above += int(np.count_nonzero(f_blk[iy0:iy1] > 0.65))

        raw_halo_index = float(np.clip((raw_sum / max(1.0, total_vals)) / 0.025, 0.0, 1.0))
        post_halo_index = float(np.clip((post_sum / max(1.0, total_vals)) / 0.025, 0.0, 1.0))
        delta_e = _compute_delta_e_pair(out, reference_rgb, h, w)
        focus_cov = (focus_above / float(max(1, h * w))) * 100.0 if focus_mask is not None else 100.0

    confidence = float(
        np.clip(
            100.0 - 26.0 * post_halo_index - 2.6 * max(0.0, delta_e - 1.8),
            40.0,
            99.8,
        )
    )

    metrics = {
        "raw_halo_index": round(raw_halo_index, 4),
        "halo_ringing_index": round(post_halo_index, 4),
        "color_drift_delta_e": round(delta_e, 3),
        "focus_coverage_pct": round(focus_cov, 1),
        "confidence_score": round(confidence, 1),
    }
    return out, metrics


def add_natural_micrograin(rgb: np.ndarray, amount: float = 0.0025, seed: int = 1337) -> np.ndarray:
    """Add ultra-subtle luminance-weighted organic grain to break up AI plastic smoothness."""
    amount = float(np.clip(amount, 0.0, 0.03))
    if amount <= 0.0:
        return rgb

    rgb_c = np.clip(rgb, 0.0, 1.0)
    h, w = rgb_c.shape[:2]
    y = (0.2126 * rgb_c[:, :, 0] + 0.7152 * rgb_c[:, :, 1] + 0.0722 * rgb_c[:, :, 2])[:, :, None]
    # Parabolic mask: peaks at midtones (0.5), vanishes cleanly in pure blacks and blown highlights.
    weight = np.clip(4.0 * y * (1.0 - y), 0.0, 1.0)

    rng = np.random.default_rng(seed)
    # Mostly monochrome luminance grain (85% luma, 15% soft chroma) feels like real camera film.
    luma_noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    chroma_noise = rng.normal(0.0, 0.2, size=(h, w, 3)).astype(np.float32)
    grain = (0.85 * luma_noise + 0.15 * chroma_noise) * amount * weight
    return np.clip(rgb_c + grain, 0.0, 1.0)


def apply_photometric_finish(
    enhanced_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    cfg: ToneDetailConfig,
    *,
    exposure_hint: float = 0.0,
    focus_mask: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    memmap_threshold_mp: float = 24.0,
    force_chunked: bool = False,
) -> np.ndarray:
    """Run full float32 finishing stack: CA suppress, wavelet color anchor, WB, tone, detail, micro-grain."""
    h, w = enhanced_rgb.shape[:2]
    use_chunked = (
        force_chunked
        or is_memmap_array(enhanced_rgb)
        or is_memmap_array(reference_rgb)
        or (focus_mask is not None and is_memmap_array(focus_mask))
        or (face_mask is not None and is_memmap_array(face_mask))
        or ((h * w) / 1_000_000.0 >= float(memmap_threshold_mp))
    )

    if not use_chunked:
        img = np.clip(enhanced_rgb, 0.0, 1.0)

        # 1. Remove lateral chromatic aberration fringes on steep optical edges before sharpening.
        if cfg.ca_suppress > 0.0:
            img = suppress_chromatic_aberration(img, strength=cfg.ca_suppress)

        # 2. Anchor low-frequency color field to prevent neural model tint shifts.
        img = wavelet_color_fix(
            img,
            reference_rgb,
            chroma_strength=cfg.color_fix_strength,
            luma_strength=cfg.color_fix_luma_strength,
        )

        # 3. Highlight-safe adaptive white balance.
        if cfg.white_balance_strength > 0.0:
            img = auto_white_balance(img, strength=cfg.white_balance_strength)

        # 4. Edge-aware shadow/highlight recovery and perceptual contrast.
        if cfg.tone_strength > 0.0:
            img = adaptive_tone_enhance(img, strength=cfg.tone_strength, exposure_hint=exposure_hint)

        # 5. Halo-free guided detail enhancement modulated by in-focus subject mask.
        if cfg.detail_strength > 0.0:
            detail_mask = focus_mask
            if cfg.bokeh_guard and detail_mask is None:
                detail_mask = compute_focus_bokeh_mask(img, face_mask=face_mask)
            if detail_mask is not None and detail_mask.shape[:2] != img.shape[:2]:
                detail_mask = cv2.resize(
                    detail_mask.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )[:, :, None]
            img = guided_detail_sharpen(img, strength=cfg.detail_strength, detail_mask=detail_mask)

        # 6. Subtle midtone organic micro-grain.
        if cfg.natural_grain > 0.0:
            img = add_natural_micrograin(img, amount=cfg.natural_grain)

        return np.clip(img, 0.0, 1.0)

    # --- Strip-chunked out-of-core execution for >24 MP / memmap canvases ---
    # Compute global photometric statistics once on bounded <=512px previews.
    enh_small = stride_preview(enhanced_rgb, max_side=512)
    ref_small = stride_preview(reference_rgb, max_side=512)

    ca_edge_scale: Optional[float] = None
    ca_active = cfg.ca_suppress > 0.0
    if ca_active:
        ca_edge_scale = _compute_ca_edge_scale(enh_small)
        if ca_edge_scale is None:
            ca_active = False
        else:
            enh_small = suppress_chromatic_aberration(
                enh_small, strength=cfg.ca_suppress, edge_scale_override=ca_edge_scale
            )

    wavelet_active = cfg.color_fix_strength > 0.0 or cfg.color_fix_luma_strength > 0.0
    low_diff_small: Optional[np.ndarray] = None
    low_x_map: Optional[np.ndarray] = None
    if wavelet_active:
        low_diff_small = _compute_wavelet_low_diff_small(enh_small, ref_small, full_h=h, full_w=w)
        low_x_map = (np.arange(w, dtype=np.float32) + 0.5) * (low_diff_small.shape[1] / float(w)) - 0.5
        enh_small = _apply_wavelet_low_diff_strip(
            enh_small,
            cv2.resize(low_diff_small, (enh_small.shape[1], enh_small.shape[0]), interpolation=cv2.INTER_LINEAR),
            chroma_strength=float(np.clip(cfg.color_fix_strength, 0.0, 1.0)),
            luma_strength=float(np.clip(cfg.color_fix_luma_strength, 0.0, 1.0)),
        )

    wb_gains: Optional[np.ndarray] = None
    if cfg.white_balance_strength > 0.0:
        wb_gains = _compute_wb_gains(enh_small, float(np.clip(cfg.white_balance_strength, 0.0, 1.0)))
        enh_small = _apply_wb_gains(enh_small, wb_gains)

    tone_stats: Optional[Tuple[float, float, float, int, int]] = None
    if cfg.tone_strength > 0.0:
        tone_stats = _compute_tone_global_stats(enh_small, h, w)

    detail_mask = focus_mask
    if cfg.detail_strength > 0.0 and cfg.bokeh_guard and detail_mask is None:
        detail_mask = compute_focus_bokeh_mask(
            enhanced_rgb, face_mask=face_mask, memmap_threshold_mp=memmap_threshold_mp
        )
    detail_radius = max(2, min(6, round(min(h, w) * 0.003)))
    d_x_map = (
        (np.arange(w, dtype=np.float32) + 0.5) * (detail_mask.shape[1] / float(w)) - 0.5
        if (detail_mask is not None and detail_mask.shape[:2] != (h, w))
        else None
    )

    out_mm = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azzani_finish_")
    strip_h = 512 if w < 10000 else 256
    pad = 80  # Strictly exceeds max guided radius (64) to prevent strip seam borders.

    for y0 in range(0, h, strip_h):
        y1 = min(h, y0 + strip_h)
        py0 = max(0, y0 - pad)
        py1 = min(h, y1 + pad)

        strip = np.clip(np.asarray(enhanced_rgb[py0:py1], dtype=np.float32), 0.0, 1.0)
        if ca_active and ca_edge_scale is not None:
            strip = suppress_chromatic_aberration(
                strip, strength=cfg.ca_suppress, edge_scale_override=ca_edge_scale
            )
        if wavelet_active and low_diff_small is not None:
            low_strip = extract_resized_strip(
                low_diff_small, h, w, py0, py1, interpolation=cv2.INTER_LINEAR, x_map=low_x_map
            )
            strip = _apply_wavelet_low_diff_strip(
                strip,
                low_strip,
                chroma_strength=float(np.clip(cfg.color_fix_strength, 0.0, 1.0)),
                luma_strength=float(np.clip(cfg.color_fix_luma_strength, 0.0, 1.0)),
            )
        if wb_gains is not None:
            strip = _apply_wb_gains(strip, wb_gains)
        if cfg.tone_strength > 0.0 and tone_stats is not None:
            mean_y, p02, p98, tone_radius, l_radius = tone_stats
            strip = _apply_tone_enhance_with_stats(
                strip,
                strength=float(np.clip(cfg.tone_strength, 0.0, 1.0)),
                exposure_hint=exposure_hint,
                mean_y=mean_y,
                p02=p02,
                p98=p98,
                radius=tone_radius,
                l_radius=l_radius,
            )
        if cfg.detail_strength > 0.0:
            d_strip = None
            if detail_mask is not None:
                if detail_mask.shape[:2] == (h, w):
                    d_strip = np.asarray(detail_mask[py0:py1], dtype=np.float32)
                else:
                    d_strip = extract_resized_strip(
                        detail_mask, h, w, py0, py1, interpolation=cv2.INTER_LINEAR, x_map=d_x_map
                    )
                if d_strip.ndim == 2:
                    d_strip = d_strip[:, :, None]
            strip = guided_detail_sharpen(
                strip,
                strength=cfg.detail_strength,
                detail_mask=d_strip,
                radius_override=detail_radius,
            )

        iy0 = y0 - py0
        iy1 = iy0 + (y1 - y0)
        core = strip[iy0:iy1]
        if cfg.natural_grain > 0.0:
            core = add_natural_micrograin(core, amount=cfg.natural_grain, seed=1337 + y0)
        out_mm[y0:y1] = np.clip(core, 0.0, 1.0)

    return out_mm

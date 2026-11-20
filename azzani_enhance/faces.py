"""MIT-licensed YuNet 5-landmark FFHQ face alignment and GFPGAN restoration with feathered blending."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .inference import EnginePool, RuntimeConfig, SpandrelEngine
from .memmap_utils import create_managed_memmap, is_memmap_array, stride_preview
from .models import ensure_model

# Canonical FFHQ 512x512 5-landmark reference points (viewer's left eye, right eye, nose, left mouth, right mouth).
FFHQ_512_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

# Pseudo YuNet detection row in 512x512 canonical FFHQ frame so OpenCV SFace alignCrop maps straight to 112x112.
_FFHQ_YUNET_ROW = np.array(
    [
        96.0, 112.0, 320.0, 352.0,
        192.98138, 239.94708,
        318.90277, 240.19360,
        256.63416, 314.01935,
        201.26117, 371.41043,
        313.08905, 371.15118,
        0.99,
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """Detected face bounding box, confidence score, and 5 canonical facial landmarks."""

    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    landmarks_5pt: np.ndarray  # shape (5, 2) float32 in image pixel coords


def detect_faces_yunet(
    rgb_f32: np.ndarray,
    detector_onnx_path: Path,
    *,
    score_threshold: float = 0.72,
    nms_threshold: float = 0.30,
    min_face_px: int = 24,
) -> List[DetectedFace]:
    """Detect faces and 5 facial landmarks using OpenCV YuNet ONNX (MIT License)."""
    h, w = rgb_f32.shape[:2]
    if h < 32 or w < 32:
        return []

    # Scale large images down cleanly for YuNet so detection stays fast & reliable across huge canvases.
    max_det_side = 1280.0
    scale = min(1.0, max_det_side / float(max(h, w)))
    det_w = max(32, int(round(w * scale)))
    det_h = max(32, int(round(h * scale)))

    src_view = (
        stride_preview(rgb_f32, max_side=2560)
        if (is_memmap_array(rgb_f32) or (h * w) > 4_000_000)
        else np.clip(rgb_f32, 0.0, 1.0)
    )
    bgr_u8 = cv2.cvtColor(np.rint(src_view * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if det_img_needed := (bgr_u8.shape[0] != det_h or bgr_u8.shape[1] != det_w):
        det_img = cv2.resize(bgr_u8, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        det_img = bgr_u8
    del det_img_needed

    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

    detector = cv2.FaceDetectorYN.create(
        model=str(detector_onnx_path),
        config="",
        input_size=(det_w, det_h),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=5000,
    )
    _, raw_faces = detector.detect(det_img)
    if raw_faces is None or len(raw_faces) == 0:
        return []

    inv_scale = 1.0 / scale
    results: List[DetectedFace] = []
    for row in raw_faces:
        bx, by, bw, bh = row[0:4] * inv_scale
        conf = float(row[14])
        if min(bw, bh) < min_face_px:
            continue

        # YuNet returns: right_eye (viewer's left), left_eye (viewer's right), nose, r_mouth, l_mouth.
        pts = row[4:14].reshape(5, 2).astype(np.float32) * inv_scale
        results.append(
            DetectedFace(
                bbox=(float(bx), float(by), float(bw), float(bh)),
                confidence=conf,
                landmarks_5pt=pts,
            )
        )

    # Process larger/higher-confidence faces deterministically.
    results.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    return results


def _umeyama_similarity(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """Compute exact 2D similarity affine matrix (no reflection) via Umeyama SVD."""
    src = src_pts.astype(np.float64)
    dst = dst_pts.astype(np.float64)
    n = src.shape[0]
    if n < 2:
        return None

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_var = float(np.sum(src_c ** 2) / n)
    if src_var < 1e-8:
        return None

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.ones(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        s[1] = -1.0

    rot = u @ np.diag(s) @ vt
    scale = float(np.sum(d * s) / src_var)
    if scale <= 1e-6 or not np.isfinite(scale):
        return None

    trans = dst_mean - scale * (rot @ src_mean)
    mat = np.zeros((2, 3), dtype=np.float32)
    mat[:, :2] = (scale * rot).astype(np.float32)
    mat[:, 2] = trans.astype(np.float32)
    return mat


def _build_face_feather_mask(size: int = 512) -> np.ndarray:
    """Create a wide, silky-smooth elliptical feather mask guaranteed zero on outer borders."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = (size - 1) * 0.5, (size - 1) * 0.5
    # In the FFHQ-aligned frame the head spans roughly y in [40, 470] and ears sit near x ~ 120 / 390.
    # Cover forehead, hair, ears and chin (GFPGAN renders them far better than generic SR, which turns
    # hair into painterly brush strokes) while fading out before the crop corners.
    ry, rx = size * 0.47, size * 0.44
    dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    # Wide cosine falloff from r=0.62 (weight 1.0) to r=1.0 (weight 0.0) prevents hairline seams.
    t = np.clip((dist - 0.62) / 0.38, 0.0, 1.0)
    mask = 0.5 * (1.0 + np.cos(np.pi * t))

    # Guarantee exact 0.0 along outer margin so inverse warp never leaks borderValue.
    edge_margin = max(8, int(round(size * 0.03)))
    border_fade_y = np.minimum(
        np.clip((yy - edge_margin) / float(edge_margin), 0.0, 1.0),
        np.clip(((size - 1 - edge_margin) - yy) / float(edge_margin), 0.0, 1.0),
    )
    border_fade_x = np.minimum(
        np.clip((xx - edge_margin) / float(edge_margin), 0.0, 1.0),
        np.clip(((size - 1 - edge_margin) - xx) / float(edge_margin), 0.0, 1.0),
    )
    mask = mask * border_fade_y * border_fade_x
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def _laplacian_pyramid_fusion(
    tone_matched: np.ndarray,
    orig_crop: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int = 4,
) -> np.ndarray:
    """Fuse restored face detail with original illumination via a 4-level Burt-Adelson Laplacian pyramid.

    Preserves crisp restored eyes/skin/hair details inside the mask while smoothly absorbing coarse
    scene illumination differences across perimeter scales so hairlines never show halo rings.
    """
    g_tone = [tone_matched.astype(np.float32)]
    g_orig = [orig_crop.astype(np.float32)]
    # Detail bands hold strong weight deep into hair/jawline transitions; base level smoothly locks lighting.
    detail_mask = np.clip(0.35 + 0.65 * np.clip((mask[:, :, 0] - 0.08) / 0.65, 0.0, 1.0), 0.0, 1.0)
    base_mask = np.clip((mask[:, :, 0] - 0.15) / 0.70, 0.0, 1.0)
    g_w_detail = [detail_mask]
    g_w_base = [base_mask]

    for _ in range(levels - 1):
        g_tone.append(cv2.pyrDown(g_tone[-1]))
        g_orig.append(cv2.pyrDown(g_orig[-1]))
        g_w_detail.append(cv2.pyrDown(g_w_detail[-1]))
        g_w_base.append(cv2.pyrDown(g_w_base[-1]))

    def _build_laplacian(gauss_pyr: List[np.ndarray]) -> List[np.ndarray]:
        lap = []
        for idx in range(len(gauss_pyr) - 1):
            th, tw = gauss_pyr[idx].shape[:2]
            up = cv2.pyrUp(gauss_pyr[idx + 1], dstsize=(tw, th))
            lap.append(gauss_pyr[idx] - up)
        lap.append(gauss_pyr[-1])
        return lap

    l_tone = _build_laplacian(g_tone)
    l_orig = _build_laplacian(g_orig)

    blended_pyr: List[np.ndarray] = []
    n_lv = len(l_tone)
    for idx in range(n_lv):
        # Coarsest residual locks perimeter illumination to original; finer bands transition gently.
        w_map = g_w_base[idx] if idx >= n_lv - 2 else g_w_detail[idx]
        w_3 = np.clip(w_map, 0.0, 1.0)[:, :, None]
        blended_pyr.append(w_3 * l_tone[idx] + (1.0 - w_3) * l_orig[idx])

    reconstructed = blended_pyr[-1]
    for idx in range(n_lv - 2, -1, -1):
        th, tw = blended_pyr[idx].shape[:2]
        reconstructed = cv2.pyrUp(reconstructed, dstsize=(tw, th)) + blended_pyr[idx]
    return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)


# Fraction of the restored-vs-original mean LAB offset removed per channel (L*, a*, b*).
_FACE_TONE_PULL = (0.5, 0.7, 0.7)


def _match_face_tone(restored_crop: np.ndarray, orig_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gently re-align restored skin tone to the scene and fuse via a 4-level Laplacian pyramid.

    Only a partial mean shift is applied in CIE LAB so the face matches the neck/ears. Contrast is
    deliberately NOT rescaled toward the (blurry) original: crisp eyes, brows, lashes and hair strands
    are exactly what face restoration is supposed to add.
    """
    core = mask[:, :, 0] > 0.45
    if np.count_nonzero(core) < 64:
        return restored_crop

    res_lab = cv2.cvtColor(np.clip(restored_crop, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    ori_lab = cv2.cvtColor(np.clip(orig_crop, 0.0, 1.0), cv2.COLOR_RGB2LAB)

    out_lab = res_lab.copy()
    for ch in range(3):
        offset = float(np.mean(ori_lab[:, :, ch][core]) - np.mean(res_lab[:, :, ch][core]))
        out_lab[:, :, ch] = res_lab[:, :, ch] + _FACE_TONE_PULL[ch] * offset

    out_lab[:, :, 0] = np.clip(out_lab[:, :, 0], 0.0, 100.0)
    out_lab[:, :, 1] = np.clip(out_lab[:, :, 1], -127.0, 127.0)
    out_lab[:, :, 2] = np.clip(out_lab[:, :, 2], -127.0, 127.0)
    tone_matched = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)

    return _laplacian_pyramid_fusion(tone_matched, orig_crop, mask, levels=4)


def _load_sface_recognizer(models_dir: Optional[Path], quiet: bool = False) -> Any:
    """Load OpenCV Zoo SFace (Apache-2.0) recognizer lazily; returns None cleanly if unavailable."""
    if not hasattr(cv2, "FaceRecognizerSF"):
        return None
    try:
        sface_path, _ = ensure_model("face-recognizer-sface", models_dir=models_dir, quiet=quiet)
        return cv2.FaceRecognizerSF.create(str(sface_path), "", 0, 0)
    except Exception:
        return None


def _measure_face_identity_cosine(
    sface: Any,
    orig_rgb_512: np.ndarray,
    restored_rgb_512: np.ndarray,
) -> float:
    """Compute canonical SFace cosine similarity in [−1, 1] between aligned original & restored 512 crops."""
    orig_bgr = cv2.cvtColor(np.rint(np.clip(orig_rgb_512, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rest_bgr = cv2.cvtColor(np.rint(np.clip(restored_rgb_512, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    crop_o = sface.alignCrop(orig_bgr, _FFHQ_YUNET_ROW)
    crop_r = sface.alignCrop(rest_bgr, _FFHQ_YUNET_ROW)
    feat_o = sface.feature(crop_o)
    feat_r = sface.feature(crop_r)
    return float(sface.match(feat_o, feat_r, cv2.FaceRecognizerSF_FR_COSINE))


def _apply_sface_identity_guard(
    sface: Any,
    aligned_orig: np.ndarray,
    restored_face: np.ndarray,
    *,
    identity_guard: float,
) -> tuple[np.ndarray, float, bool]:
    """Clamp facial likeness drift via SFace cosine similarity against aligned_orig."""
    cos_sim = _measure_face_identity_cosine(sface, aligned_orig, restored_face)
    min_cos = 0.25 + 0.35 * float(np.clip(identity_guard, 0.0, 1.0))
    if cos_sim >= min_cos:
        return restored_face, cos_sim, False

    # Progressive pull back toward the original identity until cosine clears target or floor reached.
    keep_ratio = float(np.clip((cos_sim - 0.10) / max(1e-4, min_cos - 0.10), 0.30, 0.90))
    guarded = np.clip(keep_ratio * restored_face + (1.0 - keep_ratio) * aligned_orig, 0.0, 1.0).astype(np.float32)
    post_cos = _measure_face_identity_cosine(sface, aligned_orig, guarded)
    if post_cos < min_cos:
        keep_ratio2 = max(0.20, keep_ratio * 0.55)
        guarded = np.clip(keep_ratio2 * restored_face + (1.0 - keep_ratio2) * aligned_orig, 0.0, 1.0).astype(np.float32)
        post_cos = _measure_face_identity_cosine(sface, aligned_orig, guarded)
    return guarded, post_cos, True


def _compute_affine_roi_bounds(
    affine: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    crop_size: int = 512,
    pad: int = 16,
) -> Optional[Tuple[int, int, int, int]]:
    """Return clamped (rx0, ry0, rx1, ry1) bounding box covering an inverse-warped 512x512 crop."""
    inv_mat = cv2.invertAffineTransform(affine.astype(np.float64))
    corners = np.array(
        [[0.0, 0.0, 1.0], [crop_size, 0.0, 1.0], [crop_size, crop_size, 1.0], [0.0, crop_size, 1.0]],
        dtype=np.float64,
    )
    mapped = (inv_mat @ corners.T).T
    rx0 = max(0, int(np.floor(float(np.min(mapped[:, 0])))) - pad)
    ry0 = max(0, int(np.floor(float(np.min(mapped[:, 1])))) - pad)
    rx1 = min(img_w, int(np.ceil(float(np.max(mapped[:, 0])))) + pad)
    ry1 = min(img_h, int(np.ceil(float(np.max(mapped[:, 1])))) + pad)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    return rx0, ry0, rx1, ry1


def compute_face_restoration_layer(
    rgb_f32: np.ndarray,
    *,
    device: str = "auto",
    precision: str = "auto",
    models_dir: Optional[Path] = None,
    quiet: bool = False,
    pre_detected_faces: Optional[List[DetectedFace]] = None,
    engine_pool: Optional[EnginePool] = None,
    force_reprobe: bool = False,
    identity_guard: float = 0.65,
    identity_stats_out: Optional[Dict[str, Any]] = None,
    restore_source: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    """Run YuNet + GFPGANv1.4 once at full fidelity, returning (restored_canvas, face_mask, count).

    Because fidelity blending is linear against the base canvas, downstream live UI retuning can
    adjust face fidelity instantaneously without re-running PyTorch inference.

    ``restore_source`` (same size as ``rgb_f32``) is the image faces are cropped from for GFPGAN and the
    identity check, e.g. the cleaned input resized to the output size. Generic SR output tends to paint
    faces and hair with brush-stroke textures that GFPGAN would otherwise inherit; seam fusion and
    pasting still happen against ``rgb_f32`` so the face transitions into the SR canvas consistently.
    """
    if restore_source is not None and restore_source.shape[:2] != rgb_f32.shape[:2]:
        raise ValueError("restore_source must have the same height/width as rgb_f32")
    if pre_detected_faces is not None:
        faces = pre_detected_faces
    else:
        yunet_path, _ = ensure_model("face-detector-yunet", models_dir=models_dir, quiet=quiet)
        faces = detect_faces_yunet(rgb_f32, yunet_path)
    if not faces:
        return rgb_f32, None, 0

    gfpgan_path, gfpgan_spec = ensure_model("face-gfpgan", models_dir=models_dir, quiet=quiet)
    rt = RuntimeConfig(
        device=device,
        precision=precision,
        tile_size=0,  # 512x512 aligned face runs in one pass.
        tta=1,
        tile_complexity="light",
        force_reprobe=force_reprobe,
    )
    val_range = gfpgan_spec.value_range if gfpgan_spec else "-1_1"
    if engine_pool is not None:
        engine = engine_pool.acquire(gfpgan_path, runtime=rt, value_range=val_range)
    else:
        engine = SpandrelEngine(gfpgan_path, runtime=rt, value_range=val_range)

    h, w = rgb_f32.shape[:2]
    use_memmap = is_memmap_array(rgb_f32) or ((h * w) >= 24_000_000)
    if use_memmap:
        canvas: np.ndarray = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azzani_facec_")
        strip_h = 512 if w < 10000 else 256
        for y0 in range(0, h, strip_h):
            y1 = min(h, y0 + strip_h)
            canvas[y0:y1] = np.clip(np.asarray(rgb_f32[y0:y1], dtype=np.float32), 0.0, 1.0)
        face_mask: np.ndarray = create_managed_memmap((h, w, 1), dtype=np.float32, prefix="azzani_fmask_")
    else:
        canvas = np.clip(rgb_f32, 0.0, 1.0).astype(np.float32, copy=True)
        face_mask = np.zeros((h, w, 1), dtype=np.float32)

    identity_guard_clamped = float(np.clip(identity_guard, 0.0, 1.0))
    sface = _load_sface_recognizer(models_dir, quiet=quiet) if identity_guard_clamped > 0.0 else None

    feather_512 = _build_face_feather_mask(512)
    restored_count = 0
    identity_cosines: List[float] = []
    faces_guarded = 0

    try:
        for face in faces:
            affine = _umeyama_similarity(face.landmarks_5pt, FFHQ_512_TEMPLATE)
            if affine is None:
                continue
            roi_bounds = _compute_affine_roi_bounds(affine, w, h, crop_size=512, pad=16)
            if roi_bounds is None:
                continue
            rx0, ry0, rx1, ry1 = roi_bounds
            rw, rh = rx1 - rx0, ry1 - ry0

            affine_roi = affine.copy()
            affine_roi[:, 2] = affine[:, 2] + affine[:, 0] * float(rx0) + affine[:, 1] * float(ry0)

            roi_canvas = np.clip(np.asarray(canvas[ry0:ry1, rx0:rx1], dtype=np.float32), 0.0, 1.0)

            def _align(roi: np.ndarray) -> np.ndarray:
                warped = cv2.warpAffine(
                    roi,
                    affine_roi,
                    (512, 512),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                return np.clip(warped, 0.0, 1.0).astype(np.float32)

            aligned_base = _align(roi_canvas)
            if restore_source is not None:
                roi_src = np.clip(np.asarray(restore_source[ry0:ry1, rx0:rx1], dtype=np.float32), 0.0, 1.0)
                aligned_src = _align(roi_src)
            else:
                aligned_src = aligned_base

            restored_face, _ = engine.enhance(aligned_src)
            restored_face = _match_face_tone(restored_face, aligned_base, feather_512)

            if sface is not None and identity_guard_clamped > 0.0:
                restored_face, post_cos, was_guarded = _apply_sface_identity_guard(
                    sface, aligned_src, restored_face, identity_guard=identity_guard_clamped
                )
                identity_cosines.append(post_cos)
                if was_guarded:
                    faces_guarded += 1

            inv_mask = cv2.warpAffine(
                feather_512[:, :, 0],
                affine_roi,
                (rw, rh),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            inv_face = cv2.warpAffine(
                restored_face,
                affine_roi,
                (rw, rh),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0, 0.0, 0.0),
            )

            alpha_m = np.clip(inv_mask, 0.0, 1.0)[:, :, None]
            face_mask[ry0:ry1, rx0:rx1] = np.maximum(np.asarray(face_mask[ry0:ry1, rx0:rx1]), alpha_m)
            canvas[ry0:ry1, rx0:rx1] = np.clip(
                alpha_m * inv_face + (1.0 - alpha_m) * roi_canvas,
                0.0,
                1.0,
            )
            restored_count += 1
    finally:
        if engine_pool is None:
            engine.close()

    if identity_stats_out is not None and identity_cosines:
        identity_stats_out["face_identity_cosine"] = round(float(np.mean(identity_cosines)), 3)
        identity_stats_out["faces_guarded"] = faces_guarded

    if restored_count == 0:
        return rgb_f32, None, 0
    return canvas, face_mask, restored_count


def blend_face_layer(
    base_rgb: np.ndarray,
    restored_canvas: np.ndarray,
    fidelity: float = 0.78,
) -> np.ndarray:
    """Blend cached full-fidelity face restoration canvas smoothly with base_rgb in <5ms."""
    fid = float(np.clip(fidelity, 0.0, 1.0))
    if fid <= 0.0:
        return base_rgb
    if fid >= 1.0:
        return restored_canvas
    h, w = base_rgb.shape[:2]
    if is_memmap_array(base_rgb) or is_memmap_array(restored_canvas) or (h * w) >= 24_000_000:
        out_mm = create_managed_memmap((h, w, 3), dtype=np.float32, prefix="azzani_fblend_")
        strip_h = 512 if w < 10000 else 256
        for y0 in range(0, h, strip_h):
            y1 = min(h, y0 + strip_h)
            b_strip = np.asarray(base_rgb[y0:y1], dtype=np.float32)
            r_strip = np.asarray(restored_canvas[y0:y1], dtype=np.float32)
            out_mm[y0:y1] = np.clip(b_strip + fid * (r_strip - b_strip), 0.0, 1.0)
        return out_mm
    return np.clip(base_rgb + fid * (restored_canvas - base_rgb), 0.0, 1.0).astype(np.float32)


def restore_faces(
    rgb_f32: np.ndarray,
    *,
    fidelity: float = 0.78,
    device: str = "auto",
    precision: str = "auto",
    models_dir: Optional[Path] = None,
    quiet: bool = False,
    pre_detected_faces: Optional[List[DetectedFace]] = None,
    engine_pool: Optional[EnginePool] = None,
    force_reprobe: bool = False,
    identity_guard: float = 0.65,
) -> tuple[np.ndarray, int]:
    """Detect faces via YuNet (MIT), align to FFHQ 512x512, restore via GFPGANv1.4, and feather-blend.

    Returns (restored_rgb_f32, num_faces_restored).
    """
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    if fidelity <= 0.0:
        return rgb_f32, 0

    restored_canvas, _, restored_count = compute_face_restoration_layer(
        rgb_f32,
        device=device,
        precision=precision,
        models_dir=models_dir,
        quiet=quiet,
        pre_detected_faces=pre_detected_faces,
        engine_pool=engine_pool,
        force_reprobe=force_reprobe,
        identity_guard=identity_guard,
    )
    if restored_count == 0:
        return rgb_f32, 0
    return blend_face_layer(rgb_f32, restored_canvas, fidelity=fidelity), restored_count

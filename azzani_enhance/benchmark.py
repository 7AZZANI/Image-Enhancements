"""System diagnostics and synthetic degradation-recovery benchmark suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .analyzer import analyze_image
from .hardware import probe_system_hardware
from .image_io import INPUT_EXTENSIONS, LoadedImage, load_image
from .inference import EnginePool
from .pipeline import EnhancementConfig, enhance_loaded_image


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two float32 [0, 1] RGB images."""
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) on luminance [0, 1] using Gaussian windows."""
    a = 0.2126 * img_a[:, :, 0] + 0.7152 * img_a[:, :, 1] + 0.0722 * img_a[:, :, 2]
    b = 0.2126 * img_b[:, :, 0] + 0.7152 * img_b[:, :, 1] + 0.0722 * img_b[:, :, 2]
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def compute_gmsd(img_ref: np.ndarray, img_tst: np.ndarray) -> float:
    """Compute Gradient Magnitude Similarity Deviation (GMSD, lower = better perceptual fidelity)."""
    r_y = (0.2126 * img_ref[:, :, 0] + 0.7152 * img_ref[:, :, 1] + 0.0722 * img_ref[:, :, 2]).astype(np.float32)
    t_y = (0.2126 * img_tst[:, :, 0] + 0.7152 * img_tst[:, :, 1] + 0.0722 * img_tst[:, :, 2]).astype(np.float32)

    kx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], dtype=np.float32) / 3.0
    ky = kx.T

    rx = cv2.filter2D(r_y, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
    ry = cv2.filter2D(r_y, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)
    tx = cv2.filter2D(t_y, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
    ty = cv2.filter2D(t_y, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)

    mr = np.sqrt(rx * rx + ry * ry)
    mt = np.sqrt(tx * tx + ty * ty)
    c = 0.0026
    gms = (2.0 * mr * mt + c) / (mr * mr + mt * mt + c)
    return float(np.std(gms))


def compute_perceptual_score(
    img_tst: np.ndarray,
    img_ref: Optional[np.ndarray] = None,
    *,
    halo_index: float = 0.0,
) -> float:
    """Composite perceptual quality score [0..100] combining structure, edge acuity & anti-artifact cleanliness."""
    diag = analyze_image(img_tst)
    acuity = float(np.clip(diag.sharpness_score, 0.0, 1.0))
    clean = float(np.clip(1.0 - 0.65 * diag.jpeg_blockiness - 0.80 * min(1.0, halo_index), 0.0, 1.0))
    if img_ref is not None:
        ssim = float(np.clip(compute_ssim(img_ref, img_tst), 0.0, 1.0))
        gmsd = compute_gmsd(img_ref, img_tst)
        gms_fid = float(np.clip(1.0 - gmsd * 2.5, 0.0, 1.0))
        raw = 0.35 * ssim + 0.30 * gms_fid + 0.25 * acuity + 0.10 * clean
    else:
        raw = 0.65 * acuity + 0.35 * clean
    return round(float(np.clip(raw * 100.0, 0.0, 100.0)), 1)


def generate_reference_chart(width: int = 512, height: int = 512) -> np.ndarray:
    """Synthesize a high-frequency photography-like reference image (textures, edges, gradients, text)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(1, width - 1)
    yn = yy / max(1, height - 1)

    # Smooth sky/studio gradient background.
    r = 0.16 + 0.55 * yn + 0.15 * np.sin(xn * np.pi)
    g = 0.24 + 0.45 * (1.0 - 0.5 * yn) + 0.10 * np.cos(xn * 2.0 * np.pi)
    b = 0.38 + 0.42 * (1.0 - yn)
    img = np.stack([r, g, b], axis=2)

    # Fine grass / fabric weave micro-texture in lower third.
    weave = 0.08 * np.sin(xx * 0.75) * np.cos(yy * 0.85) + 0.05 * np.sin((xx + yy) * 0.45)
    mask_tex = (yn > 0.58).astype(np.float32)[:, :, None]
    img = img + weave[:, :, None] * mask_tex

    # Crisp geometric shapes, thin lines, and Siemens-star-like high-frequency radial lines.
    u8 = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    cx, cy = width // 2, int(height * 0.40)
    for deg in range(0, 360, 10):
        rad = np.deg2rad(deg)
        x2 = int(cx + np.cos(rad) * (width * 0.20))
        y2 = int(cy + np.sin(rad) * (height * 0.20))
        cv2.line(u8, (cx, cy), (x2, y2), (240, 215, 165), 1, cv2.LINE_AA)
    cv2.circle(u8, (cx, cy), int(width * 0.06), (215, 75, 55), -1, cv2.LINE_AA)
    cv2.rectangle(u8, (28, 36), (170, 145), (35, 55, 80), -1)
    cv2.putText(u8, "AZZANI 100%", (36, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 240), 2, cv2.LINE_AA)
    return u8.astype(np.float32) / 255.0


def degrade_image(ref_rgb: np.ndarray, *, downscale: int = 2, jpeg_q: int = 28, noise_sigma: float = 0.025) -> np.ndarray:
    """Apply realistic web/camera degradation: blur, downscale, noise, and low-Q JPEG compression."""
    h, w = ref_rgb.shape[:2]
    blurred = cv2.GaussianBlur(ref_rgb, (0, 0), sigmaX=1.0)
    small = cv2.resize(blurred, (max(16, w // downscale), max(16, h // downscale)), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(123)
    noisy = np.clip(small + rng.normal(0.0, noise_sigma, small.shape).astype(np.float32), 0.0, 1.0)
    bgr_u8 = cv2.cvtColor(np.rint(noisy * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
    assert ok
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def run_doctor() -> Dict[str, Any]:
    """Inspect environment, PyTorch GPU/CUDA/ROCm/XPU/MPS capabilities, OpenCV DNN, and RAM."""
    hw = probe_system_hardware(include_torch=True)
    info: Dict[str, Any] = {
        "python": hw.python_version,
        "platform": platform.platform(),
        "torch": hw.torch_version,
        "cuda_available": hw.torch_cuda_available,
        "mps_available": hw.torch_mps_available,
        "xpu_available": hw.torch_xpu_available,
        "active_device": hw.active_device,
        "integration_status": hw.integration_status,
        "recommendation": hw.recommendation,
        "opencv": cv2.__version__,
        "yunet_supported": hasattr(cv2, "FaceDetectorYN"),
        "ram_total_gb": hw.ram_total_gb,
        "ram_available_gb": hw.ram_available_gb,
        "recommended_wheel": hw.recommended_wheel_tag,
        "available_backends": list(hw.available_backends),
        "onnx_providers": list(hw.onnx_providers),
        "torch_compile_available": hw.torch_compile_available,
        "ffmpeg_available": hw.ffmpeg_available,
        "generative_available": hw.generative_available,
        "recommend_directml": hw.recommend_directml,
    }
    if hw.gpus:
        primary = next(
            (g for g in hw.gpus if f"cuda:{g.index}" == hw.active_device),
            hw.gpus[0],
        )
        info["gpu_name"] = primary.name
        info["gpu_vendor"] = primary.vendor
        info["gpu_backend"] = primary.backend
        if primary.compute_cap:
            cc_clean = primary.compute_cap.replace(".", "")
            info["gpu_capability"] = f"sm_{cc_clean}"
        if primary.vram_total_mb is not None:
            info["vram_total_mb"] = primary.vram_total_mb
        if primary.vram_free_mb is not None:
            info["vram_free_mb"] = primary.vram_free_mb
        if len(hw.gpus) > 1:
            info["all_gpus"] = [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_free_mb": g.vram_free_mb,
                    "vram_total_mb": g.vram_total_mb,
                }
                for g in hw.gpus
            ]
    return info


def run_benchmark(
    preset: str = "fast",
    *,
    device: str = "auto",
    precision: str = "auto",
    tta: int = 1,
) -> Dict[str, Any]:
    """Run an end-to-end synthetic degradation & restoration benchmark against bicubic baseline."""
    ref = generate_reference_chart(512, 512)
    degraded = degrade_image(ref, downscale=2, jpeg_q=28, noise_sigma=0.025)
    bicubic = np.clip(cv2.resize(degraded, (512, 512), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)

    loaded = LoadedImage(
        rgb=degraded,
        alpha=None,
        source_bit_depth=8,
        source_format="JPEG",
        exif_bytes=None,
        icc_profile=None,
        original_size=(degraded.shape[1], degraded.shape[0]),
    )

    cfg = EnhancementConfig(
        preset=preset,
        scale=2.0,
        device=device,
        precision=precision,
        tta=tta,
        face_restore="off",
        quiet=True,
    )
    t0 = time.perf_counter()
    res = enhance_loaded_image(loaded, cfg=cfg)
    dt = time.perf_counter() - t0

    diag_ref = analyze_image(ref)
    diag_bic = analyze_image(bicubic)

    return {
        "preset": preset,
        "device_used": res.device_used,
        "precision_used": res.precision_used,
        "models_used": res.models_used,
        "elapsed_seconds": round(dt, 3),
        "bicubic_psnr_db": round(compute_psnr(ref, bicubic), 3),
        "azzani_psnr_db": round(compute_psnr(ref, res.output_rgb), 3),
        "bicubic_ssim": round(compute_ssim(ref, bicubic), 4),
        "azzani_ssim": round(compute_ssim(ref, res.output_rgb), 4),
        "ref_sharpness": diag_ref.sharpness_score,
        "bicubic_sharpness": diag_bic.sharpness_score,
        "azzani_sharpness": res.diagnostics_after.sharpness_score,
        "degraded_blockiness": res.diagnostics_before.jpeg_blockiness,
        "azzani_blockiness": res.diagnostics_after.jpeg_blockiness,
        "degraded_noise_sigma": res.diagnostics_before.noise_sigma,
        "azzani_noise_sigma": res.diagnostics_after.noise_sigma,
    }


def _baseline_enhance(degraded: np.ndarray, out_w: int, out_h: int, mode: str) -> np.ndarray:
    if mode == "bicubic":
        return np.clip(cv2.resize(degraded, (out_w, out_h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0).astype(np.float32)
    # Lanczos4 + gentle unsharp mask baseline
    up = np.clip(cv2.resize(degraded, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0).astype(np.float32)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
    return np.clip(up + 0.35 * (up - blur), 0.0, 1.0).astype(np.float32)


def run_benchmark_suite(
    presets: Sequence[str] = ("bicubic", "lanczos", "fast", "pro"),
    *,
    dataset_dir: Optional[Path] = None,
    scale: int = 2,
    jpeg_q: int = 28,
    noise_sigma: float = 0.025,
    max_images: int = 6,
    device: str = "auto",
    precision: str = "auto",
    tta: int = 1,
) -> Dict[str, Any]:
    """Run reproducible multi-metric comparison across baselines and AZZANI presets."""
    refs: List[tuple[str, np.ndarray]] = []
    if dataset_dir is not None:
        dpath = Path(dataset_dir).expanduser().resolve()
        if not dpath.is_dir():
            raise FileNotFoundError(f"Benchmark dataset directory does not exist: {dpath}")
        img_files = [
            p for p in sorted(dpath.iterdir())
            if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
        ][: max(1, int(max_images))]
        if not img_files:
            raise RuntimeError(f"No supported images found in '{dpath}'.")
        for p in img_files:
            loaded = load_image(p)
            refs.append((p.name, loaded.rgb))
    else:
        refs.append(("synthetic_chart_512", generate_reference_chart(512, 512)))

    pairs: List[tuple[str, np.ndarray, np.ndarray]] = []
    for name, ref in refs:
        h, w = ref.shape[:2]
        # Ensure clean divisibility for exact pixel alignment at target scale
        ch = max(32, (h // scale) * scale)
        cw = max(32, (w // scale) * scale)
        ref_crop = ref[:ch, :cw]
        deg = degrade_image(ref_crop, downscale=scale, jpeg_q=jpeg_q, noise_sigma=noise_sigma)
        pairs.append((name, ref_crop, deg))

    pool = EnginePool(max_models=2)
    hw_doc = run_doctor()
    entries: List[Dict[str, Any]] = []
    try:
        for method in presets:
            m_key = method.strip().lower()
            psnrs: List[float] = []
            ssims: List[float] = []
            gmsds: List[float] = []
            p_scores: List[float] = []
            sharps: List[float] = []
            blocks: List[float] = []
            halos: List[float] = []
            dts: List[float] = []
            mps: List[float] = []
            dev_label = "cpu"
            prec_label = "fp32"

            for _name, ref_crop, deg in pairs:
                rh, rw = ref_crop.shape[:2]
                mp_out = max(1e-4, (rh * rw) / 1_000_000.0)
                t0 = time.perf_counter()
                if m_key in {"bicubic", "lanczos"}:
                    out = _baseline_enhance(deg, rw, rh, m_key)
                    dt = time.perf_counter() - t0
                    diag_out = analyze_image(out)
                    halo = 0.0
                    if m_key == "lanczos":
                        halo = float(np.clip(np.mean(np.abs(out - ref_crop) > 0.22), 0.0, 1.0))
                else:
                    loaded = LoadedImage(
                        rgb=deg,
                        alpha=None,
                        source_bit_depth=8,
                        source_format="JPEG",
                        exif_bytes=None,
                        icc_profile=None,
                        original_size=(deg.shape[1], deg.shape[0]),
                    )
                    res = enhance_loaded_image(
                        loaded,
                        cfg=EnhancementConfig(
                            preset=m_key,
                            scale=float(scale),
                            width=rw,
                            height=rh,
                            device=device,
                            precision=precision,
                            tta=tta,
                            face_restore="off",
                            quiet=True,
                        ),
                        engine_pool=pool,
                    )
                    dt = time.perf_counter() - t0
                    out = res.output_rgb
                    diag_out = res.diagnostics_after
                    halo = float(res.critic_metrics.get("halo_ringing_index", 0.0))
                    dev_label = res.device_used
                    prec_label = res.precision_used

                psnrs.append(compute_psnr(ref_crop, out))
                ssims.append(compute_ssim(ref_crop, out))
                gmsds.append(compute_gmsd(ref_crop, out))
                p_scores.append(compute_perceptual_score(out, ref_crop, halo_index=halo))
                sharps.append(diag_out.sharpness_score)
                blocks.append(diag_out.jpeg_blockiness)
                halos.append(halo)
                dts.append(dt)
                mps.append(mp_out)

            total_dt = float(sum(dts))
            total_mp = max(1e-4, float(sum(mps)))
            entries.append(
                {
                    "method": m_key,
                    "device": dev_label,
                    "precision": prec_label,
                    "psnr_db": round(float(np.mean(psnrs)), 3),
                    "ssim": round(float(np.mean(ssims)), 4),
                    "gmsd": round(float(np.mean(gmsds)), 4),
                    "perceptual_score": round(float(np.mean(p_scores)), 1),
                    "sharpness": round(float(np.mean(sharps)), 3),
                    "blockiness": round(float(np.mean(blocks)), 4),
                    "halo_index": round(float(np.mean(halos)), 4),
                    "avg_seconds": round(float(np.mean(dts)), 3),
                    "sec_per_mp": round(total_dt / total_mp, 3),
                }
            )
    finally:
        pool.clear()

    return {
        "dataset": str(dataset_dir) if dataset_dir else "synthetic_chart_512",
        "images_evaluated": len(pairs),
        "scale": scale,
        "tta": tta,
        "hardware": hw_doc.get("gpu_name") or hw_doc.get("active_device", "cpu"),
        "optional_metrics": {
            "lpips_installed": importlib.util.find_spec("lpips") is not None,
            "pyiqa_installed": importlib.util.find_spec("pyiqa") is not None,
        },
        "results": entries,
    }


def format_leaderboard_markdown(suite: Dict[str, Any]) -> str:
    """Format run_benchmark_suite output as a clean GitHub Markdown comparison table."""
    lines = [
        f"**Benchmark Suite** — Dataset: `{suite.get('dataset')}` | Scale: `{suite.get('scale')}x` | "
        f"Hardware: `{suite.get('hardware')}` | TTA: `{suite.get('tta')}x`",
        "",
        "| Method | Device | PSNR (dB) | SSIM | GMSD (lower better) | Perceptual Score | Sharpness | Blockiness | Sec/Img | Sec/MP |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite.get("results", []):
        lines.append(
            f"| `{row['method']}` | {row['device']} | {row['psnr_db']:.2f} | {row['ssim']:.4f} | "
            f"{row['gmsd']:.4f} | {row['perceptual_score']:.1f} | {row['sharpness']:.3f} | "
            f"{row['blockiness']:.4f} | {row['avg_seconds']:.2f}s | {row['sec_per_mp']:.2f}s |"
        )
    return "\n".join(lines)

"""Unit and integration tests for AZZANI Image Enhancements."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from azzani_enhance.analyzer import analyze_image
from azzani_enhance.benchmark import compute_psnr, compute_ssim, degrade_image, generate_reference_chart
from azzani_enhance.color_tone import (
    ToneDetailConfig,
    adaptive_tone_enhance,
    apply_photometric_finish,
    auto_white_balance,
    compute_focus_bokeh_mask,
    guided_detail_sharpen,
    guided_filter_gray,
    self_critic_refinement,
    suppress_chromatic_aberration,
    wavelet_color_fix,
)
from azzani_enhance.faces import FFHQ_512_TEMPLATE, _build_face_feather_mask, _umeyama_similarity
from azzani_enhance.image_io import INPUT_EXTENSIONS, RAW_EXTENSIONS, SUPPORTED_EXTENSIONS, bleed_alpha_edges, load_image, save_image
from azzani_enhance.inference import (
    _build_feather_mask,
    _compute_tile_starts,
    _safe_reflect_pad,
    apply_tta_forward,
    apply_tta_inverse,
)
from azzani_enhance.pipeline import EnhancementConfig, enhance_file


def test_tta_all_8_symmetries_exact_inverse_non_square() -> None:
    rng = torch.Generator().manual_seed(0)
    x = torch.rand((1, 3, 19, 31), generator=rng, dtype=torch.float32)
    for k in range(8):
        fwd = apply_tta_forward(x, k)
        inv = apply_tta_inverse(fwd, k)
        assert inv.shape == x.shape
        assert torch.allclose(inv, x, atol=1e-6), f"Failed round-trip for TTA k={k}"


def test_safe_reflect_pad_handles_tiny_tensors() -> None:
    x = torch.rand((1, 3, 3, 5), dtype=torch.float32)
    padded = _safe_reflect_pad(x, pad_r=59, pad_b=61)
    assert padded.shape == (1, 3, 64, 64)
    assert torch.isfinite(padded).all()


def test_tile_starts_and_feather_mask_convex_coverage() -> None:
    h, w = 230, 310
    tile, ov = 96, 24
    ys = _compute_tile_starts(h, tile, ov)
    xs = _compute_tile_starts(w, tile, ov)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for y0 in ys:
        y1 = min(h, y0 + tile)
        for x0 in xs:
            x1 = min(w, x0 + tile)
            mask = _build_feather_mask(
                y1 - y0,
                x1 - x0,
                ov,
                fade_top=(y0 > 0),
                fade_bottom=(y1 < h),
                fade_left=(x0 > 0),
                fade_right=(x1 < w),
            )
            weight_sum[y0:y1, x0:x1] += mask

    assert np.all(weight_sum > 1e-4)
    # Outer image boundary pixels must stay strongly covered.
    assert np.min(weight_sum[0, :]) >= 0.99
    assert np.min(weight_sum[:, 0]) >= 0.99


def test_alpha_edge_bleed_eliminates_black_halo_fringe() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    alpha = np.zeros((64, 64), dtype=np.float32)
    # Bright red square inside transparent black void.
    rgb[20:44, 20:44] = [0.95, 0.15, 0.10]
    alpha[20:44, 20:44] = 1.0

    bled = bleed_alpha_edges(rgb, alpha)
    # Border transparent pixel adjacent to red square must now carry red, not black.
    assert np.allclose(bled[19, 30], [0.95, 0.15, 0.10], atol=1e-5)
    assert np.allclose(bled[30, 30], [0.95, 0.15, 0.10], atol=1e-5)


def test_16bit_png_roundtrip_and_exif_orientation(tmp_path: Path) -> None:
    # 1. 16-bit gradient round-trip.
    grad = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :, None]
    rgb_16 = np.repeat(np.repeat(grad, 32, axis=0), 3, axis=2)
    p16 = tmp_path / "grad16.png"
    save_image(p16, rgb_16, bit_depth=16, dither=False)
    loaded16 = load_image(p16)
    assert loaded16.source_bit_depth == 16
    assert np.allclose(loaded16.rgb, rgb_16, atol=2e-4)

    # 2. EXIF Orientation 6 (90 deg CW stored -> upright portrait 40x80 after transpose).
    raw_landscape = np.zeros((40, 80, 3), dtype=np.uint8)
    raw_landscape[:, :20] = [255, 0, 0]  # Top edge after 90-CW rotation
    pil = Image.fromarray(raw_landscape, mode="RGB")
    exif = Image.Exif()
    exif[274] = 6
    p_exif = tmp_path / "rotated.jpg"
    pil.save(p_exif, exif=exif.tobytes(), quality=95)

    loaded_rot = load_image(p_exif)
    assert loaded_rot.original_size == (40, 80)  # Transposed width x height
    assert loaded_rot.rgb.shape == (80, 40, 3)


def test_analyzer_detects_blur_noise_blockiness_and_exposure() -> None:
    ref = generate_reference_chart(256, 256)
    diag_clean = analyze_image(ref)
    assert diag_clean.sharpness_score > 0.35
    assert diag_clean.noise_sigma < 3.0

    # Blur test
    blurred = cv2.GaussianBlur(ref, (0, 0), sigmaX=2.8)
    diag_blur = analyze_image(blurred)
    assert diag_blur.sharpness_score < diag_clean.sharpness_score * 0.65

    # Noise test
    rng = np.random.default_rng(99)
    noisy = np.clip(ref + rng.normal(0.0, 14.0 / 255.0, ref.shape).astype(np.float32), 0.0, 1.0)
    diag_noisy = analyze_image(noisy)
    assert 9.0 <= diag_noisy.noise_sigma <= 20.0
    assert diag_noisy.recommended_denoise is True

    # Low-Q JPEG blockiness test
    degraded = degrade_image(ref, downscale=1, jpeg_q=15, noise_sigma=0.0)
    diag_jpg = analyze_image(degraded)
    assert diag_jpg.jpeg_blockiness > diag_clean.jpeg_blockiness + 0.15

    # Dark exposure test
    dark = np.clip(ref * 0.22, 0.0, 1.0)
    diag_dark = analyze_image(dark)
    assert diag_dark.recommended_exposure_boost > 0.25


def test_color_tone_guided_filter_wavelet_fix_and_flat_guard() -> None:
    ref = generate_reference_chart(128, 128)
    # Apply artificial warm green color tint drift like an imperfect GAN output.
    shifted = np.clip(ref * np.array([0.85, 1.08, 0.78], dtype=np.float32), 0.0, 1.0)
    fixed = wavelet_color_fix(shifted, ref, chroma_strength=1.0, luma_strength=0.8)
    assert compute_psnr(ref, fixed) > compute_psnr(ref, shifted) + 4.0

    # Guided filter preserves edge contrast.
    step = np.zeros((64, 64), dtype=np.float32)
    step[:, 32:] = 1.0
    gf = guided_filter_gray(step, step, radius=4, eps=0.01)
    assert gf[:, 10].mean() < 0.05
    assert gf[:, 54].mean() > 0.95

    # Flat gray image shouldn't crash or jump wildly under tone/WB/detail.
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    toned = adaptive_tone_enhance(flat, strength=0.6)
    wb = auto_white_balance(flat, strength=0.5)
    det = guided_detail_sharpen(flat, strength=0.4)
    assert np.all(np.isfinite(toned)) and np.all(np.isfinite(wb)) and np.all(np.isfinite(det))
    assert np.allclose(wb, flat, atol=1e-4)

    finished = apply_photometric_finish(ref, ref, ToneDetailConfig())
    assert finished.dtype == np.float32
    assert 0.0 <= float(finished.min()) <= float(finished.max()) <= 1.0


def test_face_umeyama_and_feather_mask_zero_borders() -> None:
    mask = _build_face_feather_mask(512)
    assert mask.shape == (512, 512, 1)
    assert float(mask[256, 256, 0]) > 0.99
    # Strict zero on all outer edges prevents dark square paste seams.
    assert float(mask[0, :, 0].max()) == 0.0
    assert float(mask[-1, :, 0].max()) == 0.0
    assert float(mask[:, 0, 0].max()) == 0.0
    assert float(mask[:, -1, 0].max()) == 0.0

    # Recover rotated + scaled + translated similarity transform accurately.
    theta = np.deg2rad(22.0)
    scale = 0.65
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    src = (FFHQ_512_TEMPLATE @ rot.T) * scale + np.array([120.0, 85.0], dtype=np.float32)
    mat = _umeyama_similarity(src, FFHQ_512_TEMPLATE)
    assert mat is not None
    assert np.linalg.det(mat[:, :2]) > 0.0
    projected = (src @ mat[:, :2].T) + mat[:, 2]
    assert np.allclose(projected, FFHQ_512_TEMPLATE, atol=1e-2)


def test_photometric_end_to_end_rgba_and_comparison(tmp_path: Path) -> None:
    ref = generate_reference_chart(192, 192)
    alpha = np.zeros((192, 192), dtype=np.float32)
    alpha[24:168, 24:168] = 1.0
    inp = tmp_path / "input_rgba.png"
    out = tmp_path / "enhanced_rgba.png"
    comp = tmp_path / "compare.png"
    save_image(inp, ref, alpha=alpha, dither=False)

    cfg = EnhancementConfig(
        preset="photometric",
        scale=2.0,
        face_restore="off",
        quiet=True,
    )
    res = enhance_file(inp, out, cfg=cfg, comparison_path=comp)
    assert out.is_file()
    assert comp.is_file()
    assert res.output_size == (384, 384)
    assert res.output_alpha is not None and res.output_alpha.shape == (384, 384)
    # Check SSIM on visible foreground region stays high with clear contrast/sharpness boost.
    up_ref = cv2.resize(ref, (384, 384), interpolation=cv2.INTER_CUBIC)
    fg = res.output_alpha > 0.5
    assert compute_ssim(up_ref * fg[:, :, None], res.output_rgb * fg[:, :, None]) > 0.88


def test_analyzer_catches_green_cast_and_coarse_grain() -> None:
    rng = np.random.default_rng(7)
    base = np.full((256, 256, 3), 0.50, dtype=np.float32)
    # Create coarse 2x2 scanned film grain + greenish cast.
    small_grain = rng.normal(0.0, 0.035, (128, 128, 1)).astype(np.float32)
    coarse_grain = cv2.resize(small_grain, (256, 256), interpolation=cv2.INTER_NEAREST)[:, :, None]
    green_cast = np.clip(base + coarse_grain + np.array([-0.04, 0.04, -0.02], dtype=np.float32), 0.0, 1.0)
    diag = analyze_image(green_cast)
    assert diag.noise_sigma >= 1.8
    assert diag.color_cast_strength > 0.20


def test_haze_removal_recovers_milky_black_point() -> None:
    ref = generate_reference_chart(128, 128)
    hazy = np.clip(ref * 0.65 + 0.18, 0.0, 1.0).astype(np.float32)
    dehazed = adaptive_tone_enhance(hazy, strength=0.35)
    assert float(np.percentile(dehazed, 2)) < float(np.percentile(hazy, 2)) - 0.05


def test_cli_entrypoints_smoke(tmp_path: Path) -> None:
    from azzani_enhance.cli import main as cli_main

    assert cli_main(["--list-models"]) == 0
    assert cli_main(["--doctor"]) == 0
    img = generate_reference_chart(64, 64)
    p_in = tmp_path / "tiny.png"
    p_out = tmp_path / "tiny_out.png"
    save_image(p_in, img, dither=False)
    assert cli_main([str(p_in), "--inspect", "--json"]) == 0
    assert cli_main([str(p_in), "-o", str(p_out), "--preset", "photometric", "--scale", "1", "-q"]) == 0
    assert p_out.is_file()


def test_hardware_wheel_selection_and_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from azzani_enhance.hardware import (
        load_precision_cache,
        probe_system_hardware,
        save_precision_entry,
        select_nvidia_wheel_tag,
    )

    assert select_nvidia_wheel_tag("13.2", ["7.5"]) == "cu126"
    assert select_nvidia_wheel_tag("12.8", ["12.0"]) == "cu128"
    assert select_nvidia_wheel_tag("12.4", ["8.6"]) == "cu124"
    assert select_nvidia_wheel_tag("12.1", ["8.6"]) == "cu121"
    assert select_nvidia_wheel_tag("11.8", ["7.5"]) == "cu118"
    assert select_nvidia_wheel_tag("11.2", ["6.1"]) == "cpu"

    monkeypatch.setenv("AZZANI_ENHANCE_CACHE_DIR", str(tmp_path))
    save_precision_entry("test-gpu|test-model", "fp16", "fp16 (verified)")
    loaded = load_precision_cache()
    assert loaded["test-gpu|test-model"]["dtype"] == "fp16"

    report = probe_system_hardware(include_torch=True)
    assert report.python_version
    assert report.integration_status in {
        "optimal",
        "cpu_only_ok",
        "torch_cpu_on_gpu_host",
        "gpu_driver_mismatch",
        "rosetta_x86_on_arm64",
        "torch_missing",
    }


def test_bootstrap_plan_handles_cuda_and_tmpdir(tmp_path: Path) -> None:
    from azzani_enhance.bootstrap import build_pip_install_plan
    from azzani_enhance.hardware import HardwareReport

    rep = HardwareReport(
        os_name="Linux",
        machine="x86_64",
        python_version="3.14.0",
        cpu_count=8,
        ram_total_gb=16.0,
        ram_available_gb=10.0,
        tmp_free_gb=2.0,
        var_tmp_free_gb=50.0,
        recommend_tmpdir="/var/tmp",
        recommended_backend="cuda",
        recommended_wheel_tag="cu126",
        recommended_index_url="https://download.pytorch.org/whl/cu126",
    )
    cmds, env = build_pip_install_plan(rep, Path("/fake/.venv/bin/python"), Path("/fake/repo"))
    assert env.get("TMPDIR") == "/var/tmp"
    assert any("https://download.pytorch.org/whl/cu126" in cmd for cmd in cmds)
    assert any("--no-cache-dir" in cmd for cmd in cmds)


def test_hardware_tier_routing_cpu_vs_high_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    import azzani_enhance.pipeline as pipeline_mod

    img = generate_reference_chart(256, 256)
    diag = analyze_image(img)
    cfg = EnhancementConfig(preset="auto")

    monkeypatch.setattr(pipeline_mod, "detect_hardware_tier", lambda _d: "cpu")
    model_cpu, scale_cpu, *_ = pipeline_mod._resolve_plan(diag, cfg)
    assert model_cpu in {"photo-fast", "anime-2x"}
    assert scale_cpu <= 2.0

    monkeypatch.setattr(pipeline_mod, "detect_hardware_tier", lambda _d: "gpu_high")
    model_hi, scale_hi, *_ = pipeline_mod._resolve_plan(diag, cfg)
    assert model_hi in {"photo-ultra", "anime-4x"}
    assert scale_hi == 4.0

    monkeypatch.setattr(pipeline_mod, "detect_hardware_tier", lambda _d: "gpu_compact")
    model_mid, scale_mid, *_ = pipeline_mod._resolve_plan(diag, cfg)
    assert model_mid in {"photo-pro", "anime-4x"}
    assert scale_mid == 4.0


def test_watch_folder_automation_once_and_idempotent(tmp_path: Path) -> None:
    from azzani_enhance.automation import WatchFolderRunner

    inbox = tmp_path / "inbox"
    outbox = inbox / "enhanced"
    inbox.mkdir()

    img = generate_reference_chart(96, 96)
    src = inbox / "photo.png"
    save_image(src, img, dither=False)
    # Temp partial download file must be ignored cleanly.
    (inbox / "downloading.png.part").write_bytes(b"partial")

    runner = WatchFolderRunner(
        inbox,
        outbox,
        cfg=EnhancementConfig(preset="photometric", scale=1.0, face_restore="off", quiet=True),
        stability_delay=0.0,
    )
    try:
        ev1 = runner.process_once(check_stability=False)
        assert len(ev1) == 1
        assert ev1[0].status == "enhanced"
        expected_out = outbox / "photo_azzani.png"
        assert expected_out.is_file()

        # Second scan must be idempotent (zero duplicate re-processing).
        ev2 = runner.process_once(check_stability=False)
        assert ev2 == []
        assert runner.processed_count == 1
    finally:
        runner.stop()


def test_web_ui_server_end_to_end(tmp_path: Path) -> None:
    import base64
    import json
    import threading
    import urllib.request

    from azzani_enhance.web_ui import create_web_server

    server, state = create_web_server("127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            assert r.status == 200
            html = r.read().decode("utf-8")
            assert "AZZANI" in html

        with urllib.request.urlopen(f"{base}/api/doctor", timeout=5) as r:
            doc = json.loads(r.read().decode("utf-8"))
            assert "python" in doc and "active_device" in doc

        img = generate_reference_chart(64, 64)
        sample = tmp_path / "web_sample.png"
        save_image(sample, img, dither=False)
        b64 = base64.b64encode(sample.read_bytes()).decode("ascii")

        req = urllib.request.Request(
            f"{base}/api/enhance",
            data=json.dumps(
                {
                    "filename": "web_sample.png",
                    "data_base64": f"data:image/png;base64,{b64}",
                    "preset": "photometric",
                    "scale": 2,
                    "tta": 1,
                    "face_restore": "off",
                    "output_ext": "png",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            res = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert res["summary"]["output_size"] == [128, 128]

        with urllib.request.urlopen(f"{base}{res['result_url']}", timeout=5) as r:
            out_bytes = r.read()
            assert out_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        server.shutdown()
        server.server_close()
        state.cleanup()
        t.join(timeout=3)


def test_ca_suppression_removes_purple_fringe_preserves_solid_colors() -> None:
    img = np.full((128, 128, 3), 0.15, dtype=np.float32)
    # High-contrast neutral vertical bars on the left so edge density triggers optical CA pass.
    for x in range(8, 64, 16):
        img[:, x : x + 8] = 0.88
    # Inject artificial purple chromatic aberration fringes along two vertical step boundaries.
    for edge_x in (24, 40):
        img[:, edge_x - 1 : edge_x + 1, 0] = np.clip(img[:, edge_x - 1 : edge_x + 1, 0] + 0.25, 0.0, 1.0)
        img[:, edge_x - 1 : edge_x + 1, 1] = np.clip(img[:, edge_x - 1 : edge_x + 1, 1] - 0.15, 0.0, 1.0)
        img[:, edge_x - 1 : edge_x + 1, 2] = np.clip(img[:, edge_x - 1 : edge_x + 1, 2] + 0.30, 0.0, 1.0)
    # Large solid saturated block on the right that must stay vivid.
    img[16:64, 80:120] = np.array([0.85, 0.35, 0.10], dtype=np.float32)

    out = suppress_chromatic_aberration(img, strength=0.95)
    raw_fringe_chroma = float(np.std(img[:, 23:25], axis=2).mean())
    fixed_fringe_chroma = float(np.std(out[:, 23:25], axis=2).mean())
    assert fixed_fringe_chroma < raw_fringe_chroma * 0.55

    # Interior of the saturated color patch must be completely preserved.
    assert float(np.max(np.abs(out[24:56, 88:112] - img[24:56, 88:112]))) < 1e-3


def test_focus_bokeh_mask_and_self_critic_halo_clamp() -> None:
    h, w = 128, 128
    # Top half: smooth bokeh background; bottom half: crisp sharp checkerboard subject.
    y_grad = np.linspace(0.25, 0.55, h, dtype=np.float32)[:, None]
    ref = np.repeat(np.repeat(y_grad, w, axis=1)[:, :, None], 3, axis=2)
    yy, xx = np.mgrid[0:64, 0:w]
    checker = (((yy // 4) + (xx // 4)) % 2).astype(np.float32)
    ref[64:, :, :] = (0.20 + 0.65 * checker)[:, :, None]

    face_mask = np.zeros((h, w, 1), dtype=np.float32)
    face_mask[10:26, 10:26] = 1.0
    focus_mask = compute_focus_bokeh_mask(ref, face_mask=face_mask)
    assert float(focus_mask[32:56, 40:100].mean()) < float(focus_mask[72:120, :].mean())
    assert float(focus_mask[14:22, 14:22].min()) > 0.95

    # Construct strong artificial SR edge overshoot halos along horizontal step boundaries.
    enh = ref.copy()
    enh[64:66, :, :] = 1.0
    enh[62:64, :, :] = 0.05
    cleaned, metrics = self_critic_refinement(
        enh,
        ref,
        clamp_strength=0.9,
        focus_mask=focus_mask,
    )
    assert metrics["raw_halo_index"] > metrics["halo_ringing_index"]
    assert 40.0 <= metrics["confidence_score"] <= 99.8
    mse_before = float(np.mean((enh - ref) ** 2))
    mse_after = float(np.mean((cleaned - ref) ** 2))
    assert mse_after < mse_before


def test_split_prepare_and_compose_matches_full_pipeline() -> None:
    from azzani_enhance import (
        compose_enhancement_result,
        enhance_loaded_image,
        prepare_enhancement_layers,
    )
    from azzani_enhance.image_io import LoadedImage

    chart = generate_reference_chart(96, 96)
    loaded = LoadedImage(
        rgb=chart,
        alpha=None,
        source_bit_depth=8,
        source_format="PNG",
        exif_bytes=None,
        icc_profile=None,
        original_size=(96, 96),
    )
    cfg = EnhancementConfig(
        preset="photometric",
        scale=1.0,
        face_restore="off",
        ca_suppress=0.5,
        bokeh_guard=True,
        halo_guard=0.75,
        quiet=True,
    )
    res_full = enhance_loaded_image(loaded, cfg=cfg)
    layers = prepare_enhancement_layers(loaded, cfg=cfg)
    res_comp = compose_enhancement_result(layers, cfg=cfg)

    assert np.allclose(res_full.output_rgb, res_comp.output_rgb, atol=1e-6)
    assert res_comp.quality_confidence == res_full.quality_confidence

    # Non-destructive retuning without touching neural stage changes output in milliseconds.
    tuned_cfg = EnhancementConfig(
        preset="photometric",
        scale=1.0,
        face_restore="off",
        tone_strength=0.80,
        detail_strength=0.75,
        bokeh_guard=False,
        quiet=True,
    )
    res_tuned = compose_enhancement_result(layers, cfg=tuned_cfg, include_neural_time=False)
    assert float(np.mean(np.abs(res_tuned.output_rgb - res_comp.output_rgb))) > 0.005


def test_raw_missing_rawpy_raises_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    assert ".dng" in RAW_EXTENSIONS
    assert ".dng" in INPUT_EXTENSIONS
    assert ".dng" not in SUPPORTED_EXTENSIONS

    raw_file = tmp_path / "sample.dng"
    raw_file.write_bytes(b"FAKE_DNG_HEADER")
    monkeypatch.setitem(sys.modules, "rawpy", None)

    with pytest.raises(RuntimeError, match=r"rawpy|\[raw\]"):
        load_image(raw_file)


def test_web_ui_live_retune_and_histogram(tmp_path: Path) -> None:
    import base64
    import json
    import threading
    import urllib.request

    from azzani_enhance.web_ui import create_web_server

    server, state = create_web_server("127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base = f"http://127.0.0.1:{port}"
        img = generate_reference_chart(64, 64)
        sample = tmp_path / "studio_test.png"
        save_image(sample, img, dither=False)
        b64 = base64.b64encode(sample.read_bytes()).decode("ascii")

        req_enh = urllib.request.Request(
            f"{base}/api/enhance",
            data=json.dumps(
                {
                    "filename": "studio_test.png",
                    "data_base64": f"data:image/png;base64,{b64}",
                    "preset": "photometric",
                    "scale": 2,
                    "tta": 1,
                    "face_restore": "off",
                    "output_ext": "png",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_enh, timeout=15) as r:
            enh = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert "job_id" in enh and "retune_defaults" in enh
            hist = enh["histogram"]
            assert hist["bins"] == 64 and len(hist["luma"]) == 64
            assert all(0.0 <= v <= 1.0 for v in hist["r"] + hist["g"] + hist["b"] + hist["luma"])
            assert "quality_confidence" in enh["summary"]
            assert "critic_metrics" in enh["summary"]

        req_rt = urllib.request.Request(
            f"{base}/api/retune",
            data=json.dumps(
                {
                    "job_id": enh["job_id"],
                    "tone": 0.72,
                    "detail": 0.65,
                    "color_fix": 0.50,
                    "wb_strength": 0.40,
                    "ca_suppress": 0.60,
                    "halo_guard": 0.80,
                    "grain": 0.005,
                    "bokeh_guard": False,
                    "output_ext": "jpg",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_rt, timeout=10) as r:
            rt = json.loads(r.read().decode("utf-8"))
            assert r.status == 200
            assert rt["retuned"] is True
            assert rt["download_name"].endswith(".jpg")
            assert len(rt["histogram"]["luma"]) == 64

        with urllib.request.urlopen(f"{base}{rt['result_url']}", timeout=5) as r:
            jpg_bytes = r.read()
            assert jpg_bytes[:3] == b"\xff\xd8\xff"
    finally:
        server.shutdown()
        server.server_close()
        state.cleanup()
        t.join(timeout=3)


def test_cli_phase1_flags_parse() -> None:
    from azzani_enhance.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["photo.jpg", "--ca-suppress", "0.65", "--halo-guard", "0.85", "--no-bokeh-guard"])
    assert ns.ca_suppress == pytest.approx(0.65)
    assert ns.halo_guard == pytest.approx(0.85)
    assert ns.no_bokeh_guard is True


def test_analyzer_fft_detects_motion_blur_vs_sharp_stripes_and_defocus() -> None:
    import cv2

    rng = np.random.default_rng(7)
    base = generate_reference_chart(384, 384)
    # Add rich isotropic multi-frequency micro-texture so natural foliage/architecture spectrum is populated.
    tex = rng.normal(0.0, 0.14, size=(384, 384, 3)).astype(np.float32)
    sharp = np.clip(base * 0.78 + 0.11 + tex, 0.0, 1.0).astype(np.float32)

    # Pure 1D horizontal stripes are completely sharp along vertical direction -> NOT camera shake!
    yy = np.linspace(0.0, 24.0 * np.pi, 384, dtype=np.float32)[:, None, None]
    stripes = np.broadcast_to(0.5 + 0.35 * np.sin(yy), (384, 384, 3)).astype(np.float32)

    # Isotropic Gaussian defocus blur lacks directional spectral notch.
    defocus = cv2.GaussianBlur(sharp, (15, 15), sigmaX=4.0)

    # Directional linear motion blur at 40 deg (length 15px).
    ksize = 15
    kern = np.zeros((ksize, ksize), dtype=np.float32)
    kern[ksize // 2, :] = 1.0 / float(ksize)
    rot_m = cv2.getRotationMatrix2D((ksize / 2.0 - 0.5, ksize / 2.0 - 0.5), 40.0, 1.0)
    kern = cv2.warpAffine(kern, rot_m, (ksize, ksize))
    kern /= max(1e-6, float(np.sum(kern)))
    motion = np.clip(cv2.filter2D(sharp, -1, kern), 0.0, 1.0).astype(np.float32)

    d_sharp = analyze_image(sharp)
    d_stripes = analyze_image(stripes)
    d_defocus = analyze_image(defocus)
    d_motion = analyze_image(motion)

    assert d_sharp.recommended_deblur is False
    assert d_stripes.recommended_deblur is False
    assert d_defocus.recommended_deblur is False
    assert d_motion.recommended_deblur is True
    assert d_motion.motion_blur_score > max(d_sharp.motion_blur_score, d_defocus.motion_blur_score) + 0.12
    assert 0.0 <= d_motion.motion_blur_angle_deg < 180.0


def test_deblur_plan_routing_and_nafnet_registry_entry() -> None:
    from azzani_enhance import pipeline as pipeline_mod
    from azzani_enhance.analyzer import ImageDiagnostics
    from azzani_enhance.models import MODEL_REGISTRY

    assert "deblur-1x" in MODEL_REGISTRY
    assert MODEL_REGISTRY["deblur-1x"].scale == 1
    assert pipeline_mod.PRESET_MODEL_MAP["deblur"] == "deblur-1x"

    diag_blur = ImageDiagnostics(
        width=640,
        height=480,
        megapixels=0.31,
        sharpness_score=0.25,
        blur_severity=0.72,
        noise_sigma=2.0,
        noise_severity=0.10,
        jpeg_blockiness=0.05,
        mean_luma=52.0,
        dynamic_range=60.0,
        shadow_clip_pct=0.5,
        highlight_clip_pct=0.5,
        color_cast_strength=0.02,
        color_cast_ab=(0.0, 0.0),
        is_illustration_likely=False,
        recommended_preset="pro",
        recommended_dejpeg=False,
        recommended_denoise=False,
        recommended_exposure_boost=0.0,
        motion_blur_score=0.78,
        motion_blur_angle_deg=35.0,
        recommended_deblur=True,
    )
    prim, scale, _, _, run_deblur, _, _ = pipeline_mod._resolve_plan(
        diag_blur, EnhancementConfig(preset="pro", deblur="auto")
    )
    assert prim == "photo-pro"
    assert run_deblur is True

    _, _, _, _, run_deblur_off, _, _ = pipeline_mod._resolve_plan(
        diag_blur, EnhancementConfig(preset="pro", deblur="off")
    )
    assert run_deblur_off is False

    # Already large photo (>2200px) routes straight to primary_model="deblur-1x" without double-running.
    diag_large_blur = ImageDiagnostics(**{**diag_blur.to_dict(), "width": 2600, "height": 2200})
    prim_large, scale_large, _, _, run_deblur_large, _, _ = pipeline_mod._resolve_plan(
        diag_large_blur, EnhancementConfig(preset="auto", deblur="auto")
    )
    assert prim_large == "deblur-1x"
    assert scale_large == pytest.approx(1.0)
    assert run_deblur_large is False


def test_face_laplacian_pyramid_fusion_and_sface_identity_guard(tmp_path: Path) -> None:
    import cv2

    from azzani_enhance.faces import (
        _apply_sface_identity_guard,
        _build_face_feather_mask,
        _load_sface_recognizer,
        _match_face_tone,
    )

    mask = _build_face_feather_mask(512)
    orig = np.full((512, 512, 3), 0.35, dtype=np.float32)
    # Place distinct textured detail inside center face region vs brighter exposure mismatch.
    orig[160:350, 170:340] = np.array([0.45, 0.35, 0.30], dtype=np.float32)
    restored = np.full((512, 512, 3), 0.62, dtype=np.float32)
    restored[180:330, 190:320] = np.array([0.75, 0.55, 0.48], dtype=np.float32)

    fused = _match_face_tone(restored, orig, mask)
    assert fused.shape == (512, 512, 3) and fused.dtype == np.float32
    # Outer perimeter closely anchors to orig lighting (0.35), eliminating hairline seams.
    assert float(np.mean(np.abs(fused[:24, :24] - orig[:24, :24]))) < 0.04

    # Test real SFace ONNX if cached locally; otherwise deterministic SFace-API check so zero skips occur.
    default_sface = Path.home() / ".cache" / "azzani-enhance" / "models" / "face_recognition_sface_2021dec.onnx"
    sface = _load_sface_recognizer(default_sface.parent, quiet=True) if default_sface.is_file() else None
    if sface is None:
        class _StubSFace:
            def alignCrop(self, img_bgr: np.ndarray, _row: np.ndarray) -> np.ndarray:
                return cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_AREA)

            def feature(self, crop: np.ndarray) -> np.ndarray:
                v = crop.reshape(-1).astype(np.float32)
                v = v - float(np.mean(v))
                n = float(np.linalg.norm(v)) + 1e-8
                return (v / n)[None, :]

            def match(self, f1: np.ndarray, f2: np.ndarray, _metric: int) -> float:
                return float(np.sum(f1 * f2))

        sface = _StubSFace()

    # Construct two completely distinct textured crops so both real SFace and stub yield low cosine.
    face_a = np.full((512, 512, 3), 0.52, dtype=np.float32)
    cv2.circle(face_a, (193, 240), 22, (0.08, 0.08, 0.08), -1)
    cv2.circle(face_a, (319, 240), 22, (0.08, 0.08, 0.08), -1)
    cv2.ellipse(face_a, (256, 372), (55, 16), 0, 0, 180, (0.75, 0.25, 0.25), -1)

    rng_f = np.random.default_rng(99)
    face_b = rng_f.uniform(0.0, 1.0, size=(512, 512, 3)).astype(np.float32)

    guarded, post_cos, was_guarded = _apply_sface_identity_guard(
        sface, face_a, face_b, identity_guard=0.80
    )
    assert was_guarded is True
    assert float(np.mean(np.abs(guarded - face_a))) < float(np.mean(np.abs(face_b - face_a)))
    assert np.isfinite(post_cos)


def test_memmap_out_of_core_tiling_and_streaming_tiff_roundtrip(tmp_path: Path) -> None:
    import gc
    from PIL import Image

    from azzani_enhance.color_tone import ToneDetailConfig, apply_photometric_finish, self_critic_refinement
    from azzani_enhance.image_io import _write_streaming_tiff
    from azzani_enhance.memmap_utils import _is_tmpfs_linux, create_managed_memmap, select_memmap_dir
    from azzani_enhance.pipeline import _resize_float32

    assert (Image.MAX_IMAGE_PIXELS or 0) >= 1_000_000_000
    selected_dir = select_memmap_dir()
    if Path("/tmp").exists() and _is_tmpfs_linux(Path("/tmp")) and Path("/var/tmp").exists():
        assert selected_dir == Path("/var/tmp")

    mm = create_managed_memmap((128, 128, 3), dtype=np.float32, prefix="azz_test_gc_")
    backing_dir = Path(str(mm.filename)).parent
    assert backing_dir.is_dir()
    mm[:] = 0.5
    mm.flush()
    del mm
    gc.collect()
    assert not backing_dir.exists()

    # Strip-chunked resize matches in-memory resize closely and returns a ManagedMemmap.
    chart = generate_reference_chart(200, 160)
    resized_mm = _resize_float32(chart, (320, 400), memmap_threshold_mp=0.05)
    assert isinstance(resized_mm, np.memmap)
    resized_mem = _resize_float32(chart, (320, 400), memmap_threshold_mp=100.0)
    assert float(np.mean(np.abs(np.asarray(resized_mm) - resized_mem))) < 0.005

    # Strip-chunked photometrics & critic match in-memory results cleanly without artifacts.
    tone_cfg = ToneDetailConfig(natural_grain=0.0)
    fin_mem = apply_photometric_finish(resized_mem, chart, tone_cfg, memmap_threshold_mp=100.0)
    fin_chk = apply_photometric_finish(resized_mm, chart, tone_cfg, force_chunked=True)
    assert isinstance(fin_chk, np.memmap)
    assert float(np.mean(np.abs(np.asarray(fin_chk) - fin_mem))) < 0.015

    crit_mem, m_mem = self_critic_refinement(fin_mem, chart, clamp_strength=0.75, memmap_threshold_mp=100.0)
    crit_chk, m_chk = self_critic_refinement(fin_chk, chart, clamp_strength=0.75, force_chunked=True)
    assert isinstance(crit_chk, np.memmap)
    assert abs(m_chk["halo_ringing_index"] - m_mem["halo_ringing_index"]) < 0.02
    assert float(np.mean(np.abs(np.asarray(crit_chk) - crit_mem))) < 0.015

    # Classic TIFF 8-bit, 16-bit, and BigTIFF streaming write round-trips cleanly.
    classic_8 = tmp_path / "stream_8bit.tif"
    _write_streaming_tiff(classic_8, fin_chk, None, bit_depth=8, dither=False, bigtiff=False)
    with Image.open(classic_8) as im8:
        arr8 = np.asarray(im8, dtype=np.float32) / 255.0
        assert arr8.shape == (320, 400, 3)
        assert float(np.mean(np.abs(arr8 - np.asarray(fin_chk)))) < 0.01

    big_16 = tmp_path / "stream_16bit_big.tiff"
    _write_streaming_tiff(big_16, fin_chk, None, bit_depth=16, dither=False, bigtiff=True)
    header_magic = big_16.read_bytes()[:4]
    assert header_magic == b"II+\x00"  # Little-endian 64-bit BigTIFF (version 43)
    with Image.open(big_16) as im16:
        assert im16.size == (400, 320)

    # End-to-end photometric enhance_loaded_image with tiny memmap threshold hands off ManagedMemmap cleanly.
    from azzani_enhance.image_io import LoadedImage, save_image
    from azzani_enhance.pipeline import enhance_loaded_image

    loaded = LoadedImage(
        rgb=chart,
        alpha=None,
        source_bit_depth=8,
        source_format="PNG",
        exif_bytes=None,
        icc_profile=None,
        original_size=(200, 160),
    )
    res_mm = enhance_loaded_image(
        loaded,
        cfg=EnhancementConfig(
            preset="photometric",
            scale=2.0,
            face_restore="off",
            memmap_threshold_mp=0.05,
            natural_grain=0.0,
            quiet=True,
        ),
    )
    assert isinstance(res_mm.output_rgb, np.memmap)
    assert res_mm.output_size == (400, 320)
    stream_tif = tmp_path / "e2e_stream.tif"
    save_image(stream_tif, res_mm.output_rgb, bit_depth=8, dither=False)
    with Image.open(stream_tif) as im_e2e:
        assert im_e2e.size == (400, 320)


def test_cli_phase2_flags_and_diagnostics() -> None:
    from azzani_enhance.cli import _build_config_from_args, build_parser

    parser = build_parser()
    ns = parser.parse_args(
        [
            "photo.jpg",
            "--preset",
            "deblur",
            "--deblur",
            "on",
            "--identity-guard",
            "0.85",
            "--memmap-mp",
            "32.0",
        ]
    )
    cfg = _build_config_from_args(ns)
    assert cfg.preset == "deblur"
    assert cfg.deblur == "on"
    assert cfg.identity_guard == pytest.approx(0.85)
    assert cfg.memmap_threshold_mp == pytest.approx(32.0)


def test_backends_provider_ranking_directml_tensorrt_and_compile_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.util
    import sys
    import types

    import spandrel

    from azzani_enhance.backends import (
        OnnxEngine,
        apply_tta_forward_np,
        apply_tta_inverse_np,
        select_onnx_providers,
    )
    from azzani_enhance.benchmark import run_doctor
    from azzani_enhance.inference import RuntimeConfig, SpandrelEngine, create_inference_engine

    # 1. NumPy TTA D4 exact algebraic inverse check
    rng = np.random.default_rng(7)
    arr_np = rng.random((1, 3, 19, 31), dtype=np.float32)
    for k in range(8):
        inv_np = apply_tta_inverse_np(apply_tta_forward_np(arr_np, k), k)
        assert inv_np.shape == arr_np.shape
        assert np.allclose(inv_np, arr_np, atol=1e-6)

    # 2. Provider ranking & missing-EP diagnostics
    assert select_onnx_providers(
        "directml",
        "auto",
        available_providers=["CPUExecutionProvider", "DmlExecutionProvider"],
    ) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="directml"):
        select_onnx_providers("directml", "auto", available_providers=["CPUExecutionProvider"])

    trt_eps = select_onnx_providers(
        "tensorrt",
        "auto",
        cache_dir=tmp_path / "trt_cache",
        available_providers=["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"],
    )
    assert isinstance(trt_eps[0], tuple) and trt_eps[0][0] == "TensorrtExecutionProvider"
    assert trt_eps[0][1]["trt_fp16_enable"] is True
    assert trt_eps[1:] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="tensorrt"):
        select_onnx_providers("tensorrt", "auto", available_providers=["CPUExecutionProvider"])

    assert select_onnx_providers(
        "auto",
        "cpu",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ) == ["CPUExecutionProvider"]
    assert select_onnx_providers(
        "auto",
        "auto",
        available_providers=["OpenVINOExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    ) == ["DmlExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]

    # 3. Mocked OnnxEngine DirectML session executing real tiling + 4x TTA
    class _FakeMeta:
        def __init__(self, name: str, shape: list):
            self.name = name
            self.shape = shape

    class _FakeSessOpts:
        graph_optimization_level = None

    class _FakeOptLevel:
        ORT_ENABLE_ALL = 99

    class _FakeOrtSession:
        def __init__(self, model_path: str, sess_options=None, providers=None):
            self.providers = [p[0] if isinstance(p, tuple) else p for p in (providers or ["CPUExecutionProvider"])]

        def get_providers(self):
            return self.providers

        def get_inputs(self):
            return [_FakeMeta("input", [1, 3, "h", "w"])]

        def get_outputs(self):
            return [_FakeMeta("output", [1, 3, "oh", "ow"])]

        def run(self, _out_names, feeds):
            x = next(iter(feeds.values()))
            return [np.repeat(np.repeat(x, 2, axis=2), 2, axis=3).astype(np.float32)]

    fake_ort = types.SimpleNamespace(
        InferenceSession=_FakeOrtSession,
        SessionOptions=_FakeSessOpts,
        GraphOptimizationLevel=_FakeOptLevel,
        get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    real_find_spec = importlib.util.find_spec

    def _mock_find_spec(name: str, package=None):
        if name == "onnxruntime":
            return object()
        return real_find_spec(name, package)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec)

    dummy_onnx = tmp_path / "upscale_2x.onnx"
    dummy_onnx.write_bytes(b"fake-onnx")
    eng_onnx = create_inference_engine(
        dummy_onnx,
        runtime=RuntimeConfig(backend="directml", tile_size=32, tile_overlap=8, tta=4),
    )
    assert isinstance(eng_onnx, OnnxEngine)
    assert eng_onnx.device.type == "dml"
    assert "directml" in eng_onnx.precision_note
    grad = np.zeros((48, 64, 3), dtype=np.float32)
    grad[:, :, 0] = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :]
    grad[:, :, 1] = np.linspace(0.0, 1.0, 48, dtype=np.float32)[:, None]
    out_onnx, tile_used = eng_onnx.enhance(grad)
    assert tile_used == 32
    assert out_onnx.shape == (96, 128, 3)
    expected_2x = np.repeat(np.repeat(grad, 2, axis=0), 2, axis=1)
    assert float(np.mean(np.abs(out_onnx - expected_2x))) < 0.025
    eng_onnx.close()

    # 4. SpandrelEngine compile fallback (both at compile init and runtime dynamic break)
    class _StubArch:
        name = "Stub2x"

    def _make_stub_descriptor():
        m = torch.nn.Upsample(scale_factor=2, mode="nearest")
        return spandrel.ImageModelDescriptor(
            model=m,
            state_dict={},
            architecture=_StubArch(),
            purpose="SR",
            tags=[],
            supports_half=False,
            supports_bfloat16=False,
            scale=2,
            input_channels=3,
            output_channels=3,
            size_requirements=spandrel.SizeRequirements(minimum=16, multiple_of=1, square=False),
            tiling=spandrel.ModelTiling.SUPPORTED,
        )

    class _StubLoader:
        def __init__(self, device=None):
            pass

        def load_from_file(self, _path):
            return _make_stub_descriptor()

    monkeypatch.setattr("azzani_enhance.inference.ModelLoader", _StubLoader)
    dummy_pth = tmp_path / "stub_2x.pth"
    dummy_pth.write_bytes(b"stub")

    # 4a: torch.compile fails at init
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("inductor off")))
    eng_c1 = SpandrelEngine(dummy_pth, runtime=RuntimeConfig(backend="compile", device="cpu", tta=1))
    assert eng_c1._compiled is False
    out_c1, _ = eng_c1.enhance(grad[:24, :24])
    assert out_c1.shape == (48, 48, 3)

    # 4b: torch.compile returns callable that breaks on runtime shape
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *a, **kw: (lambda _t: (_ for _ in ()).throw(RuntimeError("dynamic graph break"))),
    )
    eng_c2 = SpandrelEngine(dummy_pth, runtime=RuntimeConfig(backend="compile", device="cpu", tta=1))
    assert eng_c2._compiled is True
    out_c2, _ = eng_c2.enhance(grad[:24, :24])
    assert out_c2.shape == (48, 48, 3)
    assert eng_c2._compiled is False

    # 5. Doctor telemetry exposes Phase 3 backend capability fields
    doc = run_doctor()
    for field_name in (
        "available_backends",
        "onnx_providers",
        "torch_compile_available",
        "ffmpeg_available",
        "generative_available",
        "recommend_directml",
    ):
        assert field_name in doc
    assert "pytorch" in doc["available_backends"]


def test_generative_creativity_procedural_and_face_sky_protection(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from azzani_enhance.generative import GenerativeConfig, refine_generative_detail
    from azzani_enhance.image_io import LoadedImage
    from azzani_enhance.pipeline import enhance_loaded_image

    h, w = 128, 160
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Top [0:40]: ultra-smooth sky gradient
    img[:40, :, 0] = 0.42
    img[:40, :, 1] = 0.62
    img[:40, :, 2] = np.linspace(0.85, 0.95, 40, dtype=np.float32)[:, None]
    # Body background [40:]: mid-tone canvas
    img[40:, :] = [0.46, 0.44, 0.40]
    # Textured subject patch [48:112, 84:148]: crisp high-contrast stripes/foliage texture
    yy, xx = np.mgrid[0:64, 0:64]
    tex = 0.18 * np.sin(xx * 0.9) * np.cos(yy * 0.85) + 0.08 * np.sin((xx + yy) * 1.4)
    img[48:112, 84:148, 0] = np.clip(0.45 + tex, 0.05, 0.95)
    img[48:112, 84:148, 1] = np.clip(0.48 + tex * 0.9, 0.05, 0.95)
    img[48:112, 84:148, 2] = np.clip(0.36 + tex * 0.7, 0.05, 0.95)
    # Simulated face region [48:112, 14:68]: warm skin with mild contour texture
    img[48:112, 14:68, 0] = 0.78 + 0.05 * np.sin(np.mgrid[0:64, 0:54][1] * 0.5)
    img[48:112, 14:68, 1] = 0.60 + 0.04 * np.sin(np.mgrid[0:64, 0:54][0] * 0.5)
    img[48:112, 14:68, 2] = 0.50

    focus_mask = np.zeros((h, w), dtype=np.float32)
    focus_mask[44:116, 10:152] = 1.0
    face_mask = np.zeros((h, w), dtype=np.float32)
    face_mask[48:112, 14:68] = 1.0

    # Zero creativity is strictly identical no-op
    out_zero, used_zero = refine_generative_detail(
        img, img, GenerativeConfig(creativity=0.0), focus_mask=focus_mask, face_mask=face_mask
    )
    assert used_zero == "none"
    assert np.array_equal(out_zero, img)

    # Procedural refiner injects micro-texture on subject while guarding sky & face + preserving color balance
    out_gen, used_gen = refine_generative_detail(
        img,
        img,
        GenerativeConfig(creativity=0.55, backend="procedural"),
        focus_mask=focus_mask,
        face_mask=face_mask,
    )
    assert used_gen == "procedural"
    assert out_gen.shape == img.shape
    diff = np.abs(out_gen - img)
    sky_delta = float(np.mean(diff[:32, :]))
    face_delta = float(np.mean(diff[54:106, 20:62]))
    subj_delta = float(np.mean(diff[52:108, 88:144]))
    assert subj_delta > 0.005
    assert sky_delta < subj_delta * 0.5
    assert face_delta < subj_delta * 0.5
    assert float(np.max(np.abs(out_gen.mean(axis=(0, 1)) - img.mean(axis=(0, 1))))) < 0.015

    # Explicit diffusion request without diffusers raises clear install guidance
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, pkg=None: None if name in {"diffusers", "transformers"} else real_find_spec(name, pkg),
    )
    with pytest.raises(RuntimeError, match="generative"):
        refine_generative_detail(
            img,
            img,
            GenerativeConfig(creativity=0.4, backend="diffusion"),
        )

    # End-to-end pipeline wiring executes Stage D procedural refiner
    loaded = LoadedImage(
        rgb=img,
        alpha=None,
        source_bit_depth=8,
        source_format="PNG",
        exif_bytes=None,
        icc_profile=None,
        original_size=(w, h),
    )
    res_e2e = enhance_loaded_image(
        loaded,
        cfg=EnhancementConfig(
            preset="photometric",
            scale=2.0,
            face_restore="off",
            creativity=0.40,
            generative_backend="procedural",
            natural_grain=0.0,
            quiet=True,
        ),
    )
    assert res_e2e.output_size == (w * 2, h * 2)
    assert any(s.startswith("generative:procedural:") for s in res_e2e.stages_applied)


def test_video_scene_cut_optical_flow_stabilization_and_cli_e2e(tmp_path: Path) -> None:
    from azzani_enhance.cli import _build_config_from_args, build_parser, main as cli_main
    from azzani_enhance.video import VideoConfig, detect_scene_cut, enhance_video_file, temporal_stabilize_frame

    h, w = 96, 128
    chart_wide = generate_reference_chart(160, 96)

    frames_rgb: list[np.ndarray] = []
    for i in range(5):
        f = chart_wide[:, i * 4 : i * 4 + w].copy()
        if i == 3:
            # Add synthetic temporal frame flicker (exposure wobble) to verify Farneback EMA smoothing
            f = np.clip(f * 0.94 + 0.03, 0.0, 1.0).astype(np.float32)
        frames_rgb.append(f)
    # Frame 5: sudden scene cut to bright warm scene
    cut_frame = np.zeros((h, w, 3), dtype=np.float32)
    cut_frame[:, :, 0] = 0.92
    cut_frame[:, :, 1] = 0.68
    cut_frame[:, :, 2] = 0.12
    frames_rgb.append(cut_frame)

    is_cut_pan, score_pan = detect_scene_cut(frames_rgb[1], frames_rgb[2], threshold=0.30)
    is_cut_flick, score_flick = detect_scene_cut(frames_rgb[3], frames_rgb[4], threshold=0.30)
    is_cut_hard, score_hard = detect_scene_cut(frames_rgb[4], frames_rgb[5], threshold=0.30)
    assert is_cut_pan is False and score_pan < 0.30
    assert is_cut_flick is False and score_flick < 0.30
    assert is_cut_hard is True and score_hard >= 0.30

    # Optical-flow temporal stabilization pulls flickered frame 3 closer to clean panning structure
    clean_f3 = chart_wide[:, 3 * 4 : 3 * 4 + w]
    stab_f3 = temporal_stabilize_frame(
        frames_rgb[3], frames_rgb[2], frames_rgb[2], clean_f3, temporal_strength=0.55
    )
    assert float(np.mean(np.abs(stab_f3 - clean_f3))) < float(np.mean(np.abs(frames_rgb[3] - clean_f3)))

    # Write synthetic test video clip
    in_mp4 = tmp_path / "pan_clip.mp4"
    writer = cv2.VideoWriter(str(in_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
    assert writer.isOpened()
    for f in frames_rgb:
        u8_bgr = cv2.cvtColor(np.rint(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(u8_bgr)
    writer.release()

    out_mp4 = tmp_path / "pan_clip_enh.mp4"
    vres = enhance_video_file(
        in_mp4,
        out_mp4,
        cfg=EnhancementConfig(
            preset="photometric",
            scale=1.0,
            face_restore="off",
            natural_grain=0.0,
            quiet=True,
        ),
        video_cfg=VideoConfig(temporal_strength=0.45, scene_cut_threshold=0.30),
    )
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert vres.frames_processed == 6
    assert vres.scene_cuts_detected >= 1
    assert vres.input_size == (w, h)
    assert vres.output_size == (w, h)

    cap_check = cv2.VideoCapture(str(out_mp4))
    assert cap_check.isOpened()
    read_count = 0
    while True:
        ok, _ = cap_check.read()
        if not ok:
            break
        read_count += 1
    cap_check.release()
    assert read_count == 6

    # CLI Phase 3 argument parsing + --video / --max-frames end-to-end execution
    parser = build_parser()
    ns = parser.parse_args(
        [
            str(in_mp4),
            "--backend",
            "directml",
            "--creativity",
            "0.45",
            "--gen-backend",
            "procedural",
            "--video",
            "--temporal-strength",
            "0.5",
            "--scene-cut",
            "0.25",
            "--max-frames",
            "4",
            "--no-audio",
        ]
    )
    cfg_cli = _build_config_from_args(ns)
    assert cfg_cli.backend == "directml"
    assert cfg_cli.creativity == pytest.approx(0.45)
    assert cfg_cli.generative_backend == "procedural"
    assert ns.video is True
    assert ns.max_frames == 4
    assert ns.no_audio is True

    cli_out_mp4 = tmp_path / "cli_out.mp4"
    rc = cli_main(
        [
            str(in_mp4),
            "-o",
            str(cli_out_mp4),
            "--preset",
            "photometric",
            "--scale",
            "1",
            "--face-restore",
            "off",
            "--max-frames",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    assert cli_out_mp4.is_file() and cli_out_mp4.stat().st_size > 0


def test_profiles_toml_discovery_merge_and_cli_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azzani_enhance.cli import _build_config_from_args, build_parser, main as cli_main
    from azzani_enhance.profiles import (
        apply_profile_to_config,
        list_profiles,
        write_sample_profiles,
    )

    sample_path = tmp_path / "user_profiles.toml"
    written = write_sample_profiles(sample_path)
    assert written == sample_path.resolve() and written.is_file()

    # Custom workspace azzani.toml overriding & extending built-ins
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / "azzani.toml").write_text(
        '[profiles.custom-arch]\n'
        'description = "Arch render"\n'
        'preset = "pro"\n'
        'scale = 3.0\n'
        'tta = 4\n'
        'halo_guard = 0.92\n'
        '[profiles.bad-field]\n'
        'nonexistent_flag = 123\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("AZZANI_PROFILES", str(sample_path))

    profs = {name: desc for name, desc, _src in list_profiles()}
    assert {"photography", "portrait-studio", "real-estate-hdr", "custom-arch"}.issubset(profs.keys())

    cfg_arch = apply_profile_to_config("custom-arch")
    assert cfg_arch.preset == "pro"
    assert cfg_arch.scale == pytest.approx(3.0)
    assert cfg_arch.tta == 4
    assert cfg_arch.halo_guard == pytest.approx(0.92)

    with pytest.raises(KeyError, match="Unknown profile"):
        apply_profile_to_config("missing-profile-xyz")

    with pytest.raises(ValueError, match="nonexistent_flag"):
        apply_profile_to_config("bad-field")

    # CLI flag precedence: explicit CLI --tta 8 overrides profile while keeping profile defaults
    parser = build_parser()
    ns = parser.parse_args(
        ["photo.jpg", "--profile", "portrait-studio", "--tta", "8", "--profiles-file", str(sample_path)]
    )
    cfg_cli = _build_config_from_args(ns)
    assert cfg_cli.preset == "portrait"
    assert cfg_cli.face_restore == "on"
    assert cfg_cli.identity_guard == pytest.approx(0.75)
    assert cfg_cli.tta == 8

    # CLI commands --init-profiles and --list-profiles run cleanly
    init_target = tmp_path / "init_out.toml"
    assert cli_main(["--init-profiles", str(init_target), "--json"]) == 0
    assert init_target.is_file()
    assert cli_main(["--list-profiles", "--json"]) == 0


def test_headless_api_server_auth_async_queue_and_hmac_webhook(tmp_path: Path) -> None:
    import base64
    import hashlib
    import hmac
    import io
    import json
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from azzani_enhance.automation import WatchFolderRunner
    from azzani_enhance.web_ui import create_web_server

    captured_hooks: list[dict] = []
    hook_event = threading.Event()

    class _HookHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            captured_hooks.append(
                {
                    "event": self.headers.get("X-Azzani-Event"),
                    "signature": self.headers.get("X-Azzani-Signature"),
                    "raw": raw,
                    "json": json.loads(raw.decode("utf-8")),
                }
            )
            hook_event.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    hook_srv = ThreadingHTTPServer(("127.0.0.1", 0), _HookHandler)
    hook_port = hook_srv.server_address[1]
    hook_thread = threading.Thread(target=hook_srv.serve_forever, daemon=True)
    hook_thread.start()

    api_srv, api_state = create_web_server("127.0.0.1", 0, api_key="secret-token-42")
    api_port = api_srv.server_address[1]
    api_thread = threading.Thread(target=api_srv.serve_forever, daemon=True)
    api_thread.start()

    try:
        # 1. /healthz stays open without API key (Docker healthcheck)
        with urllib.request.urlopen(f"http://127.0.0.1:{api_port}/healthz", timeout=5) as r:
            assert r.status == 200
            h_data = json.loads(r.read().decode("utf-8"))
            assert h_data["status"] == "ok"

        # 2. Protected /api/profiles rejects missing key with 401 and accepts X-API-Key
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{api_port}/api/profiles", timeout=5)
        assert exc_info.value.code == 401

        req_prof = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/api/profiles",
            headers={"X-API-Key": "secret-token-42"},
        )
        with urllib.request.urlopen(req_prof, timeout=5) as r:
            p_data = json.loads(r.read().decode("utf-8"))
            assert any(p["name"] == "photography" for p in p_data["profiles"])

        # 3. Enqueue async enhancement job with Bearer token & HMAC-signed webhook
        img_arr = (generate_reference_chart(64, 64) * 255.0).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        job_payload = json.dumps(
            {
                "filename": "nas_photo.png",
                "data_base64": b64_png,
                "profile": "low-light-rescue",
                "preset": "photometric",
                "scale": 1.0,
                "face_restore": "off",
                "webhook_url": f"http://127.0.0.1:{hook_port}/cb",
                "webhook_secret": "hook-secret",
            }
        ).encode("utf-8")
        req_async = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/api/enhance/async",
            data=job_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-token-42",
            },
            method="POST",
        )
        with urllib.request.urlopen(req_async, timeout=5) as r:
            assert r.status == 202
            queued = json.loads(r.read().decode("utf-8"))
            job_id = queued["job_id"]

        # Poll /api/jobs/<job_id> until completed
        final_job = None
        for _ in range(80):
            req_st = urllib.request.Request(
                f"http://127.0.0.1:{api_port}/api/jobs/{job_id}",
                headers={"X-API-Key": "secret-token-42"},
            )
            with urllib.request.urlopen(req_st, timeout=5) as r:
                final_job = json.loads(r.read().decode("utf-8"))
            if final_job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert final_job is not None and final_job["status"] == "completed"
        assert final_job["summary"]["output_size"] == [64, 64]
        assert hook_event.wait(timeout=5.0)
        hook_rec = captured_hooks[0]
        assert hook_rec["event"] == "enhancement.completed"
        expected_sig = "sha256=" + hmac.new(b"hook-secret", hook_rec["raw"], hashlib.sha256).hexdigest()
        assert hook_rec["signature"] == expected_sig
        assert hook_rec["json"]["data"]["job_id"] == job_id

        # 4. WatchFolderRunner webhook dispatch on hot-folder process_once
        hook_event.clear()
        inbox = tmp_path / "inbox"
        outbox = tmp_path / "outbox"
        inbox.mkdir()
        Image.fromarray(img_arr).save(inbox / "drop.png")
        runner = WatchFolderRunner(
            inbox,
            outbox,
            cfg=EnhancementConfig(preset="photometric", scale=1.0, face_restore="off", quiet=True),
            webhook_url=f"http://127.0.0.1:{hook_port}/watch",
            webhook_secret="hook-secret",
        )
        try:
            events = runner.process_once()
            assert len(events) == 1 and events[0].status == "enhanced"
            assert hook_event.wait(timeout=5.0)
            assert captured_hooks[-1]["event"] == "watch.enhanced"
        finally:
            runner.stop()
    finally:
        api_srv.shutdown()
        api_srv.server_close()
        api_state.cleanup()
        hook_srv.shutdown()
        hook_srv.server_close()


def test_benchmark_suite_metrics_leaderboard_and_deployment_artifacts(tmp_path: Path) -> None:
    import subprocess

    from azzani_enhance.benchmark import (
        compute_gmsd,
        compute_perceptual_score,
        format_leaderboard_markdown,
        run_benchmark_suite,
    )
    from azzani_enhance.cli import main as cli_main

    ref = generate_reference_chart(128, 128)
    assert compute_gmsd(ref, ref) == pytest.approx(0.0, abs=1e-6)
    blurred = cv2.GaussianBlur(ref, (0, 0), sigmaX=1.8)
    assert compute_gmsd(ref, blurred) > 0.02
    assert compute_perceptual_score(ref, ref) > compute_perceptual_score(blurred, ref)

    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    Image.fromarray((ref * 255.0).astype(np.uint8)).save(ds_dir / "sample_1.png")

    suite = run_benchmark_suite(
        presets=("bicubic", "lanczos", "photometric"),
        dataset_dir=ds_dir,
        scale=2,
        max_images=1,
    )
    assert suite["images_evaluated"] == 1
    assert len(suite["results"]) == 3
    for row in suite["results"]:
        assert {"psnr_db", "ssim", "gmsd", "perceptual_score", "sharpness", "sec_per_mp"}.issubset(row.keys())
    md = format_leaderboard_markdown(suite)
    assert "| `photometric` |" in md and "GMSD" in md

    assert cli_main(["--benchmark-suite", "--suite-presets", "bicubic,lanczos,photometric", "--json"]) == 0

    # Verify zero-install scripts & Docker/NAS assets are present and syntactically valid
    root = Path(__file__).resolve().parent.parent
    install_sh = root / "scripts" / "install.sh"
    install_ps1 = root / "scripts" / "install.ps1"
    dockerfile = root / "Dockerfile"
    compose = root / "docker-compose.yml"
    for p in (install_sh, install_ps1, dockerfile, compose):
        assert p.is_file(), f"Missing deployment artifact: {p}"

    proc = subprocess.run(["bash", "-n", str(install_sh)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "/healthz" in dockerfile.read_text(encoding="utf-8")
    assert '"spandrel>=0.4.0"' in dockerfile.read_text(encoding="utf-8")
    compose_txt = compose.read_text(encoding="utf-8")
    assert "/data/inbox" in compose_txt and "--server" in compose_txt


def test_quiet_browser_launcher_low_vram_fallback_and_wb_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil
    import subprocess
    import webbrowser

    import spandrel

    from azzani_enhance.inference import RuntimeConfig, SpandrelEngine, _resolve_best_cuda_or_cpu
    from azzani_enhance.web_ui import _open_browser_quietly

    # 1. Browser launcher detaches with DEVNULL on Linux so Snap/Chromium zygote stderr never hits CLI.
    popen_calls: list[dict] = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append({"cmd": cmd, **kwargs})
        return None

    monkeypatch.setattr("azzani_enhance.web_ui.sys.platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None)
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    _open_browser_quietly("http://127.0.0.1:8765/")
    assert len(popen_calls) == 1
    assert popen_calls[0]["cmd"] == ["/usr/bin/xdg-open", "http://127.0.0.1:8765/"]
    assert popen_calls[0]["stderr"] == subprocess.DEVNULL
    assert popen_calls[0]["start_new_session"] is True

    # Fallback when xdg-open is absent uses webbrowser.open
    opened_urls: list[str] = []
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(webbrowser, "open", lambda u: opened_urls.append(u) or True)
    _open_browser_quietly("http://127.0.0.1:8765/")
    assert opened_urls == ["http://127.0.0.1:8765/"]

    # 2. Tight VRAM on device='auto' safely falls back to CPU instead of crashing on load
    assert _resolve_best_cuda_or_cpu("auto", free_gb=0.04, min_bytes=120 * 1024 * 1024) is True
    assert _resolve_best_cuda_or_cpu("cuda", free_gb=0.04, min_bytes=120 * 1024 * 1024) is False

    class _StubArch:
        name = "Stub2x"

    class _StubLoader:
        def __init__(self, device=None):
            pass

        def load_from_file(self, _path):
            m = torch.nn.Upsample(scale_factor=2, mode="nearest")
            return spandrel.ImageModelDescriptor(
                model=m,
                state_dict={},
                architecture=_StubArch(),
                purpose="SR",
                tags=[],
                supports_half=True,
                supports_bfloat16=False,
                scale=2,
                input_channels=3,
                output_channels=3,
                size_requirements=spandrel.SizeRequirements(minimum=16, multiple_of=1, square=False),
                tiling=spandrel.ModelTiling.SUPPORTED,
            )

    monkeypatch.setattr("azzani_enhance.inference.ModelLoader", _StubLoader)
    monkeypatch.setattr("azzani_enhance.inference.resolve_device", lambda _req: torch.device("cuda:0"))
    monkeypatch.setattr("azzani_enhance.inference._get_free_device_gb", lambda _dev: 0.01)
    monkeypatch.setattr("azzani_enhance.inference.configure_torch_backends", lambda _dev: None)
    monkeypatch.setattr("azzani_enhance.inference.empty_device_cache", lambda _dev: None)

    stub_pth = tmp_path / "low_vram_stub.pth"
    stub_pth.write_bytes(b"stub")
    eng = SpandrelEngine(stub_pth, runtime=RuntimeConfig(device="auto", tta=1))
    assert eng.device.type == "cpu"
    assert "CPU fallback" in eng.precision_note

    # Tile OOM fallback at min_tile falls back cleanly to CPU with >= 256px tiles (not tiny 64px)
    eng_oom = SpandrelEngine(stub_pth, runtime=RuntimeConfig(device="auto", tta=1, tile_size=64))
    eng_oom.device = torch.device("cuda:0")
    real_run_tiled = eng_oom._run_tiled
    seen_tiles: list[tuple[str, int]] = []

    def _mock_run_tiled(rgb, tile_size, overlap, progress_cb=None):
        seen_tiles.append((eng_oom.device.type, tile_size))
        if eng_oom.device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("mock OOM")
        return real_run_tiled(rgb, tile_size=tile_size, overlap=overlap, progress_cb=progress_cb)

    monkeypatch.setattr(eng_oom, "_run_tiled", _mock_run_tiled)
    sample_rgb = np.zeros((32, 32, 3), dtype=np.float32)
    out_fb, used_tile = eng_oom.enhance(sample_rgb)
    assert out_fb.shape == (64, 64, 3)
    assert eng_oom.device.type == "cpu"
    assert used_tile >= 256
    assert seen_tiles == [("cuda", 64), ("cpu", 256)]

    # EnginePool promotes stale CPU-fallback entries once free GPU VRAM recovers
    from azzani_enhance.inference import EnginePool

    pool = EnginePool(max_models=2)
    fake_free = {"gb": 0.01}
    monkeypatch.setattr("azzani_enhance.inference._get_free_device_gb", lambda _dev: fake_free["gb"])
    created: list[object] = []

    class _PoolEng:
        def __init__(self):
            self.device = torch.device("cpu" if fake_free["gb"] < 1.0 else "cuda:0")
            self.runtime = RuntimeConfig()
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "azzani_enhance.inference.create_inference_engine",
        lambda *_a, **_kw: _PoolEng(),
    )
    e1 = pool.acquire(stub_pth, runtime=RuntimeConfig(device="auto"))
    assert e1.device.type == "cpu"
    assert pool.acquire(stub_pth, runtime=RuntimeConfig(device="auto")) is e1
    fake_free["gb"] = 2.5
    e2 = pool.acquire(stub_pth, runtime=RuntimeConfig(device="auto"))
    assert e2 is not e1 and e2.device.type == "cuda"
    assert e1.closed is True
    pool.clear()

    # 3. White balance preserves colorful teal scene when neutral white highlights already balance
    scene = np.zeros((100, 100, 3), dtype=np.float32)
    scene[:] = [0.12, 0.56, 0.62]  # Vivid teal aquarium water background
    scene[:20, :30] = [0.82, 0.82, 0.82]  # Truly neutral gray/white highlight patch (> 1%)
    wb_out = auto_white_balance(scene, strength=0.45)
    assert float(np.max(np.abs(wb_out - scene))) < 0.02





def test_minus_one_one_models_bypass_spandrel_output_clamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GFPGAN-style [-1, 1] models must not lose their negative half to Spandrel's [0, 1] output clamp."""
    import spandrel

    from azzani_enhance.inference import RuntimeConfig, SpandrelEngine

    class _IdentityWithAux(torch.nn.Module):
        def forward(self, x: torch.Tensor):  # GFPGAN returns (image, intermediate_rgbs)
            return x, []

    class _StubArch:
        name = "StubFace"

    class _StubLoader:
        def __init__(self, device=None):
            pass

        def load_from_file(self, _path):
            return spandrel.ImageModelDescriptor(
                model=_IdentityWithAux(),
                state_dict={},
                architecture=_StubArch(),
                purpose="FaceSR",
                tags=[],
                supports_half=False,
                supports_bfloat16=False,
                scale=1,
                input_channels=3,
                output_channels=3,
                size_requirements=spandrel.SizeRequirements(minimum=16),
                call_fn=lambda model, image: model(image)[0],
            )

    monkeypatch.setattr("azzani_enhance.inference.ModelLoader", _StubLoader)
    stub = tmp_path / "face_stub.pth"
    stub.write_bytes(b"stub")
    eng = SpandrelEngine(stub, runtime=RuntimeConfig(device="cpu", tile_size=0), value_range="-1_1")

    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 32, axis=0), 3, axis=2)
    out, _ = eng.enhance(img)
    # Before the fix every value below mid-grey collapsed to exactly 0.5 (washed-out, bleached faces).
    assert float(out.min()) < 0.02
    assert np.allclose(out, img, atol=1e-5)


def test_neural_pass_sanity_guard_rejects_diverged_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from azzani_enhance import pipeline as pipeline_mod
    from azzani_enhance.image_io import LoadedImage
    from azzani_enhance.pipeline import neural_pass_is_sane, prepare_enhancement_layers

    chart = generate_reference_chart(96, 96)
    up = cv2.resize(chart, (192, 192), interpolation=cv2.INTER_CUBIC)
    assert neural_pass_is_sane(chart, chart)[0]
    assert neural_pass_is_sane(chart, up)[0]  # SR output at a different size compares at low resolution
    rng = np.random.default_rng(3)
    garbage = rng.uniform(0.0, 1.0, size=up.shape).astype(np.float32)
    assert not neural_pass_is_sane(chart, garbage)[0]
    nan_out = up.copy()
    nan_out[10:20, 10:20] = np.nan
    assert not neural_pass_is_sane(chart, nan_out)[0]
    # Exposure passes legitimately brighten: only structural agreement is required.
    brighter = np.clip(chart * 1.6 + 0.1, 0.0, 1.0)
    assert not neural_pass_is_sane(chart, brighter)[0]
    assert neural_pass_is_sane(chart, brighter, tone_changing=True)[0]

    # A primary model that blows up (e.g. NAFNet on a noisy thumbnail) must be discarded, not returned.
    def _blown_up_step(rgb, model_key, cfg, label, status_cb=None, engine_pool=None):
        h, w = rgb.shape[:2]
        return rng.uniform(0.0, 1.0, size=(h * 2, w * 2, 3)).astype(np.float32), 2, "cpu", "fp32", 0

    monkeypatch.setattr(pipeline_mod, "_run_model_step", _blown_up_step)
    loaded = LoadedImage(chart, None, 8, "PNG", None, None, (96, 96))
    layers = prepare_enhancement_layers(
        loaded, EnhancementConfig(preset="fast", scale=2.0, face_restore="off", quiet=True)
    )
    assert "rejected:photo-fast" in layers.neural_stages
    assert layers.primary_model is None
    bicubic = cv2.resize(chart, (192, 192), interpolation=cv2.INTER_LANCZOS4)
    assert compute_psnr(np.clip(bicubic, 0, 1), np.asarray(layers.sr_raw_rgb)) > 30.0


def test_analyzer_no_false_deblur_on_thumbnails_and_low_key_exposure() -> None:
    rng = np.random.default_rng(11)
    # Tiny noisy web thumbnail with strongly oriented content: FFT anisotropy is meaningless at this size.
    thumb = np.clip(
        0.45 + 0.3 * np.sin(np.linspace(0, 40, 128, dtype=np.float32))[None, :, None]
        + rng.normal(0.0, 0.05, (128, 128, 3)).astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    d_thumb = analyze_image(thumb)
    assert d_thumb.motion_blur_score == 0.0
    assert d_thumb.recommended_deblur is False

    # Low-key still life: dark overall but with genuine bright highlights -> exposure is intentional.
    low_key = np.full((256, 256, 3), 0.05, dtype=np.float32)
    low_key[100:140, 100:140] = [0.95, 0.90, 0.80]
    assert analyze_image(low_key).recommended_exposure_boost == 0.0

    # Whole histogram compressed into the shadows -> genuine underexposure.
    under = np.clip(generate_reference_chart(256, 256) * 0.22, 0.0, 1.0)
    assert analyze_image(under).recommended_exposure_boost > 0.25


def test_tone_config_preserves_scene_white_balance_and_tonality() -> None:
    from azzani_enhance.analyzer import ImageDiagnostics
    from azzani_enhance.pipeline import _resolve_tone_config

    # Strong "cast" (e.g. teal aquarium, golden hour) with crushed blacks in a low-key scene.
    diag = analyze_image(generate_reference_chart(128, 128))
    diag = ImageDiagnostics(
        **{
            **diag.to_dict(),
            "color_cast_strength": 0.95,
            "shadow_clip_pct": 22.0,
            "dynamic_range": 90.0,
            "recommended_exposure_boost": 0.0,
        }
    )
    auto_cfg = _resolve_tone_config(diag, EnhancementConfig(), "photo-pro")
    assert auto_cfg.white_balance_strength == 0.0
    assert auto_cfg.tone_strength == 0.0
    assert auto_cfg.detail_strength <= 0.05  # detail-rich SR models need no extra sharpening

    on_cfg = _resolve_tone_config(diag, EnhancementConfig(white_balance="on"), "photo-pro")
    assert on_cfg.white_balance_strength == pytest.approx(0.50)

    # Genuinely flat/hazy histogram still gets repaired on neural presets.
    flat = ImageDiagnostics(**{**diag.to_dict(), "dynamic_range": 25.0})
    assert _resolve_tone_config(flat, EnhancementConfig(), "photo-pro").tone_strength > 0.3


def test_critic_window_scales_with_upscale_and_protects_faces() -> None:
    from azzani_enhance.color_tone import critic_window_for_scale

    assert critic_window_for_scale(1.0) == 5
    assert critic_window_for_scale(2.0) == 5
    assert critic_window_for_scale(4.0) == 9

    # Ideal crisp 4x SR edge vs the bicubic reference ramp of a slightly soft (real-camera) source edge.
    small = cv2.GaussianBlur(
        np.concatenate([np.full((32, 16, 3), 0.2, np.float32), np.full((32, 16, 3), 0.8, np.float32)], axis=1),
        (0, 0),
        1.0,
    )
    ref = cv2.resize(small, (128, 128), interpolation=cv2.INTER_CUBIC)
    sharp = np.full((128, 128, 3), 0.2, dtype=np.float32)
    sharp[:, 64:] = 0.8

    fixed_win, _ = self_critic_refinement(sharp, ref, clamp_strength=1.0, scale_hint=1.0)
    scaled_win, _ = self_critic_refinement(sharp, ref, clamp_strength=1.0, scale_hint=4.0)
    fixed_change = float(np.mean(np.abs(fixed_win - sharp)))
    # A 5x5 window misreads legitimate SR sharpening as overshoot and blurs it back toward bicubic.
    assert fixed_change > 1e-3
    assert float(np.mean(np.abs(scaled_win - sharp))) < 0.5 * fixed_change

    # Genuine ringing overshoot beyond the plateau is still pulled back with the wider window.
    halo = sharp.copy()
    halo[:, 66:70] = 0.98
    halo[:, 58:62] = 0.02
    dehaloed, _ = self_critic_refinement(halo, ref, clamp_strength=1.0, scale_hint=4.0)
    assert float(dehaloed[:, 66:70].max()) < 0.93
    assert float(dehaloed[:, 58:62].min()) > 0.07

    protect = np.zeros((128, 128, 1), dtype=np.float32)
    protect[:, 48:80] = 1.0
    prot_out, _ = self_critic_refinement(sharp, ref, clamp_strength=1.0, scale_hint=1.0, protect_mask=protect)
    assert np.allclose(prot_out[:, 48:80], sharp[:, 48:80], atol=1e-6)


def test_face_selection_and_tone_match_keep_restored_detail() -> None:
    from azzani_enhance.analyzer import ImageDiagnostics
    from azzani_enhance.faces import DetectedFace, _match_face_tone
    from azzani_enhance.pipeline import _select_faces_to_restore

    diag = analyze_image(generate_reference_chart(128, 128))
    clean = ImageDiagnostics(**{**diag.to_dict(), "blur_severity": 0.05, "noise_severity": 0.0, "jpeg_blockiness": 0.0})
    pts = np.zeros((5, 2), dtype=np.float32)
    small = DetectedFace(bbox=(0.0, 0.0, 360.0, 400.0), confidence=0.9, landmarks_5pt=pts)  # 90px in a 4x source
    big = DetectedFace(bbox=(0.0, 0.0, 1600.0, 1800.0), confidence=0.9, landmarks_5pt=pts)  # 400px source face
    assert _select_faces_to_restore([small, big], clean, 4.0) == [small]
    degraded = ImageDiagnostics(**{**clean.to_dict(), "jpeg_blockiness": 0.8})
    assert _select_faces_to_restore([small, big], degraded, 4.0) == [small, big]

    # Restored crisp detail must not be flattened toward the blurry original's contrast.
    mask = _build_face_feather_mask(512)
    rng = np.random.default_rng(5)
    orig = np.full((512, 512, 3), 0.45, dtype=np.float32)
    restored = np.clip(0.45 + rng.normal(0.0, 0.12, (512, 512, 1)).astype(np.float32), 0.0, 1.0).repeat(3, axis=2)
    fused = _match_face_tone(restored, orig, mask)
    core = (slice(200, 312), slice(200, 312))
    assert float(np.std(fused[core])) > 0.85 * float(np.std(restored[core]))

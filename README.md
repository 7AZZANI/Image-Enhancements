# AZZANI Image Enhancements

> **Open-Source AI Image & Video Upscaler, Restorer, Deblur & Face Enhancer (Auto-GPU, MIT)**

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

**AZZANI Image Enhancements** (`azzani-enhance`) is an open-source, MIT-licensed hybrid **AI + 32-bit float photometric** image restoration, super-resolution, face recovery, and detail enhancement engine built on top of `spandrel`, PyTorch, OpenCV, and SciPy.

Unlike naive upscaler wrappers that break on modern Python, crash with CUDA OOM on 4 GB GPUs, turn Turing GTX cards into NaN/black output under FP16, strip EXIF/ICC profiles, or ruin faces and transparent borders, **AZZANI Image Enhancements** solves every link in the restoration pipeline end-to-end.

---

## Key Features

- **True Zero-Friction GPU Auto-Integration (`--setup`, `--fix-gpu`, `--doctor`)**
  Automatically detects NVIDIA CUDA drivers (`cu118`–`cu128`, including Blackwell RTX 50xx & Turing GTX 16xx), AMD ROCm (`rocm6.2`), Apple Silicon Metal (`mps`), Intel Arc (`xpu`), multi-GPU VRAM ranking (picks the GPU with the most free VRAM), and `/tmp` tmpfs disk-quota limits (`TMPDIR=/var/tmp`). Also repairs accidental `+cpu` PyTorch installs on GPU hosts automatically.
- **Persistent GPU Precision Cache & Warm `EnginePool`**
  Runs an empirical FP16-vs-FP32 self-check once per `(GPU, driver, model)` (`~/.cache/azzani-enhance/hardware_profile.json`) to block Turing `sm_75` NaN/black output, then skips redundant warmup probes on future runs while keeping weights warm in VRAM across batches, watch-folder runs, and Web UI uploads.
- **Friendly Automation: Browser Studio (`--ui`) & Hot-Folder Watcher (`--watch`)**
  - **Local Browser Studio (`--ui`):** Zero-dependency local web interface with live GPU card, drag-and-drop upload, Before/After split slider, toggleable **4× Synced Pixel-Peep Loupe**, **RGB + Luminance Histogram**, **Quality Confidence** score, and **Live <50ms Non-Destructive Studio Retuning** (adjust Tone, Detail, Color, White Balance, CA Suppression, Halo Guard, Face Fidelity, Bokeh Guard, & Grain without re-running PyTorch).
  - **Hot-Folder Automation (`--watch`):** Drop images into `./azzani_inbox` -> enhanced automatically with write-completion checks and idempotent manifest tracking into `./azzani_enhanced`.
  - **1-Click Launchers:** Run `./azzani.sh` (Linux/macOS) or double-click / drag-and-drop images onto `azzani.bat` (Windows).
- **Closed-Loop Post-SR Artifact Self-Critic & Focus/Bokeh Guard**
  Measures post-SR edge overshoot / ringing (`halo_ringing_index`) and LAB color drift (`color_drift_delta_e`) to compute a `quality_confidence` score, clamping synthetic overshoot halos relative to clean downscaled references (`--halo-guard`). Automatically builds a soft Focus/DoF mask (`--no-bokeh-guard` to disable) so smooth camera bokeh stays creamy while sharp subjects receive full micro-detail.
- **No-Reference Degradation Analyzer (`--inspect`)**
  Measures multi-scale noise ($\hat{\sigma}$ at 1× and 0.5× for sensor ISO noise and coarse scan grain), 8×8 JPEG DCT blockiness, **2D FFT directional motion-blur anisotropy** (`motion_blur_score`, `motion_blur_angle_deg`), noise-compensated sharpness, dynamic range/clipping, neutral-weighted CIE LAB color casts, and photo vs. illustration heuristics before choosing operations.
- **SOTA Permissive-License AI Restoration & Super-Resolution (`spandrel`)**
  Supports `RealPLKSR`, `DAT` / `DAT2`, `SPAN`, `HAT`, `NAFNet` (`deblur-1x`), `SwinIR`, `SRVGGNetCompact`, and `GFPGAN` natively via `spandrel` (MIT) using verified weights with cryptographic SHA-256 verification and automatic lazy caching. Also accepts any local `.safetensors` or `.pth` checkpoint via `--model`.
- **Hardware-Hardened & Out-of-Core 500+ MP Streaming Engine**
  Raised-cosine (Hann) overlap tiling with automatic disk-backed `ManagedMemmap` streaming above `--memmap-mp 64` (real-disk `/var/tmp` backing outside `tmpfs` `/tmp`), zero-dependency Classic TIFF / 64-bit BigTIFF streaming export, automatic VRAM OOM tile backoff across CUDA/MPS/XPU, and D4 dihedral Test-Time Augmentation (`--tta 1|2|4|8`).
- **MIT YuNet + GFPGANv1.4 + Apache-2.0 SFace Identity Lock**
  Zero dependency on `basicsr` or `facexlib`. Uses OpenCV's MIT-licensed `YuNet` 5-landmark detector, exact Umeyama similarity alignment into FFHQ 512×512 canonical pose, `GFPGANv1.4` restoration, **4-level Burt–Adelson Laplacian pyramid illumination fusion** for zero hairline halos, and closed-loop **SFace** cosine similarity clamping (`--identity-guard`) to guarantee faithful identity preservation.
- **Float32 Photometric, Lens CA & Color-Managed Finishing**
  Lateral Chromatic Aberration fringe suppression (`--ca-suppress`), Wavelet/CIE LAB low-frequency color anchoring, highlight-safe white balance, edge-aware Guided Retinex tone & milky-haze recovery, bokeh-guarded detail sharpening, TPDF anti-banding dithering, 16-bit PNG/TIFF I/O, EXIF orientation normalization, ICC-to-sRGB conversion, Voronoi alpha-edge bleeding, and optional Camera RAW input (`.dng`/`.cr2`/`.nef`/`.arw` via `pip install -e ".[raw]"`).

---

## One-Click Auto-GPU Setup & Launch

Just clone and run — `enhance.py`, `azzani.sh`, and `azzani.bat` auto-detect your GPU hardware, create `.venv`, install the exact matched PyTorch GPU wheel (`cu128`/`cu126`/`cu124`/`cu121`/`cu118`/`rocm6.2`/Apple `mps`), and verify CUDA/MPS/XPU automatically:

```bash
# Quick 1-Line Install (Linux / macOS):
curl -fsSL https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.sh | bash

# Or clone & auto-setup:
git clone https://github.com/7AZZANI/Image-Enhancements.git
cd Image-Enhancements

# Option A: One-click launcher (Linux / macOS) — auto-installs GPU PyTorch on 1st run
./azzani.sh

# Option B: Pure Python zero-config auto-setup & launch (Linux / macOS / Windows)
python3 enhance.py --setup

# Option C: Windows 1-Line PowerShell Installer or double-click azzani.bat
# powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.ps1 | iex"
```

<details>
<summary>Manual venv installation (optional)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # or your CUDA/ROCm wheel
pip install -e .
# Optional Camera RAW (.dng/.cr2/.nef/.arw) ingestion via LibRaw (rawpy):
pip install -e ".[raw]"
```

> **Note on `/tmp` tmpfs limits:** AZZANI's `--setup` automatically redirects `TMPDIR=/var/tmp` when `/tmp` has < 5 GB free. If installing manually with pip, pass `TMPDIR=/var/tmp`.

</details>

---

## Quick Start

```bash
# 1. Interactive Hub or Local GPU Browser Studio (drag & drop + Before/After split slider)
python3 enhance.py --ui

# 2. Hot-folder automation (continuously enhances everything dropped into ./azzani_inbox)
azzani-enhance --watch ./azzani_inbox -o ./azzani_enhanced

# 3. Smart automatic enhancement (adapts to image diagnostics AND your GPU tier)
python3 enhance.py photo.jpg --comparison

# 4. Maximum photorealistic quality (DAT2 transformer + 4x TTA self-ensemble)
azzani-enhance old_photo.jpg --preset ultra --scale 2 --tta 4 --comparison

# 5. Fast 2x upscale & web JPEG cleanup
azzani-enhance web_image.jpg --preset fast --scale 2

# 6. 1x restoration only (de-JPEG / denoise + tone/detail recovery, original dimensions)
azzani-enhance compressed.jpg --preset restore --scale 1

# 7. Pure offline float32 photometric enhancement (no AI model downloads)
azzani-enhance photo.jpg --preset photometric --scale 1

# 8. Batch process an entire folder recursively into 16-bit PNGs (warm GPU pool)
azzani-enhance ./raw_photos -o ./enhanced_photos -r --ext png --bit-depth 16

# 9. Tune Chromatic Aberration removal, Halo Self-Critic, or Bokeh DoF protection
azzani-enhance lens_photo.jpg --ca-suppress 0.65 --halo-guard 0.85
azzani-enhance flat_doc.jpg --no-bokeh-guard

# 10. Inspect diagnostics, verify GPU health, auto-repair CPU-only PyTorch, or benchmark
azzani-enhance photo.jpg --inspect
azzani-enhance --doctor
azzani-enhance --fix-gpu
azzani-enhance --list-models
azzani-enhance --benchmark --preset fast --tta 4
```

---

## Presets & Built-in Models

All default models use permissive licenses (`CC-BY-4.0`, `Apache-2.0`, `MIT`) and are downloaded on demand with SHA-256 verification (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

| Preset | Default Model | Scale | Architecture | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `auto` | Adaptive | 1×–4× | Smart routing | Best all-around automatic restoration & upscaling |
| `fast` | `2xNomosUni_span_multijpg_ldl` | 2× | `SPAN` | High-speed 2× web/photo cleanup & upscaling |
| `pro` | `4xNomosWebPhoto_RealPLKSR` | 4× | `RealPLKSR` | Crisp real-world photo deblurring, de-JPEG, & SR |
| `ultra` | `4xRealWebPhoto_v4_dat2` | 4× | `DAT2` | Maximum-fidelity photorealistic transformer SR |
| `hq` | `4xNomos2_hq_dat2` | 4× | `DAT2` | Clean, high-quality studio photography upscaling |
| `portrait` | `4xFaceUpDAT` + `GFPGANv1.4` | 4× | `DAT` + `GFPGAN` | Portraits and degraded faces |
| `anime` / `anime-2x` | `4xHFA2k_ludvae_realplksr_dysample` / `2xHFA2k_LUDVAE_SPAN` | 4× / 2× | `RealPLKSR` / `SPAN` | Anime, digital illustrations, and crisp line art |
| `deblur` | `NAFNet-GoPro-width32` | 1× | `NAFNet` | Camera shake & directional motion blur rescue |
| `restore` | `1xDeJPG_realplksr_otf` | 1× | `RealPLKSR` | Compression/noise repair at native 1× resolution |
| `photometric` | *(None — pure math)* | 1× | Float32 DSP | Instant zero-AI color, exposure, haze, & detail finish |

---

## Python API

```python
from azzani_enhance import (
    EnhancementConfig,
    compose_enhancement_result,
    enhance_file,
    load_image,
    prepare_enhancement_layers,
)

result = enhance_file(
    "input.jpg",
    "output.png",
    cfg=EnhancementConfig(
        preset="pro",
        scale=2.0,
        tta=4,
        face_restore="auto",
        bit_depth=16,
    ),
    comparison_path="before_after.png",
)
print(result.quality_confidence, result.summary_dict())

# Or cache neural layers once for <50ms non-destructive photometric & face retuning:
layers = prepare_enhancement_layers(load_image("input.jpg"), EnhancementConfig(preset="pro"))
tuned = compose_enhancement_result(layers, EnhancementConfig(tone_strength=0.45, halo_guard=0.85))
```

### Universal Hardware Backends, Hybrid Creativity & Video Mode

```bash
# Accelerated execution backends (pytorch, compile, onnx, directml, tensorrt)
azzani-enhance photo.jpg --preset fast --backend compile
azzani-enhance photo.jpg --preset pro --backend directml   # Windows AMD/Intel DX12

# Hybrid Generative Detail Refiner (--creativity > 0): face/sky/bokeh masked micro-texture + CIE LAB color lock
azzani-enhance wildlife.jpg --preset pro --creativity 0.45

# Temporal-Consistent Video Enhancer (Farneback optical-flow stabilization + scene-cut reset + audio copy)
azzani-enhance clip.mp4 -o clip_4k.mp4 --preset fast --scale 2 --temporal-strength 0.35

# Optional extras for ONNX / DirectML / Diffusers SD-Turbo / Video tooling:
pip install -e ".[onnx]"        # ONNX Runtime GPU/CPU
pip install -e ".[directml]"    # Windows AMD/Intel DirectML DX12
pip install -e ".[generative]"  # Diffusers + Transformers Apache-2.0 4-step tile refiner
```

### Zero-Install Setup, Docker / NAS Server, Webhooks & TOML Profiles

```bash
# Single-line installers (Linux/macOS & Windows):
bash scripts/install.sh
# powershell -ExecutionPolicy ByPass -File scripts/install.ps1 -Extras "directml,raw"

# Custom workflow profiles (~/.config/azzani-enhance/profiles.toml or ./azzani.toml)
azzani-enhance --init-profiles
azzani-enhance --list-profiles
azzani-enhance photo.jpg --profile portrait-studio

# Headless Docker/NAS REST API + background hot-folder automation + signed webhooks
docker compose up -d
azzani-enhance --server --watch ./inbox -o ./output --host 0.0.0.0 --port 8765 --api-key "$AZZANI_API_KEY" --webhook https://example.com/hook
```

Endpoints exposed in `--server` mode:
- `GET /healthz` — container/NAS readiness check (`status`, `version`, `queue_depth`)
- `GET /api/profiles` — list built-in and `profiles.toml` presets
- `POST /api/enhance` — synchronous enhancement (`"profile"`, `"webhook_url"`, `"webhook_secret"`)
- `POST /api/enhance/async` -> `202 Accepted` + `GET /api/jobs/<job_id>` — background `EnginePool` worker queue with optional HMAC-SHA256 `X-Azzani-Signature` webhook callback

### Reproducible Multi-Metric Benchmark Leaderboard

Run `azzani-enhance --benchmark-suite` (or `--benchmark-dir /path/to/images`) to benchmark baselines and neural presets locally:

Dataset: `synthetic_chart_512` (512x512 target, `scale=2`, `jpeg_q=28`, `noise_sigma=0.025`, `tta=1`) | Hardware: `NVIDIA GeForce GTX 1650 Ti` (`cuda:0`):

| Method | Device | PSNR (dB) | SSIM | GMSD (lower better) | Perceptual Score | Sharpness | Blockiness | Sec/Img | Sec/MP |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bicubic` | `cpu` | 26.95 | 0.7306 | 0.1966 | 64.8 | 0.706 | 0.5634 | 0.005s | 0.019s |
| `lanczos` | `cpu` | 26.93 | 0.7276 | 0.1971 | 65.3 | 0.786 | 0.7656 | 0.017s | 0.065s |
| `photometric` | `cpu` | 19.45 | 0.7088 | 0.2060 | 67.1 | 0.920 | 0.8124 | 0.446s | 1.701s |
| `fast` | `cuda:0` | 20.07 | 0.7205 | 0.1948 | 74.4 | 0.952 | 0.0000 | 1.218s | 4.647s |
| `pro` | `cuda:0` | 19.73 | 0.6247 | 0.2012 | 71.4 | 1.000 | 0.0000 | 1.592s | 6.071s |

> Note: Adaptive Retinex contrast and photometric tone finishing intentionally expand dynamic contrast relative to raw chart brightness (lowering pixelwise PSNR while boosting sharpness, perceptual fidelity `GMSD`, and reducing neural `blockiness` from `0.5634` to `0.0000`).

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
Copyright (c) 2026 Azzani (7AZZANI).

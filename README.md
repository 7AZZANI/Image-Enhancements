# AZZANI Image Enhancements

Open-source AI image upscaler and restorer: super-resolution, JPEG/noise cleanup, face restoration and natural colour finishing, with automatic GPU setup.

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

`azzani-enhance` analyzes each image (noise, blur, JPEG blocking, exposure, faces), picks suitable models, and runs them through a float32 pipeline that preserves colour, metadata and transparency. It runs on NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS), Intel (XPU) or CPU, and ships with a CLI, a local browser UI, a hot-folder watcher and a REST API.

---

## Features

- **AI upscaling and restoration** — 2× and 4× super-resolution plus 1× de-JPEG, denoise and motion deblur, using permissively licensed models (RealPLKSR, DAT2, SPAN, NAFNet) loaded through [spandrel](https://github.com/chaiNNer-org/spandrel). Any local `.safetensors`/`.pth` checkpoint that spandrel supports also works via `--model`.
- **Face restoration** — YuNet face detection, FFHQ alignment and GFPGAN v1.4, blended back with a Laplacian-pyramid seam fusion. An SFace identity check limits likeness drift.
- **Faithful by default** — keeps the original white balance and tonality; tone is only corrected for flat, hazy or underexposed images. A halo guard removes ringing without blurring real detail.
- **Safety checks** — every neural pass is compared with its input; a model that diverges (NaN, colour blow-up, unstable deblur) is discarded instead of producing a broken image.
- **Image analyzer** — `--inspect` reports noise, sharpness, JPEG blockiness, motion blur, exposure and colour-cast diagnostics.
- **Hardware-aware runtime** — automatic PyTorch wheel selection, a cached FP16/FP32 self-check (avoids black output on some GTX 16xx cards), VRAM-based tiling with out-of-memory backoff, and disk-backed processing for very large images.
- **Colour-managed I/O** — 8/16-bit PNG/TIFF/JPEG/WebP, EXIF orientation and ICC profiles, alpha-channel preservation, optional camera RAW input.
- **Tools around the engine** — browser UI with before/after slider and live retuning, hot-folder automation, REST API with async jobs and signed webhooks, TOML profiles, video mode, and a benchmark suite.

---

## Installation

```bash
git clone https://github.com/7AZZANI/Image-Enhancements.git
cd Image-Enhancements

# Detects your GPU, creates .venv and installs the matching PyTorch build
python3 enhance.py --setup
```

Alternatives:

```bash
./azzani.sh                                                   # Linux / macOS launcher
curl -fsSL https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.sh | bash
# Windows: double-click azzani.bat, or
# powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.ps1 | iex"
```

<details>
<summary>Manual installation</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # pick your CUDA/ROCm build
pip install -e .
pip install -e ".[raw]"   # optional: camera RAW input (.dng, .cr2, .nef, .arw, ...)
```

If `/tmp` is a small tmpfs, install with `TMPDIR=/var/tmp` (`--setup` does this automatically).

</details>

Models are downloaded on first use to `~/.cache/azzani-enhance/models` and verified by size and SHA-256.

---

## Quick start

```bash
# Automatic enhancement with a before/after comparison image
azzani-enhance photo.jpg --comparison

# Browser UI (drag and drop, before/after slider, live retuning)
azzani-enhance --ui

# Specific presets
azzani-enhance web_image.jpg --preset fast --scale 2       # quick 2x cleanup
azzani-enhance old_photo.jpg --preset ultra --tta 4        # highest quality, slower
azzani-enhance compressed.jpg --preset restore --scale 1   # repair only, same size

# Batch a folder into 16-bit PNGs
azzani-enhance ./photos -o ./enhanced -r --ext png --bit-depth 16

# Watch a folder and enhance new files automatically
azzani-enhance --watch ./azzani_inbox -o ./azzani_enhanced

# Diagnostics
azzani-enhance photo.jpg --inspect
azzani-enhance --doctor
azzani-enhance --list-models
```

`python3 enhance.py ...` accepts the same arguments and uses the project's `.venv` automatically.

---

## Presets

| Preset | Model | Scale | Best for |
| :--- | :--- | :--- | :--- |
| `auto` (default) | chosen per image and GPU | 1×–4× | General use |
| `fast` | `2xNomosUni_span_multijpg_ldl` (SPAN) | 2× | Quick web/photo cleanup |
| `pro` | `4xNomosWebPhoto_RealPLKSR` | 4× | Real-world photos: de-JPEG, deblur, detail |
| `ultra` | `4xRealWebPhoto_v4_dat2` (DAT2) | 4× | Maximum quality; needs a stronger GPU |
| `hq` | `4xNomos2_hq_dat2` (DAT2) | 4× | Clean, high-quality source photos |
| `portrait` | `4xFaceUpDAT` + GFPGAN v1.4 | 4× | Portraits |
| `anime` / `anime-2x` | `4xHFA2k_ludvae_realplksr_dysample` / `2xHFA2k_LUDVAE_SPAN` | 4× / 2× | Illustrations and line art |
| `deblur` | `NAFNet-GoPro-width32` | 1× | Camera shake |
| `restore` | `1xDeJPG_realplksr_otf` | 1× | Compression artifacts, same size |
| `photometric` | none (classical processing) | 1× | Colour, contrast and detail only, no AI |

All bundled models use permissive licenses (CC-BY-4.0, Apache-2.0, MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Useful options

| Option | Purpose |
| :--- | :--- |
| `--scale`, `--width`, `--height`, `--max-dim` | Output size |
| `--face-restore auto\|on\|off`, `--face-fidelity` | Face restoration (auto restores small or degraded faces) |
| `--white-balance auto\|on\|off` | `auto` keeps original colours; `on` neutralizes casts |
| `--tone`, `--detail`, `--grain` | Optional tonal contrast, sharpening and grain |
| `--halo-guard`, `--ca-suppress`, `--no-bokeh-guard` | Artifact control |
| `--tta 1\|2\|4\|8` | Test-time augmentation (higher quality, slower) |
| `--device`, `--precision`, `--tile-size`, `--backend` | Runtime control |
| `--bit-depth 0\|8\|16`, `--quality`, `--strip-metadata` | Output format |

Run `azzani-enhance --help` for the full list.

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
    cfg=EnhancementConfig(preset="pro", scale=2.0, bit_depth=16),
    comparison_path="before_after.png",
)
print(result.stages_applied, result.quality_confidence)

# Run the neural stages once, then re-tune finishing in milliseconds
layers = prepare_enhancement_layers(load_image("input.jpg"), EnhancementConfig(preset="pro"))
tuned = compose_enhancement_result(layers, EnhancementConfig(tone_strength=0.4, detail_strength=0.2))
```

---

## Advanced usage

### Backends, generative detail and video

```bash
azzani-enhance photo.jpg --backend compile             # torch.compile
azzani-enhance photo.jpg --backend directml            # Windows AMD/Intel via ONNX Runtime
azzani-enhance photo.jpg --preset pro --creativity 0.4 # optional generative texture refinement
azzani-enhance clip.mp4 -o clip_2x.mp4 --preset fast --scale 2

pip install -e ".[onnx]"        # ONNX Runtime
pip install -e ".[directml]"    # DirectML (Windows)
pip install -e ".[generative]"  # diffusers-based refiner for --creativity
pip install -e ".[video]"       # PyAV video I/O
```

### Profiles

```bash
azzani-enhance --init-profiles      # writes ~/.config/azzani-enhance/profiles.toml
azzani-enhance --list-profiles
azzani-enhance photo.jpg --profile portrait-studio
```

### REST server and Docker

```bash
docker compose up -d
# or
azzani-enhance --server --host 0.0.0.0 --port 8765 --api-key "$AZZANI_API_KEY" \
  --watch ./inbox -o ./output --webhook https://example.com/hook
```

| Endpoint | Description |
| :--- | :--- |
| `GET /healthz` | Readiness check (`status`, `version`, `queue_depth`) |
| `GET /api/profiles` | Built-in and TOML profiles |
| `POST /api/enhance` | Synchronous enhancement |
| `POST /api/enhance/async` | Queued job (`202 Accepted`), poll `GET /api/jobs/<job_id>`; optional HMAC-SHA256 signed webhook |

---

## Benchmark

`azzani-enhance --benchmark-suite` compares baselines and presets on a degraded synthetic chart (or your own images with `--benchmark-dir`).

Synthetic chart, 512×512 target, 2× upscale, JPEG q28, noise σ 0.025, measured on CPU:

| Method | PSNR (dB) ↑ | SSIM ↑ | GMSD ↓ | JPEG blockiness ↓ | Time (CPU) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `bicubic` | 26.95 | 0.7306 | 0.1966 | 0.563 | 0.01 s |
| `lanczos` | 26.93 | 0.7276 | 0.1971 | 0.766 | 0.02 s |
| `photometric` | 19.45 | 0.7088 | 0.2060 | 0.812 | 0.77 s |
| `fast` | 26.81 | 0.7399 | 0.1954 | 0.000 | 3.1 s |
| `pro` | 25.28 | 0.6707 | 0.1918 | 0.000 | 24.5 s |

Neural presets remove all JPEG blocking and give the best gradient fidelity (GMSD). GAN-based models such as `pro` trade some PSNR for synthesized detail, which is expected. `photometric` deliberately changes tone and contrast, so it scores lower on pixel fidelity. A CUDA GPU is typically 10–20× faster than the CPU times shown.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE). Model weights are distributed under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (c) 2026 Azzani (7AZZANI).

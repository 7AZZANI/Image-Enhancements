"""Curated permissive-license model registry and SHA-256 verified downloader."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and cryptographic verification spec for a supported restoration model."""

    key: str
    filename: str
    url: str
    scale: int
    architecture: str
    purpose: str
    license: str
    attribution: str
    size_bytes: int
    sha256: Optional[str]
    tile_complexity: str = "medium"  # "light" | "medium" | "heavy"
    value_range: str = "0_1"  # "0_1" or "-1_1"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "dejpeg-1x": ModelSpec(
        key="dejpeg-1x",
        filename="1xDeJPG_realplksr_otf.safetensors",
        url="https://huggingface.co/Phips/1xDeJPG_realplksr_otf/resolve/main/1xDeJPG_realplksr_otf.safetensors",
        scale=1,
        architecture="RealPLKSR",
        purpose="1x JPEG/WebP compression artifact and mosquito-noise removal",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=29_486_580,
        sha256="bfc0b9a1163bf20eb18479e1f4faa0ece1b4d5ce59759be8b843d6ef32d1890f",
        tile_complexity="medium",
    ),
    "denoise-1x": ModelSpec(
        key="denoise-1x",
        filename="1xDeNoise_realplksr_otf.safetensors",
        url="https://huggingface.co/Phips/1xDeNoise_realplksr_otf/resolve/main/1xDeNoise_realplksr_otf.safetensors",
        scale=1,
        architecture="RealPLKSR",
        purpose="1x blind sensor/ISO denoising while preserving fine natural texture",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=29_486_580,
        sha256="0a3e22bdcb70dd790fce66f45df8d07e774e269b17c92a2835162b150d4a365c",
        tile_complexity="medium",
    ),
    "exposure-1x": ModelSpec(
        key="exposure-1x",
        filename="1xExposureCorrection_compact.safetensors",
        url="https://huggingface.co/Phips/1xExposureCorrection_compact/resolve/main/1xExposureCorrection_compact.safetensors",
        scale=1,
        architecture="SRVGGNetCompact",
        purpose="1x neural exposure correction for under/over-exposed photography",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=2_386_044,
        sha256="5ef6bc73ecc0a1c9fe447a4191b0207f4c1777dd318eac06979c86462340d9ce",
        tile_complexity="light",
    ),
    "deblur-1x": ModelSpec(
        key="deblur-1x",
        filename="NAFNet-GoPro-width32.pth",
        url="https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth",
        scale=1,
        architecture="NAFNet",
        purpose="1x blind directional camera-shake and motion deblurring",
        license="MIT",
        attribution="Megvii Research (NAFNet)",
        size_bytes=68_671_121,
        sha256="19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c",
        tile_complexity="medium",
    ),
    "photo-fast": ModelSpec(
        key="photo-fast",
        filename="2xNomosUni_span_multijpg_ldl.safetensors",
        url="https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl/resolve/main/2xNomosUni_span_multijpg_ldl.safetensors",
        scale=2,
        architecture="SPAN",
        purpose="Fast 2x universal photo/web upscaling with multi-JPG cleanup and DoF preservation",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=8_903_384,
        sha256="a3d35e01b8b71b4b3041ad1686f8ebd7bc4e1f3a10378319c2ac61c78b67012a",
        tile_complexity="light",
    ),
    "photo-pro": ModelSpec(
        key="photo-pro",
        filename="4xNomosWebPhoto_RealPLKSR.safetensors",
        url="https://huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR/resolve/main/4xNomosWebPhoto_RealPLKSR.safetensors",
        scale=4,
        architecture="RealPLKSR",
        purpose="4x realistic web-photo restoration (deblur, denoise, JPEG/WebP cleanup + crisp SR)",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=29_590_440,
        sha256="9be0228f98156a100d6636d99b373ed2785b999723f9adc4cca504329ab157f2",
        tile_complexity="medium",
    ),
    "photo-ultra": ModelSpec(
        key="photo-ultra",
        filename="4xRealWebPhoto_v4_dat2.safetensors",
        url="https://huggingface.co/Phips/4xRealWebPhoto_v4_dat2/resolve/main/4xRealWebPhoto_v4_dat2.safetensors",
        scale=4,
        architecture="DAT2",
        purpose="4x Dual Aggregation Transformer v2 maximum-fidelity real-world photo restoration",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=139_792_588,
        sha256="819047acf43073f63ea7289ed94d6991ba8446a3bdf38e274fadfcbaa2b7381d",
        tile_complexity="heavy",
    ),
    "photo-hq": ModelSpec(
        key="photo-hq",
        filename="4xNomos2_hq_dat2.safetensors",
        url="https://huggingface.co/Phips/4xNomos2_hq_dat2/resolve/main/4xNomos2_hq_dat2.safetensors",
        scale=4,
        architecture="DAT2",
        purpose="4x high-detail DAT2 upscaler for clean or moderately sharp photos",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=139_792_588,
        sha256="278c78ba9fd333ace4b02b3aeaa0254428baea20a63d3e995e2e4f8a60173f6f",
        tile_complexity="heavy",
    ),
    "portrait-4x": ModelSpec(
        key="portrait-4x",
        filename="4xFaceUpDAT.safetensors",
        url="https://huggingface.co/Phips/4xFaceUpDAT/resolve/main/4xFaceUpDAT.safetensors",
        scale=4,
        architecture="DAT",
        purpose="4x portrait and facial detail restoration upscaler",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=154_152_604,
        sha256="1b2b269509ca58b99c282a4158fb05e58dfa87bb3453f00c0ad178c762c04c5a",
        tile_complexity="heavy",
    ),
    "anime-2x": ModelSpec(
        key="anime-2x",
        filename="2xHFA2k_LUDVAE_SPAN.safetensors",
        url="https://huggingface.co/Phips/2xHFA2k_LUDVAE_SPAN/resolve/main/2xHFA2k_LUDVAE_SPAN.safetensors",
        scale=2,
        architecture="SPAN",
        purpose="Fast 2x anime, digital art, and sharp line-art upscaling",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=4_461_056,
        sha256="457bca87dd84de0726e8f7a2e04a976c73530c23be0f63ad859b4cff48e9c298",
        tile_complexity="light",
    ),
    "anime-4x": ModelSpec(
        key="anime-4x",
        filename="4xHFA2k_ludvae_realplksr_dysample.safetensors",
        url="https://huggingface.co/Phips/4xHFA2k_ludvae_realplksr_dysample/resolve/main/4xHFA2k_ludvae_realplksr_dysample.safetensors",
        scale=4,
        architecture="RealPLKSR",
        purpose="4x high-precision anime, illustration, and graphic upscaling",
        license="CC-BY-4.0",
        attribution="Philip Hofmann (Phips)",
        size_bytes=29_641_748,
        sha256="392c1e812c1092ec3947a45d04c62f5a1b07c2e5c317b7796cd8d6f095e5a03b",
        tile_complexity="medium",
    ),
    "face-gfpgan": ModelSpec(
        key="face-gfpgan",
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        scale=1,
        architecture="GFPGANv1.4",
        purpose="512x512 aligned blind facial reconstruction and identity-safe eye/skin recovery",
        license="Apache-2.0",
        attribution="Tencent ARC (GFPGAN)",
        size_bytes=348_632_874,
        sha256=None,
        tile_complexity="light",
        value_range="-1_1",
    ),
    "face-detector-yunet": ModelSpec(
        key="face-detector-yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        scale=1,
        architecture="YuNet",
        purpose="Lightweight 5-landmark face detection ONNX model for FFHQ alignment",
        license="MIT",
        attribution="OpenCV Zoo / Shiqi Yu",
        size_bytes=232_589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        tile_complexity="light",
    ),
    "face-recognizer-sface": ModelSpec(
        key="face-recognizer-sface",
        filename="face_recognition_sface_2021dec.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        scale=1,
        architecture="SFace",
        purpose="Lightweight ONNX face identity recognizer for closed-loop likeness protection",
        license="Apache-2.0",
        attribution="OpenCV Zoo / Zhong et al.",
        size_bytes=38_696_353,
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        tile_complexity="light",
    ),
}


def get_default_models_dir() -> Path:
    """Resolve model cache directory via env var or XDG cache home."""
    env_dir = os.environ.get("AZZANI_ENHANCE_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (base / "azzani-enhance" / "models").resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_cached_file(path: Path, spec: ModelSpec) -> bool:
    if not path.is_file():
        return False
    if spec.size_bytes > 0 and path.stat().st_size != spec.size_bytes:
        return False
    if spec.sha256:
        return _sha256_file(path).lower() == spec.sha256.lower()
    return True


def _download_with_progress(url: str, dest_path: Path, expected_size: int, quiet: bool = False) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "azzani-image-enhancements/1.0"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or expected_size or 0)
        if quiet:
            with part_path.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
        else:
            progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
            with progress, part_path.open("wb") as out:
                task_id = progress.add_task(f"Downloading {dest_path.name}", total=total or None)
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    progress.update(task_id, advance=len(chunk))

    part_path.replace(dest_path)


def ensure_model(
    model_key_or_path: str | Path,
    *,
    models_dir: Optional[Path] = None,
    quiet: bool = False,
) -> tuple[Path, Optional[ModelSpec]]:
    """Resolve a built-in model key (downloading & verifying if needed) or custom file path."""
    candidate = Path(str(model_key_or_path)).expanduser()
    if candidate.is_file():
        return candidate.resolve(), None

    key = str(model_key_or_path)
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model key or missing file '{model_key_or_path}'. "
            f"Built-in keys: {available}"
        )

    spec = MODEL_REGISTRY[key]
    cache_dir = (models_dir or get_default_models_dir()).resolve()
    local_path = cache_dir / spec.filename

    if _verify_cached_file(local_path, spec):
        return local_path, spec

    if local_path.exists():
        # Corrupted partial or mismatched checksum — remove before clean fetch.
        local_path.unlink()

    _download_with_progress(spec.url, local_path, spec.size_bytes, quiet=quiet)

    if not _verify_cached_file(local_path, spec):
        actual_size = local_path.stat().st_size if local_path.exists() else 0
        actual_sha = _sha256_file(local_path) if local_path.exists() else "missing"
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(
            f"Downloaded model '{spec.filename}' failed integrity check "
            f"(size={actual_size} expected={spec.size_bytes}, sha256={actual_sha})."
        )

    return local_path, spec

"""Color-managed, alpha-safe, high-bit-depth image I/O utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.ndimage import distance_transform_edt

from .memmap_utils import create_managed_memmap, is_memmap_array

# Permit up to 1.5 Gigapixel trusted local images/prints without DecompressionBombError.
Image.MAX_IMAGE_PIXELS = 1_500_000_000

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}

RAW_EXTENSIONS = {
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
}

INPUT_EXTENSIONS = SUPPORTED_EXTENSIONS | RAW_EXTENSIONS

_SRGB_PROFILE_BYTES: Optional[bytes] = None


def _get_srgb_icc_bytes() -> Optional[bytes]:
    global _SRGB_PROFILE_BYTES
    if _SRGB_PROFILE_BYTES is None:
        try:
            srgb = ImageCms.createProfile("sRGB")
            _SRGB_PROFILE_BYTES = ImageCms.ImageCmsProfile(srgb).tobytes()
        except Exception:
            _SRGB_PROFILE_BYTES = b""
    return _SRGB_PROFILE_BYTES or None


@dataclass
class LoadedImage:
    """Container holding a normalized float32 RGB image and its metadata."""

    rgb: np.ndarray  # float32 [0.0, 1.0], shape (H, W, 3)
    alpha: Optional[np.ndarray]  # float32 [0.0, 1.0], shape (H, W) or None
    source_bit_depth: int
    source_format: str
    exif_bytes: Optional[bytes]
    icc_profile: Optional[bytes]
    original_size: tuple[int, int]  # (width, height) after EXIF orientation


def _apply_exif_orientation_np(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation (1..8) directly to a HxWxC NumPy array."""
    if orientation == 2:
        return np.fliplr(arr)
    if orientation == 3:
        return np.rot90(arr, 2)
    if orientation == 4:
        return np.flipud(arr)
    if orientation == 5:
        return np.transpose(arr, (1, 0, 2))
    if orientation == 6:
        return np.rot90(arr, 3)
    if orientation == 7:
        return np.rot90(np.fliplr(arr), 1)
    if orientation == 8:
        return np.rot90(arr, 1)
    return arr


def _strip_orientation_from_exif(pil_img: Image.Image) -> Optional[bytes]:
    """Return EXIF bytes with Orientation tag reset to 1 without mutating pil_img in place."""
    try:
        raw_exif = pil_img.info.get("exif")
        exif = Image.Exif()
        if raw_exif:
            exif.load(raw_exif)
        else:
            src_exif = pil_img.getexif()
            if not src_exif:
                return None
            exif.load(src_exif.tobytes())
        if not exif:
            return None
        if 274 in exif:
            exif[274] = 1
        return exif.tobytes()
    except Exception:
        return pil_img.info.get("exif")


def _convert_icc_to_srgb_pil(pil_img: Image.Image) -> tuple[Image.Image, Optional[bytes]]:
    """Convert non-sRGB ICC profiles (e.g., Display P3, AdobeRGB) cleanly to sRGB."""
    icc_bytes = pil_img.info.get("icc_profile")
    if not icc_bytes:
        return pil_img, _get_srgb_icc_bytes()
    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        dst_profile = ImageCms.createProfile("sRGB")
        # Convert RGB only; preserve alpha separately if present.
        if pil_img.mode == "RGBA":
            r, g, b, a = pil_img.split()
            rgb_img = Image.merge("RGB", (r, g, b))
            converted = ImageCms.profileToProfile(
                rgb_img,
                src_profile,
                dst_profile,
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
                outputMode="RGB",
            )
            cr, cg, cb = converted.split()
            return Image.merge("RGBA", (cr, cg, cb, a)), _get_srgb_icc_bytes()
        if pil_img.mode == "RGB":
            converted = ImageCms.profileToProfile(
                pil_img,
                src_profile,
                dst_profile,
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
                outputMode="RGB",
            )
            return converted, _get_srgb_icc_bytes()
    except Exception:
        # Malformed or exotic profile; leave pixels untouched and keep existing profile.
        return pil_img, icc_bytes
    return pil_img, icc_bytes


def bleed_alpha_edges(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Propagate nearest foreground RGB color into transparent pixels to prevent edge halos.

    Uses exact Euclidean distance transform Voronoi nearest-neighbor mapping so transparent
    fringes never bleed black or white into upscaled borders.
    """
    opaque = alpha > (1.0 / 255.0)
    if np.all(opaque) or not np.any(opaque):
        return rgb
    iy, ix = distance_transform_edt(~opaque, return_distances=False, return_indices=True)
    bled = rgb[iy, ix]
    out = rgb.copy()
    out[~opaque] = bled[~opaque]
    return out


def _load_camera_raw(path: Path) -> LoadedImage:
    """Decode camera RAW formats (DNG/CR2/CR3/NEF/ARW/etc.) into linear-headroom 16-bit sRGB float32."""
    try:
        import rawpy  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            f"Camera RAW input '{path.name}' requires optional dependency 'rawpy'. "
            "Install it with: pip install 'rawpy>=0.20.0' (or pip install -e '.[raw]')"
        ) from err

    with rawpy.imread(str(path)) as raw:
        # Output 16-bit sRGB without aggressive auto-brightening so our float32 Retinex engine controls tone cleanly.
        rgb_u16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            highlight_mode=rawpy.HighlightMode.Blend,
        )
    arr_f = np.ascontiguousarray(rgb_u16.astype(np.float32) / 65535.0)
    arr_f = np.clip(arr_f, 0.0, 1.0)
    h, w = arr_f.shape[:2]
    return LoadedImage(
        rgb=arr_f,
        alpha=None,
        source_bit_depth=16,
        source_format=path.suffix.lstrip(".").upper() or "RAW",
        exif_bytes=None,
        icc_profile=_get_srgb_icc_bytes(),
        original_size=(w, h),
    )


def load_image(path: str | Path) -> LoadedImage:
    """Load an image file into float32 [0, 1] RGB (+ optional alpha) with full metadata handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_camera_raw(path)

    with Image.open(path) as pil_raw:
        source_format = (pil_raw.format or path.suffix.lstrip(".") or "PNG").upper()
        exif_raw = pil_raw.getexif()
        orientation = int(exif_raw.get(274, 1)) if exif_raw else 1
        exif_bytes = _strip_orientation_from_exif(pil_raw)
        raw_icc = pil_raw.info.get("icc_profile")

        # Probe OpenCV for 16-bit PNG/TIFF precision (Pillow struggles with 16-bit RGB/RGBA).
        cv_arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if cv_arr is not None and cv_arr.dtype == np.uint16:
            if cv_arr.ndim == 2:
                cv_arr = np.repeat(cv_arr[:, :, None], 3, axis=2)
            elif cv_arr.shape[2] == 2:
                gray = cv_arr[:, :, 0:1]
                alpha_u16 = cv_arr[:, :, 1:2]
                cv_arr = np.concatenate([gray, gray, gray, alpha_u16], axis=2)
            elif cv_arr.shape[2] == 3:
                cv_arr = cv2.cvtColor(cv_arr, cv2.COLOR_BGR2RGB)
            elif cv_arr.shape[2] == 4:
                cv_arr = cv2.cvtColor(cv_arr, cv2.COLOR_BGRA2RGBA)

            cv_arr = _apply_exif_orientation_np(cv_arr, orientation)
            arr_f = np.ascontiguousarray(cv_arr.astype(np.float32) / 65535.0)
            h, w = arr_f.shape[:2]
            if arr_f.shape[2] == 4:
                alpha = np.clip(arr_f[:, :, 3], 0.0, 1.0)
                rgb = np.clip(arr_f[:, :, :3], 0.0, 1.0)
            else:
                alpha = None
                rgb = np.clip(arr_f[:, :, :3], 0.0, 1.0)
            return LoadedImage(
                rgb=rgb,
                alpha=alpha,
                source_bit_depth=16,
                source_format=source_format,
                exif_bytes=exif_bytes,
                icc_profile=raw_icc or _get_srgb_icc_bytes(),
                original_size=(w, h),
            )

        # Standard 8-bit path via Pillow (handles EXIF, palette transparency, CMYK, ICC profiles).
        pil_img = ImageOps.exif_transpose(pil_raw)
        has_alpha = (
            "A" in pil_img.getbands()
            or pil_img.mode == "PA"
            or (pil_img.mode == "P" and "transparency" in pil_img.info)
        )
        target_mode = "RGBA" if has_alpha else "RGB"
        if pil_img.mode != target_mode:
            pil_img = pil_img.convert(target_mode)

        pil_img, out_icc = _convert_icc_to_srgb_pil(pil_img)
        arr_u8 = np.asarray(pil_img, dtype=np.uint8)
        arr_f = np.ascontiguousarray(arr_u8.astype(np.float32) / 255.0)
        h, w = arr_f.shape[:2]
        if arr_f.shape[2] == 4:
            rgb = arr_f[:, :, :3]
            alpha = arr_f[:, :, 3]
        else:
            rgb = arr_f[:, :, :3]
            alpha = None

        return LoadedImage(
            rgb=rgb,
            alpha=alpha,
            source_bit_depth=8,
            source_format=source_format,
            exif_bytes=exif_bytes,
            icc_profile=out_icc,
            original_size=(w, h),
        )


def upscale_alpha(alpha: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Smoothly upscale an alpha matte to target (H, W) in float32 [0, 1]."""
    th, tw = target_hw
    if alpha.shape[:2] == (th, tw):
        return np.clip(alpha, 0.0, 1.0)
    if (th * tw) >= 24_000_000 or is_memmap_array(alpha):
        out_mm = create_managed_memmap((th, tw), dtype=np.float32, prefix="azz_alpha_")
        sh, sw = alpha.shape[:2]
        scale_y = sh / float(th)
        strip_h = 512
        for y0 in range(0, th, strip_h):
            y1 = min(th, y0 + strip_h)
            sy0 = max(0, int(np.floor(y0 * scale_y)) - 4)
            sy1 = min(sh, int(np.ceil(y1 * scale_y)) + 4)
            patch = np.asarray(alpha[sy0:sy1], dtype=np.float32)
            # Resize local source strip into padded intermediate then slice exact output rows.
            sub_th = max(1, int(round((sy1 - sy0) / scale_y)))
            up_sub = cv2.resize(patch, (tw, sub_th), interpolation=cv2.INTER_CUBIC)
            off = int(round(y0 - sy0 / scale_y))
            off = max(0, min(max(0, sub_th - (y1 - y0)), off))
            take = min(y1 - y0, sub_th - off)
            out_mm[y0 : y0 + take] = np.clip(up_sub[off : off + take], 0.0, 1.0)
        out_mm.flush()
        return out_mm
    up = cv2.resize(alpha.astype(np.float32), (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return np.clip(up, 0.0, 1.0)


def apply_tpdf_dither(rgb_f32: np.ndarray, seed: int = 42) -> np.ndarray:
    """Quantize float32 [0, 1] RGB to uint8 with triangular-PDF dither to prevent gradient banding."""
    rng = np.random.default_rng(seed)
    # TPDF noise in [-1.0, +1.0] * 0.5 LSB eliminates quantization banding without visible grain.
    noise = (
        rng.random(rgb_f32.shape, dtype=np.float32)
        - rng.random(rgb_f32.shape, dtype=np.float32)
    ) * (0.5 / 255.0)
    quantized = np.rint(np.clip(rgb_f32 + noise, 0.0, 1.0) * 255.0)
    return quantized.astype(np.uint8)


def _quantize_strip(
    rgb_strip: np.ndarray,
    alpha_strip: Optional[np.ndarray],
    *,
    bit_depth: int,
    dither: bool,
    seed: int,
) -> np.ndarray:
    c_strip = np.clip(np.asarray(rgb_strip, dtype=np.float32), 0.0, 1.0)
    if bit_depth == 16:
        rgb_q = np.rint(c_strip * 65535.0).astype("<u2")
        if alpha_strip is not None:
            a_q = np.rint(np.clip(np.asarray(alpha_strip, dtype=np.float32), 0.0, 1.0) * 65535.0).astype("<u2")[:, :, None]
            return np.concatenate([rgb_q, a_q], axis=2)
        return rgb_q

    if dither:
        rgb_u8 = apply_tpdf_dither(c_strip, seed=seed)
    else:
        rgb_u8 = np.rint(c_strip * 255.0).astype(np.uint8)
    if alpha_strip is not None:
        a_u8 = np.rint(np.clip(np.asarray(alpha_strip, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, None]
        return np.concatenate([rgb_u8, a_u8], axis=2)
    return rgb_u8


def _write_streaming_tiff(
    out_path: Path,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    *,
    bit_depth: int,
    dither: bool,
    bigtiff: Optional[bool] = None,
) -> Path:
    """Stream RGB/RGBA strips directly to disk as Classic TIFF or BigTIFF (>3.8 GB) using <32 MB RAM."""
    h, w = rgb.shape[:2]
    samples = 4 if alpha is not None else 3
    bps = 16 if bit_depth == 16 else 8
    bytes_per_pixel = samples * (2 if bps == 16 else 1)
    est_bytes = int(w) * int(h) * bytes_per_pixel
    use_big = bool(bigtiff) if bigtiff is not None else (est_bytes > 3_800_000_000)

    rows_per_strip = min(h, 256)
    strip_offsets: list[int] = []
    strip_counts: list[int] = []

    with out_path.open("wb") as fh:
        if use_big:
            fh.write(struct.pack("<HHHQ", 0x4949, 43, 8, 0))
            ifd_ptr_pos = 8
        else:
            fh.write(struct.pack("<HHI", 0x4949, 42, 0))
            ifd_ptr_pos = 4

        for idx, y0 in enumerate(range(0, h, rows_per_strip)):
            y1 = min(h, y0 + rows_per_strip)
            a_sub = alpha[y0:y1] if alpha is not None else None
            q = _quantize_strip(
                rgb[y0:y1],
                a_sub,
                bit_depth=bit_depth,
                dither=dither,
                seed=42 + idx,
            )
            raw = np.ascontiguousarray(q).tobytes()
            strip_offsets.append(fh.tell())
            fh.write(raw)
            strip_counts.append(len(raw))

        num_strips = len(strip_offsets)
        bps_vals = struct.pack(f"<{samples}H", *([bps] * samples))

        if use_big:
            off_bytes = struct.pack(f"<{num_strips}Q", *strip_offsets) if num_strips > 1 else None
            cnt_bytes = struct.pack(f"<{num_strips}Q", *strip_counts) if num_strips > 1 else None
            # Pad to even offset
            if fh.tell() % 2:
                fh.write(b"\x00")
            bps_inline = len(bps_vals) <= 8
            if not bps_inline:
                bps_off = fh.tell()
                fh.write(bps_vals)
            if off_bytes is not None:
                off_arr_off = fh.tell()
                fh.write(off_bytes)
                cnt_arr_off = fh.tell()
                fh.write(cnt_bytes)
            ifd_offset = fh.tell()

            # BigTIFF IFD entries sorted strictly by tag ID ascending: (tag, type, count, 8-byte val/off)
            # Types: 3=SHORT(2B), 4=LONG(4B), 16=LONG8(8B)
            entries: list[tuple[int, int, int, bytes]] = [
                (256, 16, 1, struct.pack("<Q", w)),
                (257, 16, 1, struct.pack("<Q", h)),
                (258, 3, samples, bps_vals.ljust(8, b"\x00") if bps_inline else struct.pack("<Q", bps_off)),
                (259, 3, 1, struct.pack("<H6x", 1)),
                (262, 3, 1, struct.pack("<H6x", 2)),
                (273, 16, num_strips, struct.pack("<Q", strip_offsets[0] if num_strips == 1 else off_arr_off)),
                (277, 3, 1, struct.pack("<H6x", samples)),
                (278, 16, 1, struct.pack("<Q", rows_per_strip)),
                (279, 16, num_strips, struct.pack("<Q", strip_counts[0] if num_strips == 1 else cnt_arr_off)),
                (284, 3, 1, struct.pack("<H6x", 1)),
            ]
            if alpha is not None:
                entries.append((338, 3, 1, struct.pack("<H6x", 2)))
            fh.write(struct.pack("<Q", len(entries)))
            for tag, ftype, cnt, val8 in entries:
                fh.write(struct.pack("<HHQ8s", tag, ftype, cnt, val8))
            fh.write(struct.pack("<Q", 0))
            fh.seek(ifd_ptr_pos)
            fh.write(struct.pack("<Q", ifd_offset))
        else:
            if fh.tell() % 2:
                fh.write(b"\x00")
            bps_off = fh.tell()
            fh.write(bps_vals)
            if num_strips > 1:
                off_bytes = struct.pack(f"<{num_strips}I", *strip_offsets)
                cnt_bytes = struct.pack(f"<{num_strips}I", *strip_counts)
                off_arr_off = fh.tell()
                fh.write(off_bytes)
                cnt_arr_off = fh.tell()
                fh.write(cnt_bytes)
            ifd_offset = fh.tell()

            entries32: list[tuple[int, int, int, bytes]] = [
                (256, 4, 1, struct.pack("<I", w)),
                (257, 4, 1, struct.pack("<I", h)),
                (258, 3, samples, struct.pack("<I", bps_off)),
                (259, 3, 1, struct.pack("<H2x", 1)),
                (262, 3, 1, struct.pack("<H2x", 2)),
                (273, 4, num_strips, struct.pack("<I", strip_offsets[0] if num_strips == 1 else off_arr_off)),
                (277, 3, 1, struct.pack("<H2x", samples)),
                (278, 4, 1, struct.pack("<I", rows_per_strip)),
                (279, 4, num_strips, struct.pack("<I", strip_counts[0] if num_strips == 1 else cnt_arr_off)),
                (284, 3, 1, struct.pack("<H2x", 1)),
            ]
            if alpha is not None:
                entries32.append((338, 3, 1, struct.pack("<H2x", 2)))
            fh.write(struct.pack("<H", len(entries32)))
            for tag, ftype, cnt, val4 in entries32:
                fh.write(struct.pack("<HHII", tag, ftype, cnt, struct.unpack("<I", val4)[0]))
            fh.write(struct.pack("<I", 0))
            fh.seek(ifd_ptr_pos)
            fh.write(struct.pack("<I", ifd_offset))
    return out_path


def save_image(
    path: str | Path,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    *,
    bit_depth: int = 8,
    jpeg_quality: int = 98,
    webp_lossless: bool = False,
    dither: bool = True,
    exif_bytes: Optional[bytes] = None,
    icc_profile: Optional[bytes] = None,
    stream_tiff: Optional[bool] = None,
) -> Path:
    """Save float32 [0, 1] RGB (+ optional alpha) cleanly in PNG, JPEG, WebP, or TIFF/BigTIFF."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported output extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    h, w = rgb.shape[:2]
    is_huge = is_memmap_array(rgb) or (h * w) >= 16_000_000

    # Composite alpha cleanly over white before streaming/saving JPEG or BMP.
    compose_white = False
    if alpha is not None and ext in {".jpg", ".jpeg", ".bmp"}:
        if is_huge:
            compose_white = True
        else:
            has_meaningful_alpha = not np.all(alpha >= (1.0 - 1.0 / 65535.0))
            if has_meaningful_alpha:
                compose_white = True
            else:
                alpha = None

    use_stream_tiff = bool(stream_tiff) if stream_tiff is not None else (is_huge and ext in {".tif", ".tiff"})
    if use_stream_tiff and ext in {".tif", ".tiff"}:
        return _write_streaming_tiff(
            out_path,
            rgb,
            alpha,
            bit_depth=16 if bit_depth == 16 else 8,
            dither=dither,
        )

    # Chunked out-of-core quantization for giant non-TIFF exports avoids float32 RAM spikes.
    if is_huge:
        c_count = 4 if (alpha is not None and not compose_white) else 3
        q_dtype = np.uint16 if (bit_depth == 16 and ext == ".png") else np.uint8
        q_mm = create_managed_memmap((h, w, c_count), dtype=q_dtype, prefix="azz_enc_")
        strip_h = 512
        for idx, y0 in enumerate(range(0, h, strip_h)):
            y1 = min(h, y0 + strip_h)
            rgb_s = np.clip(np.asarray(rgb[y0:y1], dtype=np.float32), 0.0, 1.0)
            a_s = None
            if alpha is not None:
                a_raw = np.clip(np.asarray(alpha[y0:y1], dtype=np.float32), 0.0, 1.0)
                if compose_white:
                    a3 = a_raw[:, :, None]
                    rgb_s = rgb_s * a3 + (1.0 - a3)
                else:
                    a_s = a_raw
            q_strip = _quantize_strip(
                rgb_s,
                a_s,
                bit_depth=16 if q_dtype == np.uint16 else 8,
                dither=dither,
                seed=42 + idx,
            )
            if q_dtype == np.uint16:
                q_mm[y0:y1] = cv2.cvtColor(q_strip, cv2.COLOR_RGBA2BGRA if c_count == 4 else cv2.COLOR_RGB2BGR)
            else:
                q_mm[y0:y1] = q_strip
        q_mm.flush()

        if q_dtype == np.uint16:
            ok = cv2.imwrite(str(out_path), q_mm)
            del q_mm
            if not ok:
                raise OSError(f"Failed to write 16-bit image to {out_path}")
            return out_path

        pil_out = Image.fromarray(q_mm, mode="RGBA" if c_count == 4 else "RGB")
    else:
        rgb_c = np.clip(rgb, 0.0, 1.0)
        if compose_white and alpha is not None:
            a3 = np.clip(alpha, 0.0, 1.0)[:, :, None]
            rgb_c = rgb_c * a3 + (1.0 - a3)
            alpha = None

        # 16-bit output path for PNG / TIFF.
        if bit_depth == 16 and ext in {".png", ".tif", ".tiff"}:
            rgb_u16 = np.rint(rgb_c * 65535.0).astype(np.uint16)
            if alpha is not None:
                a_u16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)[:, :, None]
                rgba_u16 = np.concatenate([rgb_u16, a_u16], axis=2)
                bgra_u16 = cv2.cvtColor(rgba_u16, cv2.COLOR_RGBA2BGRA)
                ok = cv2.imwrite(str(out_path), bgra_u16)
            else:
                bgr_u16 = cv2.cvtColor(rgb_u16, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(str(out_path), bgr_u16)
            if not ok:
                raise OSError(f"Failed to write 16-bit image to {out_path}")
            return out_path

        # 8-bit output path with TPDF anti-banding dither.
        if dither:
            rgb_u8 = apply_tpdf_dither(rgb_c)
        else:
            rgb_u8 = np.rint(rgb_c * 255.0).astype(np.uint8)

        if alpha is not None:
            a_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, None]
            pil_out = Image.fromarray(np.concatenate([rgb_u8, a_u8], axis=2), mode="RGBA")
        else:
            pil_out = Image.fromarray(rgb_u8, mode="RGB")

    save_kwargs: dict = {}
    if icc_profile and ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        save_kwargs["icc_profile"] = icc_profile
    if exif_bytes and ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        save_kwargs["exif"] = exif_bytes

    if ext in {".jpg", ".jpeg"}:
        save_kwargs.update(
            quality=int(np.clip(jpeg_quality, 1, 100)),
            subsampling=0,  # 4:4:4 full chroma fidelity
            optimize=True,
        )
    elif ext == ".png":
        save_kwargs.update(optimize=False, compress_level=6)
    elif ext == ".webp":
        if webp_lossless:
            save_kwargs.update(lossless=True, method=4)
        else:
            save_kwargs.update(quality=int(np.clip(jpeg_quality, 1, 100)), method=4)
    elif ext in {".tif", ".tiff"}:
        save_kwargs.update(compression="tiff_lzw")

    pil_out.save(out_path, **save_kwargs)
    return out_path

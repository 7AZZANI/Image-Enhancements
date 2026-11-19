"""TOML user profile presets (~/.config/azzani-enhance/profiles.toml & ./azzani.toml)."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .pipeline import EnhancementConfig

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "photography": {
        "description": "Balanced pro photo restoration (4x TTA, halo guard, auto faces)",
        "preset": "pro",
        "tta": 4,
        "face_restore": "auto",
        "halo_guard": 0.75,
        "bokeh_guard": True,
    },
    "portrait-studio": {
        "description": "Identity-locked portrait recovery (SFace guard + Laplacian blend)",
        "preset": "portrait",
        "tta": 4,
        "face_restore": "on",
        "face_fidelity": 0.72,
        "identity_guard": 0.75,
        "bokeh_guard": True,
    },
    "product-ecommerce": {
        "description": "Crisp high-detail product master with subtle texture & 16-bit PNG/TIFF",
        "preset": "ultra",
        "tta": 4,
        "face_restore": "off",
        "creativity": 0.15,
        "bit_depth": 16,
    },
    "low-light-rescue": {
        "description": "Night/indoor rescue: motion deblur + denoising + 1x exposure boost",
        "preset": "auto",
        "deblur": "auto",
        "denoise": "auto",
        "exposure_ai": "on",
        "tta": 2,
    },
    "print-poster": {
        "description": "Out-of-core memmap giant print master (4x scale, 16-bit, unlimited MP)",
        "preset": "pro",
        "scale": 4.0,
        "bit_depth": 16,
        "memmap_threshold_mp": 48.0,
        "max_megapixels": 0.0,
        "tta": 2,
    },
}

_SAMPLE_TOML = """# AZZANI Image Enhancements — Custom Profiles (TOML)
# Usage: azzani-enhance photo.jpg --profile real-estate-hdr

[profiles.real-estate-hdr]
description = "Crisp architectural interiors, zero face rewrite, wide-angle CA cleanup"
preset = "pro"
scale = 2.0
tta = 4
face_restore = "off"
ca_suppress = 0.65
tone_strength = 0.35
detail_strength = 0.22
halo_guard = 0.85

[profiles.vintage-film]
description = "Gentle archival grain + warm portrait likeness preservation"
preset = "restore"
tta = 2
face_restore = "on"
face_fidelity = 0.75
identity_guard = 0.80
natural_grain = 0.008
bit_depth = 16
"""


def default_user_profiles_path() -> Path:
    """Return OS-standard user profile path (~/.config/azzani-enhance/profiles.toml)."""
    env_override = os.environ.get("AZZANI_PROFILES")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return (base / "azzani-enhance" / "profiles.toml").resolve()


def _load_toml_file(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "Reading profiles.toml on Python <3.11 requires 'tomli'. Install with: pip install tomli"
        )
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    nested = raw.get("profiles")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if isinstance(v, dict):
                out[str(k)] = dict(v)
    for k, v in raw.items():
        if k != "profiles" and isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def discover_profiles(
    *,
    profiles_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load merged profiles: built-in defaults <- user profiles.toml <- ./azzani.toml <- explicit path."""
    merged: Dict[str, Dict[str, Any]] = {
        k: dict(v) for k, v in BUILTIN_PROFILES.items()
    }
    candidates: List[Path] = [default_user_profiles_path()]
    work_dir = Path(cwd) if cwd is not None else Path.cwd()
    candidates.append(work_dir / "azzani.toml")
    if profiles_path is not None:
        candidates.append(Path(profiles_path).expanduser().resolve())

    for cand in candidates:
        if cand.is_file():
            loaded = _load_toml_file(cand)
            for name, body in loaded.items():
                base = dict(merged.get(name, {}))
                base.update(body)
                base.setdefault("source", str(cand))
                merged[name] = base
    return merged


def list_profiles(
    *,
    profiles_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> List[Tuple[str, str, str]]:
    """Return sorted [(name, description, source_label)] tuples."""
    all_p = discover_profiles(profiles_path=profiles_path, cwd=cwd)
    rows: List[Tuple[str, str, str]] = []
    for name in sorted(all_p):
        body = all_p[name]
        desc = str(body.get("description") or "Custom TOML enhancement profile")
        src = str(body.get("source") or "built-in")
        rows.append((name, desc, src))
    return rows


def apply_profile_to_config(
    profile_name: str,
    base_cfg: Optional[EnhancementConfig] = None,
    *,
    profiles_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    ignore_keys: Iterable[str] = ("description", "source"),
) -> EnhancementConfig:
    """Resolve named profile onto EnhancementConfig and re-apply explicit caller overrides."""
    all_p = discover_profiles(profiles_path=profiles_path, cwd=cwd)
    key = profile_name.strip()
    if key not in all_p:
        avail = ", ".join(sorted(all_p))
        raise KeyError(f"Unknown profile '{profile_name}'. Available profiles: {avail}")

    valid_fields = {f.name for f in fields(EnhancementConfig)}
    raw_profile = all_p[key]
    updates: Dict[str, Any] = {}
    ignored = set(ignore_keys)

    for k, v in raw_profile.items():
        if k in ignored:
            continue
        if k not in valid_fields:
            raise ValueError(
                f"Profile '{key}' contains unknown EnhancementConfig option '{k}'."
            )
        if k == "models_dir" and v is not None:
            updates[k] = Path(str(v)).expanduser()
        else:
            updates[k] = v

    if cli_overrides:
        for k, v in cli_overrides.items():
            if k in valid_fields and v is not None:
                updates[k] = v

    cfg = base_cfg or EnhancementConfig()
    return replace(cfg, **updates)


def write_sample_profiles(target_path: Optional[Path] = None, *, overwrite: bool = False) -> Path:
    """Create starter TOML profiles file on disk and return its resolved path."""
    out = (
        Path(target_path).expanduser().resolve()
        if target_path is not None
        else default_user_profiles_path()
    )
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_SAMPLE_TOML, encoding="utf-8")
    return out

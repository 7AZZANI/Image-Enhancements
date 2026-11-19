"""Zero-friction GPU-matched .venv bootstrapper and broken-CUDA PyTorch auto-repair."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .hardware import HardwareReport, probe_system_hardware

REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "spandrel",
    "cv2",
    "PIL",
    "numpy",
    "scipy",
    "safetensors",
    "rich",
)


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the AZZANI repository root (containing pyproject.toml or enhance.py)."""
    cur = (start or Path(__file__).resolve().parent.parent).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "azzani_enhance").is_dir():
            return candidate
    return None


def venv_python_path(project_root: Path) -> Path:
    """Return path to the Python binary inside project .venv for the host OS."""
    if os.name == "nt":
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def is_in_virtualenv() -> bool:
    """Return True if current interpreter is already running inside a venv/conda env."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(
        os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    )


def missing_dependencies() -> List[str]:
    """Return names of required runtime modules missing from the active interpreter."""
    missing = []
    for mod in REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing


def build_pip_install_plan(
    report: HardwareReport,
    python_exe: Path,
    project_root: Optional[Path],
    *,
    fix_gpu_only: bool = False,
) -> Tuple[List[List[str]], Dict[str, str]]:
    """Build exact pip install commands and safe environment overrides (like TMPDIR=/var/tmp)."""
    py = str(python_exe)
    env = dict(os.environ)
    if report.recommend_tmpdir and not env.get("TMPDIR"):
        env["TMPDIR"] = report.recommend_tmpdir

    extra_cache_flag = ["--no-cache-dir"] if report.recommend_tmpdir else []
    commands: List[List[str]] = []

    index_url = report.recommended_index_url
    torch_cmd = [py, "-m", "pip", "install", "--upgrade", *extra_cache_flag]
    if fix_gpu_only:
        torch_cmd.append("--force-reinstall")
    torch_cmd.extend(["torch", "torchvision"])
    if index_url:
        torch_cmd.extend(["--index-url", index_url])
    commands.append(torch_cmd)

    if not fix_gpu_only:
        if project_root and (project_root / "pyproject.toml").is_file():
            commands.append([py, "-m", "pip", "install", *extra_cache_flag, "-e", str(project_root)])
        else:
            commands.append(
                [
                    py,
                    "-m",
                    "pip",
                    "install",
                    *extra_cache_flag,
                    "spandrel>=0.4.0",
                    "opencv-python-headless>=4.8.0",
                    "pillow>=10.0.0",
                    "numpy>=1.26.0",
                    "scipy>=1.11.0",
                    "safetensors>=0.4.0",
                    "rich>=13.0.0",
                    "huggingface-hub>=0.23.0",
                ]
            )

    return commands, env


def run_auto_setup(
    *,
    project_root: Optional[Path] = None,
    use_venv: bool = True,
    fix_gpu_only: bool = False,
    quiet: bool = False,
) -> int:
    """Create .venv (when appropriate), detect GPU, install matched PyTorch, and verify."""
    root = project_root or find_project_root()
    report = probe_system_hardware(include_torch=True)

    if not quiet:
        gpu_desc = (
            ", ".join(f"{g.name} ({g.vendor}/{g.backend})" for g in report.gpus)
            if report.gpus
            else "None detected (CPU mode)"
        )
        print("=" * 68)
        print("AZZANI Image Enhancements — GPU & Environment Auto-Setup")
        print(f"  OS / Arch       : {report.os_name} / {report.machine} (Python {report.python_version})")
        print(f"  Detected GPU(s) : {gpu_desc}")
        if report.nvidia_driver_cuda:
            print(f"  NVIDIA CUDA Cap : {report.nvidia_driver_cuda}")
        print(f"  RAM             : {report.ram_total_gb or '?'} GB")
        print(f"  Selected Backend: {report.recommended_backend} (wheel={report.recommended_wheel_tag})")
        if report.recommend_tmpdir:
            print(
                f"  Temp Guard      : /tmp has {report.tmp_free_gb} GB free -> "
                f"auto-using TMPDIR={report.recommend_tmpdir}"
            )
        print("=" * 68)

    if report.rosetta_warning:
        print(
            "WARNING: Running x86_64 Python under Rosetta on Apple Silicon disables Metal (MPS) GPU.\n"
            "Install native arm64 Python 3.10+ for full Apple Silicon GPU speed.",
            file=sys.stderr,
        )

    target_python = Path(sys.executable)
    if fix_gpu_only:
        # Always repair the currently active Python/venv unless a repo .venv exists outside venv.
        if root and not is_in_virtualenv():
            cand = venv_python_path(root)
            if cand.is_file():
                target_python = cand
    elif use_venv and root is not None and not is_in_virtualenv():
        venv_dir = root / ".venv"
        cand = venv_python_path(root)
        if not cand.is_file():
            if not quiet:
                print(f"[setup] Creating isolated environment at {venv_dir} ...")
            venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir))
        target_python = cand

    commands, env = build_pip_install_plan(
        report,
        python_exe=target_python,
        project_root=root,
        fix_gpu_only=fix_gpu_only,
    )

    if not fix_gpu_only:
        # Ensure pip is current enough for PEP 517 + CUDA wheels.
        subprocess.run(
            [str(target_python), "-m", "pip", "install", "--upgrade", "pip"],
            env=env,
            check=False,
        )

    for cmd in commands:
        if not quiet:
            print(f"[setup] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(
                f"[setup] Command failed with exit code {proc.returncode}: {' '.join(cmd)}",
                file=sys.stderr,
            )
            return proc.returncode

    if not quiet:
        print("[setup] Verifying live GPU & library integration...")
    verify = subprocess.run([str(target_python), "-m", "azzani_enhance", "--doctor"], env=env)
    return verify.returncode

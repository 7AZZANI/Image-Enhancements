"""Pure-stdlib GPU/hardware discovery, PyTorch wheel selector, and persistent precision cache."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PYTORCH_INDEX_URLS: Dict[str, Optional[str]] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "rocm6.2": "https://download.pytorch.org/whl/rocm6.2",
    "cpu": "https://download.pytorch.org/whl/cpu",
    "mps": None,
}


@dataclass
class GpuDeviceInfo:
    """Normalized physical or PyTorch-visible GPU descriptor."""

    index: int
    vendor: str
    name: str
    vram_total_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None
    driver_version: Optional[str] = None
    compute_cap: Optional[str] = None
    backend: str = "unknown"


@dataclass
class HardwareReport:
    """Comprehensive system, GPU, and PyTorch integration diagnostics."""

    os_name: str
    machine: str
    python_version: str
    cpu_count: int
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    tmp_free_gb: Optional[float]
    var_tmp_free_gb: Optional[float]
    recommend_tmpdir: Optional[str]
    gpus: List[GpuDeviceInfo] = field(default_factory=list)
    nvidia_driver_cuda: Optional[str] = None
    recommended_backend: str = "cpu"
    recommended_wheel_tag: str = "cpu"
    recommended_index_url: Optional[str] = None
    rosetta_warning: bool = False
    torch_installed: bool = False
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    torch_hip_version: Optional[str] = None
    torch_cuda_available: bool = False
    torch_mps_available: bool = False
    torch_xpu_available: bool = False
    active_device: str = "cpu"
    integration_status: str = "unknown"
    recommendation: str = ""
    available_backends: List[str] = field(default_factory=lambda: ["pytorch"])
    onnx_providers: List[str] = field(default_factory=list)
    torch_compile_available: bool = False
    ffmpeg_available: bool = False
    generative_available: bool = False
    recommend_directml: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gpus"] = [asdict(g) for g in self.gpus]
        return payload


def get_cache_root() -> Path:
    """Return user cache root for AZZANI Image Enhancements (~/.cache/azzani-enhance)."""
    env_dir = os.environ.get("AZZANI_ENHANCE_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (base / "azzani-enhance").resolve()


def _precision_cache_path() -> Path:
    return get_cache_root() / "hardware_profile.json"


def load_precision_cache() -> Dict[str, Dict[str, str]]:
    """Load persisted CUDA FP16/FP32 self-test probe outcomes per GPU and model."""
    path = _precision_cache_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("precisions"), dict):
            return data["precisions"]
    except Exception:
        pass
    return {}


def save_precision_entry(cache_key: str, dtype_name: str, precision_note: str) -> None:
    """Persist verified dtype decision atomically so subsequent runs skip warmup probes."""
    path = _precision_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        precisions = existing.get("precisions")
        if not isinstance(precisions, dict):
            precisions = {}
        precisions[cache_key] = {"dtype": dtype_name, "note": precision_note}
        existing["precisions"] = precisions
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def get_system_ram_gb() -> Tuple[Optional[float], Optional[float]]:
    """Return (total_ram_gb, available_ram_gb) using pure stdlib across Linux, macOS, Windows."""
    sys_name = platform.system()
    try:
        if sys_name == "Linux" and Path("/proc/meminfo").is_file():
            total_kb = None
            avail_kb = None
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            tot = round(total_kb / (1024 ** 2), 2) if total_kb else None
            avl = round(avail_kb / (1024 ** 2), 2) if avail_kb else None
            return tot, avl

        if sys_name == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return (
                    round(stat.ullTotalPhys / (1024 ** 3), 2),
                    round(stat.ullAvailPhys / (1024 ** 3), 2),
                )

        if sys_name == "Darwin":
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip().isdigit():
                tot_bytes = int(res.stdout.strip())
                return round(tot_bytes / (1024 ** 3), 2), None
    except Exception:
        pass
    return None, None


def _run_cmd(cmd: List[str], timeout: float = 4.0) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return 1, ""


def _parse_version_tuple(ver_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not ver_str:
        return None
    m = re.search(r"(\d+)\.(\d+)", ver_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def select_nvidia_wheel_tag(
    driver_cuda: Optional[str],
    compute_caps: Optional[List[str]] = None,
) -> str:
    """Pick the best official PyTorch CUDA wheel tag given nvidia-smi driver CUDA & compute caps."""
    ver = _parse_version_tuple(driver_cuda)
    if ver is None:
        return "cu126"

    max_cc = 0.0
    for cc in compute_caps or []:
        try:
            max_cc = max(max_cc, float(cc))
        except ValueError:
            pass

    # Blackwell (RTX 50xx, compute >= 10.0 / 12.0) requires cu128+ on CUDA >= 12.8 drivers.
    if ver >= (12, 8) and max_cc >= 10.0:
        return "cu128"
    if ver >= (12, 6):
        return "cu126"
    if ver >= (12, 4):
        return "cu124"
    if ver >= (12, 1):
        return "cu121"
    if ver >= (11, 8):
        return "cu118"
    return "cpu"


def _probe_nvidia_gpus() -> Tuple[List[GpuDeviceInfo], Optional[str]]:
    if not shutil.which("nvidia-smi"):
        return [], None

    rc, header_out = _run_cmd(["nvidia-smi"])
    driver_cuda = None
    if rc == 0:
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", header_out)
        if m:
            driver_cuda = m.group(1)

    rc_q, csv_out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    has_cc = rc_q == 0
    if not has_cc:
        rc_q, csv_out = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
    gpus: List[GpuDeviceInfo] = []
    if rc_q == 0:
        for line in csv_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            try:
                idx = int(parts[0])
                vram_tot = float(parts[3]) if parts[3] not in ("", "[N/A]") else None
                vram_free = float(parts[4]) if parts[4] not in ("", "[N/A]") else None
            except ValueError:
                continue
            cc_val = parts[5] if has_cc and len(parts) >= 6 and parts[5] not in ("", "[N/A]") else None
            gpus.append(
                GpuDeviceInfo(
                    index=idx,
                    vendor="nvidia",
                    name=parts[1],
                    driver_version=parts[2],
                    vram_total_mb=vram_tot,
                    vram_free_mb=vram_free,
                    compute_cap=cc_val,
                    backend="cuda",
                )
            )
    return gpus, driver_cuda


def _probe_amd_or_intel_os() -> List[GpuDeviceInfo]:
    """Detect non-NVIDIA GPUs using OS tools when nvidia-smi is absent."""
    gpus: List[GpuDeviceInfo] = []
    sys_name = platform.system()

    if sys_name == "Linux":
        if shutil.which("rocm-smi") or Path("/opt/rocm").exists():
            rc, out = _run_cmd(["rocm-smi", "--showproductname"]) if shutil.which("rocm-smi") else (1, "")
            name = "AMD Radeon / Instinct GPU (ROCm)"
            for line in out.splitlines():
                if "Card Series" in line or "Card Model" in line:
                    name = line.split(":", 1)[-1].strip() or name
                    break
            gpus.append(GpuDeviceInfo(index=0, vendor="amd", name=name, backend="rocm"))
            return gpus

        if shutil.which("lspci"):
            rc, out = _run_cmd(["lspci"])
            if rc == 0:
                idx = 0
                for line in out.splitlines():
                    low = line.lower()
                    if ("vga compatible" in low or "3d controller" in low or "display controller" in low):
                        desc = line.split(":", 2)[-1].strip()
                        if "nvidia" in low:
                            gpus.append(GpuDeviceInfo(index=idx, vendor="nvidia", name=desc, backend="cuda"))
                            idx += 1
                        elif "amd" in low or "ati" in low or "radeon" in low:
                            gpus.append(GpuDeviceInfo(index=idx, vendor="amd", name=desc, backend="rocm"))
                            idx += 1
                        elif "intel" in low:
                            gpus.append(GpuDeviceInfo(index=idx, vendor="intel", name=desc, backend="xpu"))
                            idx += 1

    elif sys_name == "Darwin":
        is_arm = platform.machine().lower() in ("arm64", "aarch64")
        rc, out = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        name = "Apple GPU (Metal)" if is_arm else "macOS GPU"
        if rc == 0:
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    name = line.split(":", 1)[1].strip()
                    break
        gpus.append(
            GpuDeviceInfo(
                index=0,
                vendor="apple" if is_arm else "unknown",
                name=name,
                backend="mps" if is_arm else "cpu",
            )
        )

    elif sys_name == "Windows" and shutil.which("powershell"):
        rc, out = _run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            timeout=5.0,
        )
        if rc == 0:
            idx = 0
            for raw in out.splitlines():
                name = raw.strip()
                if not name:
                    continue
                low = name.lower()
                if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
                    gpus.append(GpuDeviceInfo(index=idx, vendor="nvidia", name=name, backend="cuda"))
                    idx += 1
                elif "radeon" in low or "amd" in low:
                    gpus.append(GpuDeviceInfo(index=idx, vendor="amd", name=name, backend="rocm"))
                    idx += 1
                elif "intel" in low and ("arc" in low or "uhd" in low or "iris" in low):
                    gpus.append(GpuDeviceInfo(index=idx, vendor="intel", name=name, backend="xpu"))
                    idx += 1

    return gpus


def _check_tmpdir_space() -> Tuple[Optional[float], Optional[float], Optional[str]]:
    tmp_free = None
    var_tmp_free = None
    rec = None
    if platform.system() != "Windows":
        try:
            tmp_free = round(shutil.disk_usage("/tmp").free / (1024 ** 3), 2)
        except Exception:
            pass
        try:
            if Path("/var/tmp").exists():
                var_tmp_free = round(shutil.disk_usage("/var/tmp").free / (1024 ** 3), 2)
        except Exception:
            pass
        if tmp_free is not None and tmp_free < 5.0 and var_tmp_free is not None and var_tmp_free > tmp_free + 2.0:
            rec = "/var/tmp"
    return tmp_free, var_tmp_free, rec


def _check_rosetta() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() == "arm64":
        return False
    rc, out = _run_cmd(["sysctl", "-n", "sysctl.proc_translated"])
    return rc == 0 and out.strip() == "1"


def probe_system_hardware(*, include_torch: bool = True) -> HardwareReport:
    """Detect GPUs, RAM, temp space, PyTorch backend status, and recommend GPU configuration."""
    sys_name = platform.system()
    machine = platform.machine()
    ram_tot, ram_avail = get_system_ram_gb()
    tmp_free, var_tmp_free, rec_tmpdir = _check_tmpdir_space()
    rosetta = _check_rosetta()

    gpus, driver_cuda = _probe_nvidia_gpus()
    if not gpus:
        gpus = _probe_amd_or_intel_os()

    # Determine recommended PyTorch backend & wheel URL.
    nvidia_gpus = [g for g in gpus if g.vendor == "nvidia"]
    amd_gpus = [g for g in gpus if g.vendor == "amd"]
    intel_gpus = [g for g in gpus if g.vendor == "intel"]
    has_rocm_stack = sys_name == "Linux" and (bool(shutil.which("rocm-smi")) or Path("/opt/rocm").exists())

    if nvidia_gpus:
        wheel_tag = select_nvidia_wheel_tag(
            driver_cuda,
            [g.compute_cap for g in nvidia_gpus if g.compute_cap],
        )
        rec_backend = "cuda" if wheel_tag != "cpu" else "cpu"
        rec_index = PYTORCH_INDEX_URLS.get(wheel_tag)
    elif amd_gpus and has_rocm_stack:
        rec_backend = "rocm"
        wheel_tag = "rocm6.2"
        rec_index = PYTORCH_INDEX_URLS["rocm6.2"]
    elif sys_name == "Darwin" and (machine.lower() in ("arm64", "aarch64") or rosetta):
        rec_backend = "mps"
        wheel_tag = "mps"
        rec_index = None
    elif intel_gpus and any("arc" in g.name.lower() or "ultra" in g.name.lower() for g in intel_gpus):
        rec_backend = "xpu"
        wheel_tag = "cpu"  # Standard PyPI torch>=2.5 bundles XPU or intel extension
        rec_index = None
    else:
        rec_backend = "cpu"
        wheel_tag = "cpu"
        rec_index = None if sys_name == "Darwin" else PYTORCH_INDEX_URLS["cpu"]

    report = HardwareReport(
        os_name=sys_name,
        machine=machine,
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=ram_tot,
        ram_available_gb=ram_avail,
        tmp_free_gb=tmp_free,
        var_tmp_free_gb=var_tmp_free,
        recommend_tmpdir=rec_tmpdir,
        gpus=gpus,
        nvidia_driver_cuda=driver_cuda,
        recommended_backend=rec_backend,
        recommended_wheel_tag=wheel_tag,
        recommended_index_url=rec_index,
        rosetta_warning=rosetta,
    )

    if include_torch:
        try:
            import torch  # type: ignore

            report.torch_installed = True
            report.torch_version = str(torch.__version__)
            report.torch_cuda_version = getattr(torch.version, "cuda", None)
            report.torch_hip_version = getattr(torch.version, "hip", None)
            report.torch_cuda_available = bool(torch.cuda.is_available())
            report.torch_mps_available = bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            report.torch_xpu_available = bool(
                hasattr(torch, "xpu") and torch.xpu.is_available()
            )

            # Enrich/populate GPU telemetry straight from live PyTorch when available.
            if report.torch_cuda_available:
                count = torch.cuda.device_count()
                host_driver = gpus[0].driver_version if gpus else None
                live_gpus: List[GpuDeviceInfo] = []
                for idx in range(count):
                    props = torch.cuda.get_device_properties(idx)
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    vendor = "amd" if report.torch_hip_version else "nvidia"
                    live_gpus.append(
                        GpuDeviceInfo(
                            index=idx,
                            vendor=vendor,
                            name=props.name,
                            vram_total_mb=round(total_b / (1024 ** 2), 1),
                            vram_free_mb=round(free_b / (1024 ** 2), 1),
                            driver_version=host_driver,
                            compute_cap=f"{props.major}.{props.minor}",
                            backend="rocm" if report.torch_hip_version else "cuda",
                        )
                    )
                if live_gpus:
                    report.gpus = live_gpus
                    best_gpu = max(
                        live_gpus,
                        key=lambda g: (g.vram_free_mb or 0.0, g.vram_total_mb or 0.0),
                    )
                    report.active_device = f"cuda:{best_gpu.index}"
                else:
                    report.active_device = "cuda:0"
            elif report.torch_xpu_available:
                report.active_device = "xpu"
            elif report.torch_mps_available:
                report.active_device = "mps"
            else:
                report.active_device = "cpu"
        except Exception:
            report.torch_installed = False

    # Classify integration health and craft human-friendly guidance.
    if rosetta:
        report.integration_status = "rosetta_x86_on_arm64"
        report.recommendation = (
            "Apple Silicon Mac is running Python under Rosetta x86_64. "
            "Use native arm64 Python 3.10+ to unlock Metal (MPS) GPU acceleration."
        )
    elif not report.torch_installed:
        report.integration_status = "torch_missing"
        if rec_backend in ("cuda", "rocm"):
            report.recommendation = (
                f"Detected GPU ({gpus[0].name}). Run --setup to install PyTorch [{wheel_tag}] automatically."
            )
        elif rec_backend == "mps":
            report.recommendation = "Detected Apple Silicon GPU. Run --setup to install PyTorch with Metal MPS."
        else:
            report.recommendation = "Run --setup to install AZZANI dependencies automatically."
    elif report.active_device != "cpu":
        report.integration_status = "optimal"
        primary = report.gpus[0].name if report.gpus else report.active_device
        report.recommendation = f"Auto-integrated with GPU [{primary}] on {report.active_device}."
    elif nvidia_gpus and not report.torch_cuda_available:
        if report.torch_cuda_version is None:
            report.integration_status = "torch_cpu_on_gpu_host"
            report.recommendation = (
                f"NVIDIA GPU detected ({nvidia_gpus[0].name}), but PyTorch is CPU-only ({report.torch_version}). "
                f"Run 'azzani-enhance --fix-gpu' to auto-install CUDA PyTorch ({wheel_tag})."
            )
        else:
            report.integration_status = "gpu_driver_mismatch"
            report.recommendation = (
                f"NVIDIA GPU detected ({nvidia_gpus[0].name}), but CUDA failed to initialize "
                f"(torch={report.torch_version}, driver CUDA={driver_cuda}). "
                "Update NVIDIA driver or run 'azzani-enhance --fix-gpu'."
            )
    elif amd_gpus and has_rocm_stack and not report.torch_cuda_available:
        report.integration_status = "torch_cpu_on_gpu_host"
        report.recommendation = (
            f"AMD ROCm GPU detected ({amd_gpus[0].name}), but ROCm PyTorch is inactive. "
            "Run 'azzani-enhance --fix-gpu' to install rocm6.2 PyTorch."
        )
    else:
        report.integration_status = "cpu_only_ok"
        report.recommendation = (
            f"No supported discrete GPU active; tuned cleanly for multi-core CPU ({report.cpu_count} threads)."
        )

    # Probe universal Phase 3 acceleration & optional capabilities safely without heavy imports.
    backends = ["pytorch"] if report.torch_installed else []
    if report.torch_installed:
        try:
            import torch  # type: ignore

            if hasattr(torch, "compile"):
                report.torch_compile_available = True
                backends.append("compile")
        except Exception:
            pass

    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            import onnxruntime as ort  # type: ignore

            providers = list(ort.get_available_providers())
            report.onnx_providers = providers
            backends.append("onnx")
            if "DmlExecutionProvider" in providers:
                backends.append("directml")
            if "TensorrtExecutionProvider" in providers:
                backends.append("tensorrt")
        except Exception:
            pass

    report.ffmpeg_available = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    report.generative_available = bool(
        importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("transformers") is not None
    )
    if (
        sys_name == "Windows"
        and report.active_device == "cpu"
        and (amd_gpus or intel_gpus)
        and "directml" not in backends
    ):
        report.recommend_directml = True
        report.recommendation = (
            "Windows AMD/Intel GPU detected without DirectML runtime. "
            "Run 'pip install -e \".[directml]\"' to unlock GPU acceleration via DirectML DX12."
        )
    report.available_backends = backends

    return report


def recommend_safe_max_megapixels(ram_total_gb: Optional[float]) -> float:
    """Choose a memory-safe output megapixel ceiling based on host system RAM."""
    if ram_total_gb is None:
        return 48.0
    if ram_total_gb <= 4.5:
        return 24.0
    if ram_total_gb <= 9.0:
        return 48.0
    if ram_total_gb <= 20.0:
        return 64.0
    return 96.0

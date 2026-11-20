"""Rich command-line interface for AZZANI Image Enhancements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .analyzer import analyze_image
from .automation import WatchEvent, WatchFolderRunner
from .benchmark import format_leaderboard_markdown, run_benchmark, run_benchmark_suite, run_doctor
from .bootstrap import run_auto_setup
from .image_io import INPUT_EXTENSIONS, SUPPORTED_EXTENSIONS, load_image
from .inference import EnginePool
from .models import MODEL_REGISTRY
from .pipeline import EnhancementConfig, enhance_file
from .profiles import apply_profile_to_config, list_profiles, write_sample_profiles
from .server import dispatch_webhook, launch_api_server
from .video import VIDEO_EXTENSIONS, VideoConfig, enhance_video_file, is_video_file
from .web_ui import launch_web_ui

console = Console()


def _collect_inputs(input_path: Path, recursive: bool = False) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        files = [
            p
            for p in sorted(iterator)
            if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
        ]
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _resolve_output_path(
    src: Path,
    input_root: Path,
    output_arg: Optional[Path],
    is_batch: bool,
    ext: Optional[str],
    suffix: str,
) -> Path:
    target_ext = (f".{ext.lstrip('.')}" if ext else src.suffix) or ".png"
    if target_ext.lower() not in SUPPORTED_EXTENSIONS:
        target_ext = ".png"

    if output_arg is None:
        return src.with_name(f"{src.stem}{suffix}{target_ext}")

    if not is_batch and (output_arg.suffix.lower() in SUPPORTED_EXTENSIONS):
        return output_arg

    # Directory target (batch or single file dropped into directory)
    if input_root.is_dir():
        rel = src.relative_to(input_root)
        return (output_arg / rel).with_name(f"{src.stem}{suffix}{target_ext}")
    return (output_arg / src.name).with_name(f"{src.stem}{suffix}{target_ext}")


def _resolve_video_output_path(src: Path, output_arg: Optional[Path], suffix: str) -> Path:
    if output_arg is None:
        return src.with_name(f"{src.stem}{suffix}.mp4")
    if output_arg.suffix.lower() in VIDEO_EXTENSIONS:
        return output_arg
    return output_arg / f"{src.stem}{suffix}.mp4"


def _print_models_table() -> None:
    table = Table(title="AZZANI Image Enhancements — Permissive Built-in Model Zoo")
    table.add_column("Key", style="bold cyan")
    table.add_column("Scale", justify="right")
    table.add_column("Arch")
    table.add_column("Size", justify="right")
    table.add_column("License", style="green")
    table.add_column("Purpose")
    for spec in MODEL_REGISTRY.values():
        mb = f"{spec.size_bytes / (1024 * 1024):.1f} MB"
        table.add_row(spec.key, f"{spec.scale}x", spec.architecture, mb, spec.license, spec.purpose)
    console.print(table)


def _print_diagnostics(path: Path, diag) -> None:
    table = Table(title=f"Quality Diagnostics — {path.name}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Dimensions", f"{diag.width} x {diag.height} ({diag.megapixels} MP)")
    table.add_row("Sharpness / Blur", f"sharpness={diag.sharpness_score:.3f}  blur={diag.blur_severity:.3f}")
    table.add_row("Motion Blur (FFT)", f"score={diag.motion_blur_score:.3f}  angle={diag.motion_blur_angle_deg:.1f}deg (recommended={diag.recommended_deblur})")
    table.add_row("Estimated Noise", f"sigma={diag.noise_sigma:.2f}  severity={diag.noise_severity:.3f}")
    table.add_row("JPEG Blockiness", f"{diag.jpeg_blockiness:.3f}")
    table.add_row("Exposure / Range", f"mean L*={diag.mean_luma:.1f}  DR={diag.dynamic_range:.1f}  clip(S/H)={diag.shadow_clip_pct:.1f}%/{diag.highlight_clip_pct:.1f}%")
    table.add_row("Color Cast (a*, b*)", f"strength={diag.color_cast_strength:.3f}  offset={diag.color_cast_ab}")
    table.add_row("Illustration Likely", str(diag.is_illustration_likely))
    table.add_row("Recommended Preset", f"{diag.recommended_preset} (dejpeg={diag.recommended_dejpeg}, denoise={diag.recommended_denoise})")
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azzani-enhance",
        description=(
            "AZZANI Image Enhancements (MIT) — Open-source hybrid AI + 32-bit float photometric "
            "image restoration, super-resolution, face recovery, and detail enhancement."
        ),
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input image file or directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file or directory")
    parser.add_argument("--suffix", default="_azzani", help="Filename suffix when auto-naming outputs (default: _azzani)")
    parser.add_argument("--ext", default=None, help="Force output extension (e.g. png, jpg, webp, tiff)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process directories")

    parser.add_argument(
        "-p",
        "--preset",
        default="auto",
        help="Preset: auto, fast, pro, ultra, hq, portrait, anime, anime-2x, deblur, restore, photometric (default: auto)",
    )
    parser.add_argument("--profile", default=None, help="Named TOML/built-in workflow profile (e.g. photography, portrait-studio)")
    parser.add_argument("--profiles-file", type=Path, default=None, help="Custom TOML profiles file (defaults to ~/.config/azzani-enhance/profiles.toml)")
    parser.add_argument("-m", "--model", default=None, help="Custom built-in model key or local .safetensors/.pth path")
    parser.add_argument("-s", "--scale", type=float, default=None, help="Target scale factor (e.g. 1, 2, 4, 8)")
    parser.add_argument("--width", type=int, default=None, help="Explicit output width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Explicit output height in pixels")
    parser.add_argument("--max-dim", type=int, default=None, help="Cap longest output side to N pixels")
    parser.add_argument("--max-megapixels", type=float, default=48.0, help="RAM safety cap in megapixels (0 = unlimited, default: 48)")

    parser.add_argument("--dejpeg", choices=["auto", "on", "off"], default="auto", help="Pre-clean 8x8 JPEG/WebP blocking")
    parser.add_argument("--denoise", choices=["auto", "on", "off"], default="auto", help="Pre-clean sensor/ISO noise")
    parser.add_argument("--deblur", choices=["auto", "on", "off"], default="auto", help="Pre-clean directional camera shake via 1x NAFNet")
    parser.add_argument("--exposure-ai", choices=["auto", "on", "off"], default="off", help="Optional neural 1x exposure pass")

    parser.add_argument("--face-restore", choices=["auto", "on", "off"], default="auto", help="YuNet + GFPGANv1.4 face restoration (auto restores small or degraded faces)")
    parser.add_argument("--face-fidelity", type=float, default=0.78, help="Restored face blend weight [0..1] (default: 0.78)")
    parser.add_argument("--identity-guard", type=float, default=0.65, help="SFace cosine identity likeness guard [0..1] (default: 0.65, 0 = off)")

    parser.add_argument("--color-fix", type=float, default=0.85, help="Wavelet low-frequency color anchor [0..1]")
    parser.add_argument("--white-balance", choices=["auto", "on", "off"], default="auto", help="White balance: auto keeps the scene's original colours (default), on = neutral-gray balancing, off")
    parser.add_argument("--tone", type=float, default=None, help="Override Retinex shadow/highlight & contrast strength [0..1]")
    parser.add_argument("--detail", type=float, default=None, help="Override guided detail sharpening strength [0..1.5]")
    parser.add_argument("--grain", type=float, default=None, help="Override subtle natural micro-grain [0..0.02]")
    parser.add_argument("--ca-suppress", type=float, default=0.45, help="Lateral chromatic aberration fringe suppression [0..1] (default: 0.45)")
    parser.add_argument("--halo-guard", type=float, default=0.75, help="Closed-loop self-critic halo/ringing clamp strength [0..1] (default: 0.75)")
    parser.add_argument("--no-bokeh-guard", action="store_true", help="Disable Focus/DoF mask so out-of-focus bokeh receives full sharpening")

    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "compile", "onnx", "directml", "tensorrt"], help="Execution backend (auto selects GPU PyTorch or Windows DirectML DX12)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "xpu", "cpu"], help="Execution device (auto selects best GPU)")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32", "bf16"], help="Inference dtype with cached GPU self-check")
    parser.add_argument("--reprobe", action="store_true", help="Re-run CUDA FP16/FP32 self-test bypassing cached GPU profile")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size in pixels (0 = auto VRAM sizing)")
    parser.add_argument("--tile-overlap", type=int, default=32, help="Overlap pixels between tiles (default: 32)")
    parser.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="Geometric self-ensemble passes (1=off, 2, 4, 8=max)")
    parser.add_argument("--memmap-mp", type=float, default=64.0, help="Switch to disk-backed memmap streaming above N megapixels (default: 64, 0 = off)")

    parser.add_argument("--creativity", type=float, default=0.0, help="Hybrid generative micro-texture refinement [0..1] (0=faithful, >0=generative)")
    parser.add_argument("--gen-backend", default="auto", choices=["auto", "diffusion", "procedural"], help="Generative synthesis backend when --creativity > 0")
    parser.add_argument("--gen-model", default=None, help="Custom Diffusers model ID for generative --creativity diffusion mode")

    parser.add_argument("--video", action="store_true", help="Enhance video frame-by-frame with optical-flow temporal stabilization")
    parser.add_argument("--temporal-strength", type=float, default=0.35, help="Optical-flow temporal stabilization strength [0..0.85] (default: 0.35)")
    parser.add_argument("--scene-cut", type=float, default=0.32, help="Scene-cut histogram reset threshold [0..1] (default: 0.32)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max video frames to process")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy source audio stream when outputting enhanced video")

    parser.add_argument("--bit-depth", type=int, default=8, choices=[0, 8, 16], help="Output bit depth (0=match source, 8, or 16 for PNG/TIFF)")
    parser.add_argument("--quality", type=int, default=98, help="JPEG/WebP quality (default: 98)")
    parser.add_argument("--lossless", action="store_true", help="Use lossless WebP output")
    parser.add_argument("--no-dither", action="store_true", help="Disable anti-banding TPDF dither on 8-bit export")
    parser.add_argument("--strip-metadata", action="store_true", help="Do not copy EXIF/ICC metadata to output")
    parser.add_argument("-c", "--comparison", action="store_true", help="Also save a Before/After side-by-side comparison image")
    parser.add_argument("--models-dir", type=Path, default=None, help="Custom directory to store/cache downloaded models")

    parser.add_argument("--ui", "--web", dest="ui", action="store_true", help="Launch local drag-and-drop GPU Web Studio in your browser")
    parser.add_argument("--server", action="store_true", help="Run headless REST API server (/healthz, /api/enhance, /api/enhance/async) for Docker/NAS")
    parser.add_argument("--api-key", default=None, help="Optional API key guard (X-API-Key / Bearer token) for /api/* server endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI / API bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Web UI / API port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser when starting --ui")

    parser.add_argument("--watch", nargs="?", const=Path("./azzani_inbox"), default=None, type=Path, help="Auto-watch folder and enhance incoming images continuously (default: ./azzani_inbox)")
    parser.add_argument("--watch-interval", type=float, default=1.0, help="Watch-folder poll interval in seconds (default: 1.0)")
    parser.add_argument("--move-done", type=Path, default=None, help="Optional archive folder for processed watch-folder originals")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already inside --watch folder on startup")
    parser.add_argument("--webhook", default=None, help="Optional webhook HTTP/HTTPS URL called when CLI/watch/API jobs complete")
    parser.add_argument("--webhook-secret", default=None, help="Optional HMAC-SHA256 secret signing X-Azzani-Signature webhook header")

    parser.add_argument("--setup", action="store_true", help="Auto-detect GPU and install/configure .venv + matched PyTorch wheel")
    parser.add_argument("--fix-gpu", action="store_true", help="Reinstall GPU-matched PyTorch (CUDA/ROCm) if CPU PyTorch was installed")
    parser.add_argument("-y", "--yes", action="store_true", help="Non-interactive yes for bootstrap prompts")

    parser.add_argument("--inspect", action="store_true", help="Analyze image quality diagnostics without modifying")
    parser.add_argument("--list-models", action="store_true", help="List built-in open-source models and exit")
    parser.add_argument("--list-profiles", action="store_true", help="List available built-in and TOML profiles and exit")
    parser.add_argument("--init-profiles", nargs="?", const="__DEFAULT__", default=None, help="Write a starter TOML profiles file (default: ~/.config/azzani-enhance/profiles.toml)")
    parser.add_argument("--doctor", action="store_true", help="Print hardware/GPU/CUDA diagnostics and exit")
    parser.add_argument("--benchmark", action="store_true", help="Run synthetic degradation & quality benchmark")
    parser.add_argument("--benchmark-suite", action="store_true", help="Run multi-baseline perceptual leaderboard (PSNR, SSIM, GMSD, Sec/MP)")
    parser.add_argument("--benchmark-dir", type=Path, default=None, help="Optional real image directory for --benchmark-suite evaluation")
    parser.add_argument("--suite-presets", default="bicubic,lanczos,fast,pro", help="Comma-separated baselines/presets for --benchmark-suite")
    parser.add_argument("--json", action="store_true", help="Output structured JSON report")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal terminal output")
    parser.add_argument("-V", "--version", action="version", version=f"azzani-enhance {__version__}")
    return parser


def _args_to_raw_cfg_dict(args: argparse.Namespace) -> dict:
    return {
        "preset": args.preset,
        "model": args.model,
        "scale": args.scale,
        "width": args.width,
        "height": args.height,
        "max_dim": args.max_dim,
        "max_megapixels": args.max_megapixels,
        "dejpeg": args.dejpeg,
        "denoise": args.denoise,
        "deblur": args.deblur,
        "exposure_ai": args.exposure_ai,
        "face_restore": args.face_restore,
        "face_fidelity": args.face_fidelity,
        "identity_guard": max(0.0, min(1.0, float(args.identity_guard))),
        "color_fix": args.color_fix,
        "white_balance": args.white_balance,
        "tone_strength": args.tone,
        "detail_strength": args.detail,
        "natural_grain": args.grain,
        "ca_suppress": max(0.0, min(1.0, float(args.ca_suppress))),
        "bokeh_guard": not args.no_bokeh_guard,
        "halo_guard": max(0.0, min(1.0, float(args.halo_guard))),
        "backend": args.backend,
        "device": args.device,
        "precision": args.precision,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tta": args.tta,
        "force_reprobe": args.reprobe,
        "memmap_threshold_mp": max(0.0, float(args.memmap_mp)),
        "creativity": max(0.0, min(1.0, float(args.creativity))),
        "generative_backend": args.gen_backend,
        "generative_model": args.gen_model,
        "bit_depth": args.bit_depth,
        "jpeg_quality": args.quality,
        "webp_lossless": args.lossless,
        "preserve_metadata": not args.strip_metadata,
        "dither": not args.no_dither,
        "models_dir": args.models_dir,
        "quiet": args.quiet,
    }


def _build_config_from_args(args: argparse.Namespace) -> EnhancementConfig:
    raw = _args_to_raw_cfg_dict(args)
    profile_name = getattr(args, "profile", None)
    if profile_name:
        default_ns = build_parser().parse_args([])
        defaults = _args_to_raw_cfg_dict(default_ns)
        overrides = {k: v for k, v in raw.items() if v != defaults.get(k)}
        return apply_profile_to_config(
            profile_name,
            EnhancementConfig(),
            profiles_path=getattr(args, "profiles_file", None),
            cli_overrides=overrides,
        )
    return EnhancementConfig(**raw)


def _interactive_hub(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """User-friendly launch hub shown when run interactively without arguments."""
    doc = run_doctor()
    gpu_line = (
        f"[green]{doc['gpu_name']}[/green] ({doc.get('vram_free_mb', '?')}/{doc.get('vram_total_mb', '?')} MB free on {doc['active_device']})"
        if doc.get("gpu_name")
        else f"[yellow]{doc['active_device']}[/yellow] — {doc.get('recommendation', '')}"
    )
    console.print(
        Panel.fit(
            f"[bold cyan]AZZANI Image Enhancements v{__version__}[/bold cyan] [dim](MIT Open-Source)[/dim]\n"
            f"GPU Integration: {gpu_line}\n\n"
            "[bold]1[/bold]  Launch Local Browser Studio ([cyan]Drag & Drop + Split Slider[/cyan])\n"
            "[bold]2[/bold]  Start Hot-Folder Automation ([cyan]./azzani_inbox -> ./azzani_enhanced[/cyan])\n"
            "[bold]3[/bold]  Enhance an Image or Folder now\n"
            "[bold]4[/bold]  System Doctor / Auto-Repair GPU PyTorch\n"
            "[bold]5[/bold]  Run Live GPU Benchmark\n"
            "[bold]q[/bold]  Show CLI Usage & Exit",
            border_style="cyan",
        )
    )
    choice = input("Choose action [1]: ").strip().lower() or "1"
    if choice == "1":
        return launch_web_ui(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            models_dir=args.models_dir,
        )
    if choice == "2":
        watch_dir = Path(input("Inbox folder [./azzani_inbox]: ").strip() or "./azzani_inbox")
        out_dir = Path(input(f"Output folder [{watch_dir / 'enhanced'}]: ").strip() or str(watch_dir / "enhanced"))
        return _run_watch_cli(watch_dir, out_dir, args)
    if choice == "3":
        target = input("Path to image or folder: ").strip().strip('"').strip("'")
        if not target:
            return 1
        preset = input("Preset [auto/fast/pro/ultra/portrait/anime/restore] (default auto): ").strip() or "auto"
        return main([target, "--preset", preset, "--comparison"])
    if choice == "4":
        console.print(Panel.fit(json.dumps(doc, indent=2), title="AZZANI System Doctor"))
        if doc.get("integration_status") in {"torch_cpu_on_gpu_host", "gpu_driver_mismatch"}:
            ans = input("GPU mismatch detected. Auto-install matched GPU PyTorch now? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                return run_auto_setup(fix_gpu_only=True)
        return 0
    if choice == "5":
        res = run_benchmark(preset="fast", device=args.device, precision=args.precision, tta=4)
        console.print(Panel.fit(json.dumps(res, indent=2), title="AZZANI Benchmark (fast)"))
        return 0
    parser.print_help()
    return 0


def _run_watch_cli(watch_dir: Path, out_dir: Optional[Path], args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    resolved_out = out_dir or (watch_dir / "enhanced")

    def _on_event(ev: WatchEvent) -> None:
        if args.quiet:
            return
        if ev.status == "enhanced":
            console.print(
                f"[green]AUTO OK[/green] {Path(ev.input_path).name} -> [bold]{ev.output_path}[/bold] "
                f"({ev.device_used}, {ev.elapsed_seconds}s)"
            )
        else:
            console.print(f"[red]AUTO ERR[/red] {Path(ev.input_path).name}: {ev.error}")

    runner = WatchFolderRunner(
        watch_dir=watch_dir,
        output_dir=resolved_out,
        archive_dir=args.move_done,
        cfg=cfg,
        ext=args.ext,
        suffix=args.suffix,
        recursive=args.recursive,
        comparison=args.comparison,
        poll_interval=args.watch_interval,
        skip_existing=args.skip_existing,
        event_cb=_on_event,
        webhook_url=getattr(args, "webhook", None),
        webhook_secret=getattr(args, "webhook_secret", None),
    )
    if not args.quiet:
        console.print(
            Panel.fit(
                f"[bold cyan]AZZANI Hot-Folder Automation Active[/bold cyan]\n"
                f"Watching Inbox : [bold]{runner.watch_dir}[/bold]\n"
                f"Saving Outputs : [bold]{runner.output_dir}[/bold]\n"
                f"Preset={cfg.preset} | Device={cfg.device} | TTA={cfg.tta}x | Warm GPU Pool=ON\n"
                f"Drop images into the inbox folder anytime. Press Ctrl+C to stop.",
                border_style="cyan",
            )
        )
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        if not args.quiet:
            console.print("\n[cyan]Stopping automation watcher cleanly...[/cyan]")
    finally:
        runner.stop()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.setup or args.fix_gpu:
        return run_auto_setup(fix_gpu_only=(args.fix_gpu and not args.setup), quiet=args.quiet)

    if args.list_models:
        if args.json:
            payload = {k: vars(v) for k, v in MODEL_REGISTRY.items()}
            print(json.dumps(payload, indent=2))
        else:
            _print_models_table()
        return 0

    if args.init_profiles is not None:
        target = None if args.init_profiles == "__DEFAULT__" else Path(args.init_profiles)
        written = write_sample_profiles(target, overwrite=False)
        if args.json:
            print(json.dumps({"profiles_file": str(written)}, indent=2))
        else:
            console.print(f"[green]Profiles file ready:[/green] [bold]{written}[/bold]")
        return 0

    if args.list_profiles:
        profs = list_profiles(profiles_path=args.profiles_file)
        if args.json:
            print(
                json.dumps(
                    [{"name": n, "description": d, "source": s} for n, d, s in profs],
                    indent=2,
                )
            )
        else:
            table = Table(title="AZZANI Enhancement Profiles")
            table.add_column("Profile", style="bold cyan")
            table.add_column("Description")
            table.add_column("Source", style="dim")
            for n, d, s in profs:
                table.add_row(n, d, s)
            console.print(table)
        return 0

    if args.doctor:
        doc = run_doctor()
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            console.print(Panel.fit(json.dumps(doc, indent=2), title="AZZANI System Doctor"))
        return 0

    if args.benchmark:
        bench_preset = "fast" if args.preset == "auto" else args.preset
        res = run_benchmark(
            preset=bench_preset,
            device=args.device,
            precision=args.precision,
            tta=args.tta,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            console.print(Panel.fit(json.dumps(res, indent=2), title=f"AZZANI Benchmark ({bench_preset})"))
        return 0

    if args.benchmark_suite:
        presets_list = [p.strip() for p in args.suite_presets.split(",") if p.strip()]
        suite = run_benchmark_suite(
            presets=presets_list or ("bicubic", "lanczos", "fast", "pro"),
            dataset_dir=args.benchmark_dir,
            scale=int(args.scale) if args.scale else 2,
            device=args.device,
            precision=args.precision,
            tta=args.tta,
        )
        if args.json:
            print(json.dumps(suite, indent=2))
        else:
            console.print(format_leaderboard_markdown(suite))
        return 0

    if args.server:
        return launch_api_server(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            models_dir=args.models_dir,
            watch_dir=args.watch,
            output_dir=args.output,
            webhook_url=args.webhook,
            webhook_secret=args.webhook_secret,
        )

    if args.ui:
        return launch_web_ui(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            models_dir=args.models_dir,
        )

    if args.watch is not None:
        return _run_watch_cli(args.watch, args.output, args)

    if args.input is None:
        if sys.stdin.isatty() and not args.json and not args.quiet and argv is None:
            return _interactive_hub(parser, args)
        parser.print_help()
        return 1

    if args.input.is_file() and not args.inspect and (args.video or is_video_file(args.input)):
        cfg = _build_config_from_args(args)
        vcfg = VideoConfig(
            temporal_strength=max(0.0, min(0.85, float(args.temporal_strength))),
            scene_cut_threshold=max(0.05, min(1.0, float(args.scene_cut))),
            max_frames=args.max_frames,
            preserve_audio=not args.no_audio,
        )
        out_vid = _resolve_video_output_path(args.input, args.output, args.suffix)
        if not args.quiet and not args.json:
            console.print(
                Panel.fit(
                    f"[bold cyan]AZZANI Video Enhancer v{__version__}[/bold cyan]  "
                    f"preset=[bold]{cfg.preset}[/bold] | backend=[bold]{cfg.backend}[/bold] | "
                    f"temporal=[bold]{vcfg.temporal_strength:.2f}[/bold]",
                    border_style="cyan",
                )
            )
        vres = enhance_video_file(
            args.input,
            out_vid,
            cfg=cfg,
            video_cfg=vcfg,
            status_cb=None if (args.quiet or args.json) else lambda m: console.print(f"  [dim]{m}[/dim]"),
        )
        payload = {
            "input": str(args.input),
            "output": str(vres.output_path),
            "frames_processed": vres.frames_processed,
            "scene_cuts_detected": vres.scene_cuts_detected,
            "fps": round(vres.fps, 3),
            "input_size": list(vres.input_size),
            "output_size": list(vres.output_size),
            "audio_copied": vres.audio_copied,
            "io_backend": vres.io_backend,
            "elapsed_seconds": round(vres.elapsed_seconds, 3),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        elif not args.quiet:
            console.print(
                f"[green]VIDEO OK[/green] {args.input.name} -> [bold]{vres.output_path}[/bold] | "
                f"{vres.frames_processed} frames ({vres.scene_cuts_detected} cuts) | "
                f"{vres.input_size[0]}x{vres.input_size[1]} -> {vres.output_size[0]}x{vres.output_size[1]} | "
                f"io={vres.io_backend} | {vres.elapsed_seconds:.2f}s"
            )
        return 0

    inputs = _collect_inputs(args.input, recursive=args.recursive)
    if not inputs:
        console.print(f"[red]No supported image files found at:[/red] {args.input}")
        return 1

    if args.inspect:
        reports = []
        for p in inputs:
            loaded = load_image(p)
            diag = analyze_image(loaded.rgb)
            if args.json:
                reports.append({"path": str(p), **diag.to_dict()})
            else:
                _print_diagnostics(p, diag)
        if args.json:
            print(json.dumps(reports, indent=2))
        return 0

    is_batch = len(inputs) > 1 or args.input.is_dir()
    cfg = _build_config_from_args(args)

    if not args.quiet and not args.json:
        console.print(
            Panel.fit(
                f"[bold cyan]AZZANI Image Enhancements v{__version__}[/bold cyan]  "
                f"[dim](MIT Open-Source)[/dim]\n"
                f"Processing {len(inputs)} image(s) | preset=[bold]{cfg.preset}[/bold] | "
                f"device=[bold]{cfg.device}[/bold] | tta=[bold]{cfg.tta}x[/bold]",
                border_style="cyan",
            )
        )

    json_results = []
    engine_pool = EnginePool(max_models=2) if len(inputs) > 1 else None
    try:
        for idx, src in enumerate(inputs, start=1):
            out_path = _resolve_output_path(
                src,
                args.input,
                args.output,
                is_batch=is_batch,
                ext=args.ext,
                suffix=args.suffix,
            )
            comp_path = (
                out_path.with_name(f"{out_path.stem}_compare.png")
                if args.comparison
                else None
            )

            def _status(msg: str) -> None:
                if not args.quiet and not args.json:
                    console.print(f"  [dim][{idx}/{len(inputs)}][/dim] {msg}")

            res = enhance_file(
                src,
                out_path,
                cfg=cfg,
                comparison_path=comp_path,
                status_cb=_status,
                engine_pool=engine_pool,
            )

            if args.webhook:
                dispatch_webhook(
                    args.webhook,
                    "cli.enhanced",
                    {"input": str(src), "output": str(out_path), **res.summary_dict()},
                    secret=args.webhook_secret,
                )

            if args.json:
                json_results.append({"input": str(src), "output": str(out_path), **res.summary_dict()})
            elif not args.quiet:
                id_note = (
                    f" | id_cos={res.critic_metrics['face_identity_cosine']:.3f}"
                    if "face_identity_cosine" in res.critic_metrics
                    else ""
                )
                console.print(
                    f"[green]OK[/green] {src.name} -> [bold]{out_path}[/bold] | "
                    f"{res.input_size[0]}x{res.input_size[1]} -> {res.output_size[0]}x{res.output_size[1]} | "
                    f"sharpness {res.diagnostics_before.sharpness_score:.2f}->{res.diagnostics_after.sharpness_score:.2f} | "
                    f"conf={res.quality_confidence:.1f}%{id_note} | "
                    f"tile={res.tile_size_used} | {res.precision_used} | {res.elapsed_seconds:.2f}s"
                )
                if res.comparison_path:
                    console.print(f"     [dim]Comparison sheet:[/dim] {res.comparison_path}")
    finally:
        if engine_pool is not None:
            engine_pool.clear()

    if args.json:
        print(json.dumps(json_results if len(json_results) > 1 else json_results[0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

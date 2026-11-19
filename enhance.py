#!/usr/bin/env python3
"""AZZANI Image Enhancements — Zero-friction auto-GPU standalone launcher (MIT License)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from azzani_enhance.bootstrap import (
    is_in_virtualenv,
    missing_dependencies,
    run_auto_setup,
    venv_python_path,
)


def _maybe_reexec_in_venv(root: Path, argv: list[str]) -> int | None:
    """If a project .venv exists and we are running system Python outside it, hand off cleanly."""
    if os.environ.get("AZZANI_BOOTSTRAPPED") == "1" or is_in_virtualenv():
        return None
    if "--no-venv" in argv:
        return None
    vpy = venv_python_path(root)
    if vpy.is_file():
        env = dict(os.environ)
        env["AZZANI_BOOTSTRAPPED"] = "1"
        cmd = [str(vpy), str(Path(__file__).resolve()), *argv]
        if os.name == "nt":
            return subprocess.run(cmd, env=env).returncode
        os.execve(str(vpy), cmd, env)
    return None


def main() -> int:
    root = Path(__file__).resolve().parent
    argv = list(sys.argv[1:])

    reexec_rc = _maybe_reexec_in_venv(root, argv)
    if reexec_rc is not None:
        return reexec_rc

    if "--setup" in argv or "--fix-gpu" in argv:
        fix_only = "--fix-gpu" in argv and "--setup" not in argv
        rc = run_auto_setup(project_root=root, fix_gpu_only=fix_only, quiet="--quiet" in argv or "-q" in argv)
        if rc != 0:
            return rc
        # If user passed only setup flags, stop cleanly.
        remaining = [a for a in argv if a not in ("--setup", "--fix-gpu", "--yes", "-y")]
        if not remaining:
            return 0
        # Otherwise re-exec into newly provisioned .venv to run the rest of the command!
        vpy = venv_python_path(root)
        target_py = str(vpy if (vpy.is_file() and not is_in_virtualenv()) else sys.executable)
        env = dict(os.environ)
        env["AZZANI_BOOTSTRAPPED"] = "1"
        return subprocess.run([target_py, str(Path(__file__).resolve()), *remaining], env=env).returncode

    missing = missing_dependencies()
    if missing:
        auto_yes = (
            os.environ.get("AZZANI_AUTO_SETUP") == "1"
            or "--yes" in argv
            or "-y" in argv
        )
        print(
            f"AZZANI Image Enhancements needs first-time setup (missing: {', '.join(missing)})."
        )
        proceed = auto_yes
        if not proceed and sys.stdin.isatty():
            ans = input("Auto-detect GPU and install into .venv now? [Y/n]: ").strip().lower()
            proceed = ans in ("", "y", "yes")
        if not proceed:
            print("Run `python3 enhance.py --setup` to auto-configure your GPU environment.")
            return 1

        rc = run_auto_setup(project_root=root)
        if rc != 0:
            return rc
        vpy = venv_python_path(root)
        target_py = str(vpy if vpy.is_file() else sys.executable)
        env = dict(os.environ)
        env["AZZANI_BOOTSTRAPPED"] = "1"
        filtered = [a for a in argv if a not in ("--yes", "-y")]
        return subprocess.run([target_py, str(Path(__file__).resolve()), *filtered], env=env).returncode

    from azzani_enhance.cli import main as cli_main

    return cli_main(None if not argv else argv)


if __name__ == "__main__":
    sys.exit(main())

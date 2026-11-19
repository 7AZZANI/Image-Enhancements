#!/usr/bin/env bash
set -euo pipefail

# AZZANI Image Enhancements — One-Click Auto-GPU Launcher (Linux & macOS)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    PYTHON_BIN="$cand"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Error: Python 3.10+ not found in PATH. Please install Python 3.10 or newer."
  exit 1
fi

export AZZANI_AUTO_SETUP=1

if [ $# -eq 0 ]; then
  # Interactive mode: opens GPU Web Studio by default when launched via double-click / no args.
  if [ -t 0 ]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/enhance.py"
  else
    exec "$PYTHON_BIN" "$SCRIPT_DIR/enhance.py" --ui
  fi
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/enhance.py" "$@"
fi

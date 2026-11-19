#!/usr/bin/env bash
set -euo pipefail

# AZZANI Image Enhancements — Zero-Install Bootstrapping Script (Linux & macOS, MIT)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.sh | bash
#   bash scripts/install.sh --extras "raw,onnx" --dir "$HOME/.local/share/azzani-enhance"

REPO_URL="${AZZANI_REPO_URL:-https://github.com/7AZZANI/Image-Enhancements.git}"
INSTALL_DIR="${AZZANI_HOME:-$HOME/.local/share/azzani-enhance}"
BIN_DIR="${AZZANI_BIN_DIR:-$HOME/.local/bin}"
EXTRAS=""
DRY_RUN=0
ASSUME_YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--dir PATH] [--repo URL] [--extras raw,onnx,generative] [--dry-run] [-y]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[azzani-install] %s\n' "$*"; }

# Respect small tmpfs /tmp mounts automatically (avoid pip wheel No-Space errors)
if [ -z "${TMPDIR:-}" ] && [ -d /var/tmp ]; then
  TMP_AVAIL_KB="$(df -Pk /tmp 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)"
  if [ "${TMP_AVAIL_KB:-0}" -lt 6291456 ]; then
    export TMPDIR="/var/tmp"
    log "Detected small /tmp; routing TMPDIR=/var/tmp for safe wheel extraction."
  fi
fi

ensure_python() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv"
    return 0
  fi
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  echo ""
}

PY_PROVIDER="$(ensure_python)"
if [ "$DRY_RUN" -eq 1 ]; then
  log "Dry-run plan:"
  log "  Install dir : $INSTALL_DIR"
  log "  Bin shim    : $BIN_DIR/azzani-enhance"
  log "  Python/uv   : ${PY_PROVIDER:-uv-bootstrap}"
  log "  Extras      : ${EXTRAS:-none}"
  exit 0
fi

if [ -z "$PY_PROVIDER" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: Python 3.10+ or uv/curl is required." >&2
    exit 1
  fi
  log "Python 3.10+ not found; installing standalone uv toolchain..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  PY_PROVIDER="uv"
fi

SCRIPT_ROOT=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  CAND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -f "$CAND_ROOT/pyproject.toml" ] && [ -f "$CAND_ROOT/enhance.py" ]; then
    SCRIPT_ROOT="$CAND_ROOT"
  fi
fi

if [ -n "$SCRIPT_ROOT" ]; then
  INSTALL_DIR="$SCRIPT_ROOT"
  log "Installing directly from local checkout: $INSTALL_DIR"
else
  mkdir -p "$(dirname "$INSTALL_DIR")"
  if [ -d "$INSTALL_DIR/.git" ]; then
    log "Updating existing repository at $INSTALL_DIR..."
    git -C "$INSTALL_DIR" pull --ff-only
  else
    log "Cloning repository into $INSTALL_DIR..."
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
  fi
fi

cd "$INSTALL_DIR"
if [ "$PY_PROVIDER" = "uv" ]; then
  if [ ! -d "$INSTALL_DIR/.venv" ]; then
    log "Provisioning isolated Python 3.12 environment via uv..."
    uv venv --python 3.12 "$INSTALL_DIR/.venv"
  fi
  PY_BIN="$INSTALL_DIR/.venv/bin/python"
else
  PY_BIN="$PY_PROVIDER"
fi

export AZZANI_AUTO_SETUP=1
"$PY_BIN" "$INSTALL_DIR/enhance.py" --setup ${ASSUME_YES:+--yes}

if [ -n "$EXTRAS" ]; then
  log "Installing optional extras: [$EXTRAS]"
  "$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR[$EXTRAS]"
fi

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/azzani-enhance" <<EOF
#!/usr/bin/env bash
exec "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/enhance.py" "\$@"
EOF
chmod +x "$BIN_DIR/azzani-enhance"

log "Installed! Command shim written to $BIN_DIR/azzani-enhance"
"$BIN_DIR/azzani-enhance" --doctor

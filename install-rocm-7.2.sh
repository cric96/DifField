#!/usr/bin/env bash
set -euo pipefail

BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)"

log() {
  printf '[install-rocm72] %s\n' "$*"
}

fail() {
  printf '[install-rocm72] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "ROCm 7.2 is supported by this script only on Linux."
fi

require_cmd uv
require_cmd bash

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "Python executable '$PYTHON_BIN' not found. Set PYTHON_BIN=/path/to/python3.12 if needed."
fi

log "Project root: $PROJECT_ROOT"
log "Creating virtual environment in $VENV_DIR using $PYTHON_BIN"
cd "$PROJECT_ROOT"
uv venv --python "$PYTHON_BIN" "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "3.12" ]]; then
  fail "This script requires Python 3.12 inside the venv, found $PY_VER"
fi

log "Python version: $(python -V 2>&1)"

if command -v rocminfo >/dev/null 2>&1; then
  log "rocminfo detected"
else
  log "rocminfo not found in PATH; continuing, but verify ROCm runtime is installed"
fi

log "Removing any pre-existing torch stack"
python -m pip uninstall -y torch torchvision torchaudio triton pytorch-triton-rocm triton-rocm || true

log "Installing ROCm 7.2.1 wheel set from AMD"
uv pip install --no-cache-dir \
  "${BASE}/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  "${BASE}/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
  "${BASE}/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
  "${BASE}/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

log "Pinning NumPy below 2.0 for ROCm wheel compatibility"
uv pip install --no-cache-dir "numpy<2"

log "Ensuring project runtime dependencies are present"
uv pip install --no-cache-dir \
  "torch-geometric>=2.6" \
  "matplotlib>=3.7" \
  "optuna>=3.6" \
  "scikit-learn>=1.8.0"

log "Running verification"
uv run python - <<'PY'
import sys
import torch

print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print('device_count:', torch.cuda.device_count())
        print('device_name:', torch.cuda.get_device_name(0))
    except Exception as exc:
        print('device_query_error:', exc)
PY

log "Done. Activate with: source $VENV_DIR/bin/activate"
#!/usr/bin/env bash
set -euo pipefail

# Create venv and install packages with uv
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

PYTHON_BIN=""
if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN="python3.9"
elif command -v python3.8 >/dev/null 2>&1; then
  PYTHON_BIN="python3.8"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No compatible Python found. Install python3.10 or python3.9 and re-run."
  exit 1
fi

uv venv "$VENV_DIR" --python "$PYTHON_BIN"

# Core dependencies for 01_global_basic_metrics.py
uv pip install --python "$VENV_DIR/bin/python" \
  numpy \
  pandas \
  pyarrow \
  bctpy \
  scipy \
  matplotlib \
  seaborn \
  scikit-learn

echo "✓ Virtual environment created at: $VENV_DIR"
echo "✓ Packages installed via uv"

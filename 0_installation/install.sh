#!/usr/bin/env bash
# =============================================================================
# bctpy_mrtrix Installation Script
# =============================================================================
#
# This script sets up a complete Python environment for bctpy_mrtrix
# using UV (astral-sh/uv) as the package manager.
#
# Usage:
#   ./0_installation/install.sh
#
# What it does:
#   1. Checks for UV and installs it if missing
#   2. Detects compatible Python version
#   3. Creates virtual environment at .venv/
#   4. Installs all core dependencies
#   5. Installs temporal analysis dependencies
#   6. Generates uv.lock for reproducibility
#
# Author: bctpy_mrtrix Team
# Date: 2026-09-10
# =============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

echo_header() {
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}  bctpy_mrtrix Installation${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    echo ""
}

echo_step() {
    echo -e "${YELLOW}=== $1 ===${NC}"
}

echo_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

echo_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

echo_info() {
    echo "  ${1}"
}

# ============================================================================
# MAIN INSTALLATION
# ============================================================================

main() {
    echo_header

    # Step 1: Check for UV
    echo_step "Checking for UV"
    if ! command -v uv >/dev/null 2>&1; then
        echo_info "UV not found. Installing UV..."
        
        # Detect platform
        if [[ "$OSTYPE" == "linux-gnu" ]] || [[ "$OSTYPE" == "darwin" ]]; then
            # Linux or macOS
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
            # Windows (via cygwin/msys)
            echo_error "Windows detected. Please use PowerShell:"
            echo_info "  irm https://astral.sh/uv/install.ps1 | iex"
            exit 1
        fi
        
        # Add UV to PATH
        export PATH="$HOME/.local/bin:$PATH"
        echo_success "UV installed"
    else
        echo_success "UV found: $(uv --version)"
    fi
    echo ""

    # Step 2: Check Python
    echo_step "Checking Python"
    PYTHON_BIN=""
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="python3.11"
    elif command -v python3.10 >/dev/null 2>&1; then
        PYTHON_BIN="python3.10"
    elif command -v python3.9 >/dev/null 2>&1; then
        PYTHON_BIN="python3.9"
    fi
    
    if [[ -z "$PYTHON_BIN" ]]; then
        echo_error "No compatible Python found."
        echo_info "Required: Python 3.9, 3.10, or 3.11"
        echo_info "Recommended: Python 3.11"
        exit 1
    fi
    echo_success "Python found: $PYTHON_BIN"
    echo ""

    # Step 3: Create virtual environment
    echo_step "Creating Virtual Environment"
    echo_info "Location: $VENV_DIR"
    uv venv "$VENV_DIR" --python "$PYTHON_BIN"
    echo_success "Virtual environment created"
    echo ""

    # Step 4: Install Python dependencies via uv sync (reads pyproject.toml/uv.lock)
    echo_step "Installing Python Dependencies"
    echo_info "This may take a few minutes..."
    (cd "$ROOT_DIR" && uv sync --all-extras)
    echo_success "Python dependencies installed"
    echo ""

    # Step 5: Install/restore the R environment via renv
    echo_step "Installing R Dependencies (renv)"
    if ! command -v Rscript >/dev/null 2>&1; then
        echo_error "R not found. Install R first (e.g. 'brew install r'), then re-run this script."
        exit 1
    fi
    if [ ! -f "$ROOT_DIR/renv.lock" ]; then
        echo_info "No renv.lock found — generating it now (one-time, can take several minutes)..."
        (cd "$ROOT_DIR" && Rscript 0_installation/init_r_env.R)
    else
        (cd "$ROOT_DIR" && Rscript 0_installation/setup_r_env.R)
    fi
    echo_success "R dependencies installed"
    echo ""

    # Step 7: Finalize
    echo_header
    echo_success "Installation Complete!"
    echo ""
    echo_info "Virtual environment: $VENV_DIR"
    echo ""
    echo_info "To activate the environment:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo ""
    echo_info "To run the pipeline:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo_info "  python 1_utilities/..."
    echo ""
    echo_info "To run temporal analysis:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo_info "  python 7_temporal_analysis/scripts/small_worldness.py --data-dir /path/to/data --metadata-file metadata.csv"
    echo ""
    echo_info "To verify the installation:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo_info "  python 0_installation/preflight_check.py run_spec.json --temporal --r --uv-lock"
    echo ""
}

# ============================================================================
# RUN MAIN
# ============================================================================

main "$@"

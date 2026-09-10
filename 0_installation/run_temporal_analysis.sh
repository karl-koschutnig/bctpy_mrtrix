#!/usr/bin/env bash
# =============================================================================
# Temporal Analysis Workflow Script
# =============================================================================
#
# This script runs the complete temporal analysis pipeline for your
# interventional connectomics study.
#
# Requirements:
#   - Python environment activated (run install.sh first)
#   - R installed for mixed-effects modeling (via renv, see 0_installation/setup_r_env.R)
#   - Metadata file created (see data_processing/build_metadata.py)
#
# Usage:
#   ./0_installation/run_temporal_analysis.sh
#
# Author: bctpy_mrtrix Team
# Date: 2026-09-10
# =============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/processed"
METADATA_FILE="${DATA_DIR}/metadata.csv"
GROUPS_EXCEL="${ROOT_DIR}/data/raw/ID_2w_4w_groups.xlsx"

# Default connectomes path (can be overridden by setting CONNECTOMES_DIR)
# Default: use the standard study data location
CONNECTOMES_DIR="${CONNECTOMES_DIR:-/Volumes/Evo/data/129/connectomics/bct_input}"

OUTPUTS_DIR="${ROOT_DIR}/outputs/temporal_analysis"

# Atlases to process (node counts confirmed from the real connectome files)
ATLASES=(AAL3 Gordon333 HCP-MMP Schaefer200 Schaefer400)
declare -A ATLAS_NODES=(
    [AAL3]=166
    [Gordon333]=333
    [HCP-MMP]=360
    [Schaefer200]=200
    [Schaefer400]=400
)

# Study settings
TIMEPOINT_COL="session"
GROUP_COL="group"
N_TIMEPOINTS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# FUNCTIONS
# ============================================================================

echo_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}  Temporal Analysis Pipeline${NC}"
    echo -e "${BLUE}========================================================================${NC}"
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

echo_command() {
    echo -e "${BLUE}  $ ${1}${NC}"
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# ============================================================================
# CHECK ENVIRONMENT
# ============================================================================

check_environment() {
    echo_step "Checking Environment"
    
    # Check Python
    if ! check_command python; then
        echo_error "Python not found. Please activate the virtual environment."
        echo_info "Run: source ${ROOT_DIR}/.venv/bin/activate"
        exit 1
    fi
    echo_success "Python: $(python --version 2>&1)"
    
    # Check UV
    if ! check_command uv; then
        echo_error "UV not found. Please install UV first."
        echo_info "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo_success "UV: $(uv --version)"
    
    # Check if venv is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo_error "Virtual environment not activated"
        echo_info "Run: source ${ROOT_DIR}/.venv/bin/activate"
        exit 1
    fi
    echo_success "Virtual environment active: $VIRTUAL_ENV"
    
    # Check R for GAM (optional)
    HAS_R=false
    if check_command Rscript; then
        echo_success "R: $(Rscript --version 2>&1 | head -1)"
        HAS_R=true
    else
        echo_info "R not found - mixed-effects modeling will be skipped"
        echo_info "To install R: brew install r"
        echo_info "Then: Rscript 0_installation/setup_r_env.R  (restores packages via renv; run 0_installation/install.sh to set up everything)"
    fi
    
    echo ""
}

# ============================================================================
# CREATE METADATA FILE (if not exists)
# ============================================================================

create_metadata() {
    echo_step "Checking Metadata File"

    if [[ ! -f "$METADATA_FILE" ]]; then
        echo_info "Metadata file not found: $METADATA_FILE"

        if [[ ! -f "$GROUPS_EXCEL" ]]; then
            echo_error "Group-assignment spreadsheet not found: $GROUPS_EXCEL"
            echo_info "Place ID_2w_4w_groups.xlsx at that path, or set GROUPS_EXCEL."
            exit 1
        fi

        echo_info "Building metadata.csv from ${GROUPS_EXCEL}..."
        python data_processing/build_metadata.py \
            --excel "$GROUPS_EXCEL" \
            --bct-input-dir "$CONNECTOMES_DIR" \
            --atlas AAL3 \
            --out "$METADATA_FILE" || {
            echo_error "Failed to build metadata from group-assignment spreadsheet"
            exit 1
        }
        echo_success "metadata.csv built at: $METADATA_FILE"
        echo ""
    else
        echo_success "Metadata file found: $METADATA_FILE"
        echo_info "Participants: $(csvcut -c participant_id $METADATA_FILE | tail -n +2 | sort -u | wc -l)"
        echo_info "Sessions: $(csvcut -c session $METADATA_FILE | tail -n +2 | sort -u | tr '\n' ',' | sed 's/,$//')"
        echo_info "Groups: $(csvcut -c group $METADATA_FILE | tail -n +2 | sort -u | tr '\n' ',' | sed 's/,$//')"
        echo ""
    fi
}

# ============================================================================
# RUN TEMPORAL ANALYSIS PIPELINE
# ============================================================================

run_pipeline() {
    echo_step "Running Temporal Analysis Pipeline"

    for ATLAS in "${ATLASES[@]}"; do
        N_NODES="${ATLAS_NODES[$ATLAS]}"
        ATLAS_OUT="${OUTPUTS_DIR}/${ATLAS}"

        echo_step "Atlas: ${ATLAS} (${N_NODES} nodes)"

        if [[ ! -d "${CONNECTOMES_DIR}/${ATLAS}" ]]; then
            echo_error "Connectome data directory not found: ${CONNECTOMES_DIR}/${ATLAS}"
            echo_info "Set CONNECTOMES_DIR environment variable to your data location"
            exit 1
        fi

        # Step 1: Small-worldness calculation
        echo_info "Step 1/3: Calculating small-worldness..."
        python 7_temporal_analysis/scripts/small_worldness.py \
            --data-dir "${CONNECTOMES_DIR}/${ATLAS}" \
            --metadata-file "${METADATA_FILE}" \
            --output-dir "${ATLAS_OUT}/small_worldness" \
            --n-nodes "${N_NODES}" || {
            echo_error "Small-worldness calculation failed for ${ATLAS}"
            exit 1
        }
        echo_success "Small-worldness complete"

        # Step 2: Mixed-effects models (Group x Time)
        if $HAS_R; then
            echo_info "Step 2/3: Fitting mixed-effects models..."
            Rscript 7_temporal_analysis/scripts/mixed_models.R \
                --input-file "${ATLAS_OUT}/small_worldness/small_worldness_results.csv" \
                --output-dir "${ATLAS_OUT}/mixed_model_results" \
                --timepoint-col "${TIMEPOINT_COL}" \
                --group-col "${GROUP_COL}" \
                --participant-col participant_id || {
                echo_error "Mixed-effects modeling failed for ${ATLAS}"
                exit 1
            }
            echo_success "Mixed-effects models complete"
        else
            echo_info "Skipping mixed-effects models (R not available)"
        fi

        # Step 3: UMAP projection (exploratory group x time visualization)
        echo_info "Step 3/3: Running UMAP projection..."
        python 7_temporal_analysis/scripts/umap_projection.py \
            --input-file "${ATLAS_OUT}/small_worldness/small_worldness_results.csv" \
            --output-dir "${ATLAS_OUT}/umap_results" \
            --timepoint-col "${TIMEPOINT_COL}" \
            --group-col "${GROUP_COL}" || {
            echo_error "UMAP projection failed for ${ATLAS}"
            exit 1
        }
        echo_success "UMAP projection complete"
        echo ""
    done
}

# ============================================================================
# SHOW RESULTS
# ============================================================================

show_results() {
    echo_step "Pipeline Complete!"
    echo ""
    echo_success "Results saved to: ${OUTPUTS_DIR}/<atlas>/"
    echo ""

    for ATLAS in "${ATLASES[@]}"; do
        ATLAS_OUT="${OUTPUTS_DIR}/${ATLAS}"
        echo_info "${ATLAS}:"
        for dir in small_worldness mixed_model_results umap_results; do
            if [[ -d "${ATLAS_OUT}/${dir}" ]]; then
                echo_info "  ✓ ${ATLAS_OUT}/${dir}/"
            fi
        done
    done
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo_header
    
    check_environment
    create_metadata
    run_pipeline
    show_results
    
    echo_header
}

main "$@"

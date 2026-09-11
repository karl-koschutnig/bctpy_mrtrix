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
GROUPS_EXCEL="${ROOT_DIR}/data/raw/ID_2w_4w_groups.xlsx"

# Overridable via environment, so an alternative analysis (e.g. a 3-arm
# collapse) can reuse this script with its own metadata + output tree.
METADATA_FILE="${METADATA_FILE:-${DATA_DIR}/metadata.csv}"
CONNECTOMES_DIR="${CONNECTOMES_DIR:-/Volumes/Evo/data/129/connectomics/bct_input}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${ROOT_DIR}/outputs/temporal_analysis}"

# Demographics: if this file exists, age/sex/height/weight are merged into the
# metadata and the covariate-adjusted robustness models are run.
DEMOGRAPHICS_FILE="${DEMOGRAPHICS_FILE:-${ROOT_DIR}/data/raw/participants_new_groups_02.04.25.xlsx}"
COVARIATE_COLS="${COVARIATE_COLS:-age sex}"
HUB_FRAC="${HUB_FRAC:-0.15}"

# Atlases to process (node counts confirmed from the real connectome files)
ATLASES=(AAL3 Gordon333 HCP-MMP Schaefer200 Schaefer400)
declare -A ATLAS_NODES=(
    [AAL3]=166
    [Gordon333]=333
    [HCP-MMP]=360
    [Schaefer200]=200
    [Schaefer400]=400
)

# Nodes to drop before computing graph metrics: regions with >25% of the real
# cohort's subject-sessions showing zero streamlines to any other region
# (confirmed via bct_input/<atlas>/*.count.roi_normalized.csv). Leaving these
# in makes characteristic path length infinite for most/all subjects. Recompute
# if the underlying tractography/atlas data changes.
declare -A ATLAS_EXCLUDE_NODES=(
    [AAL3]="128"
    [Gordon333]="16,124,185,279,284,287"
    [HCP-MMP]="211,212,213"
    [Schaefer200]=""
    [Schaefer400]=""
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
# MERGE DEMOGRAPHICS
# ============================================================================

HAS_DEMOGRAPHICS=false

merge_demographics_step() {
    if [[ ! -f "$DEMOGRAPHICS_FILE" ]]; then
        echo_info "No demographics file at ${DEMOGRAPHICS_FILE} - skipping age/sex covariates"
        echo ""
        return 0
    fi
    echo_step "Merging demographics"
    if python 7_temporal_analysis/scripts/merge_demographics.py \
        --demographics "$DEMOGRAPHICS_FILE" --metadata "$METADATA_FILE"; then
        HAS_DEMOGRAPHICS=true
        echo_success "age / sex merged into $METADATA_FILE"
    else
        echo_error "demographics merge failed - continuing without covariates"
    fi
    echo ""
}

# ============================================================================
# RUN TEMPORAL ANALYSIS PIPELINE
# ============================================================================

declare -A ATLAS_STATUS
declare -A ATLAS_STATUS_REASON

run_pipeline() {
    echo_step "Running Temporal Analysis Pipeline"

    for ATLAS in "${ATLASES[@]}"; do
        N_NODES="${ATLAS_NODES[$ATLAS]}"
        EXCLUDE_NODES="${ATLAS_EXCLUDE_NODES[$ATLAS]}"
        ATLAS_OUT="${OUTPUTS_DIR}/${ATLAS}"

        echo_step "Atlas: ${ATLAS} (${N_NODES} nodes)"

        if [[ ! -d "${CONNECTOMES_DIR}/${ATLAS}" ]]; then
            echo_error "Connectome data directory not found: ${CONNECTOMES_DIR}/${ATLAS}"
            echo_info "Set CONNECTOMES_DIR environment variable to your data location"
            exit 1
        fi

        # Step 1: Small-worldness calculation (degree-preserving null models)
        echo_info "Step 1/4: Calculating small-worldness..."
        SW_ARGS=(
            --data-dir "${CONNECTOMES_DIR}/${ATLAS}"
            --metadata-file "${METADATA_FILE}"
            --output-dir "${ATLAS_OUT}/small_worldness"
            --n-nodes "${N_NODES}"
        )
        if [[ -n "$EXCLUDE_NODES" ]]; then SW_ARGS+=(--exclude-nodes "$EXCLUDE_NODES"); fi
        if ! python 7_temporal_analysis/scripts/small_worldness.py "${SW_ARGS[@]}"; then
            echo_error "Small-worldness calculation failed for ${ATLAS}"
            ATLAS_STATUS[$ATLAS]="failed"
            ATLAS_STATUS_REASON[$ATLAS]="small-worldness calculation failed"
            echo ""
            continue
        fi
        echo_success "Small-worldness complete"

        # Step 2: Organizational measure battery (merges small-worldness in)
        echo_info "Step 2/4: Computing organizational measures..."
        OM_ARGS=(
            --data-dir "${CONNECTOMES_DIR}/${ATLAS}"
            --metadata-file "${METADATA_FILE}"
            --output-dir "${ATLAS_OUT}/organizational_measures"
            --n-nodes "${N_NODES}"
        )
        if [[ -n "$EXCLUDE_NODES" ]]; then OM_ARGS+=(--exclude-nodes "$EXCLUDE_NODES"); fi
        if ! python 7_temporal_analysis/scripts/organizational_measures.py "${OM_ARGS[@]}"; then
            echo_error "Organizational measures failed for ${ATLAS}"
            ATLAS_STATUS[$ATLAS]="failed"
            ATLAS_STATUS_REASON[$ATLAS]="organizational measures failed"
            echo ""
            continue
        fi
        echo_success "Organizational measures complete"

        OM_CSV="${ATLAS_OUT}/organizational_measures/organizational_measures_results.csv"

        # Step 2b: Hub / rich-club measures (focused hypothesis family). Merged
        # into the organizational-measures CSV so the models below pick them up.
        echo_info "Step 2b/8: Computing hub / rich-club measures..."
        HUB_ARGS=(
            --data-dir "${CONNECTOMES_DIR}/${ATLAS}"
            --metadata-file "${METADATA_FILE}"
            --output-dir "${ATLAS_OUT}/hub_measures"
            --n-nodes "${N_NODES}" --hub-frac "${HUB_FRAC}"
        )
        if [[ -n "$EXCLUDE_NODES" ]]; then HUB_ARGS+=(--exclude-nodes "$EXCLUDE_NODES"); fi
        if python 7_temporal_analysis/scripts/hub_measures.py "${HUB_ARGS[@]}"; then
            python - "${ATLAS_OUT}/hub_measures/hub_measures_results.csv" "${OM_CSV}" <<'PY'
import sys, pandas as pd
hub, om = sys.argv[1], sys.argv[2]
h = pd.read_csv(hub)
drop = ("participant_id", "session", "group", "age", "sex", "height_cm", "weight_kg")
hub_cols = [c for c in h.columns if c not in drop]
d = pd.read_csv(om).drop(columns=[c for c in hub_cols], errors="ignore")
d.merge(h[["participant_id", "session", *hub_cols]], on=["participant_id", "session"],
        how="left").to_csv(om, index=False)
PY
            echo_success "Hub measures complete (merged into organizational measures)"
        else
            echo_error "Hub measures failed for ${ATLAS} (continuing without them)"
        fi

        COV_ARGS=()
        if $HAS_DEMOGRAPHICS; then COV_ARGS=(--covariate-cols ${COVARIATE_COLS}); fi

        # Step 3: Mixed-effects models (Group x Time) over the full measure set
        if $HAS_R; then
            echo_info "Step 3/8: Fitting mixed-effects models (primary)..."
            if ! Rscript 7_temporal_analysis/scripts/mixed_models.R \
                --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/mixed_model_results" \
                --timepoint-col "${TIMEPOINT_COL}" \
                --group-col "${GROUP_COL}" \
                --participant-col participant_id; then
                echo_error "Mixed-effects modeling failed for ${ATLAS}"
                ATLAS_STATUS[$ATLAS]="failed"
                ATLAS_STATUS_REASON[$ATLAS]="mixed-effects modeling failed"
                echo ""
                continue
            fi
            echo_success "Mixed-effects models complete"

            # Step 4: Longitudinal SEM (latent growth curves) - complementary,
            # non-fatal: an inadmissible fit for one measure shouldn't sink the atlas.
            echo_info "Step 4/8: Fitting longitudinal SEM (primary)..."
            if ! Rscript 7_temporal_analysis/scripts/longitudinal_sem.R \
                --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/sem_results" \
                --timepoint-col "${TIMEPOINT_COL}" \
                --group-col "${GROUP_COL}" \
                --participant-col participant_id; then
                echo_error "Longitudinal SEM failed for ${ATLAS} (continuing)"
            else
                echo_success "Longitudinal SEM complete"
            fi

            # Step 5: Robustness models (non-fatal). Covariate-adjusted +
            # baseline-adjusted lme, covariate-adjusted + latent-basis SEM,
            # and the pre-post difference-score model.
            echo_info "Step 5/8: Robustness models (adjusted / baseline / latent-basis / difference)..."
            Rscript 7_temporal_analysis/scripts/mixed_models.R --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/mixed_model_results_adj" \
                --timepoint-col "${TIMEPOINT_COL}" --group-col "${GROUP_COL}" \
                --participant-col participant_id "${COV_ARGS[@]}" \
                || echo_error "  adjusted lme failed (continuing)"
            Rscript 7_temporal_analysis/scripts/mixed_models.R --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/mixed_model_results_baseline" \
                --timepoint-col "${TIMEPOINT_COL}" --group-col "${GROUP_COL}" \
                --participant-col participant_id --baseline-adjust "${COV_ARGS[@]}" \
                || echo_error "  baseline-adjusted lme failed (continuing)"
            Rscript 7_temporal_analysis/scripts/longitudinal_sem.R --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/sem_results_adj" \
                --timepoint-col "${TIMEPOINT_COL}" --group-col "${GROUP_COL}" \
                --participant-col participant_id "${COV_ARGS[@]}" \
                || echo_error "  adjusted SEM failed (continuing)"
            Rscript 7_temporal_analysis/scripts/longitudinal_sem.R --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/sem_results_latentbasis" \
                --timepoint-col "${TIMEPOINT_COL}" --group-col "${GROUP_COL}" \
                --participant-col participant_id --latent-basis "${COV_ARGS[@]}" \
                || echo_error "  latent-basis SEM failed (continuing)"
            Rscript 7_temporal_analysis/scripts/difference_scores.R --input-file "${OM_CSV}" \
                --output-dir "${ATLAS_OUT}/difference_scores" \
                --timepoint-col "${TIMEPOINT_COL}" --group-col "${GROUP_COL}" \
                --participant-col participant_id "${COV_ARGS[@]}" \
                || echo_error "  difference-score model failed (continuing)"
            echo_success "Robustness models complete"
        else
            echo_info "Skipping mixed-effects models + SEM (R not available)"
        fi

        # Step 6: UMAP projection (exploratory group x time visualization)
        echo_info "Step 6/8: Running UMAP projection..."
        if ! python 7_temporal_analysis/scripts/umap_projection.py \
            --input-file "${OM_CSV}" \
            --output-dir "${ATLAS_OUT}/umap_results" \
            --timepoint-col "${TIMEPOINT_COL}" \
            --group-col "${GROUP_COL}"; then
            echo_error "UMAP projection failed for ${ATLAS}"
            ATLAS_STATUS[$ATLAS]="failed"
            ATLAS_STATUS_REASON[$ATLAS]="UMAP projection failed"
            echo ""
            continue
        fi
        echo_success "UMAP projection complete"
        echo ""

        ATLAS_STATUS[$ATLAS]="succeeded"
    done

    # Step 7/8: FDR correction across atlases / measures / designs
    if $HAS_R || true; then
        echo_step "FDR correction"
        python 7_temporal_analysis/scripts/fdr_correct.py \
            --outputs-root "$(dirname "$OUTPUTS_DIR")" \
            || echo_error "FDR correction failed (continuing)"
    fi
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
        for dir in small_worldness organizational_measures hub_measures \
                   mixed_model_results mixed_model_results_adj mixed_model_results_baseline \
                   sem_results sem_results_adj sem_results_latentbasis \
                   difference_scores umap_results; do
            if [[ -d "${ATLAS_OUT}/${dir}" ]]; then
                echo_info "  ✓ ${ATLAS_OUT}/${dir}/"
            fi
        done
    done
    echo ""

    echo_step "Per-Atlas Summary"
    for ATLAS in "${ATLASES[@]}"; do
        case "${ATLAS_STATUS[$ATLAS]:-unknown}" in
            succeeded)
                echo_success "${ATLAS}: succeeded"
                ;;
            failed)
                echo_error "${ATLAS}: failed (${ATLAS_STATUS_REASON[$ATLAS]:-unknown reason})"
                ;;
            *)
                echo_info "${ATLAS}: not run"
                ;;
        esac
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
    merge_demographics_step
    run_pipeline
    show_results
    
    echo_header
}

main "$@"

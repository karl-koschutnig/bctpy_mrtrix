# Stage 7 Statistical Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `7_temporal_analysis`'s age-GAM/turning-point machinery
(statistically inappropriate for 3 discrete timepoints) with Group×Time
linear mixed-effects models, fix a hardcoded-node-count bug and a
group-blind averaging bug along the way, and run the whole stage across
all 5 available atlases.

**Architecture:** `small_worldness.py` and `umap_projection.py` get
targeted bug fixes (not rewrites). `gam_models.R` is replaced by
`mixed_models.R` (`lmer(metric ~ group*session + (1|participant_id))` +
`emmeans` pairwise contrasts), actually exercised by a real
subprocess-executed test instead of the current text-content-only stub.
`turning_points.py` and its logic inside `umap_projection.py` are retired
from the active pipeline (left in place, unused). `run_temporal_analysis.sh`
loops the whole stage across all 5 atlases.

**Tech Stack:** Python (bctpy/bct, umap-learn, sklearn), R (lme4,
lmerTest, emmeans — via the renv environment set up in the companion
plan), bash.

**Spec:** `docs/superpowers/specs/2026-09-10-temporal-analysis-pipeline-design.md`

## Global Constraints

- **Prerequisite:** `docs/superpowers/plans/2026-09-10-environment-and-data-wiring.md`
  must be complete first — this plan's Task 4 integration test needs the
  R/renv environment it sets up, and `run_temporal_analysis.sh`'s atlas
  loop (Task 5 here) needs the real `metadata.csv` it builds.
- Group labels `g1_2w`/`g1_4w`/`g2_2w`/`g2_4w` stay blinded — never
  decode `group=1` vs `group=2` to a real modality name.
- All 5 atlases must be processed: AAL3 (166 nodes), Gordon333 (333),
  HCP-MMP (360), Schaefer200 (200), Schaefer400 (400) — confirmed node
  counts from the real connectome files.
- No turning-point derivative/gradient logic may feed into any output the
  active pipeline produces — the method requires a continuous covariate
  with many distinct values, which a 3-timepoint design doesn't have.

---

### Task 1: Fix small_worldness.py's hardcoded 200-node validation

`calculate_small_worldness()` hardcodes `expected_nodes = 200` (a
Schaefer200 assumption) regardless of what `n_nodes` the caller passes,
so it raises `ValueError` on every atlas except Schaefer200 — a hard
blocker for the 5-atlas requirement.

**Files:**
- Modify: `7_temporal_analysis/scripts/small_worldness.py:188-262` (`calculate_small_worldness`)
- Modify: `7_temporal_analysis/scripts/small_worldness.py:352` (call site inside `process_connectome_directory`)
- Modify: `7_temporal_analysis/tests/test_small_worldness.py`

**Interfaces:**
- Produces: `calculate_small_worldness(W, n_null=100, random_state=42, expected_nodes=None)` — `expected_nodes=None` skips the node-count check entirely (used when the caller doesn't know/care); `process_connectome_directory` now passes its own `n_nodes` through as `expected_nodes`.

- [ ] **Step 1: Write the failing tests**

Add to `7_temporal_analysis/tests/test_small_worldness.py`:

```python
def test_calculate_small_worldness_accepts_non_schaefer_atlas():
    """A 166x166 (AAL3) matrix must not be rejected by a hardcoded 200-node check."""
    rng = np.random.default_rng(0)
    W = rng.random((166, 166))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    # Must not raise, even though 166 != the old hardcoded 200
    calculate_small_worldness(W, n_null=2, random_state=0, expected_nodes=166)


def test_calculate_small_worldness_still_validates_expected_nodes():
    """When a mismatched expected_nodes is passed, it should still be caught."""
    rng = np.random.default_rng(0)
    W = rng.random((166, 166))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    with pytest.raises(ValueError, match="200x200"):
        calculate_small_worldness(W, n_null=2, expected_nodes=200)
```

(add `import numpy as np` and `import pytest` at the top of the test file
if not already present, and import `calculate_small_worldness` from the
existing import block used by the other tests in that file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest 7_temporal_analysis/tests/test_small_worldness.py -k "accepts_non_schaefer or still_validates" -v`
Expected: `test_calculate_small_worldness_accepts_non_schaefer_atlas` FAILS
with `ValueError: Matrix must be 200x200, got (166, 166)`.

- [ ] **Step 3: Fix the implementation**

In `7_temporal_analysis/scripts/small_worldness.py`, replace the function
signature and validation block (lines 188-224) —

```python
def calculate_small_worldness(W, n_null=100, random_state=42):
```
```python
    n_nodes = W.shape[0]
    
    # Validate matrix
    n_nodes = W.shape[0]
    expected_nodes = 200  # Schaefer200
    
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix must be square, got {W.shape}")
    
    if n_nodes != expected_nodes:
        raise ValueError(f"Matrix must be {expected_nodes}x{expected_nodes}, got {W.shape}")
```

— with:

```python
def calculate_small_worldness(W, n_null=100, random_state=42, expected_nodes=None):
```
```python
    n_nodes = W.shape[0]

    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix must be square, got {W.shape}")

    if expected_nodes is not None and n_nodes != expected_nodes:
        raise ValueError(f"Matrix must be {expected_nodes}x{expected_nodes}, got {W.shape}")
```

Then update the call site inside `process_connectome_directory` (line
352):
```python
            sigma = calculate_small_worldness(W, n_null=n_null, random_state=random_state)
```
to:
```python
            sigma = calculate_small_worldness(
                W, n_null=n_null, random_state=random_state, expected_nodes=n_nodes
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest 7_temporal_analysis/tests/test_small_worldness.py -v`
Expected: all PASS, including the whole pre-existing file (no
regressions — the default `expected_nodes=None` preserves old
callers that never passed it).

- [ ] **Step 5: Commit**

```bash
git add 7_temporal_analysis/scripts/small_worldness.py 7_temporal_analysis/tests/test_small_worldness.py
git commit -m "$(cat <<'EOF'
Fix small_worldness.py's hardcoded 200-node validation

calculate_small_worldness() ignored its own n_nodes parameter and
always validated against a hardcoded Schaefer200 assumption, rejecting
every other atlas. expected_nodes is now optional and threaded through
from the actual data being processed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix umap_projection.py — group-aware averaging, retire turning points

Two bugs/scope issues in `process_umap_projection`:
1. `calculate_timepoint_averages` groups by `session` only, silently
   averaging away the group signal — defeating the entire point of an
   intervention-study trajectory plot.
2. Turning-point detection (`detect_turning_points` + `turning_points.json`
   output) is baked into the main processing function; per the design
   decision, this needs to stop feeding the active pipeline's output.

**Files:**
- Modify: `7_temporal_analysis/scripts/umap_projection.py:148-168` (`calculate_timepoint_averages`)
- Modify: `7_temporal_analysis/scripts/umap_projection.py:352-462` (`process_umap_projection`)
- Modify: `7_temporal_analysis/scripts/umap_projection.py:469-536` (`main`)
- Modify: `7_temporal_analysis/tests/test_umap_projection.py`

**Interfaces:**
- Produces: `calculate_timepoint_averages(data, timepoint_col='session', group_col=None, metric_cols=None)` — groups by `[group_col, timepoint_col]` when `group_col` is given, else by `timepoint_col` alone (unchanged behavior). `process_umap_projection(..., group_col=None, ...)` returns `(data, averages)` — a 2-tuple, not the previous 3-tuple with turning points.

- [ ] **Step 1: Write the failing tests**

Add to `7_temporal_analysis/tests/test_umap_projection.py`:

```python
def test_calculate_timepoint_averages_is_group_aware():
    """Averaging must not collapse across groups - that erases the group signal."""
    data = pd.DataFrame({
        "session": ["ses-1", "ses-1", "ses-1", "ses-1"],
        "group":   ["ctrl", "ctrl", "g1_2w", "g1_2w"],
        "density": [0.10, 0.20, 0.50, 0.60],
    })

    averages = calculate_timepoint_averages(data, timepoint_col="session", group_col="group")

    assert len(averages) == 2  # one row per (group, session), not one row per session
    ctrl_row = averages[averages["group"] == "ctrl"].iloc[0]
    g1_row = averages[averages["group"] == "g1_2w"].iloc[0]
    assert ctrl_row["density"] == pytest.approx(0.15)
    assert g1_row["density"] == pytest.approx(0.55)


def test_process_umap_projection_does_not_emit_turning_points(tmp_path):
    """Turning-point detection is retired from the active pipeline."""
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "participant_id": [f"sub-{i:03d}" for i in range(n)],
        "session": np.tile(["ses-1", "ses-2", "ses-3"], n // 3),
        "group": np.tile(["ctrl", "g1_2w", "g2_4w"], n // 3),
        "density": rng.random(n),
        "global_efficiency": rng.random(n),
    })
    input_file = tmp_path / "metrics.csv"
    df.to_csv(input_file, index=False)
    output_dir = tmp_path / "out"

    data, averages = process_umap_projection(
        str(input_file), str(output_dir),
        timepoint_col="session", group_col="group",
        n_components=2, n_neighbors=5,
    )

    assert not (output_dir / "turning_points.json").exists()
    assert "group" in averages.columns
    assert set(averages["session"].unique()) == {"ses-1", "ses-2", "ses-3"}
```

(add `import numpy as np` and `import pandas as pd` at the top of the
test file if not already present.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest 7_temporal_analysis/tests/test_umap_projection.py -k "group_aware or does_not_emit_turning" -v`
Expected: `test_calculate_timepoint_averages_is_group_aware` FAILS
(`TypeError: calculate_timepoint_averages() got an unexpected keyword
argument 'group_col'`), and the turning-points test FAILS
(`too many values to unpack` or the file still exists).

- [ ] **Step 3: Fix `calculate_timepoint_averages`**

Replace lines 148-168 with:

```python
def calculate_timepoint_averages(data, timepoint_col='session', group_col=None, metric_cols=None):
    """
    Calculate average metrics for each (group, timepoint) combination.
    
    Args:
        data: DataFrame with metrics and timepoint (and optionally group) columns
        timepoint_col: name of the timepoint column
        group_col: name of the group column; if given, averages are computed
            per (group, timepoint) instead of collapsing across groups
        metric_cols: list of metric columns to average (None = all numeric columns)
    
    Returns:
        averages: DataFrame with one row per (group, timepoint) [or per
            timepoint if group_col is None], containing average metrics
    """
    group_keys = [group_col, timepoint_col] if group_col else [timepoint_col]

    if metric_cols is None:
        exclude = set(group_keys)
        metric_cols = [col for col in data.select_dtypes(include=[np.number]).columns
                      if col not in exclude]

    averages = data.groupby(group_keys)[metric_cols].mean().reset_index()
    return averages
```

- [ ] **Step 4: Fix `process_umap_projection`**

Replace lines 352-462 with:

```python
def process_umap_projection(input_file, output_dir, timepoint_col='session',
                           group_col=None, metric_cols=None, n_components=2, n_neighbors=15,
                           min_dist=0.1, random_state=42, color_col=None):
    """
    Main processing function for UMAP projection.
    
    Args:
        input_file: path to input CSV/parquet file with metrics
        output_dir: directory to save outputs
        timepoint_col: column name for timepoints
        group_col: column name for groups; when given, trajectory averages
            are computed per (group, timepoint) instead of collapsing
            across groups
        metric_cols: list of metric columns to use (None = auto-detect)
        n_components: number of UMAP dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: random seed
        color_col: column to use for coloring in plots (defaults to group_col or timepoint_col)
    
    Returns:
        (data, averages): DataFrames with per-subject and per-(group,timepoint)
            UMAP coordinates respectively. Turning-point detection is not run
            here - it requires a continuous covariate with many distinct
            values, which a fixed-timepoint design doesn't have.
    """
    # Load data
    if input_file.endswith('.parquet'):
        data = pd.read_parquet(input_file)
    else:
        data = pd.read_csv(input_file)
    
    # Auto-detect metric columns
    if metric_cols is None:
        exclude_cols = ['participant_id', 'session', 'group', 'age', 'sex', timepoint_col]
        metric_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Run UMAP
    embedding = run_umap(
        data[metric_cols],
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    # Add UMAP coordinates to data
    for i in range(n_components):
        data[f'umap_{i+1}'] = embedding[:, i]
    
    # Calculate group x timepoint averages (exploratory trajectory visualization)
    group_keys = [group_col, timepoint_col] if group_col else [timepoint_col]
    averages = calculate_timepoint_averages(
        data[metric_cols + group_keys],
        timepoint_col=timepoint_col,
        group_col=group_col,
        metric_cols=metric_cols
    )
    
    # Add average UMAP coordinates per (group, timepoint)
    for i in range(n_components):
        umap_avgs = data.groupby(group_keys)[f'umap_{i+1}'].mean().reset_index()
        averages = averages.merge(umap_avgs, on=group_keys, how='left')
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save UMAP coordinates
    umap_coords_file = output_dir / "umap_coordinates.csv"
    data.to_csv(umap_coords_file, index=False)
    
    # Save parquet if available
    try:
        data.to_parquet(output_dir / "umap_coordinates.parquet", index=False)
    except ImportError:
        pass
    
    # Save group x timepoint averages
    averages_file = output_dir / "umap_group_timepoint_averages.csv"
    averages.to_csv(averages_file, index=False)
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        # Plot all points
        plot_umap_trajectory(
            data,
            umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
            color_col=color_col or group_col or timepoint_col,
            output_path=output_dir / "umap_trajectory_all.png",
            show_labels=False
        )
        
        # Plot timepoint-average trajectory with arrows, one plot per group if available
        if group_col and group_col in averages.columns:
            for group_value, group_df in averages.groupby(group_col):
                plot_timepoint_averages(
                    group_df,
                    umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
                    color_col=timepoint_col,
                    output_path=output_dir / f"umap_trajectory_averages_{group_value}.png"
                )
        else:
            plot_timepoint_averages(
                averages,
                umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
                color_col=color_col or timepoint_col,
                output_path=output_dir / "umap_trajectory_averages.png"
            )
    
    return data, averages
```

Note: `detect_turning_points` stays defined in the file (unused by this
function) per the design decision to retire, not delete, the logic.

- [ ] **Step 5: Fix `main()`'s call site and prints**

Replace the call block inside `main()` (originally around lines 512-531):

```python
    # Run UMAP
    data, averages, turning_points = process_umap_projection(
        input_file=args.input_file,
        output_dir=args.output_dir,
        timepoint_col=args.timepoint_col,
        color_col=args.group_col,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )
    
    print(f"UMAP projection complete")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {len(data)}")
    print(f"Dimensions: {args.n_components}")
    print(f"Timepoints: {data[args.timepoint_col].nunique()}")
    print(f"Turning points detected: {len(turning_points)}")
    print(f"Turning timepoints: {turning_points}")
```

with:

```python
    # Run UMAP
    data, averages = process_umap_projection(
        input_file=args.input_file,
        output_dir=args.output_dir,
        timepoint_col=args.timepoint_col,
        group_col=args.group_col,
        color_col=args.group_col,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )
    
    print(f"UMAP projection complete")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {len(data)}")
    print(f"Dimensions: {args.n_components}")
    print(f"Timepoints: {data[args.timepoint_col].nunique()}")
    if args.group_col in data.columns:
        print(f"Groups: {data[args.group_col].nunique()}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest 7_temporal_analysis/tests/test_umap_projection.py -v`
Expected: all PASS, including any pre-existing tests that called
`process_umap_projection` without `group_col` (still works — defaults to
the old timepoint-only behavior) or that unpacked its old 3-tuple return
(these will need updating to the 2-tuple — if any exist, fix the unpack
in that test to `data, averages = process_umap_projection(...)`).

- [ ] **Step 7: Commit**

```bash
git add 7_temporal_analysis/scripts/umap_projection.py 7_temporal_analysis/tests/test_umap_projection.py
git commit -m "$(cat <<'EOF'
Fix umap_projection.py: group-aware averaging, retire turning points

calculate_timepoint_averages() was collapsing across groups, erasing
the group signal from trajectory plots. Turning-point detection
(designed for a continuous covariate with many distinct values) is
retired from the active pipeline per the design decision - the
function stays defined but unused.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Replace gam_models.R with mixed_models.R

Swaps the age-based GAM (never statistically meaningful with 3 discrete
timepoints) for a Group×Time linear mixed-effects model with a subject
random intercept — the standard approach for repeated-measures
intervention trials — plus `emmeans` pairwise group contrasts per
timepoint.

**Files:**
- Create: `7_temporal_analysis/scripts/mixed_models.R`
- Delete: `7_temporal_analysis/scripts/gam_models.R`
- Modify: `7_temporal_analysis/config/run_spec_temporal.json`

- [ ] **Step 1: Write mixed_models.R**

```r
#!/usr/bin/env Rscript
#' Linear Mixed-Effects Models for Network Metrics
#' ================================================
#'
#' Fits Group x Time linear mixed-effects models (subject random
#' intercept) to network metrics from a repeated-measures intervention
#' study, and computes pairwise group contrasts per timepoint via
#' emmeans.
#'
#' Replaces the age-based GAM approach used by the source
#' lifespan-turning-points pipeline, which requires a continuous
#' covariate with many distinct values and does not apply to a
#' 3-timepoint design.
#'
#' Usage:
#'   Rscript mixed_models.R --input-file metrics.csv --output-dir outputs/mixed_model_results \
#'     --timepoint-col session --group-col group --participant-col participant_id

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
  library(argparse)
})

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("lme4 package required for mixed-effects modeling. Install with: install.packages('lme4')")
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_args <- function() {
  parser <- argparse::ArgumentParser(description = "Group x Time mixed-effects modeling for network metrics")

  parser$add_argument("--input-file", type = "character", required = TRUE,
                     help = "Input CSV file with network metrics")
  parser$add_argument("--output-dir", type = "character", default = "outputs/mixed_model_results",
                     help = "Directory to save model results")
  parser$add_argument("--timepoint-col", type = "character", default = "session",
                     help = "Column name for timepoints")
  parser$add_argument("--group-col", type = "character", default = "group",
                     help = "Column name for groups")
  parser$add_argument("--participant-col", type = "character", default = "participant_id",
                     help = "Column name for the subject identifier (random effect grouping)")
  parser$add_argument("--config", type = "character", default = NULL,
                     help = "Path to configuration JSON file")
  parser$add_argument("--metric-cols", type = "character", nargs = "+", default = NULL,
                     help = "List of metric columns to model")

  parser$parse_args()
}


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

load_config <- function(config_path) {
  if (is.null(config_path)) return(list())
  if (!file.exists(config_path)) stop(paste("Configuration file not found:", config_path))
  fromJSON(config_path)
}


# ============================================================================
# LOAD DATA
# ============================================================================

load_data <- function(input_file) {
  if (grepl("\\.csv$", input_file)) {
    data <- read_csv(input_file, show_col_types = FALSE)
  } else if (grepl("\\.parquet$", input_file)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("arrow package required for parquet support")
    }
    data <- arrow::read_parquet(input_file)
  } else {
    stop(paste("Unsupported file format:", input_file))
  }
  data
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

prepare_data <- function(data, timepoint_col, group_col, participant_col, metric_cols = NULL) {
  data <- data %>%
    mutate(
      !!sym(timepoint_col)   := factor(!!sym(timepoint_col)),
      !!sym(group_col)       := factor(!!sym(group_col)),
      !!sym(participant_col) := factor(!!sym(participant_col))
    )

  if (is.null(metric_cols)) {
    metadata_cols <- c(timepoint_col, group_col, participant_col, "age", "sex")
    metadata_cols <- metadata_cols[metadata_cols %in% names(data)]
    metric_cols <- setdiff(names(data)[sapply(data, is.numeric)], metadata_cols)
  }
  metric_cols <- metric_cols[sapply(data[metric_cols], is.numeric)]

  list(data = data, metric_cols = metric_cols)
}


# ============================================================================
# FIT MIXED-EFFECTS MODELS
# ============================================================================

#' Fit metric ~ group * session + (1 | participant) for one metric.
fit_mixed_model <- function(data, group_col, timepoint_col, participant_col, metric_name) {
  formula <- as.formula(
    paste0(metric_name, " ~ ", group_col, " * ", timepoint_col, " + (1 | ", participant_col, ")")
  )
  model <- lmerTest::lmer(formula, data = data)
  contrasts <- emmeans::emmeans(
    model, specs = as.formula(paste0("pairwise ~ ", group_col, " | ", timepoint_col))
  )

  list(model = model, anova = anova(model), contrasts = contrasts, metric = metric_name)
}


#' Fit mixed models for all metrics, skipping (with a warning) any that fail to converge.
fit_all_mixed_models <- function(data, group_col, timepoint_col, participant_col, metric_cols) {
  results <- list()
  for (metric in metric_cols) {
    cat("Fitting mixed model for:", metric, "...\n")
    result <- tryCatch(
      fit_mixed_model(data, group_col, timepoint_col, participant_col, metric),
      error = function(e) {
        warning(paste("Failed to fit mixed model for", metric, ":", e$message))
        NULL
      }
    )
    results[[metric]] <- result
    if (!is.null(result)) cat("  ✓ Model fitted successfully\n")
  }
  results
}


# ============================================================================
# EXTRACT RESULTS
# ============================================================================

#' Extract the Type III ANOVA table (F, df, p) for group, session, group:session.
extract_model_stats <- function(model_result) {
  if (is.null(model_result)) return(NULL)

  anova_tbl <- as.data.frame(model_result$anova)
  anova_tbl$term <- rownames(anova_tbl)
  anova_tbl$metric <- model_result$metric
  rownames(anova_tbl) <- NULL
  anova_tbl
}


#' Extract pairwise group contrasts per timepoint as a tidy data frame.
extract_contrasts <- function(model_result) {
  if (is.null(model_result)) return(NULL)

  contrasts_df <- as.data.frame(model_result$contrasts$contrasts)
  contrasts_df$metric <- model_result$metric
  contrasts_df
}


# ============================================================================
# SAVE RESULTS
# ============================================================================

save_results <- function(results, output_dir, metric_cols) {
  output_dir <- path.expand(output_dir)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  all_stats <- list()
  all_contrasts <- list()

  for (metric in metric_cols) {
    if (!is.null(results[[metric]])) {
      stats <- extract_model_stats(results[[metric]])
      if (!is.null(stats)) all_stats[[metric]] <- stats

      contrasts <- extract_contrasts(results[[metric]])
      if (!is.null(contrasts)) all_contrasts[[metric]] <- contrasts
    }
  }

  if (length(all_stats) > 0) {
    write_csv(bind_rows(all_stats), file.path(output_dir, "mixed_model_statistics.csv"))
  }
  if (length(all_contrasts) > 0) {
    write_csv(bind_rows(all_contrasts), file.path(output_dir, "emmeans_contrasts.csv"))
  }

  saveRDS(results, file.path(output_dir, "mixed_models.RDS"))
  cat("Results saved to:", output_dir, "\n")
}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

main <- function() {
  args <- parse_args()
  config <- load_config(args$config)

  if (!is.null(config$temporal$timepoint_col))  args$timepoint_col <- config$temporal$timepoint_col
  if (!is.null(config$temporal$group_col))      args$group_col     <- config$temporal$group_col
  if (!is.null(config$mixed_model$metric_cols)) args$metric_cols   <- config$mixed_model$metric_cols

  cat("Loading data from:", args$input_file, "\n")
  data <- load_data(args$input_file)
  cat("Loaded", nrow(data), "rows\n")

  prepared <- prepare_data(data, args$timepoint_col, args$group_col, args$participant_col, args$metric_cols)
  data <- prepared$data
  metric_cols <- prepared$metric_cols
  cat("Metrics to model:", paste(metric_cols, collapse = ", "), "\n")

  results <- fit_all_mixed_models(data, args$group_col, args$timepoint_col, args$participant_col, metric_cols)
  save_results(results, args$output_dir, metric_cols)

  cat("\n=== MIXED MODEL SUMMARY ===\n")
  cat("Timepoints:", nlevels(data[[args$timepoint_col]]), "\n")
  cat("Groups:", nlevels(data[[args$group_col]]), "\n")
  cat("Metrics modeled:", sum(sapply(results, Negate(is.null))), "/", length(metric_cols), "\n")
}

if (!interactive()) {
  main()
}
```

- [ ] **Step 2: Delete gam_models.R**

```bash
rm 7_temporal_analysis/scripts/gam_models.R
```

- [ ] **Step 3: Update run_spec_temporal.json**

In `7_temporal_analysis/config/run_spec_temporal.json`, remove `"atlas"`
and `"n_nodes"` from the `"temporal"` section (the 5-atlas loop added in
Task 5 passes both explicitly via CLI flags per iteration — leaving stale
values here would silently override those CLI flags, since
`small_worldness.py`'s config-loading always takes the config value over
the CLI default when the key is present).

Replace:
```json
    "atlas": "Schaefer200",
    "n_nodes": 200
```
(inside `"temporal": { ... }`) with nothing (just remove those two lines,
keep the rest of `"temporal"` as-is).

Then replace the `"gam"` section:
```json
  "gam": {
    "smooth_terms": ["session"],
    "fixed_effects": ["group", "sex", "age"],
    "family": "gaussian"
  }
```
with:
```json
  "mixed_model": {
    "fixed_effects": ["group", "session"],
    "interaction": true,
    "random_effect": "participant_id"
  }
```

- [ ] **Step 4: Verify the script runs standalone against a tiny fixture**

```bash
mkdir -p /tmp/mixed_models_smoke
python3 - <<'EOF'
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
rows = []
for i in range(15):
    for ses in ["ses-1", "ses-2", "ses-3"]:
        rows.append({
            "participant_id": f"sub-{i:03d}",
            "session": ses,
            "group": ["ctrl", "g1_2w", "g2_4w"][i % 3],
            "density": rng.random(),
        })
pd.DataFrame(rows).to_csv("/tmp/mixed_models_smoke/metrics.csv", index=False)
EOF
Rscript 7_temporal_analysis/scripts/mixed_models.R \
    --input-file /tmp/mixed_models_smoke/metrics.csv \
    --output-dir /tmp/mixed_models_smoke/out \
    --timepoint-col session --group-col group --participant-col participant_id
```

Expected: exits 0, prints `=== MIXED MODEL SUMMARY ===` with `Metrics
modeled: 1 / 1`, and `/tmp/mixed_models_smoke/out/mixed_model_statistics.csv`
+ `emmeans_contrasts.csv` exist. (Requires the companion plan's renv
environment to already be set up.)

- [ ] **Step 5: Commit**

```bash
git add 7_temporal_analysis/scripts/mixed_models.R 7_temporal_analysis/config/run_spec_temporal.json
git rm 7_temporal_analysis/scripts/gam_models.R
git commit -m "$(cat <<'EOF'
Replace gam_models.R with mixed_models.R (Group x Time mixed-effects)

The age-GAM approach requires a continuous covariate with many distinct
values (the source pipeline's 91-age, n~3800 lifespan sample) - it isn't
statistically meaningful for 3 discrete timepoints. mixed_models.R fits
lmer(metric ~ group*session + (1|participant_id)) per metric, with
emmeans pairwise group contrasts per timepoint.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add a real, executed integration test for mixed_models.R

The existing `test_gam_models.py` never actually runs R — it only checks
that `gam_models.R`'s file contents include certain substrings. Replace
it with a test that actually executes `mixed_models.R` via `Rscript` and
checks its output.

**Files:**
- Create: `7_temporal_analysis/tests/test_mixed_models.py`
- Delete: `7_temporal_analysis/tests/test_gam_models.py`

- [ ] **Step 1: Write the test**

```python
"""
Tests for mixed_models.R.

Actually executes the script via Rscript - the model logic (a Group x
Time interaction with a subject random intercept, not an age smooth)
only means anything when it actually runs and produces the expected
output files. Skipped automatically if Rscript isn't on PATH.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "mixed_models.R"

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="Rscript not available"
)


@pytest.fixture
def sample_metrics_csv(tmp_path):
    """20 subjects x 3 sessions across 5 study-arm labels."""
    rng = np.random.default_rng(42)
    groups = ["ctrl", "g1_2w", "g1_4w", "g2_2w", "g2_4w"]
    rows = []
    for i in range(20):
        participant = f"sub-{i:03d}"
        group = groups[i % len(groups)]
        baseline = rng.normal(0.3, 0.05)
        for session_idx, session in enumerate(["ses-1", "ses-2", "ses-3"]):
            rows.append({
                "participant_id": participant,
                "session": session,
                "group": group,
                "density": baseline + 0.01 * session_idx + rng.normal(0, 0.01),
                "global_efficiency": baseline * 2 + rng.normal(0, 0.02),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "metrics.csv"
    df.to_csv(path, index=False)
    return path


def _run_mixed_models(input_file, output_dir):
    return subprocess.run(
        [
            "Rscript", str(SCRIPT_PATH),
            "--input-file", str(input_file),
            "--output-dir", str(output_dir),
            "--timepoint-col", "session",
            "--group-col", "group",
            "--participant-col", "participant_id",
        ],
        capture_output=True, text=True,
    )


def test_mixed_models_script_exists():
    assert SCRIPT_PATH.exists(), f"mixed_models.R not found at {SCRIPT_PATH}"


def test_mixed_models_runs_and_writes_statistics(sample_metrics_csv, tmp_path):
    output_dir = tmp_path / "outputs"

    result = _run_mixed_models(sample_metrics_csv, output_dir)

    assert result.returncode == 0, f"mixed_models.R failed:\n{result.stdout}\n{result.stderr}"

    stats_path = output_dir / "mixed_model_statistics.csv"
    assert stats_path.exists()

    stats = pd.read_csv(stats_path)
    assert set(stats["metric"].unique()) == {"density", "global_efficiency"}
    # Type III ANOVA table must include the group:session interaction term
    assert any("group" in t and "session" in t for t in stats["term"])


def test_mixed_models_writes_emmeans_contrasts(sample_metrics_csv, tmp_path):
    output_dir = tmp_path / "outputs"

    result = _run_mixed_models(sample_metrics_csv, output_dir)
    assert result.returncode == 0, f"mixed_models.R failed:\n{result.stdout}\n{result.stderr}"

    contrasts_path = output_dir / "emmeans_contrasts.csv"
    assert contrasts_path.exists()
    contrasts = pd.read_csv(contrasts_path)
    assert "contrast" in contrasts.columns
    assert set(contrasts["metric"].unique()) == {"density", "global_efficiency"}
```

- [ ] **Step 2: Delete the old text-content-only test**

```bash
rm 7_temporal_analysis/tests/test_gam_models.py
```

- [ ] **Step 3: Run the test**

Run: `pytest 7_temporal_analysis/tests/test_mixed_models.py -v`
Expected: all 3 PASS (requires the companion plan's renv environment;
if `Rscript` isn't on PATH the whole file is cleanly skipped rather than
erroring).

- [ ] **Step 4: Commit**

```bash
git add 7_temporal_analysis/tests/test_mixed_models.py
git rm 7_temporal_analysis/tests/test_gam_models.py
git commit -m "$(cat <<'EOF'
Add a real subprocess-executed test for mixed_models.R

test_gam_models.py only ever checked the R file's text content, never
executed it. test_mixed_models.py runs mixed_models.R via Rscript
against a synthetic 5-arm repeated-measures fixture and checks the
actual output files.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Run the whole stage across all 5 atlases, retire turning_points.py

Wraps `run_temporal_analysis.sh`'s pipeline steps in a loop over all 5
atlases (each with its correct node count), drops the standalone
turning-point-detection step, and updates references from
`gam_models.R` to `mixed_models.R`.

**Files:**
- Modify: `0_installation/run_temporal_analysis.sh` (CONFIGURATION section, `check_environment()`, `run_pipeline()`, `show_results()`)
- Modify: `7_temporal_analysis/README.md`

- [ ] **Step 1: Replace the single-atlas config with a 5-atlas map**

In `0_installation/run_temporal_analysis.sh`, replace the "Atlas
settings" block:
```bash
# Atlas settings
ATLAS="Schaefer200"
N_NODES=200
```
with:
```bash
# Atlases to process (node counts confirmed from the real connectome files)
ATLASES=(AAL3 Gordon333 HCP-MMP Schaefer200 Schaefer400)
declare -A ATLAS_NODES=(
    [AAL3]=166
    [Gordon333]=333
    [HCP-MMP]=360
    [Schaefer200]=200
    [Schaefer400]=400
)
```

- [ ] **Step 2: Update the R-missing hint text in check_environment()**

Replace:
```bash
    else
        echo_info "R not found - GAM modeling will be skipped"
        echo_info "To install R: brew install r"
        echo_info "Then: Rscript -e \"install.packages(c('mgcv', 'tidyverse', 'jsonlite', 'argparse'), repos='https://cloud.r-project.org/')\""
    fi
```
with:
```bash
    else
        echo_info "R not found - mixed-effects modeling will be skipped"
        echo_info "To install R: brew install r"
        echo_info "Then set up packages: Rscript 0_installation/setup_r_env.R"
    fi
```

- [ ] **Step 3: Replace run_pipeline() with the 5-atlas loop**

Replace the entire `run_pipeline()` function with:

```bash
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
```

- [ ] **Step 4: Replace show_results() for the per-atlas layout**

Replace the entire `show_results()` function with:

```bash
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
```

- [ ] **Step 5: Update the header comment block**

At the top of the file, replace:
```bash
# Requirements:
#   - Python environment activated (run install.sh first)
#   - R installed for GAM modeling
#   - Metadata file created
```
with:
```bash
# Requirements:
#   - Python environment activated (run install.sh first)
#   - R installed for mixed-effects modeling (via renv, see 0_installation/setup_r_env.R)
#   - Metadata file created (see data_processing/build_metadata.py)
```

- [ ] **Step 6: Run the full pipeline against real data**

```bash
source .venv/bin/activate
export CONNECTOMES_DIR=/Volumes/Evo/data/129/connectomics/bct_input
./0_installation/run_temporal_analysis.sh
```

Expected: exits 0; runs all 5 atlases in sequence; ends with "Pipeline
Complete!" listing `small_worldness/`, `mixed_model_results/`, and
`umap_results/` present under each of the 5
`outputs/temporal_analysis/<atlas>/` directories.

- [ ] **Step 7: Update 7_temporal_analysis/README.md**

Replace the whole file with:

```markdown
# 7. Temporal Analysis: Group x Time Network Trajectories

This module analyzes how graph-theory network metrics change across the
study's 3 timepoints and 5 study arms, for each of 5 parcellation
atlases.

## Overview

1. **Small-Worldness Calculation** - Network topology metric (sigma), per atlas
2. **Group x Time Mixed-Effects Models** - `lmer(metric ~ group*session + (1|participant_id))`,
   with `emmeans` pairwise group contrasts per timepoint
3. **UMAP Dimensionality Reduction** - Exploratory visualization of
   per-group network-state trajectories across timepoints

Adapted from the small-worldness/GAM/UMAP/turning-points chain in
`lifespan_topological_turning_points` (Mousley et al. 2025), but the
age-based GAM and turning-point detection are replaced by mixed-effects
models: the source method requires a continuous covariate with many
distinct values (91 ages, n~3800 cross-sectional subjects) to have
statistical meaning, and doesn't apply to a 3-timepoint repeated-measures
design. `turning_points.py` is kept in the repo but not called by the
active pipeline.

## Study Design

- **Data**: `/Volumes/Evo/data/129/connectomics/bct_input/<atlas>`
- **Atlases**: AAL3 (166 nodes), Gordon333 (333), HCP-MMP (360),
  Schaefer200 (200), Schaefer400 (400) - all 5 processed by default
- **Timepoints**: 3 (ses-1, ses-2, ses-3)
- **Groups**: 5 study arms - `ctrl`, `g1_2w`, `g1_4w`, `g2_2w`, `g2_4w`.
  `g1` vs `g2` is deliberately kept blinded (double-blind analysis); `2w`/`4w`
  is duration.
- **Subjects**: 126

## Directory Structure

```
7_temporal_analysis/
├── scripts/
│   ├── small_worldness.py      # Calculate small-worldness (sigma)
│   ├── mixed_models.R          # Group x Time mixed-effects models
│   ├── umap_projection.py      # UMAP dimensionality reduction
│   └── turning_points.py       # Retired from the active pipeline; not called
├── tests/
├── config/
│   └── run_spec_temporal.json
├── outputs/
│   └── <atlas>/
│       ├── small_worldness/
│       ├── mixed_model_results/
│       └── umap_results/
└── README.md
```

## Requirements

See `0_installation/README.md` — `install.sh` provisions both the Python
(`uv sync`) and R (`renv`) sides automatically.

## Usage

```bash
export CONNECTOMES_DIR=/path/to/your/bct_input
./0_installation/run_temporal_analysis.sh
```

Runs all 5 atlases in sequence; per-atlas output lands under
`outputs/temporal_analysis/<atlas>/`.
```

- [ ] **Step 8: Commit**

```bash
git add 0_installation/run_temporal_analysis.sh 7_temporal_analysis/README.md
git commit -m "$(cat <<'EOF'
Run the temporal analysis pipeline across all 5 atlases

run_temporal_analysis.sh now loops small-worldness -> mixed-effects
models -> UMAP over AAL3/Gordon333/HCP-MMP/Schaefer200/Schaefer400 with
their correct node counts, and drops the standalone turning-point
detection step. References updated from gam_models.R to mixed_models.R.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

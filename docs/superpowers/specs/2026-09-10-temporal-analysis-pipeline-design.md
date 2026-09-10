# Temporal analysis pipeline rework — design spec

Date: 2026-09-10
Status: approved by user, pending implementation plan

## Context

`bctpy_mrtrix` is a Python (bctpy) reimplementation of the analysis pipeline
from Mousley et al. 2025, *"Topological turning points across the human
lifespan"* (source repo:
`github.com/alexamousley/lifespan_topological_turning_points`). Stages 1–6
of `bctpy_mrtrix` already adapted that pipeline sensibly for a repeated-
measures design (SVM group/time classifiers, responder analysis, sex
interactions). Stage 7 (`7_temporal_analysis/`) is a newer, more literal
port of the source repo's small-worldness → GAM → UMAP → turning-points
chain, and is the focus of this rework.

The target dataset is a real intervention study: 126 subjects with
connectomes at 3 timepoints (`ses-1/2/3`), across 5 atlases (AAL3,
Gordon333, HCP-MMP, Schaefer200, Schaefer400), under 5 study arms (control
+ 2 modalities × 2 durations). Data lives at
`/Volumes/Evo/data/129/connectomics/bct_input/<atlas>/sub-XXX_ses-N.count*.csv`.

## Source repo assessment

Confirmed bugs in `lifespan_topological_turning_points` (informational —
we are not fixing the upstream repo, just not blindly copying its bugs):

- `A_threshold_networks.m`: references `all_connectomes` and
  `unharmonized_demographics`, neither of which is ever defined (should be
  `all_data.connectomes` / `reduced_demographics`); saves
  `controlled_density_thresholds`, a variable that is never created
  (the loop populates `controlled_thresholds` instead).
- `C_small_worldness.m`: reads from `lifespan_project_data...` instead of
  the loaded `harmonized_data...`; uses the invalid MATLAB field name
  `variable-density` (a bare hyphen in a struct field access — a syntax
  error as written); accepts a `sim` parameter it never uses (loop count is
  hardcoded to 1000).
- `D_calculate_organization_measures.m`: `global_statistics(5:end)` is
  linear indexing into a 2-D matrix rather than column selection (should be
  `global_statistics(:,5:end)`), silently producing a scrambled measure
  vector.

More fundamentally: the source repo's "turning points" method (age-binned
UMAP manifolds, degree-5 polynomial curve fits, gradient-sign-change
detection, KDE-of-turning-points) requires a continuous covariate with many
distinct values (91 ages, n≈3,800 cross-sectional subjects) to have
statistical meaning. It does not transfer to a 3-timepoint repeated-measures
design — there is no continuous axis to bin or find curvature changes
along. This is why the existing `7_temporal_analysis` stub's
`s(session)` GAM smooth over 3 discrete timepoints is not statistically
meaningful.

**Decision:** keep the reusable methodological skeleton (network
thresholding, graph-theory metrics, small-worldness via null models,
UMAP as an exploratory/visual tool, PCA), replace the age-GAM/turning-point
inference with **linear mixed-effects models** testing Group×Time, which is
the standard approach for repeated-measures intervention trials.

## Current bctpy_mrtrix state (relevant facts)

- `0_installation/install.sh`/`install_venv.sh`: `uv`-based for Python only.
  R package installation is documentation-only (README lists packages, no
  script installs them) — `Rscript`/`rpy2` is never invoked from
  `install.sh`.
- `7_temporal_analysis/scripts/gam_models.R` is the only R script in the
  repo and is never invoked programmatically anywhere (no `subprocess`/
  `rpy2` call in the codebase); its own test (`test_gam_models.py`) only
  checks the file's text content, never executes R.
- `7_temporal_analysis/config/run_spec_temporal.json` targets Schaefer200
  only and a 3-group scheme (`ctrl`, `2w`, `4w`), collapsing the
  modality (solo/social) factor — the module's own README admits this
  ("neglecting social/solo differentiation").
- `pyproject.toml` declares `readme = "README.md"`, but no root README
  exists.
- `group_detection.py` is duplicated (copy-pasted, not imported) into
  `3_nodal_network_metrics/` and `4_statistical_classification/`.
- Real group-assignment data was found at
  `.venv/ID_2w_4w_groups.xlsx` (175 rows: `subject_ID`, `interv_2w`,
  `interv_4w`, `group` ∈ {1,2,3}). `.venv/` is gitignored and gets wiped by
  any `uv venv`/env rebuild, so this file's location is itself a bug.
  Crosstab of `(interv_2w, interv_4w, group)` yields exactly 5 combinations
  → confirms the 5-arm design: `group∈{1,2}` × `{2w,4w}` + `group=3`
  (control, both interv flags 0). No age/sex column exists in this file.
- ID scheme mismatch: the Excel uses `sub-129{group}{subnum:03d}`
  (e.g. `sub-1292001`), while `bct_input` filenames use `sub-{subnum:03d}`
  (e.g. `sub-001`). Joining on the **last 3 digits** matches all 126
  `bct_input` subjects with zero collisions (175 Excel rows → 126 matched,
  49 unmatched = subjects without usable connectome data, which is
  expected attrition).
- `data/processed/metadata.csv` (currently tracked in git) is a 3-row
  synthetic placeholder, not real data.

## Data hygiene (added requirement)

The GitHub remote (`karl-koschutnig/bctpy_mrtrix`) is **public**, and real
subject-level data was already committed and pushed (since 2026-05-06):
`data/processed/metadata.csv`, `outputs/global_metrics/*`, and the entire
`test_pipeline/` tree — the latter includes **raw connectome `.npy` files**
keyed by real subject codes (e.g. `sub-119BPAF161001`), not just derived
metrics. User decision: leave the repo public, but stop tracking real data
going forward (history rewrite deferred, not in scope here).

Already done as part of this design pass:
- `.gitignore` updated: `data/`, `test_pipeline/`, `results/`, `results_*/`
  (with `results_mock_*/` re-included, since those are synthetic fixtures)
  added to the existing `outputs/`/`*.xlsx` rules.
- `git rm --cached` run on `data/processed/metadata.csv`,
  `outputs/global_metrics/`, `test_pipeline/` — untracked but left on disk.

Remaining for the implementation plan:
- Move `ID_2w_4w_groups.xlsx` from `.venv/` to `data/raw/` (now gitignored
  via the `data/` rule, so it's safe there).
- Confirm no other real-data files are still tracked (`mock_data*/` and
  `results_mock_*/` are synthetic and intentionally stay tracked).

## Architecture

### 1. Installation (`uv` + `renv`)

- `install.sh` becomes the single entry point: `uv sync` for Python
  (replacing the current manual `uv pip install <list>` steps — reads
  `pyproject.toml`/`uv.lock` directly), then invokes an R bootstrap step.
- R bootstrap: `renv`-based. If `renv` isn't installed, install it, then
  `renv::restore()` against a committed `renv.lock` covering `mgcv`,
  `lme4`, `lmerTest`, `emmeans`, `tidyverse`, `jsonlite`, `argparse`,
  `R.matlab`, `arrow`. `renv` scopes the library to the repo
  (`renv/library/`), not the user's global R library.
  `renv/library/`, `renv/staging/`, `renv/local/` are gitignored (renv's
  own `renv/.gitignore`, generated by `renv::init()`); `renv.lock` and
  `renv/activate.R` are tracked.
- Add a root `README.md` (fixes the `pyproject.toml` `readme` reference
  that currently points at a nonexistent file).
- `0_installation/preflight_check.py` extended to also validate the R/renv
  side (currently Python-only).
- `run_temporal_analysis.sh` updated to point at the real metadata path
  and 5-group config, dropping any R-related interactive prompts.

### 2. Real data wiring

- One-time script (`data_processing/build_metadata.py` or similar):
  1. Scans `bct_input/*/sub-*_ses-*` filenames for the actual
     (subject, session, atlas) inventory present on disk.
  2. Loads `data/raw/ID_2w_4w_groups.xlsx`, derives a 5-way group label
     from `(interv_2w, interv_4w, group)`. **Deliberately kept blinded:**
     labels are `g1_2w`/`g1_4w`/`g2_2w`/`g2_4w`/`ctrl` — `group=1` vs
     `group=2` is intentionally not decoded to its real modality (solo vs.
     social) anywhere in code, config, or output, so the whole analysis
     runs double-blind. Unblinding (if ever wanted) is a manual relabel
     step done outside the pipeline, after results are finalized.
  3. Joins via last-3-digits of the Excel subject ID to `bct_input`
     subject numbers.
  4. Writes real `data/processed/metadata.csv`
     (`participant_id, session, group, atlas`); leaves `age`/`sex` columns
     absent/optional (no source found yet).
  5. Asserts the join is lossless for every `bct_input` subject (fails
     loudly on any unmatched subject rather than silently dropping it).

### 3. Stage 7 rework — per-atlas, config-driven, 5 groups

Run independently for each of the 5 atlases (no cross-atlas ComBat
harmonization — single site/scanner, so not needed):

- `small_worldness.py` + graph-metrics extraction: reused, re-pointed at
  real per-atlas data, looped over all 5 atlases via config.
- `gam_models.R` → rewritten as `mixed_models.R`: for each metric,
  `lmer(metric ~ group*session + (1|subject))` (or `nlme` if the variance
  structure needs it), `emmeans` for pairwise group contrasts per
  timepoint. Invoked from Python via `subprocess.run(["Rscript", ...])`,
  closing the current "never actually executed" gap; writes CSV/JSON
  consumed downstream.
- `umap_projection.py`: kept as exploratory-only visualization of
  group×time trajectories (2–3 components), no derivative/turning-point
  logic feeding into it.
- `turning_points.py`: retired from the active pipeline (statistically
  inappropriate for 3 timepoints); code kept in place but not called,
  in case a future dataset has enough timepoints to warrant it.
- `group_detection.py`: deduplicated into `1_utilities/`, imported by
  stages 3/4 instead of copy-pasted.

### 4. Testing

- `test_mixed_models.R`/`test_mixed_models_integration.py`: actually
  executes `mixed_models.R` via `subprocess` against mock fixtures
  (replacing the current text-content-only stub).
- `test_metadata_join.py`: asserts the Excel↔`bct_input` join is lossless
  (126/126 matched, expected group counts) — protects the last-3-digit
  join key from silently breaking if ID formats ever change.

## Out of scope

- ComBat harmonization across atlases/scanners.
- MATLAB toolchain (not used; bctpy covers this in Python).
- `archive/` and legacy `data_processing/{bct_test,bct_all_test}.py` —
  left alone.
- Git history rewrite to purge already-pushed real data (user deferred
  this explicitly).
- Age/sex covariates in the mixed models (no source data found yet).

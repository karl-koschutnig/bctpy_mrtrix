# Environment & Real-Data Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `uv` install the whole Python environment and R (via `renv`)
guarantee its packages in one step, and replace the synthetic 3-row
metadata stub with a real, verified metadata table built from the actual
126-subject connectome inventory and the real group-assignment spreadsheet.

**Architecture:** `install.sh` becomes a single entry point calling `uv
sync` for Python and a new `renv`-based R bootstrap for R — no more
manual package lists or documentation-only R setup. A new
`data_processing/build_metadata.py` joins the `bct_input` file inventory
to `data/raw/ID_2w_4w_groups.xlsx` on a verified join key, replacing the
placeholder `data/processed/metadata.csv`.

**Tech Stack:** Python (uv, pandas, openpyxl, pytest), R (renv, to be
populated with mgcv/lme4/lmerTest/emmeans/tidyverse/jsonlite/argparse/
R.matlab/arrow for the companion plan), bash.

**Spec:** `docs/superpowers/specs/2026-09-10-temporal-analysis-pipeline-design.md`

## Global Constraints

- Real subject-level data (metadata, results, raw connectomes) must never
  be added to git tracking — `data/`, `outputs/`, `test_pipeline/`,
  `results_*/` (except `results_mock_*/`) are gitignored; verify new
  files land under these paths, not elsewhere.
- Group labels `g1_2w`/`g1_4w`/`g2_2w`/`g2_4w` stay blinded — never decode
  `group=1` vs `group=2` to a real modality name anywhere in code, config,
  filenames, or output.
- `renv` scopes the R library to the repo (`renv/library/`), never the
  user's global R library.
- Every script this plan touches must keep working when run directly
  (`python path/to/script.py ...` / `Rscript path/to/script.R ...`), not
  only under pytest.

---

## Note on this plan's companion

This plan covers environment setup and real-data wiring only. The
statistical rework of `7_temporal_analysis` (mixed-effects models
replacing the age-GAM, 5-atlas loop, retiring turning-point detection) is
a separate, dependent plan:
`docs/superpowers/plans/2026-09-10-stage7-statistical-rework.md`. Run this
plan first — the rework plan's integration test needs the R/renv
environment this plan sets up.

---

### Task 1: Commit the data-hygiene fix already made

This session already ran `git rm --cached` on real-data files
(`data/processed/metadata.csv`, `outputs/global_metrics/`,
`test_pipeline/`) and updated `.gitignore` to stop tracking `data/`,
`outputs/`, `test_pipeline/`, and `results_*/` (except `results_mock_*/`)
going forward, per the user's explicit request that no project-specific
data leave the machine untracked-but-still-committed. This task commits
that already-staged fix so it isn't lost or re-mixed with later changes.

**Files:**
- Modify: `.gitignore` (already edited)
- Delete (from git index only, files remain on disk): `data/processed/metadata.csv`, `outputs/global_metrics/*`, `test_pipeline/*`

- [ ] **Step 1: Verify the staged state matches expectations**

Run: `git status --short`
Expected: `.gitignore` shown as modified (`M`), and every file under
`data/processed/`, `outputs/global_metrics/`, `test_pipeline/` shown as
deleted-from-index (`D`) — confirm no *other* unrelated files are staged.

- [ ] **Step 2: Confirm nothing real-data-shaped remains tracked**

Run: `git ls-files | grep -E '^(data/|outputs/|test_pipeline/)' | grep -v '^outputs/global_metrics/\.gitkeep'`
Expected: empty output (nothing left tracked under those prefixes).

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
Stop tracking real subject data and results in git

data/, outputs/, test_pipeline/, and results_*/ (except results_mock_*/)
now gitignored. Untracks already-committed real-subject files (metadata,
per-subject graph metrics, and raw connectome matrices under
test_pipeline/Test_matrizen/) without touching prior history.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add a root README.md

`pyproject.toml` declares `readme = "README.md"` but no root README
exists. `uv sync` currently tolerates this, but a public repo with no
README is bad hygiene on its own, and this is the natural place to
document the new install flow this plan introduces.

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# bctpy_mrtrix

Python (bctpy) reimplementation of a brain-connectivity graph-theory
analysis pipeline, adapted for a repeated-measures intervention study
(3 timepoints, 5 study arms) across 5 parcellation atlases.

## Setup

```bash
./0_installation/install.sh
```

This installs the Python environment via `uv` and the R environment via
`renv` (see `0_installation/README.md` for details, including one-time R
lockfile generation and troubleshooting).

## Verify

```bash
source .venv/bin/activate
python 0_installation/preflight_check.py run_spec.json --temporal --r
```

## Pipeline stages

See the numbered directories `1_utilities/` through `7_temporal_analysis/`,
each with its own README. Stage 7 (`7_temporal_analysis/`) is documented
separately since it has the most moving parts (Python + R).

## Data

Real subject data and results are never committed to this repository —
see `.gitignore`. Point the pipeline at your own data via `run_spec.json`
/ the `CONNECTOMES_DIR` environment variable documented in
`0_installation/run_temporal_analysis.sh`.
```

- [ ] **Step 2: Verify `uv sync` still succeeds**

Run: `uv sync`
Expected: exits 0, no errors mentioning `README.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Add root README

Fixes the dangling pyproject.toml readme reference and documents the
new uv + renv install flow.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Set up the R environment via renv, unify install.sh

Replaces the current "install.sh installs Python only, R packages are
documentation you run by hand" split with one command that provisions
both, using `renv` for exact, repo-scoped R package reproducibility.

**Files:**
- Create: `0_installation/init_r_env.R`
- Create: `0_installation/setup_r_env.R`
- Modify: `0_installation/install.sh` (full rewrite of steps 4-6, lines 124-160)
- Modify: `0_installation/README.md` (R section, lines 88-100)
- Create (generated by Step 2 below, then committed): `renv.lock`, `.Rprofile`, `renv/activate.R`, `renv/settings.json`, `renv/.gitignore`

**Interfaces:**
- Produces: `renv.lock` at repo root, restorable via `Rscript 0_installation/setup_r_env.R` — Task 6 (in the companion plan) and any future R script depend on this being present and in sync.

- [ ] **Step 1: Write the one-time lockfile-generation script**

```r
#!/usr/bin/env Rscript
# One-time setup: initializes renv and creates renv.lock for the R
# packages the temporal analysis pipeline needs. Run once from the repo
# root:
#   Rscript 0_installation/init_r_env.R
# Re-run (from repo root) after adding a new library() call anywhere in
# 7_temporal_analysis/ to refresh the lock file.

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE)
}

required_packages <- c(
  "mgcv", "lme4", "lmerTest", "emmeans", "tidyverse",
  "jsonlite", "argparse", "R.matlab", "arrow"
)
install.packages(required_packages, repos = "https://cloud.r-project.org")

renv::snapshot(prompt = FALSE)
cat("\nrenv.lock written. Commit renv.lock, .Rprofile, and renv/activate.R.\n")
```

Save as `0_installation/init_r_env.R`.

- [ ] **Step 2: Run it once from the repo root to generate the lockfile**

Run (from repo root): `Rscript 0_installation/init_r_env.R`
Expected: installs `renv` if missing, creates `renv/`, `.Rprofile`,
installs the 9 required packages (this can take several minutes —
`tidyverse` and `mgcv` compile from source on macOS), ends with
`renv.lock written.`

- [ ] **Step 3: Write the every-install restore script**

```r
#!/usr/bin/env Rscript
# Restores the R environment from renv.lock. Run from the repo root
# (install.sh does this automatically). Fast and idempotent once the
# renv cache is warm.

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
if (!file.exists("renv.lock")) {
  stop("renv.lock not found. Run 'Rscript 0_installation/init_r_env.R' once first.")
}
renv::restore(prompt = FALSE)
```

Save as `0_installation/setup_r_env.R`.

- [ ] **Step 4: Verify the restore script works against the freshly generated lock**

Run: `Rscript 0_installation/setup_r_env.R`
Expected: `* The library is already synchronized with the lockfile.` (or
equivalent "nothing to do" message) — proves the lockfile Step 2 produced
is internally consistent.

- [ ] **Step 5: Verify all required packages actually load**

Run: `Rscript -e "library(mgcv); library(lme4); library(lmerTest); library(emmeans); library(tidyverse); library(jsonlite); library(argparse); library(R.matlab); library(arrow); cat('OK\n')"`
Expected: prints `OK` with no errors (tidyverse's own startup messages on
stderr are fine).

- [ ] **Step 6: Rewrite install.sh's dependency-install and lock-file steps**

Replace `0_installation/install.sh` lines 124-160 (Steps 4 "Install Core
Dependencies", 5 "Install Temporal Analysis Dependencies", 6 "Generate
Lock File") with:

```bash
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
```

Note: this also removes the manual `uv pip install <package list>` calls
and the separate `uv pip compile` lock-file step — `uv sync` maintains
`uv.lock` automatically from `pyproject.toml`, so a hand-maintained
package list duplicating `pyproject.toml` is no longer needed.

- [ ] **Step 7: Update the "Finalize" step's printed instructions**

In the same file, in the `main()` "Finalize" section (originally lines
162-183), change:
```bash
    echo_info "To verify the installation:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo_info "  python 0_installation/preflight_check.py run_spec.json --temporal --uv-lock"
```
to:
```bash
    echo_info "To verify the installation:"
    echo_info "  source $VENV_DIR/bin/activate"
    echo_info "  python 0_installation/preflight_check.py run_spec.json --temporal --r --uv-lock"
```

- [ ] **Step 8: Run install.sh end-to-end to confirm it works unattended**

Run: `./0_installation/install.sh`
Expected: exits 0; ends with `Installation Complete!`; both the `.venv`
Python environment and the R `renv` environment are present and in sync
(re-running `Rscript 0_installation/setup_r_env.R` afterward should again
report "already synchronized").

- [ ] **Step 9: Update 0_installation/README.md's R section**

Replace `0_installation/README.md` lines 88-100 (the "R Dependencies (for
GAM Modeling)" section with the manual `install.packages(...)` snippet)
with:

```markdown
## R Dependencies

R packages are managed with [renv](https://rstudio.github.io/renv/),
scoped to this repo (not your global R library). `install.sh` handles
this automatically:

- First run ever: generates `renv.lock` by installing the required
  packages (`mgcv`, `lme4`, `lmerTest`, `emmeans`, `tidyverse`,
  `jsonlite`, `argparse`, `R.matlab`, `arrow`) — can take several minutes.
- Every subsequent run: restores the exact locked versions via
  `renv::restore()` — fast.

To add a new R package: install it normally inside the project (`Rscript
-e "install.packages('somepkg')"` from the repo root — renv intercepts
this automatically once `.Rprofile` is present), then refresh the lock
file with `Rscript 0_installation/init_r_env.R`, then commit the updated
`renv.lock`.
```

- [ ] **Step 10: Commit**

```bash
git add 0_installation/init_r_env.R 0_installation/setup_r_env.R \
        0_installation/install.sh 0_installation/README.md \
        renv.lock .Rprofile renv/activate.R renv/settings.json renv/.gitignore
git commit -m "$(cat <<'EOF'
Automate R environment setup via renv, unify install.sh on uv sync

install.sh now provisions both Python (uv sync) and R (renv::restore)
in one command instead of documenting R packages for manual install.
renv.lock pins mgcv/lme4/lmerTest/emmeans/tidyverse/jsonlite/argparse/
R.matlab/arrow, scoped to this repo's renv/library/.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Extend preflight_check.py to validate the R/renv environment

Closes the gap where "everything is installed" could never actually be
verified for the R side — `preflight_check.py` only ever checked Python
packages.

**Files:**
- Modify: `0_installation/preflight_check.py`
- Modify: `tests/test_preflight_check.py`

**Interfaces:**
- Produces: `check_r_environment(repo_root: Path) -> list[str]` — empty
  list means OK, non-empty is a list of human-readable problems.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_preflight_check.py`, near the top (after existing
imports, add `from types import SimpleNamespace`):

```python
from types import SimpleNamespace
```

Then append at the end of the file:

```python
# ── check_r_environment ─────────────────────────────────────────────────────

def test_check_r_environment_rscript_missing(monkeypatch, tmp_path):
    """Fehlt Rscript im PATH, wird das als Problem gemeldet."""
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: None)
    problems = check_r_environment(tmp_path)
    assert any("Rscript not found" in p for p in problems)


def test_check_r_environment_missing_lock(monkeypatch, tmp_path):
    """Fehlende renv.lock wird gemeldet, auch wenn Rscript vorhanden ist."""
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert any("renv.lock not found" in p for p in problems)


def test_check_r_environment_missing_packages(monkeypatch, tmp_path):
    """Fehlende R-Pakete werden über die Rscript-Ausgabe erkannt."""
    (tmp_path / "renv.lock").write_text("{}")
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="lme4,emmeans", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert any("lme4" in p and "emmeans" in p for p in problems)


def test_check_r_environment_all_ok(monkeypatch, tmp_path):
    """Keine Probleme, wenn Rscript, renv.lock und alle Pakete vorhanden sind."""
    (tmp_path / "renv.lock").write_text("{}")
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert problems == []
```

Also update the import line that currently reads:
```python
from preflight_check import load_spec, resolve_path, check_imports, main
```
to:
```python
from preflight_check import load_spec, resolve_path, check_imports, check_r_environment, main
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_preflight_check.py -k check_r_environment -v`
Expected: FAIL with `ImportError: cannot import name 'check_r_environment'`

- [ ] **Step 3: Implement `check_r_environment` and wire up `--r`**

In `0_installation/preflight_check.py`, add near the top (after the
existing `import sys` at line 22):

```python
import shutil
import subprocess
```

Add after the `TEMPORAL_PACKAGES` list (after line 41):

```python
# R packages required for the temporal analysis pipeline (see renv.lock)
REQUIRED_R_PACKAGES = [
    "mgcv", "lme4", "lmerTest", "emmeans", "tidyverse",
    "jsonlite", "argparse", "R.matlab", "arrow",
]


def check_r_environment(repo_root: Path) -> list[str]:
    """Check that Rscript, renv.lock, and the required R packages are available.

    Returns a list of human-readable problem strings; empty list means OK.
    """
    problems: list[str] = []

    if shutil.which("Rscript") is None:
        problems.append("Rscript not found on PATH")
        return problems

    if not (repo_root / "renv.lock").exists():
        problems.append(f"renv.lock not found at {repo_root / 'renv.lock'}")

    check_expr = (
        "pkgs <- c(" + ", ".join(f'"{p}"' for p in REQUIRED_R_PACKAGES) + "); "
        "missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]; "
        "cat(paste(missing, collapse = ','))"
    )
    result = subprocess.run(
        ["Rscript", "-e", check_expr],
        capture_output=True, text=True, cwd=repo_root,
    )
    missing = [p for p in result.stdout.strip().split(",") if p]
    if missing:
        problems.append("Missing R packages: " + ", ".join(missing))

    return problems
```

Then replace the `--uv-lock` block in `main()` (lines 166-174) — this
also fixes a pre-existing path bug (`spec_dir.parent / "uv.lock"` points
one level *above* the repo root when `run_spec.json` lives at the repo
root, as it does here) — with:

```python
    repo_root = Path(__file__).resolve().parent.parent

    # Check uv.lock if requested
    if args.uv_lock:
        uv_lock_path = repo_root / "uv.lock"
        if uv_lock_path.exists():
            print("✓ uv.lock found (reproducible environment configured)")
        else:
            print("⚠ uv.lock not found at repo root")
            print("  Generate with: uv sync")

    # Check R/renv environment if requested
    if args.r:
        r_problems = check_r_environment(repo_root)
        if r_problems:
            print("✗ R environment issues:")
            for p in r_problems:
                print("   -", p)
            print("  Run: Rscript 0_installation/setup_r_env.R")
            sys.exit(4)
        else:
            print("✓ R environment OK (renv.lock present, all required packages available)")
```

And add the `--r` CLI flag next to the existing `--temporal`/`--uv-lock`
flags (after line 94):

```python
    parser.add_argument(
        "--r", action="store_true",
        help="Check R/renv environment for the temporal analysis pipeline"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_preflight_check.py -v`
Expected: all PASS, including the 4 new `check_r_environment` tests and
all pre-existing tests (the `--uv-lock` path-bug fix must not break
`test_main_success` etc. — none of those tests pass `--uv-lock` or `--r`,
so they're unaffected).

- [ ] **Step 5: Verify it works against the real environment**

Run: `python 0_installation/preflight_check.py run_spec.json --temporal --r --uv-lock`
Expected: `✓ R environment OK ...` (this depends on Task 3 having been
completed first, so `renv.lock` and the packages exist).

- [ ] **Step 6: Commit**

```bash
git add 0_installation/preflight_check.py tests/test_preflight_check.py
git commit -m "$(cat <<'EOF'
Extend preflight_check.py to validate the R/renv environment

Adds --r flag checking Rscript availability, renv.lock presence, and
that all required R packages actually import. Also fixes a latent path
bug in the --uv-lock check (was resolving one directory above the repo
root when run_spec.json lives at the repo root).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Build real metadata.csv from the group-assignment spreadsheet

Replaces the 3-row synthetic `data/processed/metadata.csv` with the real
126-subject table, joining the `bct_input` connectome inventory to
`ID_2w_4w_groups.xlsx` on the verified last-3-digits key, deriving the 5
blinded group labels, and failing loudly on any unmatched subject.

**Files:**
- Create: `data_processing/build_metadata.py`
- Create: `tests/test_build_metadata.py`
- Modify: `pyproject.toml` (add `data_processing` to `pythonpath`, line 46-55 area)

**Interfaces:**
- Produces: `build_metadata(bct_input_dir: Path, excel_path: Path, atlas: str = "AAL3") -> pd.DataFrame` with columns `participant_id, session, group`; `derive_group_label(interv_2w: int, interv_4w: int, group: int) -> str`; `GroupJoinError(ValueError)`.

- [ ] **Step 1: Add `data_processing` to pytest's pythonpath**

In `pyproject.toml`, in the `[tool.pytest.ini_options]` `pythonpath` list,
add `"data_processing"` alongside the existing stage directories:

```toml
[tool.pytest.ini_options]
pythonpath = [
    ".",
    "0_installation",
    "1_utilities",
    "2_global_network_metrics",
    "3_nodal_network_metrics",
    "4_statistical_classification",
    "5_responder_analysis",
    "6_visualization",
    "7_temporal_analysis",
    "data_processing"
]
testpaths = ["tests", "7_temporal_analysis/tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_build_metadata.py`:

```python
"""
Tests for build_metadata.py.

Real inputs (bct_input/ and ID_2w_4w_groups.xlsx) never enter the repo —
these tests build small synthetic stand-ins with the same filename and
spreadsheet shape, focused on the two things that can silently break:
the (interv_2w, interv_4w, group) -> blinded-label mapping, and the
last-3-digits join between the two different ID schemes.
"""

import openpyxl
import pandas as pd
import pytest

from build_metadata import GroupJoinError, build_metadata, derive_group_label


def _make_excel(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tabelle1"
    ws.append(["subject_ID", "interv_2w", "interv_4w", "group"])
    for r in rows:
        ws.append(r)
    path = tmp_path / "groups.xlsx"
    wb.save(path)
    return path


def _make_bct_input(tmp_path, atlas, subject_sessions):
    atlas_dir = tmp_path / "bct_input" / atlas
    atlas_dir.mkdir(parents=True)
    for sub, sessions in subject_sessions.items():
        for ses in sessions:
            (atlas_dir / f"sub-{sub}_ses-{ses}.count.csv").write_text("0,0\n0,0\n")
            # A same-subject/session file with a different suffix must not
            # cause double-counting.
            (atlas_dir / f"sub-{sub}_ses-{ses}.count.normalized.csv").write_text("0,0\n0,0\n")
    return tmp_path / "bct_input"


def test_derive_group_label_all_five_combos():
    assert derive_group_label(1, 0, 1) == "g1_2w"
    assert derive_group_label(0, 1, 1) == "g1_4w"
    assert derive_group_label(1, 0, 2) == "g2_2w"
    assert derive_group_label(0, 1, 2) == "g2_4w"
    assert derive_group_label(0, 0, 3) == "ctrl"


def test_derive_group_label_rejects_unknown_combo():
    with pytest.raises(ValueError):
        derive_group_label(1, 1, 1)


def test_build_metadata_joins_on_last_three_digits(tmp_path):
    excel_path = _make_excel(tmp_path, [
        ["sub-1291001", 1, 0, 1],   # -> g1_2w, last3=001
        ["sub-1292002", 0, 1, 2],   # -> g2_4w, last3=002
        ["sub-1293003", 0, 0, 3],   # -> ctrl,  last3=003
        ["sub-1291099", 1, 0, 1],   # not present in bct_input -> dropped, not an error
    ])
    bct_input_dir = _make_bct_input(tmp_path, "AAL3", {
        "001": [1, 2, 3],
        "002": [1, 2, 3],
        "003": [1, 2, 3],
    })

    meta = build_metadata(bct_input_dir, excel_path, atlas="AAL3")

    assert set(meta["participant_id"]) == {"sub-001", "sub-002", "sub-003"}
    assert set(meta["session"]) == {"ses-1", "ses-2", "ses-3"}
    assert len(meta) == 9  # 3 subjects x 3 sessions, no double-counting from the .normalized.csv variant

    g1 = meta.loc[meta["participant_id"] == "sub-001", "group"].unique()
    assert list(g1) == ["g1_2w"]


def test_build_metadata_raises_on_unmatched_bct_subject(tmp_path):
    excel_path = _make_excel(tmp_path, [["sub-1291001", 1, 0, 1]])
    bct_input_dir = _make_bct_input(tmp_path, "AAL3", {
        "001": [1, 2, 3],
        "999": [1, 2, 3],  # no Excel entry -> must raise, not silently drop
    })

    with pytest.raises(GroupJoinError, match="999"):
        build_metadata(bct_input_dir, excel_path, atlas="AAL3")


def test_build_metadata_drops_rows_with_missing_group(tmp_path):
    excel_path = _make_excel(tmp_path, [
        ["sub-1291001", 1, 0, 1],
        ["sub-1291002", None, None, None],  # unassigned participant, no matching bct_input subject
    ])
    bct_input_dir = _make_bct_input(tmp_path, "AAL3", {"001": [1]})

    meta = build_metadata(bct_input_dir, excel_path, atlas="AAL3")

    assert set(meta["participant_id"]) == {"sub-001"}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_build_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_metadata'`

- [ ] **Step 4: Implement build_metadata.py**

Create `data_processing/build_metadata.py`:

```python
#!/usr/bin/env python3
"""
build_metadata.py
==================

Builds the real subject/session/group metadata table for the temporal
analysis pipeline from two sources:

- The connectome file inventory under bct_input/<atlas>/ (which subjects
  and sessions actually have usable connectome data).
- The group-assignment spreadsheet (ID_2w_4w_groups.xlsx), which uses a
  different subject-ID format than bct_input. The two are joined on the
  last 3 digits of the subject number, which is unique and collision-free
  across both sources.

Study design: 5 arms encoded as (interv_2w, interv_4w, group) in the
spreadsheet. group 1/2 are deliberately NOT decoded to their real
modality (kept blinded); group 3 is the no-duration control.

Usage:
    python data_processing/build_metadata.py \
        --excel data/raw/ID_2w_4w_groups.xlsx \
        --bct-input-dir /Volumes/Evo/data/129/connectomics/bct_input \
        --atlas AAL3 \
        --out data/processed/metadata.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


class GroupJoinError(ValueError):
    """Raised when a bct_input subject has no matching group assignment."""


_GROUP_LABELS = {
    (1, 0, 1): "g1_2w",
    (0, 1, 1): "g1_4w",
    (1, 0, 2): "g2_2w",
    (0, 1, 2): "g2_4w",
    (0, 0, 3): "ctrl",
}

_FILENAME_RE = re.compile(r"^sub-(\d{3})_ses-(\d+)\.count\.csv$")


def derive_group_label(interv_2w: int, interv_4w: int, group: int) -> str:
    """Map (interv_2w, interv_4w, group) to one of the 5 blinded arm labels."""
    key = (int(interv_2w), int(interv_4w), int(group))
    if key not in _GROUP_LABELS:
        raise ValueError(f"Unrecognized (interv_2w, interv_4w, group) combo: {key}")
    return _GROUP_LABELS[key]


def load_group_assignments(excel_path: Path) -> pd.DataFrame:
    """Load the group-assignment spreadsheet, keyed by last-3-digits subject number."""
    df = pd.read_excel(excel_path, sheet_name=0)
    df = df.dropna(subset=["group", "interv_2w", "interv_4w"])
    df = df.copy()
    df["subject_num"] = df["subject_ID"].str[-3:]
    df["group_label"] = df.apply(
        lambda r: derive_group_label(r["interv_2w"], r["interv_4w"], r["group"]),
        axis=1,
    )
    return df[["subject_num", "group_label"]]


def scan_bct_input(bct_input_dir: Path, atlas: str) -> pd.DataFrame:
    """Scan bct_input/<atlas>/sub-XXX_ses-N.count.csv for the real subject/session inventory."""
    atlas_dir = Path(bct_input_dir) / atlas
    rows = []
    for f in sorted(atlas_dir.glob("sub-*_ses-*.count.csv")):
        m = _FILENAME_RE.match(f.name)
        if not m:
            continue
        subject_num, session_num = m.group(1), m.group(2)
        rows.append({
            "participant_id": f"sub-{subject_num}",
            "subject_num": subject_num,
            "session": f"ses-{session_num}",
        })
    return pd.DataFrame(rows)


def build_metadata(bct_input_dir: Path, excel_path: Path, atlas: str = "AAL3") -> pd.DataFrame:
    """Join the bct_input inventory to group assignments; fail loudly on any unmatched subject."""
    inventory = scan_bct_input(bct_input_dir, atlas)
    groups = load_group_assignments(excel_path)

    merged = inventory.merge(groups, on="subject_num", how="left")
    unmatched = sorted(merged.loc[merged["group_label"].isna(), "participant_id"].unique())
    if unmatched:
        raise GroupJoinError(
            "No group assignment found for bct_input subjects: " + ", ".join(unmatched)
        )

    result = merged.rename(columns={"group_label": "group"})
    result = result[["participant_id", "session", "group"]]
    result = result.sort_values(["participant_id", "session"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", required=True, help="Path to ID_2w_4w_groups.xlsx")
    parser.add_argument("--bct-input-dir", required=True, help="Path to bct_input/ directory")
    parser.add_argument(
        "--atlas", default="AAL3",
        help="Atlas subdirectory to scan for the subject/session inventory "
             "(the inventory is atlas-independent in practice; any atlas with "
             "complete data works)",
    )
    parser.add_argument("--out", required=True, help="Output path for metadata.csv")
    args = parser.parse_args()

    metadata = build_metadata(Path(args.bct_input_dir), Path(args.excel), atlas=args.atlas)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_path, index=False)

    print(f"Wrote {len(metadata)} rows ({metadata['participant_id'].nunique()} subjects) to {out_path}")
    print(metadata.groupby("group")["participant_id"].nunique())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_build_metadata.py -v`
Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add data_processing/build_metadata.py tests/test_build_metadata.py pyproject.toml
git commit -m "$(cat <<'EOF'
Add build_metadata.py to join real group assignments to the bct_input inventory

Joins the ID_2w_4w_groups.xlsx spreadsheet (different ID format) to the
bct_input connectome file inventory on a verified last-3-digits key,
derives the 5 blinded group labels, and fails loudly on any unmatched
subject rather than silently dropping data.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Move the real spreadsheet out of .venv, wire it into the workflow script

`ID_2w_4w_groups.xlsx` currently lives in `.venv/`, which is gitignored
*and* gets wiped every time the virtual environment is rebuilt (e.g. by
`uv venv` inside `install.sh`). Move it to `data/raw/` (now gitignored as
a whole directory per Task 1, so this is safe), and make
`run_temporal_analysis.sh` build `metadata.csv` from it via
`build_metadata.py` instead of writing a 3-row synthetic template.

**Files:**
- Move: `.venv/ID_2w_4w_groups.xlsx` → `data/raw/ID_2w_4w_groups.xlsx`
- Modify: `0_installation/run_temporal_analysis.sh` (lines 33-46 CONFIGURATION, lines 149-176 `create_metadata()`)

- [ ] **Step 1: Move the file**

```bash
mkdir -p data/raw
mv .venv/ID_2w_4w_groups.xlsx data/raw/ID_2w_4w_groups.xlsx
```

- [ ] **Step 2: Add the spreadsheet path to run_temporal_analysis.sh's config**

In `0_installation/run_temporal_analysis.sh`, in the CONFIGURATION
section, after the existing `METADATA_FILE="${DATA_DIR}/metadata.csv"`
line, add:

```bash
GROUPS_EXCEL="${ROOT_DIR}/data/raw/ID_2w_4w_groups.xlsx"
```

- [ ] **Step 3: Replace the template-writing `create_metadata()` function**

Replace the entire `create_metadata()` function (lines 149-176) with:

```bash
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
```

- [ ] **Step 4: Build the real metadata.csv and inspect it**

```bash
rm -f data/processed/metadata.csv   # remove the old 3-row placeholder
source .venv/bin/activate
python data_processing/build_metadata.py \
    --excel data/raw/ID_2w_4w_groups.xlsx \
    --bct-input-dir /Volumes/Evo/data/129/connectomics/bct_input \
    --atlas AAL3 \
    --out data/processed/metadata.csv
```

Expected: prints `Wrote 378 rows (126 subjects) to data/processed/metadata.csv`
(126 subjects × 3 sessions) and a per-group subject count with 5 distinct
group values (`ctrl`, `g1_2w`, `g1_4w`, `g2_2w`, `g2_4w`), no
`GroupJoinError`.

- [ ] **Step 5: Commit**

Note: `data/` is gitignored (Task 1), so `data/raw/ID_2w_4w_groups.xlsx`
and `data/processed/metadata.csv` will NOT be added by this commit — only
the script change is tracked. Verify that before committing:

```bash
git status --short data/
```

Expected: empty output (both files correctly ignored).

```bash
git add 0_installation/run_temporal_analysis.sh
git commit -m "$(cat <<'EOF'
Build real metadata.csv from the group-assignment spreadsheet

run_temporal_analysis.sh now calls build_metadata.py instead of writing
a 3-row synthetic template when metadata.csv is missing. The real
spreadsheet moved from .venv/ (gitignored and wiped on env rebuild) to
data/raw/ (gitignored, but stable).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Deduplicate group_detection.py

`1_utilities/group_detection.py`, `3_nodal_network_metrics/group_detection.py`,
and `4_statistical_classification/group_detection.py` are byte-identical
copies. Keep the one in `1_utilities/` and have the 4 scripts that import
it locate it there explicitly, instead of relying on a same-directory
copy.

**Files:**
- Delete: `3_nodal_network_metrics/group_detection.py`
- Delete: `4_statistical_classification/group_detection.py`
- Modify: `3_nodal_network_metrics/nodal_multivariate_analysis.py` (import block, line 48)
- Modify: `3_nodal_network_metrics/nodal_temporal_trajectories.py` (import block, line 40)
- Modify: `4_statistical_classification/stat_random_forest_comparison.py` (import block, line 44)
- Modify: `4_statistical_classification/stat_svm_baseline_analysis.py` (import block, line 45)

- [ ] **Step 1: Add a sys.path bootstrap before the group_detection import in each of the 4 scripts**

In each of the 4 files above, immediately before the line
`from group_detection import detect_or_ask_groups`, add:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_utilities"))
```

(all 4 files already `import sys` and `from pathlib import Path` earlier
in the file — verify this before adding; if either import is missing in
a given file, add it alongside the existing top-of-file imports.)

- [ ] **Step 2: Delete the duplicate files**

```bash
rm 3_nodal_network_metrics/group_detection.py
rm 4_statistical_classification/group_detection.py
```

- [ ] **Step 3: Run the full existing test suite to confirm nothing broke**

Run: `pytest tests/ -v`
Expected: all tests that were passing before this task still pass —
in particular `test_nodal_multivariate_analysis.py`,
`test_nodal_trajectory_analysis.py`, `test_stat_random_forest_comparison.py`,
`test_stat_svm_baseline_analysis.py`, and `test_group_detection.py`.

- [ ] **Step 4: Verify the affected scripts still run standalone (not just under pytest)**

Run: `python -c "import sys; sys.path.insert(0, '3_nodal_network_metrics'); import nodal_multivariate_analysis"`
Expected: no `ImportError` for `group_detection`.

Run: `python -c "import sys; sys.path.insert(0, '4_statistical_classification'); import stat_random_forest_comparison"`
Expected: no `ImportError` for `group_detection`.

- [ ] **Step 5: Commit**

```bash
git add 3_nodal_network_metrics/nodal_multivariate_analysis.py \
        3_nodal_network_metrics/nodal_temporal_trajectories.py \
        4_statistical_classification/stat_random_forest_comparison.py \
        4_statistical_classification/stat_svm_baseline_analysis.py \
        3_nodal_network_metrics/group_detection.py \
        4_statistical_classification/group_detection.py
git commit -m "$(cat <<'EOF'
Deduplicate group_detection.py into 1_utilities/

Three byte-identical copies existed; the 4 scripts that imported the
local copy now locate 1_utilities/group_detection.py explicitly via a
sys.path bootstrap, so there's one source of truth.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

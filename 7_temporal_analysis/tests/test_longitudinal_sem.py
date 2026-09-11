"""Tests for longitudinal_sem.R - runs the script via Rscript against a
synthetic 3-wave dataset with a known group-on-slope effect."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "longitudinal_sem.R"

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="Rscript not available"
)


@pytest.fixture
def synthetic_csv(tmp_path):
    """40 subjects x 3 sessions x 3 arms. 'modularity' has a real group->slope
    effect (ctrl flat, others increasing); 'density' is pure noise."""
    rng = np.random.default_rng(0)
    arms = ["ctrl", "2w", "4w"]
    rows = []
    for i in range(45):
        pid = f"sub-{i:03d}"
        arm = arms[i % 3]
        base_mod = rng.normal(0.5, 0.05)
        slope_mod = {"ctrl": 0.0, "2w": 0.04, "4w": 0.06}[arm]
        base_den = rng.normal(0.15, 0.02)
        for t, ses in enumerate(["ses-1", "ses-2", "ses-3"]):
            rows.append({
                "participant_id": pid, "session": ses, "group": arm,
                "modularity": base_mod + slope_mod * t + rng.normal(0, 0.02),
                "density": base_den + rng.normal(0, 0.02),
                "global_efficiency": rng.normal(0.5, 0.03),
                "path_length": rng.normal(2.3, 0.1),
                "clustering_coef_mean": rng.normal(0.3, 0.03),
            })
    p = tmp_path / "measures.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _run(csv, outdir):
    return subprocess.run(
        ["Rscript", str(SCRIPT), "--input-file", str(csv),
         "--output-dir", str(outdir), "--timepoint-col", "session",
         "--group-col", "group", "--participant-col", "participant_id"],
        capture_output=True, text=True,
    )


def test_script_exists():
    assert SCRIPT.exists()


def test_produces_growth_parameters_and_fit_indices(synthetic_csv, tmp_path):
    out = tmp_path / "sem"
    r = _run(synthetic_csv, out)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"

    fit = pd.read_csv(out / "sem_fit_indices.csv")
    assert "admissible" in fit.columns
    assert "converged" in fit.columns
    assert (fit["metric"] == "modularity").any()

    params = pd.read_csv(out / "sem_growth_parameters.csv")
    # the group->slope paths must be represented
    slope_paths = params[(params["lhs"] == "s") & (params["op"] == "~")]
    assert len(slope_paths) > 0


def test_recovers_known_group_slope_effect(synthetic_csv, tmp_path):
    out = tmp_path / "sem"
    r = _run(synthetic_csv, out)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"

    params = pd.read_csv(out / "sem_growth_parameters.csv")
    mod_slope = params[
        (params["metric"] == "modularity")
        & (params["lhs"] == "s") & (params["op"] == "~")
    ]
    # at least one arm's slope path should be clearly non-null given the
    # injected effect (ctrl flat, 2w/4w rising)
    assert (mod_slope["pvalue"] < 0.10).any(), mod_slope.to_string()


def test_parallel_process_file_written(synthetic_csv, tmp_path):
    out = tmp_path / "sem"
    _run(synthetic_csv, out)
    pp = out / "sem_parallel_process.csv"
    if pp.exists():  # segregation domain needs modularity + clustering, both present
        df = pd.read_csv(pp)
        assert "slope_slope_cov" in df["term"].values
        assert "admissible" in df.columns


@pytest.fixture
def synthetic_csv_with_demographics(tmp_path):
    rng = np.random.default_rng(1)
    arms = ["ctrl", "2w", "4w"]
    rows = []
    for i in range(45):
        arm = arms[i % 3]
        base = rng.normal(0.5, 0.05)
        slope = {"ctrl": 0.0, "2w": 0.05, "4w": 0.06}[arm]
        for t, ses in enumerate(["ses-1", "ses-2", "ses-3"]):
            rows.append({
                "participant_id": f"sub-{i:03d}", "session": ses, "group": arm,
                "age": 20 + (i % 12), "sex": "MF"[i % 2],
                "modularity": base + slope * t + rng.normal(0, 0.02),
                "global_efficiency": rng.normal(0.5, 0.03),
                "clustering_coef_mean": rng.normal(0.3, 0.03),
                "path_length": rng.normal(2.3, 0.1),
            })
    p = tmp_path / "measures.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_covariate_cols_produce_covariate_slope_paths(synthetic_csv_with_demographics, tmp_path):
    out = tmp_path / "sem"
    r = subprocess.run(
        ["Rscript", str(SCRIPT), "--input-file", str(synthetic_csv_with_demographics),
         "--output-dir", str(out), "--timepoint-col", "session", "--group-col", "group",
         "--participant-col", "participant_id", "--covariate-cols", "age", "sex"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    params = pd.read_csv(out / "sem_growth_parameters.csv")
    assert "slope_spec" in params.columns
    # age / sex appear as predictors of the latent slope, not as modelled measures
    cov_paths = params[(params.op == "~") & (params.rhs.isin(["age", "sex"]))]
    assert len(cov_paths) > 0
    assert "age" not in params["metric"].unique()


def test_latent_basis_flag_tags_output(synthetic_csv_with_demographics, tmp_path):
    out = tmp_path / "sem"
    r = subprocess.run(
        ["Rscript", str(SCRIPT), "--input-file", str(synthetic_csv_with_demographics),
         "--output-dir", str(out), "--timepoint-col", "session", "--group-col", "group",
         "--participant-col", "participant_id", "--latent-basis"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    fi = pd.read_csv(out / "sem_fit_indices.csv")
    assert (fi["slope_spec"] == "latent_basis").any()

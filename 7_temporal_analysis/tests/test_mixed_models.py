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
    """20 subjects x 3 sessions across 5 study-arm labels, with age/sex."""
    rng = np.random.default_rng(42)
    groups = ["ctrl", "g1_2w", "g1_4w", "g2_2w", "g2_4w"]
    rows = []
    for i in range(20):
        participant = f"sub-{i:03d}"
        group = groups[i % len(groups)]
        baseline = rng.normal(0.3, 0.05)
        age = rng.integers(18, 39)
        sex = "MF"[i % 2]
        for session_idx, session in enumerate(["ses-1", "ses-2", "ses-3"]):
            rows.append({
                "participant_id": participant,
                "session": session,
                "group": group,
                "age": age,
                "sex": sex,
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


def test_covariate_cols_add_age_sex_terms_and_exclude_them_as_metrics(sample_metrics_csv, tmp_path):
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        ["Rscript", str(SCRIPT_PATH), "--input-file", str(sample_metrics_csv),
         "--output-dir", str(output_dir), "--group-col", "group",
         "--participant-col", "participant_id", "--covariate-cols", "age", "sex"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    stats = pd.read_csv(output_dir / "mixed_model_statistics.csv")
    assert {"age", "sex"}.issubset(set(stats["term"]))
    # age/sex must not be modelled as outcomes
    assert set(stats["metric"].unique()) == {"density", "global_efficiency"}


def test_baseline_adjust_drops_first_session_and_adds_baseline_term(sample_metrics_csv, tmp_path):
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        ["Rscript", str(SCRIPT_PATH), "--input-file", str(sample_metrics_csv),
         "--output-dir", str(output_dir), "--group-col", "group",
         "--participant-col", "participant_id", "--baseline-adjust"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    stats = pd.read_csv(output_dir / "mixed_model_statistics.csv")
    assert ".baseline" in set(stats["term"])

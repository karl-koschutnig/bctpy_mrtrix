"""Tests for difference_scores.R - lm(last - first ~ group + covariates)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "difference_scores.R"

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="Rscript not available"
)


@pytest.fixture
def csv(tmp_path):
    """45 subjects, 3 arms. 'moved' rises only for arm 'b'; 'flat' is noise."""
    rng = np.random.default_rng(0)
    arms = ["ctrl", "a", "b"]
    rows = []
    for i in range(45):
        arm = arms[i % 3]
        slope = {"ctrl": 0.0, "a": 0.0, "b": 0.5}[arm]
        base = rng.normal(0, 1)
        for t, ses in enumerate(["ses-1", "ses-2", "ses-3"]):
            rows.append({
                "participant_id": f"sub-{i:03d}", "session": ses, "group": arm,
                "age": 20 + (i % 10), "sex": "MF"[i % 2],
                "moved": base + slope * t + rng.normal(0, 0.2),
                "flat": rng.normal(0, 1),
            })
    p = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_recovers_group_effect_and_ignores_covariates_as_outcomes(csv, tmp_path):
    out = tmp_path / "d"
    r = subprocess.run(
        ["Rscript", str(SCRIPT), "--input-file", str(csv), "--output-dir", str(out),
         "--group-col", "group", "--participant-col", "participant_id",
         "--covariate-cols", "age", "sex"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    co = pd.read_csv(out / "difference_score_coefficients.csv")
    assert set(co["metric"].unique()) == {"moved", "flat"}
    pcol = [c for c in co.columns if c.startswith("Pr")][0]
    moved_b = co[(co.metric == "moved") & (co.term.str.contains("groupb"))]
    assert (moved_b[pcol] < 0.05).any(), moved_b.to_string()
    flat_b = co[(co.metric == "flat") & (co.term.str.contains("group"))]
    assert (flat_b[pcol] > 0.05).all()

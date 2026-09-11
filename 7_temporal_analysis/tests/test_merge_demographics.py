"""Tests for merge_demographics.py - ID mapping + join behaviour."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "merge_demographics.py"


@pytest.fixture
def demo_xlsx(tmp_path):
    d = pd.DataFrame({
        "participant_id": ["sub-1291003", "sub-1292001", "sub-1293002",
                           "sub-1293044", "sub-1292044"],  # last-3 collision on 044
        "age": [20, 34, 20, 19, 19],
        "sex": ["F", "M", "M", "F", "F"],
        "size": [167, 190, 184, 170, 170],
        "weight": [54, 100, 80, 60, 60],
    })
    p = tmp_path / "participants.xlsx"
    d.to_excel(p, index=False)
    return p


@pytest.fixture
def metadata_csv(tmp_path):
    d = pd.DataFrame({
        "participant_id": ["sub-001", "sub-001", "sub-003", "sub-044"],
        "session": ["ses-1", "ses-2", "ses-1", "ses-1"],
        "group": ["g2_4w", "g2_4w", "g1_2w", "ctrl"],
    })
    p = tmp_path / "metadata.csv"
    d.to_csv(p, index=False)
    return p


def _run(demo, meta, out):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--demographics", str(demo),
         "--metadata", str(meta), "--output", str(out)],
        capture_output=True, text=True,
    )


def test_maps_encoded_ids_and_joins_age_sex(demo_xlsx, metadata_csv, tmp_path):
    out = tmp_path / "merged.csv"
    r = _run(demo_xlsx, metadata_csv, out)
    assert r.returncode == 0, r.stderr
    m = pd.read_csv(out)
    assert {"age", "sex", "height_cm", "weight_kg"}.issubset(m.columns)
    # sub-001 -> sub-1292001 -> age 34 M, on both its rows
    s1 = m[m.participant_id == "sub-001"]
    assert (s1["age"] == 34).all() and (s1["sex"] == "M").all()
    # sub-003 -> sub-1291003 -> 20 F
    assert m.loc[m.participant_id == "sub-003", "age"].iloc[0] == 20


def test_identical_collision_resolved_without_error(demo_xlsx, metadata_csv, tmp_path):
    out = tmp_path / "merged.csv"
    r = _run(demo_xlsx, metadata_csv, out)
    assert r.returncode == 0
    m = pd.read_csv(out)
    row = m[m.participant_id == "sub-044"]
    assert len(row) == 1
    assert row["age"].iloc[0] == 19 and row["sex"].iloc[0] == "F"


def test_row_count_preserved(demo_xlsx, metadata_csv, tmp_path):
    out = tmp_path / "merged.csv"
    _run(demo_xlsx, metadata_csv, out)
    assert len(pd.read_csv(out)) == 4

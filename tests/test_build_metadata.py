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

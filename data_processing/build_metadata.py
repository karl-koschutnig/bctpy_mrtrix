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

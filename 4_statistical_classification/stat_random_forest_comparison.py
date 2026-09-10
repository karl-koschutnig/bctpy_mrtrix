#!/usr/bin/env python3
"""
SCRIPT: Random Forest vs SVM Comparison with Design Variants
=============================================================

PURPOSE:
    Compares RF and SVM across three grouping strategies:
    1. Social effect: alone vs group vs control
    2. Duration effect: 2w vs 4w vs control
    3. Intervention vs Control: binary

    Groups and columns are auto-detected from the data.
    Clear error messages if group labels cannot be detected.

USAGE:
    python stat_rf_comparison.py run_spec.json
    python stat_rf_comparison.py --metrics-file /path/to/metrics.parquet --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 1.0 (Auto-detection, run_spec driven)
"""

from __future__ import annotations
from datetime import datetime

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Local
sys.path.append(str(Path(__file__).resolve().parent.parent / "1_utilities"))
from group_detection import detect_or_ask_groups

# ============================================================================
# CLI / run_spec
# ============================================================================


def _progress(current, total, desc):
    """Simple progress display without external dependencies."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {desc}: |{bar}| {current}/{total} ({pct:.0f}%)", end="\r", flush=True)
    if current == total:
        print()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF vs SVM comparison.")
    parser.add_argument("run_spec",          nargs="?", help="Path to run_spec.json")
    parser.add_argument("--metrics-file",    help="Metrics .parquet file")
    parser.add_argument("--metadata",        help="Participant metadata TSV/CSV")
    parser.add_argument("--output-dir",      help="Output directory")
    parser.add_argument("--control-group",   help="Control group label")
    parser.add_argument("--alone-groups",    nargs="+")
    parser.add_argument("--group-groups",    nargs="+")
    parser.add_argument("--short-groups",    nargs="+")
    parser.add_argument("--long-groups",     nargs="+")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config: dict = {}
    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"x run_spec not found: {spec_path}")
        with open(spec_path) as f:
            spec = json.load(f)
        inputs  = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["metrics_file"]  = inputs.get("metrics_file")
        config["metadata_file"] = inputs.get("metadata_file")
        config["output_dir"]    = outputs.get("output_dir")
        for k in ("control_group", "alone_groups", "group_groups",
                  "short_groups", "long_groups"):
            config[k] = spec.get(k, None)

    if args.metrics_file:  config["metrics_file"]  = args.metrics_file
    if args.metadata:      config["metadata_file"] = args.metadata
    if args.output_dir:    config["output_dir"]    = args.output_dir
    if args.control_group: config["control_group"] = args.control_group
    if args.alone_groups:  config["alone_groups"]  = args.alone_groups
    if args.group_groups:  config["group_groups"]  = args.group_groups
    if args.short_groups:  config["short_groups"]  = args.short_groups
    if args.long_groups:   config["long_groups"]   = args.long_groups

    missing = [k for k in ("metrics_file", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(f"x Missing: {', '.join(missing)}")

    config["metrics_file"] = Path(config["metrics_file"]).expanduser().resolve()
    config["output_dir"]   = Path(config["output_dir"]).expanduser().resolve()
    if config.get("metadata_file"):
        config["metadata_file"] = Path(config["metadata_file"]).expanduser().resolve()
    return config


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_columns(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower: return cols_lower[c]
        for c in candidates:
            for cl, co in cols_lower.items():
                if c in cl: return co
        return None

    exclude = {"subject", "participant_id", "session", "ses", "session_id",
               "node", "group", "condition", "sex", "gender", "age", "atlas",
               "hub_type", "community", "nr_sessions", "n_sessions",
               "dsi_metric", "matrix_type", "qc_passed", "qc_warnings"}

    detected = {
        "subject_col": find(["participant_id", "subject_id", "subject"]),
        "session_col": find(["session_id", "session", "ses", "timepoint"]),
        "group_col":   find(["group", "condition", "arm", "intervention"]),
        "sex_col":     find(["sex", "gender"]),
        "age_col":     find(["age"]),
        "metric_cols": [c for c in df.columns
                        if c.lower() not in exclude
                        and pd.api.types.is_numeric_dtype(df[c])
                        and c.lower() != "node"],
    }
    print("  Auto-detected columns:")
    for k, v in detected.items():
        if k != "metric_cols":
            print(f"    {k}: {v or '⚠ not found'}")
        else:
            print(f"    metric_cols: {len(v)} columns")
    return detected


def detect_groups(df: pd.DataFrame, group_col: str, config: dict) -> dict:
    all_groups   = df[group_col].dropna().unique().tolist()
    group_counts = df.groupby(group_col)[df.columns[0]].count()
    control      = config.get("control_group") or group_counts.idxmin()
    intervention = [g for g in all_groups if g != control]

    def match(groups, keywords):
        return [g for g in groups if any(k in str(g).lower() for k in keywords)]

    alone  = config.get("alone_groups")  or match(intervention, ["alone", "individual", "solo"])
    social = config.get("group_groups")  or match(intervention, ["group", "social", "team"])
    short  = config.get("short_groups")  or match(intervention, ["2w", "short", "2week"])
    long_  = config.get("long_groups")   or match(intervention, ["4w", "long",  "4week"])

    if not alone and not social and len(intervention) >= 2:
        half = len(intervention) // 2
        alone, social = intervention[:half], intervention[half:]

    if not short and not long_ and len(intervention) >= 2:
        half = len(intervention) // 2
        short, long_ = intervention[:half], intervention[half:]

    groups = {
        "all": all_groups, "control": control, "intervention": intervention,
        "alone": alone or [], "social": social or [],
        "short": short or [], "long": long_ or [],
    }

    print(f"  Control: {control}")
    print(f"  Intervention: {intervention}")
    print(f"  Alone: {groups['alone']}  Social: {groups['social']}")
    print(f"  Short: {groups['short']}  Long: {groups['long']}")
    return groups


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_merge(metrics_file: Path, metadata_file: Path | None,
                   cols: dict) -> pd.DataFrame:
    df = pd.read_parquet(metrics_file)
    if metadata_file and metadata_file.exists():
        sep  = "\t" if metadata_file.suffix == ".tsv" else ","
        meta = pd.read_csv(metadata_file, sep=sep)
        keys = [k for k in [cols["subject_col"], cols["session_col"]]
                if k and k in meta.columns and k in df.columns]
        if keys:
            df = df.merge(meta, on=keys, how="left", suffixes=("", "_meta"))
    return df


def aggregate_nodal(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    if "node" not in df.columns:
        return df
    keep  = [c for c in [cols["subject_col"], cols["session_col"],
                          cols["group_col"], cols["sex_col"], cols["age_col"]]
             if c and c in df.columns]
    avail = [c for c in cols["metric_cols"] if c in df.columns]
    return df.groupby(keep)[avail].mean().reset_index()


def calculate_slopes(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    subj_col    = cols["subject_col"]
    session_col = cols["session_col"]
    group_col   = cols["group_col"]
    sex_col     = cols["sex_col"]
    age_col     = cols["age_col"]
    metric_cols = [c for c in cols["metric_cols"] if c in df.columns]

    records = []
    _subjects_list = list(df.groupby(subj_col))
    _total_s = len(_subjects_list)
    for _i_s, (subj, sdata) in enumerate(_subjects_list):
        _g = sdata[group_col].iloc[0] if group_col and group_col in sdata.columns else ""
        _progress(_i_s + 1, _total_s, "Calculating slopes")
        print(f"    {subj} [{_g}] fitting slopes across {len(sdata)} sessions...  ", end="\n", flush=True)
        sdata = sdata.sort_values(session_col)
        if len(sdata) < 2:
            continue
        sessions = sdata[session_col].values
        try:
            sessions = pd.to_numeric(
                pd.Series(sessions).str.extract(r"(\d+)")[0].values)
        except Exception:
            sessions = np.arange(len(sessions), dtype=float)

        rec = {subj_col: subj}
        for c in [group_col, sex_col, age_col]:
            if c and c in sdata.columns:
                rec[c] = sdata[c].iloc[0]
        rec["n_sessions"] = len(sdata)

        for metric in metric_cols:
            vals  = pd.to_numeric(sdata[metric], errors="coerce").values
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                try:
                    rec[f"{metric}_slope"] = np.polyfit(
                        sessions[valid].astype(float), vals[valid], 1)[0]
                except Exception:
                    pass
        records.append(rec)

    return pd.DataFrame(records)


def remap_groups(group_series: pd.Series, group_a: list, group_b: list,
                 label_a: str, label_b: str, control) -> pd.Series:
    def _map(g):
        if g in group_a:  return label_a
        if g in group_b:  return label_b
        if g == control:  return "control"
        return np.nan
    return group_series.apply(_map)


def build_features(slopes_df: pd.DataFrame, group_new: pd.Series,
                   label_mapping: dict, sex_col: str | None,
                   age_col: str | None) -> tuple:
    df = slopes_df.copy()
    df["_label"] = group_new.values

    if sex_col and sex_col in df.columns:
        df["sex_encoded"] = df[sex_col].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0)

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    extra      = [c for c in ["sex_encoded", "n_sessions"] +
                  ([age_col] if age_col and age_col in df.columns else [])
                  if c in df.columns]
    feat_cols  = slope_cols + extra

    X     = df[feat_cols].fillna(0).values.astype(float)
    y_raw = df["_label"].values
    y     = np.array([label_mapping.get(l, np.nan) for l in y_raw], dtype=float)

    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    return X[valid], y[valid].astype(int), feat_cols


# ============================================================================
# CLASSIFIERS
# ============================================================================

def train_rf(X: np.ndarray, y: np.ndarray, target_names: list) -> dict:
    rf = RandomForestClassifier(n_estimators=500, max_depth=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X, y, cv=min(5, len(np.unique(y))))
    rf.fit(X, y)
    y_pred = rf.predict(X)
    print(f"  RF  CV={cv.mean():.3f}±{cv.std():.3f}  Train={((y_pred==y).mean()):.3f}")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
    fi = pd.DataFrame({"feature": range(X.shape[1]), "importance": rf.feature_importances_})
    return {"cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
            "train_acc": float((y_pred == y).mean()), "feature_importance": rf.feature_importances_}


def train_svm_model(X: np.ndarray, y: np.ndarray, target_names: list) -> dict:
    scaler   = StandardScaler()
    X_s      = scaler.fit_transform(X)
    svm      = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=42)
    cv = cross_val_score(svm, X_s, y, cv=min(5, len(np.unique(y))))
    svm.fit(X_s, y)
    y_pred = svm.predict(X_s)
    print(f"  SVM CV={cv.mean():.3f}±{cv.std():.3f}  Train={((y_pred==y).mean()):.3f}")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
    return {"cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
            "train_acc": float((y_pred == y).mean())}


def feature_importance_anova(X: np.ndarray, y: np.ndarray,
                              feat_cols: list) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(feat_cols):
        groups = [X[y == g, i] for g in np.unique(y)]
        try:
            f, p = f_oneway(*groups)
        except Exception:
            f, p = np.nan, np.nan
        rows.append({"feature": col, "f_score": f, "p_value": p})
    return pd.DataFrame(rows).sort_values("f_score", ascending=False)


# ============================================================================
# RUN ONE VARIANT
# ============================================================================

def run_variant(slopes_df: pd.DataFrame, group_col: str,
                sex_col: str | None, age_col: str | None,
                groups: dict, variant: str, output_dir: Path) -> dict:

    print(f"\n{'='*70}")
    print(f"VARIANT: {variant.upper()}")
    print("="*70)

    if variant == "social":
        if not groups["alone"] or not groups["social"]:
            print(f"  x Skipping — alone/social groups not detected")
            print(f"    Found: {groups['all']}")
            print(f"    Fix: add 'alone_groups' and 'group_groups' to run_spec.json")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["alone"],
                                    groups["social"], "alone", "group", groups["control"])
        label_map    = {"alone": 0, "group": 1, "control": 2}
        target_names = ["Alone", "Group", "Control"]

    elif variant == "duration":
        if not groups["short"] or not groups["long"]:
            print(f"  x Skipping — short/long groups not detected")
            print(f"    Found: {groups['all']}")
            print(f"    Fix: add 'short_groups' and 'long_groups' to run_spec.json")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["short"],
                                    groups["long"], "2w", "4w", groups["control"])
        label_map    = {"2w": 0, "4w": 1, "control": 2}
        target_names = ["2 weeks", "4 weeks", "Control"]

    elif variant == "intervention":
        if not groups["intervention"]:
            print(f"  x Skipping — no intervention groups detected")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["intervention"],
                                    [], "intervention", "_none_", groups["control"])
        label_map    = {"intervention": 0, "control": 1}
        target_names = ["Intervention", "Control"]
    else:
        return {}

    X, y, feat_cols = build_features(slopes_df, group_new, label_map,
                                      sex_col, age_col)

    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  x Not enough data for {variant}")
        return {}

    print(f"  Samples={len(X)}, Features={len(feat_cols)}")

    rf_res  = train_rf(X, y, target_names)
    svm_res = train_svm_model(X, y, target_names)

    anova_df = feature_importance_anova(X, y, feat_cols)
    print("\nTop 5 ANOVA Features:")
    print(anova_df.head(5).to_string(index=False))

    # Add feature names to RF importance
    rf_fi = pd.DataFrame({"feature": feat_cols, "importance": rf_res["feature_importance"]})
    rf_fi = rf_fi.sort_values("importance", ascending=False)

    anova_df.to_csv(output_dir / f"feature_importance_anova_{variant}.csv", index=False)
    rf_fi.to_csv(output_dir / f"feature_importance_rf_{variant}.csv", index=False)

    best = "RF" if rf_res["cv_mean"] >= svm_res["cv_mean"] else "SVM"
    print(f"\n  Best: {best}")

    return {
        "best_model": best,
        "random_forest": {k: v for k, v in rf_res.items() if k != "feature_importance"},
        "svm": svm_res,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = load_config(args)

    metrics_file  = config["metrics_file"]
    metadata_file = config.get("metadata_file")
    output_dir    = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("RF vs SVM COMPARISON")
    print("=" * 70)

    print("\nLoading data...")
    raw_df = pd.read_parquet(metrics_file)
    cols   = detect_columns(raw_df)
    df     = load_and_merge(metrics_file, metadata_file, cols)
    cols   = detect_columns(df)
    df     = aggregate_nodal(df, cols)
    cols   = detect_columns(df)

    group_col = cols["group_col"]
    if not group_col:
        sys.exit("x Could not detect group column.")

    print("\nAuto-detecting groups...")
    run_spec_path = Path(args.run_spec) if args.run_spec else None
    groups = detect_or_ask_groups(df, group_col, config, run_spec_path)

    print("\nCalculating slopes...")
    slopes_df = calculate_slopes(df, cols)
    if len(slopes_df) == 0:
        sys.exit("x No slopes — subjects need ≥2 sessions.")

    all_results = {}
    for variant in ["social", "duration", "intervention"]:
        result = run_variant(slopes_df, group_col, cols["sex_col"],
                             cols["age_col"], groups, variant, output_dir)
        if result:
            all_results[variant] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for variant, res in all_results.items():
        rf  = res["random_forest"]
        svm = res["svm"]
        print(f"  {variant:15s}  RF={rf['cv_mean']:.3f}±{rf['cv_std']:.3f}  "
              f"SVM={svm['cv_mean']:.3f}±{svm['cv_std']:.3f}  Best={res['best_model']}")

    slopes_df.to_csv(output_dir / "slopes_rf_comparison.csv", index=False)
    with open(output_dir / "rf_svm_comparison_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n+ Output: {output_dir}")
    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Finished: " + _end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Duration: " + str(_duration).split(".")[0])


if __name__ == "__main__":
    main()
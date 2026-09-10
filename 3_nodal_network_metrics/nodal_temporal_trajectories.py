#!/usr/bin/env python3
"""
SCRIPT: Node-Level Temporal Trajectory Analysis
================================================

PURPOSE:
    Analyzes how individual brain regions respond to intervention over time.
    Reads node-level metrics from 02_nodal_metrics.py output.
    Intervention and control groups are auto-detected from the data.

USAGE:
    python 03_node_trajectory.py run_spec.json
    python 03_node_trajectory.py --node-metrics-dir /path/to/node_level_analysis --output-dir /path/to/out

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
from scipy import stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Local
sys.path.append(str(Path(__file__).resolve().parent.parent / "1_utilities"))
from group_detection import detect_or_ask_groups

# ============================================================================
# CLI / run_spec LOADING
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
    parser = argparse.ArgumentParser(
        description="Node trajectory analysis. Pass run_spec.json or explicit paths."
    )
    parser.add_argument("run_spec", nargs="?", help="Path to run_spec.json")
    parser.add_argument("--node-metrics-dir", help="Directory with node_level_metrics.parquet")
    parser.add_argument("--output-dir",       help="Output directory")
    parser.add_argument("--control-group",    help="Control group label (auto-detected if omitted)")
    parser.add_argument("--single-group",  action="store_true", default=False,
                        help="Single-group mode: analyze temporal changes within one group (no control needed)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config: dict = {}

    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"x run_spec not found: {spec_path}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        inputs  = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["node_metrics_dir"] = inputs.get("node_metrics_dir")
        config["output_dir"]       = outputs.get("output_dir")
        config["control_group"]    = spec.get("control_group", None)

    if args.node_metrics_dir: config["node_metrics_dir"] = args.node_metrics_dir
    if args.output_dir:       config["output_dir"]       = args.output_dir
    if args.control_group:    config["control_group"]    = args.control_group

    missing = [k for k in ("node_metrics_dir", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(
            f"x Missing required config: {', '.join(missing)}\n"
            "  Provide via run_spec.json or CLI flags (--node-metrics-dir, --output-dir)"
        )

    config["node_metrics_dir"] = Path(config["node_metrics_dir"]).expanduser().resolve()
    config["output_dir"]       = Path(config["output_dir"]).expanduser().resolve()
    if hasattr(args, 'single_group') and args.single_group:
        config["single_group"] = True
    config.setdefault("single_group", False)
    return config


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_group_col(node_df: pd.DataFrame) -> str:
    candidates = ["group", "condition", "arm", "intervention"]
    cols_lower  = {c.lower(): c for c in node_df.columns}
    for c in candidates:
        if c in cols_lower:
            print(f"  + Group column detected: {cols_lower[c]}")
            return cols_lower[c]
    sys.exit(f"x Could not detect group column. Available: {list(node_df.columns)}")


def detect_session_col(node_df: pd.DataFrame) -> str:
    candidates = ["session", "ses", "timepoint", "visit"]
    cols_lower  = {c.lower(): c for c in node_df.columns}
    for c in candidates:
        if c in cols_lower:
            print(f"  + Session column detected: {cols_lower[c]}")
            return cols_lower[c]
    sys.exit(f"x Could not detect session column. Available: {list(node_df.columns)}")


def detect_metric_cols(node_df: pd.DataFrame) -> list:
    exclude = {"subject", "session", "ses", "node", "group", "condition",
               "arm", "intervention", "sex", "gender", "age", "atlas",
               "hub_type", "community"}
    metrics = [c for c in node_df.columns
               if c.lower() not in exclude
               and pd.api.types.is_numeric_dtype(node_df[c])]
    print(f"  + Metric columns detected: {metrics}")
    return metrics


def detect_n_nodes(node_df: pd.DataFrame) -> int:
    n = int(node_df["node"].max())
    print(f"  + Number of nodes detected: {n}")
    return n


def detect_groups(node_df: pd.DataFrame, group_col: str):
    group_counts    = node_df.groupby(group_col)["subject"].nunique()
    control_group   = group_counts.idxmin()
    interv_groups   = [g for g in group_counts.index if g != control_group]
    print(f"  + Control group detected:       {control_group} (n={group_counts[control_group]})")
    print(f"  + Intervention groups detected: {interv_groups}")
    return interv_groups, control_group


# ============================================================================
# TRAJECTORY COMPUTATION
# ============================================================================

def compute_nodal_trajectories(node_df, metric_cols, group_col, session_col, n_nodes):
    records = []
    for node in range(1, n_nodes + 1):
        _progress(node, n_nodes, "Computing trajectories")
        print(f"    node {node}/{n_nodes}: fitting trajectories per metric/group...  ", end="\n", flush=True)
        node_data = node_df[node_df["node"] == node]
        if node_data.empty:
            continue
        for metric in metric_cols:
            if metric not in node_data.columns:
                continue
            for group, group_data in node_data.groupby(group_col):
                print(f"      → node {node} | {metric} | {group}...  ", end="\r", flush=True)
                sessions = group_data[session_col].values
                values   = group_data[metric].values
                valid    = ~np.isnan(values)
                if valid.sum() < 2:
                    continue
                s_v = sessions[valid].astype(float)
                y_v = values[valid]
                try:
                    coeffs      = np.polyfit(s_v, y_v, 1)
                    slope, intercept = coeffs
                    y_pred      = np.polyval(coeffs, s_v)
                    ss_res      = np.sum((y_v - y_pred) ** 2)
                    ss_tot      = np.sum((y_v - y_v.mean()) ** 2)
                    r2          = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    records.append({
                        "node":             node,
                        "metric":           metric,
                        "group":            group,
                        "slope":            slope,
                        "intercept":        intercept,
                        "r_squared":        r2,
                        "change_magnitude": float(y_v.max() - y_v.min()),
                        "change_direction": "increasing" if slope > 0 else "decreasing",
                        "n_sessions":       int(valid.sum()),
                        "mean_value":       float(y_v.mean()),
                    })
                except Exception:
                    continue
    return pd.DataFrame(records)


# ============================================================================
# EFFECT SIZE COMPUTATION
# ============================================================================

def compute_intervention_effects(trajectory_df, intervention_groups, control_group, n_nodes):
    records = []
    for metric in trajectory_df["metric"].unique():
        for node in range(1, n_nodes + 1):
            base    = trajectory_df[(trajectory_df["node"] == node) &
                                     (trajectory_df["metric"] == metric)]
            interv  = base[base["group"].isin(intervention_groups)]["slope"].dropna().values
            control = base[base["group"] == control_group]["slope"].dropna().values
            if len(interv) == 0 or len(control) == 0:
                continue
            mean_diff  = interv.mean() - control.mean()
            pooled_std = np.sqrt((interv.std() ** 2 + control.std() ** 2) / 2)
            cohens_d   = mean_diff / pooled_std if pooled_std > 0 else 0.0
            t_stat, p_val = (stats.ttest_ind(interv, control)
                             if len(interv) > 1 and len(control) > 1
                             else (np.nan, np.nan))
            records.append({
                "node":                    node,
                "metric":                  metric,
                "intervention_mean_slope": float(interv.mean()),
                "control_mean_slope":      float(control.mean()),
                "effect_size_cohens_d":    cohens_d,
                "abs_effect_size":         abs(cohens_d),
                "t_statistic":             t_stat,
                "p_value":                 p_val,
            })
    return pd.DataFrame(records)


# ============================================================================
# HUB ANALYSIS
# ============================================================================

def compute_hub_responses(node_df, trajectory_df, intervention_groups,
                           control_group, group_col, n_nodes):
    records = []
    for node in range(1, n_nodes + 1):
        hub_counts = node_df[node_df["node"] == node]["hub_type"].value_counts()
        if hub_counts.empty:
            continue
        modal_hub  = hub_counts.idxmax()
        node_trajs = trajectory_df[trajectory_df["node"] == node]
        for metric in node_trajs["metric"].unique():
            m       = node_trajs[node_trajs["metric"] == metric]
            interv  = m[m["group"].isin(intervention_groups)]["slope"].mean()
            control = m[m["group"] == control_group]["slope"].mean()
            if np.isnan(interv) or np.isnan(control):
                continue
            records.append({
                "node":                    node,
                "metric":                  metric,
                "hub_type":                modal_hub,
                "intervention_mean_slope": interv,
                "control_mean_slope":      control,
                "slope_difference":        interv - control,
            })
    return pd.DataFrame(records)


def test_hub_type_effects(hub_df):
    hub_types = hub_df["hub_type"].unique()
    for metric in hub_df["metric"].unique():
        m_data = hub_df[hub_df["metric"] == metric]
        groups = [m_data[m_data["hub_type"] == ht]["slope_difference"].values
                  for ht in hub_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            if p_val < 0.05:
                print(f"  + Hub type effect on {metric}: F={f_stat:.3f}, p={p_val:.4f}")


# ============================================================================
# PLOTS
# ============================================================================

def make_plots(effects_df, hub_df, trajectory_df, plot_dir):
    plot_dir.mkdir(exist_ok=True)

    # 1. Regional effect heatmap
    if not effects_df.empty:
        effect_matrix = effects_df.pivot_table(
            index="node", columns="metric",
            values="effect_size_cohens_d", fill_value=0
        )
        fig, ax = plt.subplots(figsize=(12, max(8, len(effect_matrix) // 10)))
        sns.heatmap(effect_matrix, cmap="RdBu_r", center=0,
                    cbar_kws={"label": "Cohen's d"}, ax=ax)
        ax.set_title("Intervention Effect Sizes - Node x Metric", fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "regional_effect_heatmap.png", dpi=150)
        plt.close()
        print("  + regional_effect_heatmap.png")

    # 2. Per-metric node effect bars
    for metric in effects_df["metric"].unique():
        m_data = effects_df[effects_df["metric"] == metric].sort_values("effect_size_cohens_d")
        colors = ["tomato" if x > 0 else "steelblue"
                  for x in m_data["effect_size_cohens_d"].values]
        fig, ax = plt.subplots(figsize=(10, max(6, len(m_data) // 8)))
        ax.barh(range(len(m_data)), m_data["effect_size_cohens_d"].values,
                color=colors, alpha=0.7)
        ax.set_yticks(range(len(m_data)))
        ax.set_yticklabels(m_data["node"].values, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Cohen's d")
        ax.set_title(f"Intervention Effect: {metric}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / f"node_effects_{metric}.png", dpi=150)
        plt.close()
        print(f"  + node_effects_{metric}.png")

    # 3. Hub type response distributions
    if not hub_df.empty:
        metrics_to_plot = hub_df["metric"].unique()[:4]
        n = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics_to_plot):
            m_hub      = hub_df[hub_df["metric"] == metric]
            hub_types  = m_hub["hub_type"].unique()
            data_groups = [m_hub[m_hub["hub_type"] == ht]["slope_difference"].values
                           for ht in hub_types]
            bp = ax.boxplot(data_groups, labels=hub_types, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Hub Response: {metric}", fontweight="bold")
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(plot_dir / "hub_type_response_distributions.png", dpi=150)
        plt.close()
        print("  + hub_type_response_distributions.png")

    # 4. Trajectory examples (top 6 nodes)
    if not effects_df.empty and not trajectory_df.empty:
        top_nodes = effects_df.nlargest(6, "abs_effect_size")["node"].unique()
        metric_ex = trajectory_df["metric"].iloc[0]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, node in enumerate(top_nodes):
            ax = axes[i]
            node_trajs = trajectory_df[(trajectory_df["node"] == node) &
                                        (trajectory_df["metric"] == metric_ex)]
            for _, traj in node_trajs.iterrows():
                x = np.linspace(1, traj["n_sessions"], 20)
                y = traj["intercept"] + traj["slope"] * x
                ax.plot(x, y, label=str(traj["group"]), linewidth=2)
            ax.set_title(f"Node {node} - {metric_ex}", fontweight="bold")
            ax.set_xlabel("Session")
            ax.set_ylabel(metric_ex)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "trajectory_examples_top_nodes.png", dpi=150)
        plt.close()
        print("  + trajectory_examples_top_nodes.png")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = load_config(args)

    node_metrics_dir = config["node_metrics_dir"]
    output_dir       = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("NODE-LEVEL TEMPORAL TRAJECTORY ANALYSIS")
    print("=" * 70)
    print(f"Input:  {node_metrics_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading node-level metrics...")
    node_file = node_metrics_dir / "node_level_metrics.parquet"
    if not node_file.exists():
        sys.exit(f"x node_level_metrics.parquet not found in: {node_metrics_dir}")
    node_df = pd.read_parquet(node_file)
    print(f"  + {len(node_df)} node records loaded")

    # Auto-detection
    print("\nAuto-detecting data structure...")
    group_col = detect_group_col(node_df)
    session_col = detect_session_col(node_df)
    metric_cols = detect_metric_cols(node_df)
    n_nodes = detect_n_nodes(node_df)
    run_spec_path = Path(args.run_spec) if args.run_spec else None
    # ── Single-group or multi-group mode ──────────────────────────────────
    all_groups = node_df[group_col].dropna().unique().tolist() if group_col else []
    single_group_mode = config.get("single_group", False) or len(all_groups) <= 1

    if single_group_mode:
        print(f"  + Single-group mode: analyzing temporal changes within group(s): {all_groups}")
        intervention_groups = all_groups
        control_group       = all_groups[0] if all_groups else None
    else:
        groups = detect_or_ask_groups(node_df, group_col, config, run_spec_path)
        intervention_groups = groups["intervention"]
        control_group       = groups["control"]
    print()

    # Trajectories
    print("Computing nodal trajectories...")
    trajectory_df = compute_nodal_trajectories(
        node_df, metric_cols, group_col, session_col, n_nodes
    )
    print(f"  + {len(trajectory_df)} trajectories computed")
    trajectory_df.to_parquet(output_dir / "node_trajectories.parquet", index=False)
    print("  + Saved: node_trajectories.parquet")

    # Intervention effects / Single-group temporal effects
    if single_group_mode:
        print("\nComputing temporal effect sizes (single-group)...")
        # trajectory_df already has slope per node/metric/group
        # Use slope magnitude and change_direction as effect size
        records = []
        for _, row in trajectory_df.iterrows():
            # Normalize slope by mean_value to get relative change
            mean_val = abs(float(row["mean_value"])) if row["mean_value"] != 0 else 1.0
            rel_slope = float(row["slope"]) / mean_val
            records.append({
                "node":                 int(row["node"]),
                "metric":               str(row["metric"]),
                "mean_slope":           float(row["slope"]),
                "change_magnitude":     float(row["change_magnitude"]),
                "change_direction":     str(row["change_direction"]),
                "relative_slope":       float(rel_slope),
                "abs_effect_size":      float(abs(rel_slope)),
                "effect_size_cohens_d": float(rel_slope),
                "r_squared":            float(row["r_squared"]),
                "n_sessions":           int(row["n_sessions"]),
                "significant":          bool(abs(rel_slope) > 0.01),
            })
        effects_df = pd.DataFrame(records)
        print(f"  + {len(effects_df)} node x metric temporal effects computed")
        effects_df.to_parquet(output_dir / "temporal_effect_sizes.parquet", index=False)
        effects_df.to_csv(output_dir / "temporal_effect_sizes.csv", index=False)
        print("  + Saved: temporal_effect_sizes.parquet + .csv")

        # Top nodes
        if not effects_df.empty:
            available_cols = ["node", "metric", "mean_slope", "abs_effect_size",
                                "effect_size_cohens_d", "change_direction", "r_squared", "significant"]
            top_cols = [c for c in available_cols if c in effects_df.columns]
            top = effects_df.nlargest(20, "abs_effect_size")[top_cols]
            print("\nTop 20 nodes with strongest temporal changes:")
            print(top.to_string(index=False))
            top.to_csv(output_dir / "top_temporal_nodes.csv", index=False)
            print("  + Saved: top_temporal_nodes.csv")

        hub_df = pd.DataFrame()
    else:
        print("\nComputing intervention effects...")
        effects_df = compute_intervention_effects(
            trajectory_df, intervention_groups, control_group, n_nodes
        )
        print(f"  + {len(effects_df)} node x metric effects computed")
        effects_df.to_parquet(output_dir / "intervention_effect_sizes.parquet", index=False)
        print("  + Saved: intervention_effect_sizes.parquet")

    # Hub analysis
    print("\nAnalyzing hub-specific responses...")
    hub_df = compute_hub_responses(
        node_df, trajectory_df, intervention_groups, control_group, group_col, n_nodes
    ) if not single_group_mode else pd.DataFrame()
    if not hub_df.empty:
        test_hub_type_effects(hub_df)
        hub_df.to_parquet(output_dir / "hub_specific_responses.parquet", index=False)
        print("  + Saved: hub_specific_responses.parquet")

    # Plots
    print("\nCreating plots...")
    if effects_df.empty:
        print("  ⚠ No effects to plot — skipping plots")
    else:
        make_plots(effects_df, hub_df, trajectory_df, output_dir / "plots")

    # Summary
    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Finished: " + _end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Duration: " + str(_duration).split(".")[0])
    if not effects_df.empty:
        print(f"Nodes with strong effects (|d|>0.5):   {(effects_df['abs_effect_size'] > 0.5).sum()}")
        print(f"Nodes with moderate effects (|d|>0.2): {(effects_df['abs_effect_size'] > 0.2).sum()}")
        print("\nTop 5 responding nodes:")
        for _, row in effects_df.nlargest(5, "abs_effect_size").iterrows():
            print(f"  Node {int(row['node']):3d} - {row['metric']:20s}: d = {row['effect_size_cohens_d']:.3f}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
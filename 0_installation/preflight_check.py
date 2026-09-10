#!/usr/bin/env python3
"""
Preflight checks for pipeline runs.

- Validates run_spec.json exists and is readable
- Checks required input/output paths
- Verifies required Python packages are installed (static list)
- Optionally checks temporal analysis dependencies

Usage:
  python 0_installation/preflight_check.py /path/to/run_spec.json
  python 0_installation/preflight_check.py /path/to/run_spec.json --temporal
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


# Core required packages
REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "bct",
    "scipy",
    "matplotlib",
    "seaborn",
    "umap",
    "sklearn",
]

# Packages specifically for temporal analysis pipeline
TEMPORAL_PACKAGES = [
    "umap_learn",
    "sklearn",
]


def load_spec(spec_path: Path) -> dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"Run spec not found: {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    spec["_spec_dir"] = str(spec_path.parent)
    return spec


def resolve_path(spec_dir: Path, raw_path: str) -> Path:
    raw_path = os.path.expandvars(raw_path)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (spec_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def check_imports(requirements: list[str]) -> list[str]:
    """Check if packages are importable. Returns list of missing packages."""
    missing = []
    for pkg in requirements:
        try:
            # Use find_spec instead of import_module to avoid slow JIT compilation
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                missing.append(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def check_optional_imports(requirements: list[str]) -> list[str]:
    """Check optional packages but don't fail if missing."""
    return check_imports(requirements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for pipeline execution")
    parser.add_argument("run_spec", help="Path to run_spec.json")
    parser.add_argument(
        "--temporal", action="store_true", 
        help="Check temporal analysis dependencies"
    )
    parser.add_argument(
        "--uv-lock", action="store_true",
        help="Check if uv.lock exists for reproducible installs"
    )
    args = parser.parse_args()

    spec_path = Path(args.run_spec).expanduser().resolve()
    spec = load_spec(spec_path)

    # Check Python environment if specified in spec
    venv_python = spec.get("venv_python")
    if venv_python and Path(sys.executable).resolve() != Path(venv_python).resolve():
        print(f"✗ Wrong Python: {sys.executable}\n  Run with: {venv_python}", file=sys.stderr)
        sys.exit(3)

    spec_dir = Path(spec["_spec_dir"]).resolve()

    # Check script path if specified
    script_path = None
    script_raw = spec.get("script")
    if script_raw:
        script_path = resolve_path(spec_dir, script_raw)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

    # Check inputs and outputs
    inputs = spec.get("inputs", {})
    outputs = spec.get("outputs", {})

    required_inputs = [k for k in ("data_dir", "metadata_file") if not inputs.get(k)]
    if required_inputs:
        raise ValueError(
            "Missing required inputs in run_spec: " + ", ".join(required_inputs)
        )
    if not outputs.get("output_dir"):
        raise ValueError("Missing required outputs.output_dir in run_spec")

    data_dir = resolve_path(spec_dir, inputs.get("data_dir", ""))
    metadata_file = resolve_path(spec_dir, inputs.get("metadata_file", ""))
    output_dir = resolve_path(spec_dir, outputs.get("output_dir", ""))

    # Validate paths
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    # Check required packages
    missing = check_imports(REQUIRED_PACKAGES)

    # Print status
    if script_path is not None:
        print("✓ Script found:", script_path)
    else:
        print("! Script not specified in run_spec (path checks only)")
    print("✓ Data directory:", data_dir)
    print("✓ Metadata file:", metadata_file)
    print("✓ Output directory (will be created if needed):", output_dir)

    # Report missing packages
    if missing:
        print("✗ Missing required packages:", ", ".join(missing))
        sys.exit(2)

    print("✓ All required packages are installed")
    
    # Check temporal analysis packages if requested
    if args.temporal:
        temporal_missing = check_optional_imports(TEMPORAL_PACKAGES)
        if temporal_missing:
            print("⚠ Optional temporal analysis packages missing:", ", ".join(temporal_missing))
            print("  Install with: uv pip install umap-learn scikit-learn")
        else:
            print("✓ All temporal analysis packages are installed")
    
    # Check uv.lock if requested
    if args.uv_lock:
        # uv.lock is at repository root, not in 0_installation/
        uv_lock_path = spec_dir.parent / "uv.lock"
        if uv_lock_path.exists():
            print("✓ uv.lock found (reproducible environment configured)")
        else:
            print("⚠ uv.lock not found at repo root")
            print("  Generate with: uv pip compile pyproject.toml --all-extras -o uv.lock")


if __name__ == "__main__":
    main()

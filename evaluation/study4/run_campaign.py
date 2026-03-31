#!/usr/bin/env python3
"""Experiment 4.1: Full Campaign Comparison.

Runs BBA benchmark campaigns in two modes:
  1. Baseline: no freeze (standard DeepDriveMD workflow)
  2. Dynamic: signal monitor + freeze + GPU reclaim

Both run with the same wall-clock budget and GPU allocation.
Results are saved for analysis by analyze_results.py.

Usage:
    python run_campaign.py --config campaign.yaml --mode [baseline|dynamic|both]
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


def create_baseline_config(base_config: dict, run_dir: Path) -> Path:
    """Create a config for baseline (no freeze) campaign."""
    config = base_config.copy()
    config["run_dir"] = str(run_dir)
    # No freeze mechanism
    config.pop("freeze_after_n_trains", None)
    config.pop("policy_name", None)
    config.pop("cooldown_sec", None)

    config_path = run_dir / "config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def create_dynamic_config(base_config: dict, run_dir: Path) -> Path:
    """Create a config for dynamic provisioning campaign."""
    config = base_config.copy()
    config["run_dir"] = str(run_dir)
    # Enable dynamic provisioning defaults if not set
    config.setdefault("policy_name", "sliding_window")
    config.setdefault("policy_window_size", 10)
    config.setdefault("policy_loss_threshold", 0.05)
    config.setdefault("policy_min_improvement", 0.001)
    config.setdefault("cooldown_sec", 30.0)
    config.setdefault("signal_window_size", 50)

    config_path = run_dir / "config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def run_workflow(config_path: Path, workflow_script: str, label: str) -> dict:
    """Run a workflow campaign and return timing metadata."""
    logging.info(f"Starting {label} campaign with config: {config_path}")

    t_start = time.time()
    result = subprocess.run(
        [sys.executable, workflow_script, "-c", str(config_path)],
        capture_output=True, text=True, timeout=7200,  # 2 hour max
    )
    t_end = time.time()

    run_dir = config_path.parent
    metadata = {
        "label": label,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "wall_clock_s": t_end - t_start,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }

    # Save stdout/stderr
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "stderr.log").write_text(result.stderr)

    # Save metadata
    with open(run_dir / "campaign_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if result.returncode != 0:
        logging.error(f"{label} campaign failed (rc={result.returncode})")
        logging.error(f"stderr: {result.stderr[:500]}")
    else:
        logging.info(f"{label} campaign completed in {t_end - t_start:.1f}s")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Run comparison campaigns")
    parser.add_argument("--config", type=str, required=True,
                        help="Base campaign configuration YAML")
    parser.add_argument("--mode", choices=["baseline", "dynamic", "both"],
                        default="both", help="Which campaigns to run")
    parser.add_argument("--output-dir", type=str, default="campaign_results",
                        help="Base output directory for results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    workflow_baseline = str(
        Path(__file__).parent.parent.parent / "deepdrivemd" / "workflows" / "openmm_cvae.py"
    )
    workflow_dynamic = str(
        Path(__file__).parent.parent.parent / "deepdrivemd" / "workflows" / "openmm_cvae_dynamic.py"
    )

    if args.mode in ("baseline", "both"):
        baseline_dir = output_base / "baseline"
        baseline_config = create_baseline_config(base_config, baseline_dir)
        meta = run_workflow(baseline_config, workflow_baseline, "baseline")
        results.append(meta)

    if args.mode in ("dynamic", "both"):
        dynamic_dir = output_base / "dynamic"
        dynamic_config = create_dynamic_config(base_config, dynamic_dir)
        meta = run_workflow(dynamic_config, workflow_dynamic, "dynamic")
        results.append(meta)

    # Save combined results
    summary_path = output_base / "campaign_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Campaign summary saved to {summary_path}")

    # Print summary
    print("\n=== Campaign Results ===")
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        print(f"  {r['label']}: {status} ({r['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()

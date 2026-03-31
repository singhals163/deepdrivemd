#!/usr/bin/env python3
"""Experiment 4.2: Campaign Analysis.

Parses Colmena result JSONs (simulation.json, train.json, inference.json)
from baseline and dynamic campaigns, computes comparison metrics, and
generates a summary table.

Metrics:
  - Total wall-clock time
  - GPU-hours (total and AI-only)
  - Training cycles completed
  - Simulation throughput (sims/min)
  - Near-native yield (if RMSD data available)
  - AI staleness (time since last training)

Usage:
    python analyze_results.py --baseline /path/to/baseline/result --dynamic /path/to/dynamic/result
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def parse_colmena_results(result_dir: Path) -> Dict[str, Any]:
    """Parse Colmena result JSON files from a campaign run."""
    metrics = {
        "result_dir": str(result_dir),
        "simulation": {"tasks": [], "count": 0, "total_time_s": 0},
        "train": {"tasks": [], "count": 0, "total_time_s": 0},
        "inference": {"tasks": [], "count": 0, "total_time_s": 0},
    }

    for task_type in ["simulation", "train", "inference"]:
        json_path = result_dir / f"{task_type}.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if not record.get("success", False):
                    continue

                timing = record.get("time", {})
                task_info = {
                    "method": record.get("method", ""),
                    "created": timing.get("created", 0),
                    "running_start": timing.get("running_start", 0),
                    "running_end": timing.get("running_end", 0),
                    "completed": timing.get("completed", 0),
                    "running": timing.get("running", 0),
                }
                metrics[task_type]["tasks"].append(task_info)
                metrics[task_type]["count"] += 1
                metrics[task_type]["total_time_s"] += task_info["running"]

    return metrics


def compute_campaign_metrics(raw: Dict[str, Any], n_gpus: int = 8) -> Dict[str, Any]:
    """Compute high-level metrics from parsed Colmena results."""
    sim_tasks = raw["simulation"]["tasks"]
    train_tasks = raw["train"]["tasks"]

    # Wall clock: from first task created to last task completed
    all_tasks = sim_tasks + train_tasks + raw["inference"]["tasks"]
    if not all_tasks:
        return {"error": "No tasks found"}

    t_start = min(t["created"] for t in all_tasks)
    t_end = max(t["completed"] for t in all_tasks)
    wall_clock_s = t_end - t_start

    # Simulation throughput
    sim_count = raw["simulation"]["count"]
    sim_throughput = sim_count / (wall_clock_s / 60) if wall_clock_s > 0 else 0

    # GPU-hours
    total_gpu_hours = (wall_clock_s * n_gpus) / 3600
    ai_gpu_seconds = raw["train"]["total_time_s"] + raw["inference"]["total_time_s"]
    ai_gpu_hours = ai_gpu_seconds / 3600
    ai_fraction = ai_gpu_seconds / (wall_clock_s * n_gpus) if wall_clock_s > 0 else 0

    # Training metrics
    train_count = raw["train"]["count"]
    avg_train_time = (raw["train"]["total_time_s"] / train_count) if train_count > 0 else 0

    # Staleness: average time between training completions
    if len(train_tasks) >= 2:
        train_completions = sorted(t["completed"] for t in train_tasks)
        gaps = np.diff(train_completions)
        avg_staleness = float(np.mean(gaps))
        max_staleness = float(np.max(gaps))
    else:
        avg_staleness = wall_clock_s
        max_staleness = wall_clock_s

    return {
        "wall_clock_s": wall_clock_s,
        "wall_clock_min": wall_clock_s / 60,
        "simulation_count": sim_count,
        "simulation_throughput_spm": sim_throughput,
        "train_count": train_count,
        "avg_train_time_s": avg_train_time,
        "inference_count": raw["inference"]["count"],
        "total_gpu_hours": total_gpu_hours,
        "ai_gpu_hours": ai_gpu_hours,
        "ai_fraction": ai_fraction,
        "avg_staleness_s": avg_staleness,
        "max_staleness_s": max_staleness,
    }


def load_dynamic_stats(run_dir: Path) -> Optional[Dict]:
    """Load dynamic provisioning stats if available."""
    stats_path = run_dir / "dynamic_provisioning_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze campaign results")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline campaign result directory")
    parser.add_argument("--dynamic", type=str, required=True,
                        help="Path to dynamic campaign result directory")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Total number of GPUs in the allocation")
    parser.add_argument("--output", type=str, default="comparison_results.json")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    dynamic_dir = Path(args.dynamic)

    print("=== Parsing Campaign Results ===")

    baseline_raw = parse_colmena_results(baseline_dir)
    dynamic_raw = parse_colmena_results(dynamic_dir)

    baseline_metrics = compute_campaign_metrics(baseline_raw, args.n_gpus)
    dynamic_metrics = compute_campaign_metrics(dynamic_raw, args.n_gpus)

    # Load dynamic provisioning stats
    dynamic_stats = load_dynamic_stats(dynamic_dir.parent)

    # Comparison
    comparison = {}
    if "error" not in baseline_metrics and "error" not in dynamic_metrics:
        bm = baseline_metrics
        dm = dynamic_metrics

        comparison = {
            "throughput_change_pct": (
                (dm["simulation_throughput_spm"] - bm["simulation_throughput_spm"])
                / bm["simulation_throughput_spm"] * 100
                if bm["simulation_throughput_spm"] > 0 else 0
            ),
            "ai_gpu_hours_saved": bm["ai_gpu_hours"] - dm["ai_gpu_hours"],
            "ai_gpu_hours_saved_pct": (
                (bm["ai_gpu_hours"] - dm["ai_gpu_hours"]) / bm["ai_gpu_hours"] * 100
                if bm["ai_gpu_hours"] > 0 else 0
            ),
            "staleness_change_s": dm["avg_staleness_s"] - bm["avg_staleness_s"],
            "extra_simulations": dm["simulation_count"] - bm["simulation_count"],
        }

    results = {
        "baseline": baseline_metrics,
        "dynamic": dynamic_metrics,
        "comparison": comparison,
        "dynamic_provisioning_stats": dynamic_stats,
    }

    # Print comparison table
    print("\n=== Campaign Comparison ===")
    print(f"{'Metric':<35} {'Baseline':>15} {'Dynamic':>15} {'Delta':>15}")
    print("-" * 82)

    if "error" not in baseline_metrics and "error" not in dynamic_metrics:
        rows = [
            ("Wall clock (min)", "wall_clock_min", ".1f"),
            ("Simulations", "simulation_count", "d"),
            ("Throughput (sims/min)", "simulation_throughput_spm", ".2f"),
            ("Training cycles", "train_count", "d"),
            ("Avg train time (s)", "avg_train_time_s", ".1f"),
            ("Total GPU-hours", "total_gpu_hours", ".2f"),
            ("AI GPU-hours", "ai_gpu_hours", ".2f"),
            ("AI fraction", "ai_fraction", ".3f"),
            ("Avg staleness (s)", "avg_staleness_s", ".1f"),
            ("Max staleness (s)", "max_staleness_s", ".1f"),
        ]
        for label, key, fmt in rows:
            bv = baseline_metrics.get(key, 0)
            dv = dynamic_metrics.get(key, 0)
            delta = dv - bv
            sign = "+" if delta > 0 else ""
            print(f"{label:<35} {bv:>15{fmt}} {dv:>15{fmt}} {sign}{delta:>14{fmt}}")

        if comparison:
            print(f"\n{'Key Comparisons':}")
            print(f"  Throughput change: {comparison['throughput_change_pct']:+.1f}%")
            print(f"  AI GPU-hours saved: {comparison['ai_gpu_hours_saved']:.2f} "
                  f"({comparison['ai_gpu_hours_saved_pct']:.1f}%)")
            print(f"  Extra simulations: {comparison['extra_simulations']}")
    else:
        for label, metrics in [("Baseline", baseline_metrics), ("Dynamic", dynamic_metrics)]:
            if "error" in metrics:
                print(f"  {label}: {metrics['error']}")

    # Dynamic provisioning details
    if dynamic_stats:
        sm = dynamic_stats.get("signal_monitor", {})
        rb = dynamic_stats.get("resource_broker", {})
        print(f"\n{'Dynamic Provisioning Details':}")
        print(f"  Signals processed: {sm.get('signals_received', 'N/A')}")
        print(f"  State transitions: {sm.get('transitions_issued', 'N/A')}")
        print(f"  GPU freezes: {rb.get('freeze_count', 'N/A')}")
        print(f"  GPU resumes: {rb.get('resume_count', 'N/A')}")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

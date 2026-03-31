#!/usr/bin/env python3
"""Experiment 3.2: Throughput Recovery.

Simulates the throughput impact of reclaiming the ML GPU for simulation
work. Compares simulation throughput before freeze (N GPUs) vs after
freeze (N+1 GPUs) using a synthetic workload model.

In standalone mode, models throughput analytically based on GPU count
and measured task latencies. With real Colmena results, parses
simulation.json task timestamps to compute actual throughput.

Usage:
    python bench_throughput.py [--sim-json /path/to/simulation.json] [--output results_throughput.json]
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def analytical_throughput(n_gpus: int, sim_time_s: float, overhead_s: float = 0.1) -> float:
    """Compute expected throughput (sims/minute) for a given GPU count.

    Assumes each GPU runs one simulation at a time with uniform task duration.
    """
    task_time = sim_time_s + overhead_s
    sims_per_gpu_per_min = 60.0 / task_time
    return n_gpus * sims_per_gpu_per_min


def parse_colmena_throughput(sim_json_path: Path, freeze_time_s: Optional[float] = None) -> dict:
    """Parse simulation.json and compute throughput before/after freeze.

    Parameters
    ----------
    sim_json_path : Path
        Path to Colmena simulation.json results file.
    freeze_time_s : float, optional
        Wall-clock time of freeze event (seconds from campaign start).
        If None, splits at the midpoint of task timestamps.
    """
    tasks = []
    with open(sim_json_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("success"):
                start = record.get("time", {}).get("created", 0)
                end = record.get("time", {}).get("completed", start)
                running = record.get("time", {}).get("running", end - start)
                tasks.append({
                    "start": start,
                    "end": end,
                    "running": running,
                    "method": record.get("method", ""),
                })

    if not tasks:
        return {"error": "No successful simulation tasks found"}

    tasks.sort(key=lambda t: t["start"])
    t_min = tasks[0]["start"]
    t_max = tasks[-1]["end"]

    if freeze_time_s is None:
        freeze_time_s = (t_min + t_max) / 2

    before = [t for t in tasks if t["end"] <= freeze_time_s]
    after = [t for t in tasks if t["start"] >= freeze_time_s]

    before_duration = (freeze_time_s - t_min) / 60.0 if before else 0
    after_duration = (t_max - freeze_time_s) / 60.0 if after else 0

    return {
        "total_tasks": len(tasks),
        "before_freeze": {
            "task_count": len(before),
            "duration_min": before_duration,
            "throughput_sims_per_min": len(before) / before_duration if before_duration > 0 else 0,
        },
        "after_freeze": {
            "task_count": len(after),
            "duration_min": after_duration,
            "throughput_sims_per_min": len(after) / after_duration if after_duration > 0 else 0,
        },
        "freeze_time_s": freeze_time_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Throughput recovery benchmark")
    parser.add_argument("--sim-json", type=str, default=None,
                        help="Path to Colmena simulation.json for real throughput analysis")
    parser.add_argument("--n-sim-gpus", type=int, default=7,
                        help="Number of simulation GPUs before freeze")
    parser.add_argument("--sim-time-s", type=float, default=1620.0,
                        help="Mean simulation task duration in seconds (default: 27 min)")
    parser.add_argument("--output", type=str, default="results_throughput.json")
    args = parser.parse_args()

    results = {}

    # Analytical model
    n_sim = args.n_sim_gpus
    n_total = n_sim + 1  # After reclaim

    tp_before = analytical_throughput(n_sim, args.sim_time_s)
    tp_after = analytical_throughput(n_total, args.sim_time_s)
    improvement = (tp_after - tp_before) / tp_before * 100

    results["analytical"] = {
        "sim_gpus_before": n_sim,
        "sim_gpus_after": n_total,
        "sim_time_s": args.sim_time_s,
        "throughput_before_spm": tp_before,
        "throughput_after_spm": tp_after,
        "improvement_pct": improvement,
        "expected_improvement_pct": (1.0 / n_sim) * 100,
    }

    print("=== Analytical Throughput Model ===")
    print(f"  Before freeze: {n_sim} GPUs -> {tp_before:.2f} sims/min")
    print(f"  After freeze:  {n_total} GPUs -> {tp_after:.2f} sims/min")
    print(f"  Improvement:   {improvement:.1f}% (expected: {(1.0/n_sim)*100:.1f}%)")

    # Also test with shorter sim times typical of BBA
    short_configs = [
        ("BBA (1 min)", 60.0),
        ("CLN025 (5 min)", 300.0),
        ("KRAS (27 min)", 1620.0),
    ]

    scaling_results = []
    print("\n=== Scaling Across Simulation Durations ===")
    print(f"{'System':<20} {'Sim (s)':>8} {'Before (spm)':>13} {'After (spm)':>12} {'Improvement':>12}")
    print("-" * 68)
    for name, sim_s in short_configs:
        tp_b = analytical_throughput(n_sim, sim_s)
        tp_a = analytical_throughput(n_total, sim_s)
        imp = (tp_a - tp_b) / tp_b * 100
        print(f"{name:<20} {sim_s:>8.0f} {tp_b:>13.2f} {tp_a:>12.2f} {imp:>11.1f}%")
        scaling_results.append({
            "system": name,
            "sim_time_s": sim_s,
            "throughput_before": tp_b,
            "throughput_after": tp_a,
            "improvement_pct": imp,
        })

    results["scaling"] = scaling_results

    # Real data analysis if available
    if args.sim_json:
        sim_path = Path(args.sim_json)
        if sim_path.exists():
            print(f"\n=== Real Throughput from {sim_path.name} ===")
            real = parse_colmena_throughput(sim_path)
            results["real_data"] = real
            if "error" not in real:
                print(f"  Before freeze: {real['before_freeze']['throughput_sims_per_min']:.2f} sims/min "
                      f"({real['before_freeze']['task_count']} tasks)")
                print(f"  After freeze:  {real['after_freeze']['throughput_sims_per_min']:.2f} sims/min "
                      f"({real['after_freeze']['task_count']} tasks)")
            else:
                print(f"  {real['error']}")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

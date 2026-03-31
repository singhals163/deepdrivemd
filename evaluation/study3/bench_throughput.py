#!/usr/bin/env python3
"""Experiment 3.2: Throughput Recovery.

Measures real simulation throughput from Colmena experiment runs, comparing
throughput across campaigns. Also computes analytical throughput model
using real measured simulation durations.

Usage:
    python bench_throughput.py [--runs-dir /path/to/runs] [--output results_throughput.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    discover_experiments,
    load_all_task_durations,
)


def analytical_throughput(n_gpus: int, sim_time_s: float, overhead_s: float = 0.1) -> float:
    """Compute expected throughput (sims/minute) for a given GPU count."""
    task_time = sim_time_s + overhead_s
    return n_gpus * 60.0 / task_time


def parse_colmena_throughput(sim_json_path: Path) -> dict:
    """Parse simulation.json and compute throughput."""
    tasks = []
    with open(sim_json_path) as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if record.get("success"):
                start = record.get("time_created", 0)
                end = record.get("time_compute_ended", start)
                running = record.get("time_running", end - start)
                tasks.append({
                    "start": start,
                    "end": end,
                    "running": running,
                })

    if not tasks:
        return {"error": "No successful simulation tasks found"}

    tasks.sort(key=lambda t: t["start"])
    t_min = tasks[0]["start"]
    t_max = tasks[-1]["end"]
    wall_clock_s = t_max - t_min
    wall_clock_min = wall_clock_s / 60.0

    return {
        "total_tasks": len(tasks),
        "wall_clock_s": wall_clock_s,
        "wall_clock_min": wall_clock_min,
        "throughput_sims_per_min": len(tasks) / wall_clock_min if wall_clock_min > 0 else 0,
        "mean_task_duration_s": float(np.mean([t["running"] for t in tasks])),
    }


def main():
    parser = argparse.ArgumentParser(description="Throughput recovery benchmark")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--n-sim-gpus", type=int, default=7)
    parser.add_argument("--output", type=str, default="results_throughput.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results = {}

    # Parse real throughput from all experiments
    print("=== Real Throughput from Colmena Runs ===")
    experiments = discover_experiments(runs_dir)
    per_experiment = []
    throughputs = []

    for exp_dir in experiments:
        sim_json = exp_dir / "result" / "simulation.json"
        if not sim_json.exists():
            continue
        tp = parse_colmena_throughput(sim_json)
        if "error" in tp:
            continue
        per_experiment.append({
            "experiment": exp_dir.name,
            **tp,
        })
        throughputs.append(tp["throughput_sims_per_min"])
        print(f"  {exp_dir.name}: {tp['total_tasks']} sims in {tp['wall_clock_min']:.1f}min "
              f"= {tp['throughput_sims_per_min']:.2f} sims/min "
              f"(mean task={tp['mean_task_duration_s']:.1f}s)")

    if throughputs:
        results["real_data"] = {
            "experiments": per_experiment,
            "throughput": compute_stats(np.array(throughputs), unit="spm"),
        }
        print(f"\n  Aggregate: {results['real_data']['throughput']['mean_spm']:.2f} "
              f"+/- {results['real_data']['throughput']['ci95_spm']:.2f} sims/min "
              f"(N={len(throughputs)} experiments)")

    # Analytical model using real simulation durations
    sim_durations = load_all_task_durations(runs_dir, method="run_simulation")
    real_sim_time = float(np.mean(sim_durations)) if len(sim_durations) > 0 else 60.0
    print(f"\n=== Analytical Model (real sim duration: {real_sim_time:.1f}s) ===")

    n_sim = args.n_sim_gpus
    n_total = n_sim + 1
    tp_before = analytical_throughput(n_sim, real_sim_time)
    tp_after = analytical_throughput(n_total, real_sim_time)
    improvement = (tp_after - tp_before) / tp_before * 100

    results["analytical"] = {
        "sim_gpus_before": n_sim,
        "sim_gpus_after": n_total,
        "real_sim_time_s": real_sim_time,
        "n_sim_measurements": len(sim_durations),
        "throughput_before_spm": tp_before,
        "throughput_after_spm": tp_after,
        "improvement_pct": improvement,
        "expected_improvement_pct": (1.0 / n_sim) * 100,
    }

    print(f"  Before freeze: {n_sim} GPUs -> {tp_before:.2f} sims/min")
    print(f"  After freeze:  {n_total} GPUs -> {tp_after:.2f} sims/min")
    print(f"  Improvement:   {improvement:.1f}%")

    results["data_source"] = "real_colmena_runs"

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

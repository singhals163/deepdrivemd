#!/usr/bin/env python3
"""Experiment 1.1: Signal Monitor Interface Overhead.

Measures wall-clock latency of submit_telemetry() for three policies of
increasing computational complexity.

Supports --synthetic flag to use generated workloads alongside real data.

Usage:
    python bench_latency.py --runs-dir /path/to/runs --output real/results_latency.json
    python bench_latency.py --synthetic --output synthetic/results_latency.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import (
    MannKendallPolicy,
    SlidingWindowPolicy,
    ThresholdPolicy,
)
from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    load_loss_curves,
    synthetic_loss_curves,
)


def measure_latency(policy, telemetry_pool: np.ndarray, n_repeats: int) -> dict:
    """Measure per-call latency for a given policy."""
    monitor = SignalMonitor(policy=policy, window_size=50)

    n_pool = len(telemetry_pool)
    telemetry = [
        np.array([telemetry_pool[i % n_pool], i])
        for i in range(n_repeats)
    ]

    for v in telemetry[:50]:
        monitor.submit_telemetry(v)

    latencies_ns = []
    for v in telemetry[50:]:
        t0 = time.perf_counter_ns()
        monitor.submit_telemetry(v)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    latencies_ms = np.array(latencies_ns) / 1e6
    return compute_stats(latencies_ms, unit="ms")


def run_benchmark(telemetry_pool, data_source, n_curves, policies, trials, repeats):
    """Run the latency benchmark and return results list."""
    results = []
    for label, policy in policies:
        print(f"\n  {label} ({trials} trials x {repeats} repeats)")
        trial_means, trial_p99s, trial_maxs = [], [], []

        for _ in range(trials):
            stats = measure_latency(policy, telemetry_pool, repeats)
            trial_means.append(stats["mean_ms"])
            trial_p99s.append(stats["p99_ms"])
            trial_maxs.append(stats["max_ms"])

        entry = {
            "label": label,
            "policy": policy.name,
            "data_source": data_source,
            "n_loss_curves": n_curves,
            "n_trials": trials,
            "n_repeats_per_trial": repeats,
            **compute_stats(np.array(trial_means), unit="ms"),
            "p99": compute_stats(np.array(trial_p99s), unit="ms"),
            "max": compute_stats(np.array(trial_maxs), unit="ms"),
        }
        results.append(entry)
        print(f"    mean={entry['mean_ms']:.4f} +/- {entry['ci95_ms']:.4f}ms  "
              f"p99={entry['p99']['mean_ms']:.4f}ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark signal monitor latency")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic workload instead of real data")
    parser.add_argument("--output", type=str, default="results_latency.json")
    args = parser.parse_args()

    policies = [
        ("Threshold O(1)", ThresholdPolicy(loss_threshold=0.05)),
        ("SlidingWindow O(Nk)", SlidingWindowPolicy(window_size=10)),
        ("MannKendall O(Nk²)", MannKendallPolicy(window_size=20)),
    ]

    if args.synthetic:
        print("=== Synthetic workload ===")
        curves = synthetic_loss_curves(n_curves=10, n_epochs=20)
        pool = np.concatenate(curves)
        print(f"Generated {len(curves)} synthetic loss curves ({len(pool)} values)")
        results = run_benchmark(pool, "synthetic", len(curves),
                                policies, args.trials, args.repeats)
    else:
        print("=== Real data ===")
        curves = load_loss_curves(Path(args.runs_dir))
        if not curves:
            print("ERROR: No training loss data found.")
            return
        pool = np.concatenate(curves)
        print(f"Loaded {len(curves)} real loss curves ({len(pool)} values)")
        results = run_benchmark(pool, "real_training_loss", len(curves),
                                policies, args.trials, args.repeats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

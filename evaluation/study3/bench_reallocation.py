#!/usr/bin/env python3
"""Experiment 3.1: Reallocation Latency.

Measures the broker's internal transition latency with in-flight tasks
parameterized by real ML task durations from DeepDriveMD runs.

Usage:
    python bench_reallocation.py [--runs-dir /path/to/runs] [--repeats 20] [--output results_reallocation.json]
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from deepdrivemd.resource_broker.broker import GPUState, ResourceBroker
from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    load_all_task_durations,
)

SYNTHETIC_DRAIN_DELAYS = [0.0, 0.01, 0.01, 0.01]  # Original synthetic values

# Scale factor: real train durations are ~10s, scale down for test speed
SCALE_FACTOR = 0.001


def measure_transition_latency(n_inflight: int, drain_delay_s: float = 0.01) -> dict:
    """Measure freeze transition latency with simulated in-flight ML tasks."""
    broker = ResourceBroker(cooldown_sec=0.0)

    for _ in range(n_inflight):
        broker.register_ml_task_start()

    t_request = time.perf_counter()
    initiated = broker.freeze()
    t_initiated = time.perf_counter()

    if n_inflight > 0:
        def drain_tasks():
            for _ in range(n_inflight):
                time.sleep(drain_delay_s)
                broker.register_ml_task_complete()

        drainer = threading.Thread(target=drain_tasks)
        drainer.start()
        drainer.join()

    t_complete = time.perf_counter()

    return {
        "n_inflight": n_inflight,
        "drain_delay_s": drain_delay_s,
        "initiated": initiated,
        "initiation_ms": (t_initiated - t_request) * 1000,
        "total_ms": (t_complete - t_request) * 1000,
        "final_state": broker.state.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Reallocation latency benchmark")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default="results_reallocation.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    if args.synthetic:
        print("=== Synthetic workload ===\n")
        scenarios = [
            ("No in-flight tasks", 0, 0.0, None),
            ("1 in-flight task (10ms drain)", 1, 0.01, None),
            ("3 in-flight tasks (10ms each)", 3, 0.01, None),
            ("5 in-flight tasks (10ms each)", 5, 0.01, None),
        ]
        data_source = "synthetic"
    else:
        print("=== Real data ===")
        ml_durations = load_all_task_durations(runs_dir, method="run_train")
        if len(ml_durations) > 0:
            p10 = float(np.percentile(ml_durations, 10))
            p50 = float(np.percentile(ml_durations, 50))
            p90 = float(np.percentile(ml_durations, 90))
            print(f"Real ML task durations: N={len(ml_durations)}, "
                  f"p10={p10:.1f}s, p50={p50:.1f}s, p90={p90:.1f}s")
            print(f"Scale factor: {SCALE_FACTOR}\n")
        else:
            p10, p50, p90 = 5.0, 10.0, 20.0
            print("No real ML durations found, using defaults\n")
        scenarios = [
            ("No in-flight tasks", 0, 0.0, None),
            (f"1 task (p10={p10:.1f}s real)", 1, p10 * SCALE_FACTOR, p10),
            (f"3 tasks (p50={p50:.1f}s real)", 3, p50 * SCALE_FACTOR, p50),
            (f"5 tasks (p90={p90:.1f}s real)", 5, p90 * SCALE_FACTOR, p90),
        ]
        data_source = "real_train_durations"

    results = []
    for label, n_inflight, drain_delay, real_duration in scenarios:
        print(f"=== {label} ===")
        initiation_vals = []
        total_vals = []

        for _ in range(args.repeats):
            m = measure_transition_latency(n_inflight, drain_delay)
            initiation_vals.append(m["initiation_ms"])
            total_vals.append(m["total_ms"])

        result = {
            "label": label,
            "n_inflight": n_inflight,
            "drain_delay_s": drain_delay,
            "real_task_duration_s": real_duration,
            "scale_factor": SCALE_FACTOR,
            "repeats": args.repeats,
            "data_source": data_source,
            "initiation": compute_stats(np.array(initiation_vals), unit="ms"),
            "total": compute_stats(np.array(total_vals), unit="ms"),
            "all_reclaimed": all(
                measure_transition_latency(n_inflight, drain_delay)["final_state"] == "SIM_RECLAIMED"
                for _ in range(3)  # Quick sanity check
            ),
        }
        results.append(result)

        print(f"  Initiation: {result['initiation']['mean_ms']:.4f} +/- {result['initiation']['ci95_ms']:.4f} ms")
        print(f"  Total:      {result['total']['mean_ms']:.2f} +/- {result['total']['ci95_ms']:.2f} ms\n")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

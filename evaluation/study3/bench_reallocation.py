#!/usr/bin/env python3
"""Experiment 3.1: Reallocation Latency.

Measures the time between a freeze decision and the first simulation task
executing on the reclaimed ML GPU. This includes draining any in-flight
ML tasks, state transition overhead, and task dispatch latency.

In standalone mode (without Parsl/Colmena), measures the broker's internal
transition latency with simulated in-flight tasks.

Usage:
    python bench_reallocation.py [--repeats 20] [--output results_reallocation.json]
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from deepdrivemd.resource_broker.broker import GPUState, ResourceBroker


def measure_transition_latency(n_inflight: int, drain_delay_s: float = 0.01) -> dict:
    """Measure freeze transition latency with simulated in-flight ML tasks.

    Parameters
    ----------
    n_inflight : int
        Number of simulated in-flight ML tasks at freeze time.
    drain_delay_s : float
        Simulated time for each ML task to complete.
    """
    freeze_records = []

    def on_freeze(record):
        freeze_records.append(record)

    broker = ResourceBroker(cooldown_sec=0.0, on_freeze=on_freeze)

    # Register in-flight tasks
    for _ in range(n_inflight):
        broker.register_ml_task_start()

    # Time the freeze
    t_request = time.perf_counter()
    initiated = broker.freeze()
    t_initiated = time.perf_counter()

    if n_inflight > 0:
        # Simulate tasks completing after drain_delay
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=str, default="results_reallocation.json")
    args = parser.parse_args()

    # Test scenarios: varying number of in-flight tasks
    scenarios = [
        ("No in-flight tasks", 0, 0.0),
        ("1 in-flight task (10ms drain)", 1, 0.01),
        ("3 in-flight tasks (10ms each)", 3, 0.01),
        ("5 in-flight tasks (10ms each)", 5, 0.01),
    ]

    results = []
    for label, n_inflight, drain_delay in scenarios:
        print(f"\n=== {label} ===")
        measurements = []
        for i in range(args.repeats):
            m = measure_transition_latency(n_inflight, drain_delay)
            measurements.append(m)

        initiation_ms = np.array([m["initiation_ms"] for m in measurements])
        total_ms = np.array([m["total_ms"] for m in measurements])

        result = {
            "label": label,
            "n_inflight": n_inflight,
            "drain_delay_s": drain_delay,
            "repeats": args.repeats,
            "initiation": {
                "mean_ms": float(np.mean(initiation_ms)),
                "p99_ms": float(np.percentile(initiation_ms, 99)),
                "max_ms": float(np.max(initiation_ms)),
            },
            "total": {
                "mean_ms": float(np.mean(total_ms)),
                "p99_ms": float(np.percentile(total_ms, 99)),
                "max_ms": float(np.max(total_ms)),
            },
            "all_initiated": all(m["initiated"] for m in measurements),
            "all_reclaimed": all(m["final_state"] == "SIM_RECLAIMED" for m in measurements),
        }
        results.append(result)

        print(f"  Initiation: mean={result['initiation']['mean_ms']:.4f}ms")
        print(f"  Total:      mean={result['total']['mean_ms']:.2f}ms")
        print(f"  All reclaimed: {result['all_reclaimed']}")

    # Summary
    print("\n--- Summary ---")
    print(f"{'Scenario':<40} {'Initiation (ms)':>16} {'Total (ms)':>12} {'Status':>10}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r["all_reclaimed"] else "FAIL"
        print(f"{r['label']:<40} {r['initiation']['mean_ms']:>16.4f} "
              f"{r['total']['mean_ms']:>12.2f} {status:>10}")
    print(f"\nReference: simulation cycle is ~27 min = 1,620,000 ms")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

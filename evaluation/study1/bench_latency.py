#!/usr/bin/env python3
"""Experiment 1.1: Signal Monitor Interface Overhead.

Measures wall-clock latency of submit_telemetry() for three policies of
increasing computational complexity. Proves the management plane introduces
no meaningful overhead relative to the ~27-minute simulation cycle.

Usage:
    python bench_latency.py [--repeats 1000] [--output results_latency.json]
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


def measure_latency(policy, n_repeats: int, telemetry_dim: int = 4) -> dict:
    """Measure per-call latency for a given policy.

    Generates synthetic telemetry vectors that slowly decrease (simulating
    converging training loss) so the policy actually exercises its logic.
    """
    monitor = SignalMonitor(policy=policy, window_size=50)

    # Pre-generate telemetry: slowly decreasing loss + noise
    rng = np.random.default_rng(42)
    base_loss = np.linspace(0.5, 0.01, n_repeats)
    telemetry = [
        np.array([base_loss[i] + rng.normal(0, 0.01)] + [rng.random() for _ in range(telemetry_dim - 1)])
        for i in range(n_repeats)
    ]

    # Warm up (fill the window)
    for v in telemetry[:50]:
        monitor.submit_telemetry(v)

    # Measure
    latencies_ns = []
    for v in telemetry[50:]:
        t0 = time.perf_counter_ns()
        monitor.submit_telemetry(v)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    latencies_ms = np.array(latencies_ns) / 1e6

    return {
        "policy": policy.name,
        "n_repeats": len(latencies_ms),
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "signals_received": monitor.signals_received,
        "transitions_issued": monitor.transitions_issued,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark signal monitor latency")
    parser.add_argument("--repeats", type=int, default=1000, help="Number of telemetry submissions")
    parser.add_argument("--output", type=str, default="results_latency.json", help="Output file")
    args = parser.parse_args()

    policies = [
        ("Trivial (O(1))", ThresholdPolicy(loss_threshold=0.05)),
        ("Moderate (O(Nk), k=10)", SlidingWindowPolicy(window_size=10)),
        ("Heavy (O(Nk²), k=20)", MannKendallPolicy(window_size=20)),
    ]

    results = []
    for label, policy in policies:
        print(f"Benchmarking: {label} ...")
        result = measure_latency(policy, args.repeats)
        result["label"] = label
        results.append(result)
        print(f"  mean={result['mean_ms']:.4f}ms  p99={result['p99_ms']:.4f}ms  max={result['max_ms']:.4f}ms")

    # Summary
    print("\n--- Summary ---")
    print(f"{'Policy':<30} {'Mean (ms)':>10} {'P99 (ms)':>10} {'Max (ms)':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r['label']:<30} {r['mean_ms']:>10.4f} {r['p99_ms']:>10.4f} {r['max_ms']:>10.4f}")
    print(f"\nReference: simulation cycle is ~27 min = 1,620,000 ms")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

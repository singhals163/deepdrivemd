#!/usr/bin/env python3
"""Experiment 1.2: Correctness Under Concurrency.

Stress-tests the Signal Monitor with concurrent telemetry submissions
from multiple threads, verifying zero dropped signals, zero duplicate
state transitions, and correct ordering.

Usage:
    python bench_concurrency.py [--threads 50] [--signals-per-thread 100]
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import SlidingWindowPolicy, WorkflowState


def stress_test(n_threads: int, signals_per_thread: int) -> dict:
    """Blast the signal monitor from multiple threads simultaneously."""

    transitions = []
    transition_lock = threading.Lock()

    def on_transition(old_state, new_state):
        with transition_lock:
            transitions.append({
                "time": time.perf_counter(),
                "from": old_state.name,
                "to": new_state.name,
                "thread": threading.current_thread().name,
            })

    policy = SlidingWindowPolicy(window_size=10, loss_threshold=0.05, min_improvement=0.001)
    monitor = SignalMonitor(policy=policy, window_size=100, on_state_change=on_transition)

    total_signals = n_threads * signals_per_thread
    barrier = threading.Barrier(n_threads)
    errors = []

    def worker(thread_id: int):
        try:
            rng = np.random.default_rng(thread_id)
            # Generate telemetry that crosses the threshold midway
            losses = np.linspace(0.5, 0.01, signals_per_thread)
            barrier.wait()  # Synchronize start
            for i in range(signals_per_thread):
                v = np.array([losses[i] + rng.normal(0, 0.01), rng.random()])
                monitor.submit_telemetry(v)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

    # Launch threads
    threads = []
    t0 = time.perf_counter()
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i,), name=f"worker-{i}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    stats = monitor.get_stats()

    # Validate invariants
    checks = {
        "zero_dropped_signals": stats["signals_received"] == total_signals,
        "zero_errors": len(errors) == 0,
        "no_self_transitions": all(t["from"] != t["to"] for t in transitions),
    }

    # Check that consecutive transitions alternate states
    consecutive_ok = True
    for i in range(1, len(transitions)):
        if transitions[i]["from"] != transitions[i - 1]["to"]:
            consecutive_ok = False
            break
    checks["consistent_state_chain"] = consecutive_ok

    all_passed = all(checks.values())

    return {
        "n_threads": n_threads,
        "signals_per_thread": signals_per_thread,
        "total_signals_expected": total_signals,
        "total_signals_received": stats["signals_received"],
        "transitions_issued": stats["transitions_issued"],
        "elapsed_sec": elapsed,
        "signals_per_sec": total_signals / elapsed,
        "errors": errors,
        "checks": checks,
        "all_passed": all_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency stress test")
    parser.add_argument("--threads", type=int, default=50)
    parser.add_argument("--signals-per-thread", type=int, default=100)
    parser.add_argument("--output", type=str, default="results_concurrency.json")
    args = parser.parse_args()

    print(f"Stress test: {args.threads} threads x {args.signals_per_thread} signals = "
          f"{args.threads * args.signals_per_thread} total signals")

    result = stress_test(args.threads, args.signals_per_thread)

    print(f"\nResults:")
    print(f"  Signals received: {result['total_signals_received']}/{result['total_signals_expected']}")
    print(f"  Transitions: {result['transitions_issued']}")
    print(f"  Throughput: {result['signals_per_sec']:.0f} signals/sec")
    print(f"  Elapsed: {result['elapsed_sec']:.3f}s")

    print(f"\nInvariant checks:")
    for check, passed in result["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    print(f"\nOverall: {'ALL PASSED' if result['all_passed'] else 'FAILED'}")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

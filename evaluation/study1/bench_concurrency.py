#!/usr/bin/env python3
"""Experiment 1.2: Correctness Under Concurrency.

Stress-tests the Signal Monitor with concurrent telemetry submissions.

Usage:
    python bench_concurrency.py --runs-dir /path/to/runs --output real/results_concurrency.json
    python bench_concurrency.py --synthetic --output synthetic/results_concurrency.json
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import SlidingWindowPolicy, WorkflowState
from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    load_loss_curves,
    synthetic_loss_curves,
)


def stress_test(n_threads: int, signals_per_thread: int,
                loss_curves: list) -> dict:
    """Blast the signal monitor from multiple threads."""
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

    policy = SlidingWindowPolicy(
        window_size=10, loss_threshold=0.05, min_improvement=0.001
    )
    monitor = SignalMonitor(
        policy=policy, window_size=100, on_state_change=on_transition
    )

    total_signals = n_threads * signals_per_thread
    barrier = threading.Barrier(n_threads)
    errors = []

    def worker(thread_id: int):
        try:
            curve = loss_curves[thread_id % len(loss_curves)]
            n_tile = (signals_per_thread // len(curve)) + 1
            losses = np.tile(curve, n_tile)[:signals_per_thread]
            rng = np.random.default_rng(thread_id)
            barrier.wait()
            for i in range(signals_per_thread):
                v = np.array([losses[i], rng.random()])
                monitor.submit_telemetry(v)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

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
    checks = {
        "zero_dropped_signals": stats["signals_received"] == total_signals,
        "zero_errors": len(errors) == 0,
        "no_self_transitions": all(t["from"] != t["to"] for t in transitions),
    }
    consecutive_ok = True
    for i in range(1, len(transitions)):
        if transitions[i]["from"] != transitions[i - 1]["to"]:
            consecutive_ok = False
            break
    checks["consistent_state_chain"] = consecutive_ok

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
        "all_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrency stress test")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--threads", type=int, default=50)
    parser.add_argument("--signals-per-thread", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default="results_concurrency.json")
    args = parser.parse_args()

    if args.synthetic:
        loss_curves = synthetic_loss_curves(n_curves=10)
        data_source = "synthetic"
        print(f"Using {len(loss_curves)} synthetic loss curves")
    else:
        loss_curves = load_loss_curves(Path(args.runs_dir))
        data_source = "real_training_loss"
        if not loss_curves:
            print("ERROR: No training loss data found.")
            return
        print(f"Loaded {len(loss_curves)} real loss curves")

    print(f"Stress test: {args.threads} threads x {args.signals_per_thread} signals, "
          f"{args.trials} trials\n")

    throughputs = []
    trial_results = []
    for trial in range(args.trials):
        result = stress_test(args.threads, args.signals_per_thread, loss_curves)
        result["data_source"] = data_source
        trial_results.append(result)
        throughputs.append(result["signals_per_sec"])
        status = "PASS" if result["all_passed"] else "FAIL"
        print(f"  Trial {trial+1}: [{status}] {result['signals_per_sec']:.0f} signals/sec")

    output = {
        "data_source": data_source,
        "n_trials": args.trials,
        "n_loss_curves": len(loss_curves),
        "config": {
            "n_threads": args.threads,
            "signals_per_thread": args.signals_per_thread,
            "total_signals": args.threads * args.signals_per_thread,
        },
        "throughput": compute_stats(np.array(throughputs), unit="sps"),
        "all_trials_passed": all(r["all_passed"] for r in trial_results),
        "checks": trial_results[0]["checks"],
        "trials": trial_results,
    }

    print(f"\nThroughput: {output['throughput']['mean_sps']:.0f} +/- "
          f"{output['throughput']['ci95_sps']:.0f} signals/sec")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

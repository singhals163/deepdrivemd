#!/usr/bin/env python3
"""Experiment 1.3: Policy Pluggability.

Demonstrates that two different policies produce different freeze decisions
on the same telemetry stream without any changes to the workflow code.

Usage:
    python bench_pluggability.py --runs-dir /path/to/runs --output real/results_pluggability.json
    python bench_pluggability.py --synthetic --output synthetic/results_pluggability.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import (
    SlidingWindowPolicy,
    ThresholdPolicy,
    WorkflowState,
)
from evaluation.data_loader import (
    RUNS_DIR,
    load_loss_curves_by_experiment,
    synthetic_loss_profiles,
)


def replay_through_policy(policy, telemetry: List[np.ndarray]) -> Tuple[List[str], int, list]:
    """Replay telemetry through a policy and record state at each step."""
    transitions = []

    def on_change(old, new):
        transitions.append({
            "step": monitor.signals_received,
            "from": old.name,
            "to": new.name,
        })

    monitor = SignalMonitor(policy=policy, window_size=50, on_state_change=on_change)
    states = []
    for v in telemetry:
        state = monitor.submit_telemetry(v)
        states.append(state.name)

    freeze_step = None
    for t in transitions:
        if t["to"] == "DORMANT":
            freeze_step = t["step"]
            break

    return states, freeze_step, transitions


def run_real(runs_dir: Path, policy_a, policy_b) -> list:
    """Run pluggability test with real training loss curves."""
    grouped = load_loss_curves_by_experiment(runs_dir, min_trains=1)
    if not grouped:
        print("ERROR: No training loss data found.")
        return []

    print(f"Loaded loss curves from {len(grouped)} experiments")
    results = []

    for exp_name, curves in grouped.items():
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        mean_curve = stacked.mean(axis=0)
        norm_curve = mean_curve / mean_curve[0]

        telemetry = [np.array([loss, epoch]) for epoch, loss in enumerate(norm_curve)]
        states_a, freeze_a, trans_a = replay_through_policy(policy_a, telemetry)
        states_b, freeze_b, trans_b = replay_through_policy(policy_b, telemetry)

        result = {
            "profile": exp_name,
            "data_source": "real_training_loss",
            "n_training_runs": len(curves),
            "n_epochs": len(mean_curve),
            "loss_curve": [float(v) for v in mean_curve],
            "loss_curve_normalized": [float(v) for v in norm_curve],
            "policy_a": {"name": policy_a.name, "freeze_step": freeze_a, "transitions": len(trans_a)},
            "policy_b": {"name": policy_b.name, "freeze_step": freeze_b, "transitions": len(trans_b)},
            "different_decisions": freeze_a != freeze_b,
        }
        results.append(result)

        print(f"\n  {exp_name} ({len(curves)} runs): "
              f"A={freeze_a}, B={freeze_b}, different={freeze_a != freeze_b}")

    return results


def run_synthetic(policy_a, policy_b) -> list:
    """Run pluggability test with canonical synthetic profiles."""
    profiles = synthetic_loss_profiles()
    results = []

    for profile_name, losses in profiles.items():
        telemetry = [np.array([loss, i]) for i, loss in enumerate(losses)]
        states_a, freeze_a, trans_a = replay_through_policy(policy_a, telemetry)
        states_b, freeze_b, trans_b = replay_through_policy(policy_b, telemetry)

        result = {
            "profile": profile_name,
            "data_source": "synthetic",
            "n_epochs": len(losses),
            "loss_curve": [float(v) for v in losses],
            "policy_a": {"name": policy_a.name, "freeze_step": freeze_a, "transitions": len(trans_a)},
            "policy_b": {"name": policy_b.name, "freeze_step": freeze_b, "transitions": len(trans_b)},
            "different_decisions": freeze_a != freeze_b,
        }
        results.append(result)

        print(f"\n  {profile_name}: A={freeze_a}, B={freeze_b}, "
              f"different={freeze_a != freeze_b}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Policy pluggability demo")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default="results_pluggability.json")
    args = parser.parse_args()

    policy_a = ThresholdPolicy(loss_threshold=0.05, loss_index=0)
    policy_b = SlidingWindowPolicy(
        window_size=10, loss_threshold=0.1, min_improvement=0.005, loss_index=0
    )

    if args.synthetic:
        print("=== Synthetic profiles ===")
        results = run_synthetic(policy_a, policy_b)
    else:
        print("=== Real data ===")
        results = run_real(Path(args.runs_dir), policy_a, policy_b)

    any_different = any(r["different_decisions"] for r in results)
    print(f"\n--- Pluggability Verification ---")
    print(f"Policies produced different decisions: {any_different}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

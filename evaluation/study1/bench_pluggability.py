#!/usr/bin/env python3
"""Experiment 1.3: Policy Pluggability.

Demonstrates that two different policies produce different freeze decisions
on the same telemetry stream without any changes to the workflow code.
Replays telemetry from Colmena result JSON files or generates synthetic
streams matching real convergence profiles.

Usage:
    python bench_pluggability.py [--telemetry-file train.json] [--output results_pluggability.json]
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from deepdrivemd.signal_monitor.monitor import SignalMonitor
from deepdrivemd.signal_monitor.policy import (
    SlidingWindowPolicy,
    ThresholdPolicy,
    WorkflowState,
)


def generate_synthetic_telemetry(profile: str, n_cycles: int = 50) -> List[np.ndarray]:
    """Generate synthetic telemetry matching real convergence profiles.

    Profiles:
    - 'fast_converge': KRAS-like, loss drops quickly then plateaus
    - 'slow_converge': BBA-like, loss drops slowly, never fully plateaus
    - 'distribution_shift': CLN025-like, loss drops then rises again
    """
    rng = np.random.default_rng(42)

    if profile == "fast_converge":
        # KRAS: rapid convergence, plateau after ~15 cycles
        losses = 0.5 * np.exp(-0.15 * np.arange(n_cycles)) + 0.02
    elif profile == "slow_converge":
        # BBA: slow convergence, still improving at end
        losses = 0.5 * np.exp(-0.03 * np.arange(n_cycles)) + 0.08
    elif profile == "distribution_shift":
        # CLN025: converges then loss increases (shift)
        converge = 0.5 * np.exp(-0.1 * np.arange(n_cycles // 2))
        diverge = converge[-1] + 0.02 * np.arange(n_cycles - n_cycles // 2)
        losses = np.concatenate([converge, diverge])
    else:
        raise ValueError(f"Unknown profile: {profile}")

    # Add noise
    losses += rng.normal(0, 0.005, n_cycles)
    losses = np.clip(losses, 0.001, 1.0)

    return [np.array([loss, i]) for i, loss in enumerate(losses)]


def load_telemetry_from_colmena(path: Path) -> List[np.ndarray]:
    """Load training loss telemetry from Colmena train.json results."""
    vectors = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("success"):
                # Extract timing as a proxy for training complexity
                wall_time = record.get("time", {}).get("running", 0)
                vectors.append(np.array([wall_time, len(vectors)]))
    return vectors


def replay_through_policy(policy, telemetry: List[np.ndarray]) -> Tuple[List[str], int]:
    """Replay telemetry through a policy and record state at each step."""
    transitions = []

    def on_change(old, new):
        transitions.append({"step": monitor.signals_received, "from": old.name, "to": new.name})

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


def main():
    parser = argparse.ArgumentParser(description="Policy pluggability demo")
    parser.add_argument("--telemetry-file", type=str, default=None,
                        help="Path to Colmena train.json for real telemetry replay")
    parser.add_argument("--output", type=str, default="results_pluggability.json")
    args = parser.parse_args()

    # Define two policies (different parameters, same interface)
    policy_a = ThresholdPolicy(loss_threshold=0.05, loss_index=0)
    policy_b = SlidingWindowPolicy(window_size=10, loss_threshold=0.1, min_improvement=0.005, loss_index=0)

    profiles = ["fast_converge", "slow_converge", "distribution_shift"]
    results = []

    for profile in profiles:
        telemetry = generate_synthetic_telemetry(profile, n_cycles=50)

        states_a, freeze_a, trans_a = replay_through_policy(policy_a, telemetry)
        states_b, freeze_b, trans_b = replay_through_policy(policy_b, telemetry)

        different_decisions = freeze_a != freeze_b
        result = {
            "profile": profile,
            "n_cycles": len(telemetry),
            "policy_a": {"name": policy_a.name, "freeze_step": freeze_a, "transitions": len(trans_a)},
            "policy_b": {"name": policy_b.name, "freeze_step": freeze_b, "transitions": len(trans_b)},
            "different_decisions": different_decisions,
        }
        results.append(result)

        print(f"\nProfile: {profile}")
        print(f"  Policy A ({policy_a.name}): freeze at step {freeze_a}")
        print(f"  Policy B ({policy_b.name}): freeze at step {freeze_b}")
        print(f"  Different decisions: {different_decisions}")

    # Also replay from real data if available
    if args.telemetry_file:
        path = Path(args.telemetry_file)
        if path.exists():
            telemetry = load_telemetry_from_colmena(path)
            if telemetry:
                states_a, freeze_a, trans_a = replay_through_policy(policy_a, telemetry)
                states_b, freeze_b, trans_b = replay_through_policy(policy_b, telemetry)
                results.append({
                    "profile": f"real_data ({path.name})",
                    "n_cycles": len(telemetry),
                    "policy_a": {"name": policy_a.name, "freeze_step": freeze_a, "transitions": len(trans_a)},
                    "policy_b": {"name": policy_b.name, "freeze_step": freeze_b, "transitions": len(trans_b)},
                    "different_decisions": freeze_a != freeze_b,
                })

    print(f"\n--- Pluggability Verification ---")
    print("Both policies use the SAME SignalMonitor interface.")
    print("Swapping policies requires changing only the policy object, not workflow code.")
    any_different = any(r["different_decisions"] for r in results)
    print(f"Policies produced different decisions on at least one profile: {any_different}")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

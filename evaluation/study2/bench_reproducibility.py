#!/usr/bin/env python3
"""Experiment 2.2: Bit-Reproducibility of Redis Serialization.

Verifies that model checkpoints survive Redis roundtrip bit-identically.

Usage:
    python bench_reproducibility.py --runs-dir /path/to/runs --output real/results_reproducibility.json
    python bench_reproducibility.py --synthetic --output synthetic/results_reproducibility.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepdrivemd.redis_io import init_redis, store_torch, load_torch
from evaluation.data_loader import (
    RUNS_DIR,
    load_checkpoint_paths,
    load_checkpoint_state_dict,
    synthetic_model,
)


def test_roundtrip(state_dict: dict, key: str) -> dict:
    """Round-trip a state_dict through Redis and verify bit-identity."""
    store_torch(key, state_dict)
    loaded = load_torch(key, map_location="cpu")

    max_diff = 0.0
    all_match = True
    for k in state_dict:
        diff = torch.abs(state_dict[k].float() - loaded[k].float()).max().item()
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            all_match = False

    return {"max_diff": max_diff, "bit_identical": all_match}


def run_real(runs_dir):
    """Test with real CVAE checkpoints."""
    ckpt_paths = load_checkpoint_paths(runs_dir)
    if not ckpt_paths:
        print("ERROR: No checkpoints found.")
        return None

    print(f"Testing {len(ckpt_paths)} real checkpoints")
    trials = []
    all_passed = True
    for i, ckpt_path in enumerate(ckpt_paths):
        state_dict = load_checkpoint_state_dict(ckpt_path)
        result = test_roundtrip(state_dict, f"bench:repro:real:{i}")
        result["checkpoint"] = str(ckpt_path)
        result["n_params"] = sum(v.numel() for v in state_dict.values())
        trials.append(result)
        if not result["bit_identical"]:
            all_passed = False
        if (i + 1) % 10 == 0 or i == len(ckpt_paths) - 1:
            print(f"  [{('PASS' if result['bit_identical'] else 'FAIL')}] "
                  f"{i + 1}/{len(ckpt_paths)}")

    return {
        "data_source": "real_checkpoints",
        "n_checkpoints_tested": len(ckpt_paths),
        "all_bit_identical": all_passed,
        "max_diff_across_all": max(t["max_diff"] for t in trials),
        "trials": trials,
    }


def run_synthetic():
    """Test with synthetic models at different sizes."""
    configs = [
        ("BBA-scale (~200K)", 200_000),
        ("CLN025-scale (~2M)", 2_000_000),
        ("KRAS-scale (~66M)", 66_000_000),
    ]
    trials = []
    all_passed = True
    for name, n_params in configs:
        print(f"  Testing {name}...")
        model = synthetic_model(n_params, name)
        state_dict = model.state_dict()
        result = test_roundtrip(state_dict, f"bench:repro:synth:{name}")
        result["model_name"] = name
        result["n_params"] = sum(v.numel() for v in state_dict.values())
        trials.append(result)
        if not result["bit_identical"]:
            all_passed = False
        status = "PASS" if result["bit_identical"] else "FAIL"
        print(f"  [{status}] {name} (max_diff={result['max_diff']:.2e})")

    return {
        "data_source": "synthetic",
        "n_models_tested": len(configs),
        "all_bit_identical": all_passed,
        "max_diff_across_all": max(t["max_diff"] for t in trials),
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproducibility test")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--output", type=str, default="results_reproducibility.json")
    args = parser.parse_args()

    init_redis(args.redis_host, args.redis_port)

    if args.synthetic:
        print("=== Synthetic models ===")
        output = run_synthetic()
    else:
        print("=== Real data ===")
        output = run_real(Path(args.runs_dir))

    if output is None:
        return

    print(f"\nAll bit-identical: {output['all_bit_identical']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

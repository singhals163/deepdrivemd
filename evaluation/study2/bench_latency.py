#!/usr/bin/env python3
"""Experiment 2.1: Transition Latency vs Model Size.

Measures serialize/transfer/deserialize latency for model checkpoints
of different sizes through the Stateful Service (Redis-backed).

Tests with synthetic models matching BBA CVAE (~200K params) and
optionally KRAS CVAE (~66M params) parameter counts.

Usage:
    python bench_latency.py [--repeats 20] [--redis-host 127.0.0.1] [--output results_latency.json]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deepdrivemd.redis_io import init_redis, store_torch, load_torch, reset_io_stats, get_io_stats


def make_synthetic_model(n_params: int, name: str) -> nn.Module:
    """Create a synthetic model with approximately n_params parameters."""
    # Use a simple linear stack to hit the target parameter count
    # Each Linear(in, out) has in*out + out parameters
    dim = int(np.sqrt(n_params / 2))
    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )
    actual = sum(p.numel() for p in model.parameters())
    print(f"  {name}: target={n_params:,} actual={actual:,} params, "
          f"~{sum(p.nbytes for p in model.parameters()) / 1024:.1f} KB")
    return model


def measure_roundtrip(model: nn.Module, key: str, repeats: int) -> dict:
    """Measure serialize and deserialize latency for a model checkpoint."""
    state_dict = model.state_dict()
    serialize_times = []
    deserialize_times = []
    data_sizes = []

    for i in range(repeats):
        reset_io_stats()

        # Serialize
        t0 = time.perf_counter()
        stats = store_torch(f"{key}:{i}", state_dict)
        t1 = time.perf_counter()
        serialize_times.append(t1 - t0)
        data_sizes.append(stats["data_bytes"])

        # Deserialize
        t2 = time.perf_counter()
        loaded = load_torch(f"{key}:{i}", map_location="cpu")
        t3 = time.perf_counter()
        deserialize_times.append(t3 - t2)

        # Verify correctness
        for k in state_dict:
            assert torch.equal(state_dict[k], loaded[k]), f"Mismatch in {k} on repeat {i}"

    serialize_ms = np.array(serialize_times) * 1000
    deserialize_ms = np.array(deserialize_times) * 1000

    return {
        "key": key,
        "repeats": repeats,
        "data_bytes": int(np.mean(data_sizes)),
        "data_kb": float(np.mean(data_sizes) / 1024),
        "serialize": {
            "mean_ms": float(np.mean(serialize_ms)),
            "median_ms": float(np.median(serialize_ms)),
            "p99_ms": float(np.percentile(serialize_ms, 99)),
            "std_ms": float(np.std(serialize_ms)),
        },
        "deserialize": {
            "mean_ms": float(np.mean(deserialize_ms)),
            "median_ms": float(np.median(deserialize_ms)),
            "p99_ms": float(np.percentile(deserialize_ms, 99)),
            "std_ms": float(np.std(deserialize_ms)),
        },
        "roundtrip_mean_ms": float(np.mean(serialize_ms) + np.mean(deserialize_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark stateful service latency")
    parser.add_argument("--repeats", type=int, default=20, help="Repetitions per model size")
    parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--output", type=str, default="results_latency.json")
    args = parser.parse_args()

    init_redis(args.redis_host, args.redis_port)

    # Model configurations matching real DeepDriveMD workloads
    configs = [
        ("BBA CVAE (~200K)", 200_000),
        ("CLN025 CVAE (~2M)", 2_000_000),
        ("KRAS CVAE (~66M)", 66_000_000),
    ]

    results = []
    for name, n_params in configs:
        print(f"\nBenchmarking: {name}")
        try:
            model = make_synthetic_model(n_params, name)
        except Exception as e:
            print(f"  Skipping (too large for available memory): {e}")
            continue

        result = measure_roundtrip(model, f"bench:{name}", args.repeats)
        result["model_name"] = name
        result["target_params"] = n_params
        results.append(result)

        print(f"  Serialize:   mean={result['serialize']['mean_ms']:.2f}ms  "
              f"p99={result['serialize']['p99_ms']:.2f}ms")
        print(f"  Deserialize: mean={result['deserialize']['mean_ms']:.2f}ms  "
              f"p99={result['deserialize']['p99_ms']:.2f}ms")
        print(f"  Roundtrip:   {result['roundtrip_mean_ms']:.2f}ms  "
              f"({result['data_kb']:.1f} KB)")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Model':<25} {'Size (KB)':>10} {'Serialize (ms)':>15} {'Deserialize (ms)':>17} {'Roundtrip (ms)':>15}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_name']:<25} {r['data_kb']:>10.1f} "
              f"{r['serialize']['mean_ms']:>15.2f} "
              f"{r['deserialize']['mean_ms']:>17.2f} "
              f"{r['roundtrip_mean_ms']:>15.2f}")
    print(f"\nReference: simulation cycle is ~27 min = 1,620,000 ms")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

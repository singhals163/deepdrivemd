#!/usr/bin/env python3
"""Experiment 2.1: Transition Latency vs Model Size.

Measures serialize/transfer/deserialize latency through Redis,
grouped by real model size.

Usage:
    python bench_latency.py --runs-dir /path/to/runs --output real/results_latency.json
    python bench_latency.py --synthetic --output synthetic/results_latency.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from deepdrivemd.redis_io import init_redis, store_torch, load_torch, reset_io_stats
from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    load_checkpoint_paths,
    load_checkpoint_state_dict,
    synthetic_model,
)


def measure_roundtrip(state_dict: dict, key: str, repeats: int) -> dict:
    """Measure serialize and deserialize latency for a model state_dict."""
    serialize_times = []
    deserialize_times = []
    data_sizes = []

    for i in range(repeats):
        reset_io_stats()

        t0 = time.perf_counter()
        stats = store_torch(f"{key}:{i}", state_dict)
        t1 = time.perf_counter()
        serialize_times.append(t1 - t0)
        data_sizes.append(stats["data_bytes"])

        t2 = time.perf_counter()
        loaded = load_torch(f"{key}:{i}", map_location="cpu")
        t3 = time.perf_counter()
        deserialize_times.append(t3 - t2)

        for k in state_dict:
            assert torch.equal(state_dict[k], loaded[k]), f"Mismatch in {k}"

    ser_ms = np.array(serialize_times) * 1000
    deser_ms = np.array(deserialize_times) * 1000
    return {
        "data_bytes": int(np.mean(data_sizes)),
        "data_kb": float(np.mean(data_sizes) / 1024),
        "serialize_ms": float(np.mean(ser_ms)),
        "deserialize_ms": float(np.mean(deser_ms)),
        "roundtrip_ms": float(np.mean(ser_ms) + np.mean(deser_ms)),
    }


def run_real(runs_dir, repeats):
    """Benchmark with real CVAE checkpoints, grouped by model size."""
    ckpt_paths = load_checkpoint_paths(runs_dir)
    if not ckpt_paths:
        print("ERROR: No checkpoints found.")
        return None

    # Group checkpoints by param count
    groups = defaultdict(list)
    for p in ckpt_paths:
        sd = load_checkpoint_state_dict(p)
        n_params = sum(v.numel() for v in sd.values())
        groups[n_params].append((p, sd))

    print(f"Found {len(ckpt_paths)} checkpoints in {len(groups)} size groups")

    results = []
    for n_params in sorted(groups.keys()):
        items = groups[n_params]
        n_bytes = sum(v.nbytes for v in items[0][1].values())
        label = f"Real CVAE ({n_params:,} params, {n_bytes/1024:.0f} KB)"
        print(f"\n  {label}: {len(items)} checkpoints")

        all_ser, all_deser, all_rt = [], [], []
        for i, (path, sd) in enumerate(items):
            m = measure_roundtrip(sd, f"bench:real:{n_params}:{i}", repeats)
            all_ser.append(m["serialize_ms"])
            all_deser.append(m["deserialize_ms"])
            all_rt.append(m["roundtrip_ms"])

        results.append({
            "model_name": label,
            "data_source": "real_checkpoints",
            "n_checkpoints": len(items),
            "n_params": n_params,
            "data_kb": n_bytes / 1024,
            "repeats_per_checkpoint": repeats,
            "serialize": compute_stats(np.array(all_ser), unit="ms"),
            "deserialize": compute_stats(np.array(all_deser), unit="ms"),
            "roundtrip": compute_stats(np.array(all_rt), unit="ms"),
        })

        r = results[-1]
        print(f"    Serialize:   {r['serialize']['mean_ms']:.2f} +/- {r['serialize']['ci95_ms']:.2f} ms")
        print(f"    Deserialize: {r['deserialize']['mean_ms']:.2f} +/- {r['deserialize']['ci95_ms']:.2f} ms")
        print(f"    Roundtrip:   {r['roundtrip']['mean_ms']:.2f} +/- {r['roundtrip']['ci95_ms']:.2f} ms")

    return results


def run_synthetic(repeats):
    """Benchmark with synthetic models at different sizes."""
    configs = [
        ("BBA-scale (~200K)", 200_000),
        ("CLN025-scale (~2M)", 2_000_000),
        ("KRAS-scale (~66M)", 66_000_000),
    ]
    results = []
    for name, n_params in configs:
        print(f"\n  {name}")
        model = synthetic_model(n_params, name)
        state_dict = model.state_dict()
        actual_params = sum(v.numel() for v in state_dict.values())
        actual_bytes = sum(v.nbytes for v in state_dict.values())
        print(f"    {actual_params:,} params, {actual_bytes / 1024:.1f} KB")

        m = measure_roundtrip(state_dict, f"bench:synth:{name}", repeats)
        results.append({
            "model_name": name,
            "data_source": "synthetic",
            "n_params": actual_params,
            "data_kb": actual_bytes / 1024,
            "repeats_per_checkpoint": repeats,
            "serialize": compute_stats(np.array([m["serialize_ms"]]), unit="ms"),
            "deserialize": compute_stats(np.array([m["deserialize_ms"]]), unit="ms"),
            "roundtrip": compute_stats(np.array([m["roundtrip_ms"]]), unit="ms"),
        })
        print(f"    Roundtrip: {m['roundtrip_ms']:.2f}ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark stateful service latency")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--output", type=str, default="results_latency.json")
    args = parser.parse_args()

    init_redis(args.redis_host, args.redis_port)

    if args.synthetic:
        print("=== Synthetic models ===")
        output = run_synthetic(args.repeats)
    else:
        print("=== Real data ===")
        output = run_real(Path(args.runs_dir), args.repeats)

    if output is None:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

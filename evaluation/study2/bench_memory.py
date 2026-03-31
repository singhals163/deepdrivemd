#!/usr/bin/env python3
"""Experiment 2.3: Memory Recovery During Freeze/Resume.

Measures whether the Stateful Service correctly frees model memory
during a freeze transition and restores it during resume. This validates
that the GPU can be fully reclaimed for simulation work when the AI
model is serialized to Redis.

Lifecycle measured:
  1. Baseline RSS (no model loaded)
  2. Load real CVAE model + run forward pass (simulates active training)
  3. Serialize model to Redis + delete local model (simulates freeze)
  4. Measure RSS — should return to baseline (GPU memory freed)
  5. Deserialize model from Redis (simulates resume)
  6. Measure RSS — model is back in memory

Usage:
    python bench_memory.py --runs-dir /path/to/runs --output real/results_memory.json
    python bench_memory.py --synthetic --output synthetic/results_memory.json
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation.data_loader import (
    RUNS_DIR,
    compute_stats,
    load_checkpoint_paths,
    load_checkpoint_state_dict,
    load_contact_maps_dense,
    build_cvae_model,
)


def get_rss_mb() -> float:
    """Get current process RSS in MB via /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _gc_and_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_resume_lifecycle(ckpt_path: Path, contact_maps: torch.Tensor) -> dict:
    """Simulate a full freeze/resume lifecycle with the Stateful Service.

    1. Load model (active state)
    2. Serialize to Redis + delete (freeze)
    3. Deserialize from Redis (resume)
    """
    from deepdrivemd.redis_io import init_redis, store_torch, load_torch

    init_redis("127.0.0.1", 6379)

    _gc_and_clear()
    rss_baseline = get_rss_mb()

    # --- ACTIVE: Load model, run forward pass ---
    trainer = build_cvae_model(ckpt_path)
    model = trainer.model
    with torch.no_grad():
        _ = model(contact_maps[:16].clone())

    rss_active = get_rss_mb()
    state_dict = model.state_dict()
    n_params = sum(v.numel() for v in state_dict.values())
    model_bytes = sum(v.nbytes for v in state_dict.values())

    # --- FREEZE: Serialize to Redis, delete local model ---
    redis_key = f"bench:memory:{id(model)}"
    store_torch(redis_key, state_dict)
    del state_dict, model, trainer, _
    _gc_and_clear()

    rss_frozen = get_rss_mb()

    # --- RESUME: Deserialize from Redis ---
    loaded_state = load_torch(redis_key, map_location="cpu")
    resume_trainer = build_cvae_model()  # fresh model
    resume_trainer.model.load_state_dict(loaded_state)
    with torch.no_grad():
        _ = resume_trainer.model(contact_maps[:16].clone())

    rss_resumed = get_rss_mb()

    # Cleanup
    del loaded_state, resume_trainer, _
    _gc_and_clear()

    memory_increase = rss_active - rss_baseline
    memory_freed = rss_active - rss_frozen
    memory_restored = rss_resumed - rss_frozen

    return {
        "n_params": n_params,
        "model_bytes": model_bytes,
        "model_mb": model_bytes / (1024 * 1024),
        "rss_baseline_mb": rss_baseline,
        "rss_active_mb": rss_active,
        "rss_frozen_mb": rss_frozen,
        "rss_resumed_mb": rss_resumed,
        "memory_increase_mb": memory_increase,
        "memory_freed_mb": memory_freed,
        "memory_restored_mb": memory_restored,
        "freeze_recovery_pct": (memory_freed / max(memory_increase, 0.01)) * 100,
    }


def freeze_resume_state_dict(ckpt_path: Path) -> dict:
    """Freeze/resume lifecycle using raw state_dict (no model architecture needed).

    Works for any checkpoint size — just loads tensors into memory,
    serializes to Redis, deletes, deserializes back.
    """
    from deepdrivemd.redis_io import init_redis, store_torch, load_torch

    init_redis("127.0.0.1", 6379)

    _gc_and_clear()
    rss_baseline = get_rss_mb()

    # ACTIVE: load state_dict tensors into memory
    state_dict = load_checkpoint_state_dict(ckpt_path)
    # Force tensors into contiguous memory
    tensors = {k: v.clone() for k, v in state_dict.items()}
    n_params = sum(v.numel() for v in tensors.values())
    model_bytes = sum(v.nbytes for v in tensors.values())

    rss_active = get_rss_mb()

    # FREEZE: serialize to Redis, delete local tensors
    redis_key = f"bench:memory:sd:{id(tensors)}"
    store_torch(redis_key, tensors)
    del tensors, state_dict
    _gc_and_clear()

    rss_frozen = get_rss_mb()

    # RESUME: deserialize from Redis
    loaded = load_torch(redis_key, map_location="cpu")
    # Force into memory
    resumed_tensors = {k: v.clone() for k, v in loaded.items()}
    del loaded

    rss_resumed = get_rss_mb()

    del resumed_tensors
    _gc_and_clear()

    memory_increase = rss_active - rss_baseline
    memory_freed = rss_active - rss_frozen
    memory_restored = rss_resumed - rss_frozen

    return {
        "n_params": n_params,
        "model_bytes": model_bytes,
        "model_mb": model_bytes / (1024 * 1024),
        "rss_baseline_mb": rss_baseline,
        "rss_active_mb": rss_active,
        "rss_frozen_mb": rss_frozen,
        "rss_resumed_mb": rss_resumed,
        "memory_increase_mb": memory_increase,
        "memory_freed_mb": memory_freed,
        "memory_restored_mb": memory_restored,
        "freeze_recovery_pct": (memory_freed / max(memory_increase, 0.01)) * 100,
    }


def freeze_resume_synthetic(n_params: int, label: str) -> dict:
    """Synthetic version using nn.Linear models."""
    from deepdrivemd.redis_io import init_redis, store_torch, load_torch

    init_redis("127.0.0.1", 6379)

    dim = max(int(np.sqrt(n_params / 2)), 4)
    _gc_and_clear()
    rss_baseline = get_rss_mb()

    # ACTIVE
    model = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    data = torch.rand(16, dim)
    with torch.no_grad():
        _ = model(data)

    rss_active = get_rss_mb()
    state_dict = model.state_dict()
    actual_params = sum(v.numel() for v in state_dict.values())
    model_bytes = sum(v.nbytes for v in state_dict.values())

    # FREEZE
    redis_key = f"bench:memory:synth:{label}"
    store_torch(redis_key, state_dict)
    del state_dict, model, data, _
    _gc_and_clear()

    rss_frozen = get_rss_mb()

    # RESUME
    loaded = load_torch(redis_key, map_location="cpu")
    model2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    model2.load_state_dict(loaded)
    with torch.no_grad():
        _ = model2(torch.rand(16, dim))

    rss_resumed = get_rss_mb()
    del loaded, model2, _
    _gc_and_clear()

    memory_increase = rss_active - rss_baseline
    memory_freed = rss_active - rss_frozen

    return {
        "n_params": actual_params,
        "model_bytes": model_bytes,
        "model_mb": model_bytes / (1024 * 1024),
        "rss_baseline_mb": rss_baseline,
        "rss_active_mb": rss_active,
        "rss_frozen_mb": rss_frozen,
        "rss_resumed_mb": rss_resumed,
        "memory_increase_mb": memory_increase,
        "memory_freed_mb": memory_freed,
        "memory_restored_mb": rss_resumed - rss_frozen,
        "freeze_recovery_pct": (memory_freed / max(memory_increase, 0.01)) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory freeze/resume benchmark")
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default="results_memory.json")
    args = parser.parse_args()

    if args.synthetic:
        print("=== Synthetic freeze/resume lifecycle ===")
        configs = [
            ("BBA-scale (~200K)", 200_000),
            ("CLN025-scale (~2M)", 2_000_000),
            ("KRAS-scale (~66M)", 66_000_000),
        ]
        all_results = []
        for label, n_params in configs:
            print(f"\n  {label}")
            trials = []
            for i in range(args.trials):
                result = freeze_resume_synthetic(n_params, f"{label}:{i}")
                trials.append(result)

            # Drop first trial (cold start)
            if len(trials) > 2:
                trials = trials[1:]

            freed = [t["memory_freed_mb"] for t in trials]
            recovery = [t["freeze_recovery_pct"] for t in trials]
            all_results.append({
                "config_name": label,
                "data_source": "synthetic",
                "n_trials": len(trials),
                "n_params": trials[0]["n_params"],
                "model_mb": trials[0]["model_mb"],
                "memory_freed": compute_stats(np.array(freed), unit="mb"),
                "freeze_recovery_pct": compute_stats(np.array(recovery), unit="pct"),
                "trials": trials,
            })
            print(f"    Freed: {all_results[-1]['memory_freed']['mean_mb']:.1f} MB, "
                  f"Recovery: {all_results[-1]['freeze_recovery_pct']['mean_pct']:.0f}%")
        output = all_results

    else:
        print("=== Real freeze/resume lifecycle ===")
        runs_dir = Path(args.runs_dir)
        ckpt_paths = load_checkpoint_paths(runs_dir)
        if not ckpt_paths:
            print("ERROR: No checkpoints found.")
            return

        # Group by model size
        from collections import defaultdict
        groups = defaultdict(list)
        for p in ckpt_paths:
            sd = load_checkpoint_state_dict(p)
            n_params = sum(v.numel() for v in sd.values())
            groups[n_params].append(p)
        del sd

        print(f"Found {len(groups)} model sizes: "
              + ", ".join(f"{n:,} ({len(ps)} ckpts)" for n, ps in sorted(groups.items())))

        # Load contact maps for BBA forward pass
        contact_maps = load_contact_maps_dense(runs_dir, max_sims=5)
        print(f"Loaded {len(contact_maps)} contact map frames\n")

        all_configs = []
        for n_params in sorted(groups.keys()):
            paths = groups[n_params]
            sd = load_checkpoint_state_dict(paths[0])
            n_bytes = sum(v.nbytes for v in sd.values())
            is_bba = n_params < 1_000_000
            del sd

            if is_bba:
                label = f"BBA CVAE ({n_params:,} params, {n_bytes/(1024*1024):.1f} MB)"
            elif n_params < 10_000_000:
                label = f"CVAE ({n_params:,} params, {n_bytes/(1024*1024):.1f} MB)"
            else:
                label = f"Large CVAE ({n_params:,} params, {n_bytes/(1024*1024):.1f} MB)"

            n_trials = min(args.trials, len(paths))
            print(f"{label}: {n_trials} trials")

            trials = []
            for i in range(n_trials):
                if is_bba:
                    result = freeze_resume_lifecycle(paths[i], contact_maps)
                else:
                    result = freeze_resume_state_dict(paths[i])
                trials.append(result)
                print(f"  Trial {i+1}: freed={result['memory_freed_mb']:.1f}MB "
                      f"restored={result['memory_restored_mb']:.1f}MB")

            # Drop first trial (cold start)
            if len(trials) > 2:
                trials = trials[1:]

            freed = [t["memory_freed_mb"] for t in trials]
            increases = [t["memory_increase_mb"] for t in trials]
            restored = [t["memory_restored_mb"] for t in trials]

            config = {
                "config_name": label,
                "data_source": "real_checkpoints",
                "n_trials": len(trials),
                "n_params": n_params,
                "model_mb": n_bytes / (1024 * 1024),
                "memory_increase": compute_stats(np.array(increases), unit="mb"),
                "memory_freed": compute_stats(np.array(freed), unit="mb"),
                "memory_restored": compute_stats(np.array(restored), unit="mb"),
                "trials": trials,
            }
            all_configs.append(config)
            print(f"  Summary: freed={config['memory_freed']['mean_mb']:.1f} MB, "
                  f"restored={config['memory_restored']['mean_mb']:.1f} MB\n")

        output = all_configs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

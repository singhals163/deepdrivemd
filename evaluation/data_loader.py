"""Shared data loading utilities for evaluation benchmarks.

Provides functions to load real experiment data (training loss curves,
model checkpoints, contact maps, Colmena task timing) from DeepDriveMD
runs, plus a compute_stats() helper for consistent JSON output with
error bar support.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

RUNS_DIR = Path(os.environ.get("DEEPDRIVEMD_RUNS_DIR", "/shivam/deepdrivemd/runs"))

# BBA CVAE production settings (from examples/bba-folding-workstation/cvae-prod-settings.yaml)
BBA_CVAE_CONFIG = dict(
    input_shape=(1, 28, 28),
    filters=[16, 16, 16, 16],
    kernels=[3, 3, 3, 3],
    strides=[1, 1, 1, 2],
    affine_widths=[128],
    affine_dropouts=[0.5],
    latent_dim=3,
    lambda_rec=1.0,
    batch_size=64,
    device="cpu",
    optimizer_name="RMSprop",
    optimizer_hparams={"lr": 0.001, "weight_decay": 0.00001},
    epochs=1,
    checkpoint_log_every=1,
    plot_log_every=1,
    plot_n_samples=0,
    num_data_workers=0,
    prefetch_factor=2,
)


def discover_experiments(
    runs_dir: Path = RUNS_DIR, min_trains: int = 1
) -> List[Path]:
    """Return sorted list of experiment dirs with at least min_trains
    completed training runs (each having a checkpoint-epoch-20.pt)."""
    experiments = []
    if not runs_dir.exists():
        return experiments
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("experiment-"):
            continue
        train_dir = d / "train"
        if not train_dir.exists():
            continue
        n_complete = sum(
            1
            for td in train_dir.iterdir()
            if td.is_dir()
            and (td / "model" / "checkpoints" / "checkpoint-epoch-20.pt").exists()
        )
        if n_complete >= min_trains:
            experiments.append(d)
    return experiments


def _train_run_dirs(experiment_dir: Path) -> List[Path]:
    """Return sorted train run dirs that have a completed checkpoint."""
    train_dir = experiment_dir / "train"
    if not train_dir.exists():
        return []
    return sorted(
        td
        for td in train_dir.iterdir()
        if td.is_dir()
        and (td / "model" / "checkpoints" / "checkpoint-epoch-20.pt").exists()
    )


def load_loss_curves(
    runs_dir: Path = RUNS_DIR, min_trains: int = 1
) -> List[np.ndarray]:
    """Return list of loss arrays, one per training run found across all
    experiments. Each array has shape (N_epochs,) of train_loss values
    from loss.csv. Sorted by experiment then run timestamp."""
    curves = []
    for exp_dir in discover_experiments(runs_dir, min_trains):
        for td in _train_run_dirs(exp_dir):
            loss_csv = td / "model" / "loss.csv"
            if loss_csv.exists():
                df = pd.read_csv(loss_csv)
                if "train_loss" in df.columns:
                    curves.append(df["train_loss"].values)
    return curves


def load_loss_curves_by_experiment(
    runs_dir: Path = RUNS_DIR, min_trains: int = 1
) -> Dict[str, List[np.ndarray]]:
    """Return dict mapping experiment name -> list of loss curve arrays."""
    grouped = {}
    for exp_dir in discover_experiments(runs_dir, min_trains):
        curves = []
        for td in _train_run_dirs(exp_dir):
            loss_csv = td / "model" / "loss.csv"
            if loss_csv.exists():
                df = pd.read_csv(loss_csv)
                if "train_loss" in df.columns:
                    curves.append(df["train_loss"].values)
        if curves:
            grouped[exp_dir.name] = curves
    return grouped


def load_checkpoint_paths(
    runs_dir: Path = RUNS_DIR, min_trains: int = 1
) -> List[Path]:
    """Return list of paths to checkpoint-epoch-20.pt files."""
    paths = []
    for exp_dir in discover_experiments(runs_dir, min_trains):
        for td in _train_run_dirs(exp_dir):
            ckpt = td / "model" / "checkpoints" / "checkpoint-epoch-20.pt"
            if ckpt.exists():
                paths.append(ckpt)
    return paths


def load_checkpoint_state_dict(ckpt_path: Path) -> dict:
    """Load and return model_state_dict from a .pt checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["model_state_dict"]


def load_contact_maps(
    runs_dir: Path = RUNS_DIR, max_sims: Optional[int] = None
) -> np.ndarray:
    """Return concatenated contact map array (dtype=object, sparse COO)
    from all simulation/*/contact_map.npy files found."""
    all_maps = []
    for exp_dir in discover_experiments(runs_dir):
        sim_dir = exp_dir / "simulation"
        if not sim_dir.exists():
            continue
        for sd in sorted(sim_dir.iterdir()):
            if not sd.is_dir():
                continue
            cm_path = sd / "contact_map.npy"
            if cm_path.exists():
                cm = np.load(cm_path, allow_pickle=True)
                all_maps.append(cm)
                if max_sims and len(all_maps) >= max_sims:
                    break
        if max_sims and len(all_maps) >= max_sims:
            break
    if not all_maps:
        return np.array([], dtype=object)
    return np.concatenate(all_maps)


def load_contact_maps_dense(
    runs_dir: Path = RUNS_DIR,
    max_sims: Optional[int] = None,
    shape: Tuple[int, int, int] = (1, 28, 28),
) -> torch.Tensor:
    """Return (N, 1, 28, 28) float32 tensor of dense contact maps."""
    from mdlearn.data.datasets.contact_map import ContactMapDataset

    sparse = load_contact_maps(runs_dir, max_sims)
    if len(sparse) == 0:
        return torch.zeros(0, *shape)
    ds = ContactMapDataset(sparse, shape=shape)
    return torch.stack([ds[i]["X"] for i in range(len(ds))])


def load_task_durations(result_json: Path, method: str) -> np.ndarray:
    """Parse a Colmena JSONL result file and return array of time_running
    values for successful tasks matching `method`."""
    durations = []
    if not result_json.exists():
        return np.array([])
    with open(result_json) as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if record.get("success") and record.get("method", "") == method:
                t = record.get("time_running", 0)
                if t > 0:
                    durations.append(t)
    return np.array(durations)


def load_all_task_durations(
    runs_dir: Path = RUNS_DIR, method: str = "run_train"
) -> np.ndarray:
    """Aggregate task durations across all experiments."""
    task_type_map = {
        "run_train": "train.json",
        "run_simulation": "simulation.json",
        "run_inference": "inference.json",
    }
    filename = task_type_map.get(method, "train.json")
    all_durations = []
    for exp_dir in discover_experiments(runs_dir):
        result_json = exp_dir / "result" / filename
        durations = load_task_durations(result_json, method)
        if len(durations) > 0:
            all_durations.append(durations)
    if not all_durations:
        return np.array([])
    return np.concatenate(all_durations)


def build_cvae_model(ckpt_path: Optional[Path] = None):
    """Instantiate a SymmetricConv2dVAETrainer with the BBA architecture.
    If ckpt_path is given, loads model_state_dict into trainer.model."""
    from mdlearn.nn.models.vae.symmetric_conv2d_vae import (
        SymmetricConv2dVAETrainer,
    )

    trainer = SymmetricConv2dVAETrainer(**BBA_CVAE_CONFIG)
    if ckpt_path is not None:
        state_dict = load_checkpoint_state_dict(ckpt_path)
        trainer.model.load_state_dict(state_dict)
    return trainer


# ── Synthetic workload generators ──────────────────────────


def synthetic_loss_curves(n_curves: int = 10, n_epochs: int = 20) -> List[np.ndarray]:
    """Generate synthetic training loss curves for benchmarking.

    Produces curves with different convergence profiles:
    fast (KRAS-like), slow (BBA-like), and distribution shift (CLN025-like).
    """
    rng = np.random.default_rng(42)
    curves = []
    for i in range(n_curves):
        rate = rng.uniform(0.03, 0.2)
        base = rng.uniform(0.3, 0.8)
        offset = rng.uniform(0.01, 0.1)
        loss = base * np.exp(-rate * np.arange(n_epochs)) + offset
        loss += rng.normal(0, 0.005, n_epochs)
        loss = np.clip(loss, 0.001, 2.0)
        curves.append(loss)
    return curves


def synthetic_loss_profiles() -> Dict[str, np.ndarray]:
    """Generate the 3 canonical synthetic convergence profiles."""
    rng = np.random.default_rng(42)
    n = 50
    profiles = {
        "fast_converge (KRAS-like)": np.clip(
            0.5 * np.exp(-0.15 * np.arange(n)) + 0.02 + rng.normal(0, 0.005, n),
            0.001, 1.0,
        ),
        "slow_converge (BBA-like)": np.clip(
            0.5 * np.exp(-0.03 * np.arange(n)) + 0.08 + rng.normal(0, 0.005, n),
            0.001, 1.0,
        ),
        "distribution_shift (CLN025-like)": np.clip(
            np.concatenate([
                0.5 * np.exp(-0.1 * np.arange(n // 2)),
                0.5 * np.exp(-0.1 * (n // 2 - 1)) + 0.02 * np.arange(n - n // 2),
            ]) + rng.normal(0, 0.005, n),
            0.001, 1.0,
        ),
    }
    return profiles


def synthetic_model(n_params: int, name: str = "synthetic"):
    """Create a synthetic PyTorch model with approximately n_params parameters."""
    import torch.nn as nn
    dim = max(int(np.sqrt(n_params / 2)), 4)
    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )
    return model


def compute_stats(values: np.ndarray, unit: str = "") -> dict:
    """Return dict with mean, std, median, p99, min, max, ci95, n, raw.
    Keys are suffixed with _<unit> if unit is provided."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    suffix = f"_{unit}" if unit else ""
    if n == 0:
        return {
            f"mean{suffix}": 0.0,
            f"std{suffix}": 0.0,
            f"median{suffix}": 0.0,
            f"p99{suffix}": 0.0,
            f"min{suffix}": 0.0,
            f"max{suffix}": 0.0,
            f"ci95{suffix}": 0.0,
            "n": 0,
            f"raw{suffix}": [],
        }
    return {
        f"mean{suffix}": float(np.mean(values)),
        f"std{suffix}": float(np.std(values)),
        f"median{suffix}": float(np.median(values)),
        f"p99{suffix}": float(np.percentile(values, 99)),
        f"min{suffix}": float(np.min(values)),
        f"max{suffix}": float(np.max(values)),
        f"ci95{suffix}": float(1.96 * np.std(values) / np.sqrt(n)) if n > 1 else 0.0,
        "n": n,
        f"raw{suffix}": [float(v) for v in values],
    }


if __name__ == "__main__":
    print("=== Smoke Testing data_loader ===\n")

    exps = discover_experiments()
    print(f"Experiments found: {len(exps)}")
    for e in exps:
        trains = _train_run_dirs(e)
        print(f"  {e.name}: {len(trains)} training runs")

    curves = load_loss_curves()
    print(f"\nLoss curves loaded: {len(curves)}")
    if curves:
        print(f"  First curve shape: {curves[0].shape}")
        print(f"  First curve range: {curves[0].min():.2f} - {curves[0].max():.2f}")

    ckpts = load_checkpoint_paths()
    print(f"\nCheckpoints found: {len(ckpts)}")
    if ckpts:
        sd = load_checkpoint_state_dict(ckpts[0])
        n_params = sum(v.numel() for v in sd.values())
        n_bytes = sum(v.nbytes for v in sd.values())
        print(f"  First checkpoint: {n_params:,} params, {n_bytes:,} bytes")

    train_dur = load_all_task_durations(method="run_train")
    sim_dur = load_all_task_durations(method="run_simulation")
    inf_dur = load_all_task_durations(method="run_inference")
    print(f"\nTask durations:")
    print(f"  Train: N={len(train_dur)}, mean={train_dur.mean():.2f}s, std={train_dur.std():.2f}s" if len(train_dur) else "  Train: none")
    print(f"  Sim:   N={len(sim_dur)}, mean={sim_dur.mean():.2f}s, std={sim_dur.std():.2f}s" if len(sim_dur) else "  Sim: none")
    print(f"  Inf:   N={len(inf_dur)}, mean={inf_dur.mean():.2f}s, std={inf_dur.std():.2f}s" if len(inf_dur) else "  Inf: none")

    sparse = load_contact_maps(max_sims=2)
    print(f"\nContact maps (2 sims): {len(sparse)} frames")

    stats = compute_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), unit="ms")
    print(f"\ncompute_stats test: {stats}")

    print("\n=== All smoke tests passed ===")

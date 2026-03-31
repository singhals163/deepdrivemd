#!/usr/bin/env python3
"""Experiment 2.2: Bit-Reproducibility of Redis Serialization.

Verifies that serializing a model checkpoint to Redis and deserializing
it back produces bit-identical results. Trains a small CVAE for a few
epochs, round-trips through Redis, trains one more epoch, and compares
the loss against a control that was never serialized.

Usage:
    python bench_reproducibility.py [--epochs 5] [--redis-host 127.0.0.1] [--output results_reproducibility.json]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deepdrivemd.redis_io import init_redis, store_torch, load_torch


class SimpleCVAE(nn.Module):
    """Minimal CVAE matching BBA architecture for reproducibility testing."""

    def __init__(self, input_dim: int = 28, latent_dim: int = 10):
        super().__init__()
        flat = input_dim * input_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, flat),
            nn.Sigmoid(),
        )
        self.input_dim = input_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, self.input_dim, self.input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train_epoch(model, optimizer, data, seed: int) -> float:
    """Train one epoch with deterministic seed. Returns mean loss."""
    torch.manual_seed(seed)
    model.train()
    total_loss = 0.0
    n_batches = 0
    batch_size = 16
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Reproducibility test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--output", type=str, default="results_reproducibility.json")
    args = parser.parse_args()

    init_redis(args.redis_host, args.redis_port)

    # Generate synthetic contact maps (BBA-like, 28x28)
    torch.manual_seed(0)
    np.random.seed(0)
    n_samples = 128
    data = torch.rand(n_samples, 28, 28)

    # Create two identical models
    torch.manual_seed(42)
    model_test = SimpleCVAE(input_dim=28, latent_dim=10)
    torch.manual_seed(42)
    model_control = SimpleCVAE(input_dim=28, latent_dim=10)

    # Verify initial state is identical
    for k in model_test.state_dict():
        assert torch.equal(model_test.state_dict()[k], model_control.state_dict()[k]), \
            f"Initial mismatch in {k}"
    print("Initial models are identical: PASS")

    # Train both for args.epochs epochs with same seeds
    opt_test = torch.optim.Adam(model_test.parameters(), lr=1e-3)
    opt_control = torch.optim.Adam(model_control.parameters(), lr=1e-3)

    losses_test = []
    losses_control = []
    for epoch in range(args.epochs):
        loss_t = train_epoch(model_test, opt_test, data, seed=epoch + 100)
        loss_c = train_epoch(model_control, opt_control, data, seed=epoch + 100)
        losses_test.append(loss_t)
        losses_control.append(loss_c)
        print(f"  Epoch {epoch + 1}: test_loss={loss_t:.4f} control_loss={loss_c:.4f}")

    # Verify models match after training
    for k in model_test.state_dict():
        assert torch.equal(model_test.state_dict()[k], model_control.state_dict()[k]), \
            f"Post-training mismatch in {k}"
    print(f"After {args.epochs} epochs, models still identical: PASS")

    # Round-trip test model through Redis
    redis_key = "bench:reproducibility:test"
    test_state = model_test.state_dict()
    opt_state = opt_test.state_dict()

    store_torch(f"{redis_key}:model", test_state)
    store_torch(f"{redis_key}:optimizer", opt_state)

    loaded_model_state = load_torch(f"{redis_key}:model", map_location="cpu")
    loaded_opt_state = load_torch(f"{redis_key}:optimizer", map_location="cpu")

    # Verify bit-identical after round-trip
    max_diff = 0.0
    for k in test_state:
        diff = torch.abs(test_state[k].float() - loaded_model_state[k].float()).max().item()
        max_diff = max(max_diff, diff)
        assert diff == 0.0, f"Redis round-trip mismatch in {k}: max_diff={diff}"

    print(f"Redis round-trip bit-identical: PASS (max_diff={max_diff})")

    # Load into test model and train one more epoch
    model_test.load_state_dict(loaded_model_state)
    opt_test.load_state_dict(loaded_opt_state)

    final_epoch = args.epochs
    final_loss_test = train_epoch(model_test, opt_test, data, seed=final_epoch + 100)
    final_loss_control = train_epoch(model_control, opt_control, data, seed=final_epoch + 100)

    loss_diff = abs(final_loss_test - final_loss_control)
    print(f"\nFinal epoch after Redis round-trip:")
    print(f"  Test loss:    {final_loss_test:.6f}")
    print(f"  Control loss: {final_loss_control:.6f}")
    print(f"  Difference:   {loss_diff:.2e}")
    print(f"  Bit-identical: {'PASS' if loss_diff == 0.0 else 'FAIL'}")

    result = {
        "epochs_before_roundtrip": args.epochs,
        "losses_test": losses_test,
        "losses_control": losses_control,
        "redis_roundtrip_max_diff": max_diff,
        "final_loss_test": final_loss_test,
        "final_loss_control": final_loss_control,
        "final_loss_diff": loss_diff,
        "bit_identical": loss_diff == 0.0,
        "all_passed": max_diff == 0.0 and loss_diff == 0.0,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

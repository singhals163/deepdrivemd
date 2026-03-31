#!/usr/bin/env python3
"""Generate figures for Study 1: Signal Monitor.

Produces:
  - fig_study1a_latency: Bar chart of per-policy latency (mean + p99)
  - fig_study1b_pluggability: Timeline showing different freeze points per policy
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plot_config import (
    RESULTS_DIR, apply_style, savefig,
    BLUE, ORANGE, RED, GRAY_DARK,
)

apply_style()
S1_DIR = RESULTS_DIR / "study1"


def plot_latency():
    with open(S1_DIR / "results_latency.json") as f:
        data = json.load(f)

    short_labels = ["Threshold\nO(1)", "Sliding Window\nO(Nk)", "Mann-Kendall\nO(Nk²)"]
    means = [d["mean_ms"] for d in data]
    p99s = [d["p99_ms"] for d in data]
    maxes = [d["max_ms"] for d in data]

    x = np.arange(len(short_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width, means, width, label="Mean", color=BLUE, edgecolor="white")
    bars2 = ax.bar(x, p99s, width, label="P99", color=ORANGE, edgecolor="white")
    bars3 = ax.bar(x + width, maxes, width, label="Max", color=RED, edgecolor="white")

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Policy Complexity")
    ax.set_title("Signal Monitor Interface Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.legend(loc="upper left")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.annotate("Simulation cycle: 1,620,000 ms (27 min)",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", fontsize=8, fontstyle="italic", color="gray")

    ax.set_ylim(0, max(maxes) * 1.4)
    fig.tight_layout()
    savefig(fig, "fig_study1a_latency")
    plt.close(fig)


def plot_pluggability():
    with open(S1_DIR / "results_pluggability.json") as f:
        data = json.load(f)

    rng = np.random.default_rng(42)
    profiles = {
        "fast_converge": 0.5 * np.exp(-0.15 * np.arange(50)) + 0.02,
        "slow_converge": 0.5 * np.exp(-0.03 * np.arange(50)) + 0.08,
        "distribution_shift": np.concatenate([
            0.5 * np.exp(-0.1 * np.arange(25)),
            0.5 * np.exp(-0.1 * 24) + 0.02 * np.arange(25),
        ]),
    }
    profile_labels = {
        "fast_converge": "Fast Convergence\n(KRAS-like)",
        "slow_converge": "Slow Convergence\n(BBA-like)",
        "distribution_shift": "Distribution Shift\n(CLN025-like)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, (profile_key, losses), result in zip(axes, profiles.items(), data):
        noise = rng.normal(0, 0.005, len(losses))
        noisy_losses = np.clip(losses + noise, 0.001, 1.0)
        ax.plot(noisy_losses, color=GRAY_DARK, linewidth=1.2, label="Training loss")

        freeze_a = result["policy_a"]["freeze_step"]
        freeze_b = result["policy_b"]["freeze_step"]
        if freeze_a is not None:
            ax.axvline(x=freeze_a, color=BLUE, linestyle="--", linewidth=1.5,
                       label=f"Threshold (step {freeze_a})")
        if freeze_b is not None:
            ax.axvline(x=freeze_b, color=ORANGE, linestyle="-.", linewidth=1.5,
                       label=f"Sliding Window (step {freeze_b})")

        ax.set_title(profile_labels[profile_key], fontsize=10)
        ax.set_xlabel("Training Cycle")
        if ax == axes[0]:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 50)

    fig.suptitle("Policy Pluggability: Same Interface, Different Decisions", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_study1b_pluggability")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Study 1 Figures ===")
    plot_latency()
    plot_pluggability()
    print("Done.")

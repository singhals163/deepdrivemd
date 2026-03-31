#!/usr/bin/env python3
"""Generate figures for Study 1: Signal Monitor.

Produces:
  - fig_study1a_latency: Bar chart of per-policy latency with CI95 error bars
  - fig_study1b_pluggability: Real training loss curves with freeze-point markers
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
S1_DIR = RESULTS_DIR / "study1" / "real"


def plot_latency():
    with open(S1_DIR / "results_latency.json") as f:
        data = json.load(f)

    labels = [d["label"] for d in data]
    means = [d["mean_ms"] for d in data]
    ci95s = [d["ci95_ms"] for d in data]
    p99_means = [d["p99"]["mean_ms"] for d in data]
    p99_ci95s = [d["p99"]["ci95_ms"] for d in data]
    max_means = [d["max"]["mean_ms"] for d in data]
    max_ci95s = [d["max"]["ci95_ms"] for d in data]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width, means, width, yerr=ci95s, capsize=3,
           label="Mean", color=BLUE, edgecolor="white")
    ax.bar(x, p99_means, width, yerr=p99_ci95s, capsize=3,
           label="P99", color=ORANGE, edgecolor="white")
    ax.bar(x + width, max_means, width, yerr=max_ci95s, capsize=3,
           label="Max", color=RED, edgecolor="white")

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Policy Complexity")
    ax.set_title("Signal Monitor Interface Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper left")

    n_trials = data[0].get("n_trials", "?")
    n_repeats = data[0].get("n_repeats_per_trial", "?")
    ax.annotate(f"N={n_trials} trials x {n_repeats} repeats, real training loss",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", fontsize=7, fontstyle="italic", color="gray")

    ax.set_ylim(0, max(max_means) * 1.5)
    fig.tight_layout()
    savefig(fig, "fig_study1a_latency")
    plt.close(fig)


def plot_pluggability():
    with open(S1_DIR / "results_pluggability.json") as f:
        data = json.load(f)

    n_panels = min(len(data), 5)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, result in zip(axes, data[:n_panels]):
        # Read real loss curve from JSON (normalized)
        if "loss_curve_normalized" in result:
            losses = np.array(result["loss_curve_normalized"])
        elif "loss_curve" in result:
            losses = np.array(result["loss_curve"])
            losses = losses / losses[0]  # normalize
        else:
            continue

        ax.plot(losses, color=GRAY_DARK, linewidth=1.2, label="Training loss")

        freeze_a = result["policy_a"]["freeze_step"]
        freeze_b = result["policy_b"]["freeze_step"]
        if freeze_a is not None:
            ax.axvline(x=freeze_a, color=BLUE, linestyle="--", linewidth=1.5,
                       label=f"Threshold (step {freeze_a})")
        if freeze_b is not None:
            ax.axvline(x=freeze_b, color=ORANGE, linestyle="-.", linewidth=1.5,
                       label=f"SlidingWindow (step {freeze_b})")

        exp_name = result["profile"].replace("experiment-310326-", "")
        n_runs = result.get("n_training_runs", "?")
        ax.set_title(f"{exp_name}\n({n_runs} runs)", fontsize=9)
        ax.set_xlabel("Epoch")
        if ax == axes[0]:
            ax.set_ylabel("Normalized Loss")
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Policy Pluggability: Same Interface, Different Decisions\n(real training data)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    savefig(fig, "fig_study1b_pluggability")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Study 1 Figures ===")
    plot_latency()
    print("  fig_study1a_latency done")
    plot_pluggability()
    print("  fig_study1b_pluggability done")
    print("Done.")

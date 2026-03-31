#!/usr/bin/env python3
"""Generate figures for Study 2: Stateful Service.

Produces:
  - fig_study2a_latency: Serialize/deserialize latency by model size with CI95
  - fig_study2b_memory: Memory lifecycle with error bars across trials
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plot_config import (
    RESULTS_DIR, apply_style, savefig,
    BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT,
    GREEN, GREEN_DARK, RED, GRAY,
)

apply_style()
S2_DIR = RESULTS_DIR / "study2" / "real"


def plot_latency():
    with open(S2_DIR / "results_latency.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: grouped bar chart by model size
    n = len(data)
    labels = []
    for d in data:
        p = d["n_params"]
        if p < 1_000_000:
            labels.append(f"{p/1000:.0f}K")
        else:
            labels.append(f"{p/1_000_000:.1f}M")

    ser_mean = [d["serialize"]["mean_ms"] for d in data]
    ser_ci = [d["serialize"]["ci95_ms"] for d in data]
    deser_mean = [d["deserialize"]["mean_ms"] for d in data]
    deser_ci = [d["deserialize"]["ci95_ms"] for d in data]

    x = np.arange(n)
    width = 0.35
    ax1.bar(x - width/2, ser_mean, width, yerr=ser_ci, capsize=4,
            label="Serialize", color=BLUE, edgecolor="white")
    ax1.bar(x + width/2, deser_mean, width, yerr=deser_ci, capsize=4,
            label="Deserialize", color=ORANGE, edgecolor="white")

    for i in range(n):
        n_ckpts = data[i]["n_checkpoints"]
        ax1.annotate(f"N={n_ckpts}", xy=(i, 0), xytext=(0, -18),
                     textcoords="offset points", ha="center", fontsize=7, color="gray")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xlabel("Model Size (params)")
    ax1.set_title("Redis Serialization Latency\n(real CVAE checkpoints)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)

    # Right: log-log scaling plot
    sizes_kb = [d["data_kb"] for d in data]
    rt_mean = [d["roundtrip"]["mean_ms"] for d in data]
    rt_ci = [d["roundtrip"]["ci95_ms"] for d in data]

    ax2.errorbar(sizes_kb, rt_mean, yerr=rt_ci, fmt="o-", color=GREEN,
                 linewidth=2, markersize=8, capsize=5, label="Roundtrip")
    ax2.errorbar(sizes_kb, ser_mean, yerr=ser_ci, fmt="s--", color=BLUE,
                 linewidth=1.5, markersize=6, capsize=4, label="Serialize")
    ax2.errorbar(sizes_kb, deser_mean, yerr=deser_ci, fmt="^--", color=ORANGE,
                 linewidth=1.5, markersize=6, capsize=4, label="Deserialize")

    for i, label in enumerate(labels):
        ax2.annotate(label, (sizes_kb[i], rt_mean[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax2.set_xlabel("Checkpoint Size (KB)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Scaling with Model Size")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig_study2a_latency")
    plt.close(fig)


def plot_memory():
    with open(S2_DIR / "results_memory.json") as f:
        data = json.load(f)

    # Handle both list (multi-config) and dict (single-config) formats
    if isinstance(data, dict):
        configs = [data]
    else:
        configs = data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: memory freed per model size
    labels = []
    for c in configs:
        p = c["n_params"]
        if p < 1_000_000:
            labels.append(f"{p/1000:.0f}K\n({c['model_mb']:.1f} MB)")
        else:
            labels.append(f"{p/1_000_000:.1f}M\n({c['model_mb']:.1f} MB)")

    x = np.arange(len(labels))
    freed_mean = [c["memory_freed"]["mean_mb"] for c in configs]
    freed_ci = [c["memory_freed"]["ci95_mb"] for c in configs]
    restored_mean = [c["memory_restored"]["mean_mb"] for c in configs]
    restored_ci = [c["memory_restored"]["ci95_mb"] for c in configs]

    width = 0.35
    ax1.bar(x - width/2, freed_mean, width, yerr=freed_ci, capsize=4,
            label="Freed (freeze)", color=GREEN, edgecolor="white")
    ax1.bar(x + width/2, restored_mean, width, yerr=restored_ci, capsize=4,
            label="Restored (resume)", color=BLUE, edgecolor="white")

    for i in range(len(labels)):
        ax1.annotate(f"{freed_mean[i]:.1f}",
                     xy=(x[i] - width/2, freed_mean[i] + freed_ci[i]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8)
        ax1.annotate(f"{restored_mean[i]:.1f}",
                     xy=(x[i] + width/2, restored_mean[i] + restored_ci[i]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8)

    ax1.set_ylabel("Memory (MB)")
    ax1.set_xlabel("Model Size")
    ax1.set_title("Freeze/Resume Memory Transfer\n(real CVAE checkpoints via Redis)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend(fontsize=8)

    # Right: lifecycle phases for the largest model
    biggest = configs[-1]
    trials = biggest["trials"]
    actives = [t["rss_active_mb"] for t in trials]
    frozens = [t["rss_frozen_mb"] for t in trials]
    resumeds = [t["rss_resumed_mb"] for t in trials]

    bp = ax2.boxplot(
        [actives, frozens, resumeds],
        tick_labels=["Active\n(loaded)", "Frozen\n(to Redis)", "Resumed\n(from Redis)"],
        patch_artist=True,
    )
    colors_box = [RED, GREEN, BLUE]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    p = biggest["n_params"]
    if p < 1_000_000:
        size_label = f"{p/1000:.0f}K"
    else:
        size_label = f"{p/1_000_000:.1f}M"
    ax2.set_ylabel("RSS (MB)")
    ax2.set_title(f"Lifecycle Phases — {size_label} params\n({biggest['model_mb']:.0f} MB checkpoint)")

    fig.tight_layout()
    savefig(fig, "fig_study2b_memory")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Study 2 Figures ===")
    plot_latency()
    print("  fig_study2a_latency done")
    plot_memory()
    print("  fig_study2b_memory done")
    print("Done.")

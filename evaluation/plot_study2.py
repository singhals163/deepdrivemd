#!/usr/bin/env python3
"""Generate figures for Study 2: Stateful Service.

Produces:
  - fig_study2a_latency: Serialize/deserialize latency vs model size
  - fig_study2b_memory: Memory lifecycle showing worker cleanup
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
S2_DIR = RESULTS_DIR / "study2"


def plot_latency():
    with open(S2_DIR / "results_latency.json") as f:
        data = json.load(f)

    short_names = ["BBA\n(200K)", "CLN025\n(2M)", "KRAS\n(66M)"]
    sizes_kb = [d["data_kb"] for d in data]
    ser_mean = [d["serialize"]["mean_ms"] for d in data]
    ser_p99 = [d["serialize"]["p99_ms"] for d in data]
    deser_mean = [d["deserialize"]["mean_ms"] for d in data]
    deser_p99 = [d["deserialize"]["p99_ms"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(short_names))
    width = 0.2
    ax1.bar(x - 1.5*width, ser_mean, width, label="Serialize (mean)", color=BLUE)
    ax1.bar(x - 0.5*width, ser_p99, width, label="Serialize (p99)", color=BLUE_LIGHT)
    ax1.bar(x + 0.5*width, deser_mean, width, label="Deserialize (mean)", color=ORANGE)
    ax1.bar(x + 1.5*width, deser_p99, width, label="Deserialize (p99)", color=ORANGE_LIGHT)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xlabel("Model Size")
    ax1.set_title("Redis Serialization Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names)
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    roundtrip = [d["roundtrip_mean_ms"] for d in data]
    ax2.plot(sizes_kb, roundtrip, "o-", color=GREEN, linewidth=2, markersize=8,
             label="Roundtrip")
    ax2.plot(sizes_kb, ser_mean, "s--", color=BLUE, linewidth=1.5, markersize=6,
             label="Serialize")
    ax2.plot(sizes_kb, deser_mean, "^--", color=ORANGE, linewidth=1.5, markersize=6,
             label="Deserialize")
    for i, name in enumerate(["BBA", "CLN025", "KRAS"]):
        ax2.annotate(name, (sizes_kb[i], roundtrip[i]),
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    configs = [d["config_name"] for d in data]
    baselines = [d["rss_baseline_mb"] for d in data]
    loaded = [d["rss_loaded_mb"] for d in data]
    cleaned = [d["rss_cleaned_mb"] for d in data]

    x = np.arange(len(configs))
    width = 0.25
    ax1.bar(x - width, baselines, width, label="Baseline", color=GRAY, edgecolor="gray")
    ax1.bar(x, loaded, width, label="Model Loaded", color=RED, edgecolor="gray")
    ax1.bar(x + width, cleaned, width, label="After Cleanup", color=GREEN, edgecolor="gray")
    ax1.set_ylabel("RSS (MB)")
    ax1.set_xlabel("Model Configuration")
    ax1.set_title("Worker Memory at Each Phase")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.legend(fontsize=8)

    recovery_pct = [d["recovery_pct"] for d in data]
    increase_mb = [d["memory_increase_mb"] for d in data]
    recovered_mb = [d["memory_recovered_mb"] for d in data]
    bar_colors = ["#FFCDD2", ORANGE_LIGHT, GREEN]
    bars = ax2.bar(x, recovery_pct, 0.5, color=bar_colors, edgecolor="gray")
    for i, (bar, inc, rec) in enumerate(zip(bars, increase_mb, recovered_mb)):
        ax2.annotate(f"+{inc:.0f}MB\n-{rec:.0f}MB",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=8)
    ax2.set_ylabel("Memory Recovered (%)")
    ax2.set_xlabel("Model Configuration")
    ax2.set_title("Memory Recovery After Worker Cleanup")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    savefig(fig, "fig_study2b_memory")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Study 2 Figures ===")
    plot_latency()
    plot_memory()
    print("Done.")

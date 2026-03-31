#!/usr/bin/env python3
"""Generate 2x2 overview figure combining key results from all studies.

Produces:
  - fig_overview: Combined panel (a) latency, (b) serialization, (c) throughput, (d) reallocation
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plot_config import (
    RESULTS_DIR, apply_style, savefig,
    BLUE, ORANGE, GREEN, GREEN_DARK, GRAY,
)

apply_style()


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ── (a) Study 1: Signal Monitor Overhead ──
    ax = axes[0, 0]
    with open(RESULTS_DIR / "study1" / "results_latency.json") as f:
        s1 = json.load(f)
    policies = ["Threshold\nO(1)", "Sliding\nWindow\nO(Nk)", "Mann-\nKendall\nO(Nk²)"]
    means = [d["mean_ms"] for d in s1]
    p99s = [d["p99_ms"] for d in s1]
    x = np.arange(len(policies))
    width = 0.3
    ax.bar(x - width/2, means, width, label="Mean", color=BLUE, edgecolor="white")
    ax.bar(x + width/2, p99s, width, label="P99", color=ORANGE, edgecolor="white")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(a) Signal Monitor Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=7)
    ax.legend(loc="upper left", fontsize=7)
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.annotate(f"{m:.3f}", xy=(i - width/2, m), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6)

    # ── (b) Study 2: Serialization Latency ──
    ax = axes[0, 1]
    with open(RESULTS_DIR / "study2" / "results_latency.json") as f:
        s2 = json.load(f)
    models = ["BBA\n(200K)", "CLN025\n(2M)", "KRAS\n(66M)"]
    roundtrips = [d["roundtrip_mean_ms"] for d in s2]
    ax.bar(np.arange(len(models)), roundtrips, 0.5,
           color=["#81C784", GREEN, GREEN_DARK], edgecolor="white")
    for i, rt in enumerate(roundtrips):
        ax.annotate(f"{rt:.1f}ms", xy=(i, rt), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("Roundtrip Latency (ms)")
    ax.set_title("(b) Redis Serialization Latency")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_yscale("log")

    # ── (c) Study 3: Throughput Improvement ──
    ax = axes[1, 0]
    with open(RESULTS_DIR / "study3" / "results_throughput.json") as f:
        s3 = json.load(f)
    scaling = s3["scaling"]
    systems = [s["system"].split(" (")[0] for s in scaling]
    before = [s["throughput_before"] for s in scaling]
    after = [s["throughput_after"] for s in scaling]
    x = np.arange(len(systems))
    width = 0.3
    ax.bar(x - width/2, before, width, label="7 GPUs (before)", color=GRAY, edgecolor="gray")
    ax.bar(x + width/2, after, width, label="8 GPUs (after)", color=GREEN, edgecolor="gray")
    ax.set_ylabel("Throughput (sims/min)")
    ax.set_title("(c) Throughput Recovery After GPU Reclaim")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    for i in range(len(systems)):
        ax.annotate("+14.3%", xy=(i + width/2, after[i]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=GREEN_DARK, fontweight="bold")

    # ── (d) Study 3: Reallocation Latency ──
    ax = axes[1, 1]
    with open(RESULTS_DIR / "study3" / "results_reallocation.json") as f:
        s3r = json.load(f)
    n_inflight = [d["n_inflight"] for d in s3r]
    init_ms = [d["initiation"]["mean_ms"] for d in s3r]
    total_ms = [d["total"]["mean_ms"] for d in s3r]
    ax.plot(n_inflight, total_ms, "o-", color=ORANGE, linewidth=2,
            markersize=8, label="Total (with drain)")
    ax.plot(n_inflight, init_ms, "s--", color=BLUE, linewidth=1.5,
            markersize=6, label="Broker overhead only")
    expected = [n * 10 for n in n_inflight]
    ax.plot(n_inflight, expected, "r:", linewidth=1, label="Expected drain")
    ax.set_xlabel("In-flight ML Tasks")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(d) GPU Reallocation Latency")
    ax.legend(fontsize=7)

    fig.suptitle("Dynamic Provisioning: Component Evaluation Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig_overview")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Overview Figure ===")
    main()
    print("Done.")

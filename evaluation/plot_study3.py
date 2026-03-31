#!/usr/bin/env python3
"""Generate figures for Study 3: Resource Broker.

Produces:
  - fig_study3a_reallocation: Reallocation latency vs in-flight tasks
  - fig_study3b_throughput: Throughput improvement from GPU reclaim
  - fig_study3c_cooldown: Cooldown throttling effectiveness
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plot_config import (
    RESULTS_DIR, apply_style, savefig,
    BLUE, ORANGE, GREEN, GREEN_DARK, RED, GRAY,
)

apply_style()
S3_DIR = RESULTS_DIR / "study3"


def plot_reallocation():
    with open(S3_DIR / "results_reallocation.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    n_inflight = [d["n_inflight"] for d in data]
    init_mean = [d["initiation"]["mean_ms"] for d in data]
    total_mean = [d["total"]["mean_ms"] for d in data]

    x = np.arange(len(n_inflight))
    ax1.bar(x, init_mean, 0.5, color=BLUE, edgecolor="white")
    for i, v in enumerate(init_mean):
        ax1.annotate(f"{v:.4f}", xy=(i, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
    ax1.set_ylabel("Initiation Latency (ms)")
    ax1.set_xlabel("In-flight ML Tasks")
    ax1.set_title("Broker Decision Overhead")
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_inflight)
    ax1.set_ylim(0, max(init_mean) * 2)

    colors = [GREEN, "#8BC34A", "#FFC107", ORANGE]
    ax2.bar(x, total_mean, 0.5, color=colors, edgecolor="white")
    for i, v in enumerate(total_mean):
        ax2.annotate(f"{v:.1f}", xy=(i, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
    expected = [d["n_inflight"] * d["drain_delay_s"] * 1000 for d in data]
    ax2.plot(x, expected, "ro--", markersize=6, label="Expected (N × 10ms)")
    ax2.set_ylabel("Total Transition Time (ms)")
    ax2.set_xlabel("In-flight ML Tasks")
    ax2.set_title("End-to-End Reallocation Latency\n(includes graceful drain)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_inflight)
    ax2.legend()

    fig.tight_layout()
    savefig(fig, "fig_study3a_reallocation")
    plt.close(fig)


def plot_throughput():
    with open(S3_DIR / "results_throughput.json") as f:
        data = json.load(f)

    scaling = data["scaling"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    systems = [s["system"].split(" (")[0] for s in scaling]
    before = [s["throughput_before"] for s in scaling]
    after = [s["throughput_after"] for s in scaling]

    x = np.arange(len(systems))
    width = 0.3
    ax1.bar(x - width/2, before, width, label="7 GPUs (before)", color=GRAY, edgecolor="gray")
    ax1.bar(x + width/2, after, width, label="8 GPUs (after)", color=GREEN, edgecolor="gray")
    ax1.set_ylabel("Throughput (sims/min)")
    ax1.set_title("Simulation Throughput: GPU Reclaim")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    for i in range(len(systems)):
        ax1.annotate("+14.3%", xy=(i + width/2, after[i]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7, color=GREEN_DARK, fontweight="bold")

    # GPU allocation timeline
    colors_sim = "#64B5F6"
    colors_ml = "#FF7043"
    for i in range(7):
        ax2.barh(i, 60, left=0, height=0.7, color=colors_sim, edgecolor="white")
        ax2.barh(i, 40, left=60, height=0.7, color=colors_sim, edgecolor="white")
    ax2.barh(7, 60, left=0, height=0.7, color=colors_ml, edgecolor="white")
    ax2.barh(7, 40, left=60, height=0.7, color=colors_sim, edgecolor="white")
    ax2.axvline(x=60, color="red", linestyle="--", linewidth=2, label="Freeze Event")
    ax2.set_xlabel("Time (arbitrary)")
    ax2.set_ylabel("GPU")
    ax2.set_title("GPU Allocation Timeline")
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([f"GPU {i}" for i in range(8)], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.annotate("ML Training", xy=(30, 7.3), ha="center", fontsize=8,
                 color="white", fontweight="bold")
    ax2.annotate("Simulations", xy=(80, 7.3), ha="center", fontsize=8,
                 color="white", fontweight="bold")
    ax2.annotate("+14.3%\nthroughput", xy=(80, -0.8), ha="center", fontsize=9,
                 color=GREEN, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig_study3b_throughput")
    plt.close(fig)


def plot_cooldown():
    with open(S3_DIR / "results_cooldown.json") as f:
        data = json.load(f)

    tests = data["tests"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    test_names = ["No Cooldown\n(0s)", "With Cooldown\n(0.1s)", "Stress Test\n(0.05s)"]
    test_data = [tests[0], tests[1], tests[3]]
    attempted = []
    succeeded = []
    for t in test_data:
        att = t.get("total_attempts", t.get("attempted_transitions", 0))
        suc = t["successful_transitions"]
        attempted.append(att)
        succeeded.append(suc)

    x = np.arange(len(test_names))
    width = 0.3
    ax1.bar(x - width/2, attempted, width, label="Attempted", color=GRAY, edgecolor="gray")
    ax1.bar(x + width/2, succeeded, width, label="Succeeded", color=GREEN, edgecolor="gray")
    for i in range(len(test_names)):
        pct = (succeeded[i] / attempted[i] * 100) if attempted[i] > 0 else 0
        ax1.annotate(f"{pct:.0f}%", xy=(i + width/2, succeeded[i]),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Number of Transitions")
    ax1.set_title("Cooldown Throttling Effect")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend()

    expiry_test = tests[2]
    actions = [r["action"].replace("_", "\n") for r in expiry_test["results"]]
    success = [r["success"] for r in expiry_test["results"]]
    colors = [GREEN if s else RED for s in success]
    x2 = np.arange(len(actions))
    ax2.bar(x2, [1]*len(actions), 0.6, color=colors, edgecolor="white")
    for i, s in enumerate(success):
        label = "OK" if s else "Blocked"
        ax2.annotate(label, xy=(i, 0.5), ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
    ax2.set_yticks([])
    ax2.set_title("Cooldown Expiry Correctness\n(green=allowed, red=blocked)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(actions, fontsize=7)

    fig.tight_layout()
    savefig(fig, "fig_study3c_cooldown")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Study 3 Figures ===")
    plot_reallocation()
    plot_throughput()
    plot_cooldown()
    print("Done.")

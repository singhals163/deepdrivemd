#!/usr/bin/env python3
"""Generate figures for Study 3: Resource Broker.

Produces:
  - fig_study3a_reallocation: Reallocation latency with CI95 error bars
  - fig_study3b_throughput: Real throughput with error bars + analytical model
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
S3_DIR = RESULTS_DIR / "study3" / "real"


def plot_reallocation():
    with open(S3_DIR / "results_reallocation.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = [d["label"].split("(")[0].strip() for d in data]
    n_inflight = [d["n_inflight"] for d in data]
    init_mean = [d["initiation"]["mean_ms"] for d in data]
    init_ci = [d["initiation"]["ci95_ms"] for d in data]
    total_mean = [d["total"]["mean_ms"] for d in data]
    total_ci = [d["total"]["ci95_ms"] for d in data]

    x = np.arange(len(labels))
    ax1.bar(x, init_mean, 0.5, yerr=init_ci, capsize=3,
            color=BLUE, edgecolor="white")
    for i, (v, ci) in enumerate(zip(init_mean, init_ci)):
        ax1.annotate(f"{v:.4f}", xy=(i, v + ci), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
    ax1.set_ylabel("Initiation Latency (ms)")
    ax1.set_xlabel("In-flight ML Tasks")
    ax1.set_title("Broker Decision Overhead")
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_inflight)

    colors = [GREEN, "#8BC34A", "#FFC107", ORANGE]
    ax2.bar(x, total_mean, 0.5, yerr=total_ci, capsize=3,
            color=colors, edgecolor="white")
    for i, (v, ci) in enumerate(zip(total_mean, total_ci)):
        ax2.annotate(f"{v:.1f}", xy=(i, v + ci), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
    ax2.set_ylabel("Total Transition Time (ms)")
    ax2.set_xlabel("In-flight ML Tasks")
    ax2.set_title("End-to-End Reallocation Latency\n(drain delays from real train durations)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_inflight)

    fig.tight_layout()
    savefig(fig, "fig_study3a_reallocation")
    plt.close(fig)


def plot_throughput():
    with open(S3_DIR / "results_throughput.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: real throughput per experiment
    if "real_data" in data:
        rd = data["real_data"]
        exps = rd["experiments"]
        exp_names = [e["experiment"].replace("experiment-310326-", "") for e in exps]
        tps = [e["throughput_sims_per_min"] for e in exps]

        x = np.arange(len(exp_names))
        ax1.bar(x, tps, 0.6, color=BLUE, edgecolor="white")
        mean_tp = rd["throughput"]["mean_spm"]
        ci95_tp = rd["throughput"]["ci95_spm"]
        ax1.axhline(y=mean_tp, color=RED, linestyle="--", linewidth=1.5,
                     label=f"Mean: {mean_tp:.2f} +/- {ci95_tp:.2f}")
        ax1.fill_between([-0.5, len(exp_names) - 0.5],
                         mean_tp - ci95_tp, mean_tp + ci95_tp,
                         alpha=0.15, color=RED)
        ax1.set_ylabel("Throughput (sims/min)")
        ax1.set_title("Real Simulation Throughput\n(per experiment)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_names, fontsize=7, rotation=45, ha="right")
        ax1.legend(fontsize=8)

    # Right: analytical model
    ana = data["analytical"]
    categories = [f"Before\n({ana['sim_gpus_before']} GPUs)",
                  f"After\n({ana['sim_gpus_after']} GPUs)"]
    values = [ana["throughput_before_spm"], ana["throughput_after_spm"]]
    colors = [GRAY, GREEN]
    bars = ax2.bar([0, 1], values, 0.5, color=colors, edgecolor="white")
    imp = ana["improvement_pct"]
    ax2.annotate(f"+{imp:.1f}%", xy=(1, values[1]),
                 xytext=(0, 5), textcoords="offset points",
                 ha="center", fontsize=10, color=GREEN_DARK, fontweight="bold")
    ax2.set_ylabel("Throughput (sims/min)")
    ax2.set_title(f"Analytical: GPU Reclaim Impact\n(real sim duration: {ana['real_sim_time_s']:.0f}s)")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, max(values) * 1.3)

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
    colors_bar = [GREEN if s else RED for s in success]
    x2 = np.arange(len(actions))
    ax2.bar(x2, [1]*len(actions), 0.6, color=colors_bar, edgecolor="white")
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
    print("  fig_study3a_reallocation done")
    plot_throughput()
    print("  fig_study3b_throughput done")
    plot_cooldown()
    print("  fig_study3c_cooldown done")
    print("Done.")

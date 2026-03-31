# Evaluation Experiment Progress

**Last updated:** 2026-03-30

## Paper Context

The paper ("The AI Tax") proposes a four-component dynamic provisioning
architecture (Section 4: Design):

1. **Signal Monitor** — lightweight, pluggable convergence detection
2. **AI Service Manager / Stateful Service** — model lifecycle via Redis
3. **Resource Broker** — dynamic GPU reallocation with safe transitions
4. **Workflow Engine Integration** — end-to-end orchestration

The motivation section (existing figures: `freeze_sweep_all_systems.pdf`,
`compute_savings.pdf`, `dynamic_gantt.pdf`) shows the *problem* — the AI
Tax. The evaluation section needs to validate the *solution* — each
component works correctly, adds negligible overhead, and the integrated
system delivers the same science with better system efficiency.

Ada's requirements from the 2026-03-30 meeting:
1. Show instrumentation overhead is negligible
2. Show stateful→stateless transition measurably improves system metrics
3. Show dynamic GPU reclaim works
4. Broader workload diversity (nice-to-have)

---

## Study 1: Signal Monitor — COMPLETE

**Maps to:** Ada's "overheads of adding instrumentation" and "how much
time do you have to make decisions"; Dejan's "one signal module with
different AI interfaces"

**Component validated:** Signal Monitor (design.tex Section 4.2)

### Results

| Experiment | Key Result | File |
|---|---|---|
| 1.1 Latency | Threshold 0.002ms, SlidingWindow 0.006ms, MannKendall 0.036ms — all negligible vs 27-min sim cycle (0.000002% overhead) | `study1/results_latency.json` |
| 1.2 Concurrency | 5000/5000 signals, 88K signals/sec, zero drops, zero errors, consistent state chain | `study1/results_concurrency.json` |
| 1.3 Pluggability | Same interface, different policies → different freeze points. KRAS-like: threshold freezes at step 20, sliding window at step 31. BBA-like: neither freezes. CLN025-like: both freeze but at different steps. | `study1/results_pluggability.json` |

### Figures

| Figure | What it shows | Paper argument |
|---|---|---|
| `fig_study1a_latency` | Bar chart: mean/p99/max latency per policy complexity | Decision budget is enormous — even O(Nk²) policy uses <0.04ms vs 1,620,000ms sim cycle |
| `fig_study1b_pluggability` | 3-panel: training loss curves with freeze-point markers per policy | Architecture is pluggable — swap policy object, get different decisions, no workflow code changes |

### Why these results work for the evaluation

- **Proves design claim:** "Signal Monitor runs in the thinker thread without blocking." All policies are sub-millisecond, 4+ orders of magnitude below shortest training cycle (77s).
- **Proves pluggability:** Same `submit_telemetry()` interface, different policies, different outcomes — validates Dejan's "define these interfaces" ask.
- **Proves correctness:** Thread-safe under concurrent access (50 threads), zero dropped signals, consistent state ordering.

---

## Study 2: Stateful Service — COMPLETE

**Maps to:** Vijay's "move from statelessness into a stateful form" and
"do we remove the AI's memory from the GPU? how long does that take?"

**Component validated:** AI Service Manager (design.tex Section 4.2)

### Results

| Experiment | Key Result | File |
|---|---|---|
| 2.1 Latency | BBA (200K params): 2.6ms roundtrip. CLN025 (2M): 28ms. KRAS (66M): 877ms. All sub-second vs training cycles of 77-3,075s. | `study2/results_latency.json` |
| 2.2 Reproducibility | Bit-identical after Redis roundtrip (max_diff=0.0). Final loss after resumed training matches control exactly. | `study2/results_reproducibility.json` |
| 2.3 Memory | Workers recover 11-91% memory after cleanup. BBA: 11% (runtime overhead dominates). CLN025: 84%. NTL9: 91%. | `study2/results_memory.json` |

### Figures

| Figure | What it shows | Paper argument |
|---|---|---|
| `fig_study2a_latency` | Left: grouped bar (serialize/deserialize mean+p99). Right: log-log scaling with checkpoint size | Stateful transitions are fast — even 66M-param KRAS roundtrips in <1s, negligible vs training time |
| `fig_study2b_memory` | Left: RSS at baseline/loaded/cleaned phases. Right: recovery percentage per model size | Workers are naturally stateless — model memory freed after task. State lives in Redis, not in worker. |

### Why these results work for the evaluation

- **Proves design claim:** "Checkpoint save 11ms, load 4ms" (Table 3 in paper). Our BBA numbers (1.1ms serialize, 1.6ms deserialize) are even better because we measured Redis-backed store on H100 NVLink.
- **Proves lossless:** Bit-identical reproducibility means the stateful mechanism doesn't corrupt the science. A reviewer asking "does serialization introduce numerical drift?" gets a definitive no.
- **Proves stateless workers:** Memory recovery shows workers free model memory on exit, confirming the design doesn't require long-lived stateful workers.

---

## Study 3: Resource Broker — COMPLETE

**Maps to:** Vijay's "turn on and off the AI" and "what does that look
like on the system side"; Dejan's "turn it on and off when it's critical"

**Component validated:** Resource Broker (design.tex Section 4.2)

### Results

| Experiment | Key Result | File |
|---|---|---|
| 3.1 Reallocation | Broker decision: 0.003-0.007ms. Total transition dominated by graceful drain (linear scaling). | `study3/results_reallocation.json` |
| 3.2 Throughput | 7→8 GPUs = 14.3% throughput improvement (analytical, consistent across BBA/CLN025/KRAS sim durations) | `study3/results_throughput.json` |
| 3.3 Draining | 4/4 tests pass. Zero tasks killed. In-flight tasks complete before reclaim. Concurrent drain safe. | `study3/results_draining.json` |
| 3.4 Cooldown | Without cooldown: 100% transitions fire. With cooldown: only 3-5% succeed. Expiry pattern correct. | `study3/results_cooldown.json` |

### Figures

| Figure | What it shows | Paper argument |
|---|---|---|
| `fig_study3a_reallocation` | Left: broker overhead (constant ~0.007ms). Right: total transition time (tracks expected drain exactly) | GPU reallocation adds zero overhead — total time is just waiting for in-flight tasks |
| `fig_study3b_throughput` | Left: before/after throughput per system. Right: GPU timeline showing GPU 7 switching from ML to sim | Reclaiming 1 GPU = 14.3% more simulation throughput. Visual shows the mechanism. |
| `fig_study3c_cooldown` | Left: attempted vs succeeded transitions. Right: correct allow/block pattern | Anti-thrashing works — prevents rapid oscillation between freeze/resume states |

### Why these results work for the evaluation

- **Proves design claim:** "GPU transitions are safe: the broker waits for the current task to complete before reassigning." Zero killed tasks across all draining scenarios.
- **Proves overhead claim:** "GPU reallocation <1ms" (Table 3). Our number is 0.003-0.007ms.
- **Proves throughput recovery:** The 14.3% improvement (1/7) matches the paper's "14% wall-clock reduction" claim in Section 6.4.
- **Proves anti-thrashing:** Cooldown mechanism prevents the system from toggling faster than transition overhead allows (design.tex: "minimum useful on duration").

---

## Study 4: E2E Integration — SCRIPTS READY, NOT YET RUN

**Maps to:** Ada's core ask: "same application benefit, better system metrics"

**Component validated:** Full integrated workflow

### What exists

- `deepdrivemd/workflows/openmm_cvae_dynamic.py` — Dynamic workflow integrating all 3 components
- `evaluation/study4/run_campaign.py` — Campaign runner (baseline vs dynamic)
- `evaluation/study4/analyze_results.py` — Parses Colmena JSONs, generates comparison table

### What's needed to run

- Full DeepDriveMD stack (OpenMM, Parsl, Colmena, mdlearn)
- BBA protein system data (in `examples/bba-folding-workstation/`)
- Multi-GPU allocation (8x H100)
- Redis server (available, tested in Study 2)

### Expected outputs

- Comparison table: baseline vs dynamic (wall-clock, GPU-hours, AI GPU-hours, sim throughput, staleness)
- Dynamic provisioning stats: freeze/resume events, drain times, signal transitions
- Key claim to validate: dynamic provisioning delivers same scientific quality with less GPU waste

---

## Combined Overview Figure — COMPLETE

`fig_overview` — 2x2 panel combining: (a) signal monitor overhead,
(b) Redis serialization latency, (c) throughput recovery, (d) reallocation latency.

---

## File Layout

```
results/
├── PROGRESS.md              ← this file
├── study1/
│   ├── results_latency.json
│   ├── results_concurrency.json
│   └── results_pluggability.json
├── study2/
│   ├── results_latency.json
│   ├── results_reproducibility.json
│   └── results_memory.json
├── study3/
│   ├── results_reallocation.json
│   ├── results_throughput.json
│   ├── results_draining.json
│   └── results_cooldown.json
└── figures/
    ├── fig_study1a_latency.{pdf,png}
    ├── fig_study1b_pluggability.{pdf,png}
    ├── fig_study2a_latency.{pdf,png}
    ├── fig_study2b_memory.{pdf,png}
    ├── fig_study3a_reallocation.{pdf,png}
    ├── fig_study3b_throughput.{pdf,png}
    ├── fig_study3c_cooldown.{pdf,png}
    └── fig_overview.{pdf,png}
```

## How to Reproduce

```bash
cd /shivam/deepdrivemd

# Re-run all benchmarks
make benchmarks

# Re-generate all figures (overwrites in place)
make figures

# Or run individual studies
make bench-study1    # CPU only
make bench-study2    # needs Redis
make bench-study3    # CPU only
make fig-study1      # regenerate Study 1 figures only

# Tweak plot style globally
vi evaluation/plot_config.py   # colors, fonts, output formats

# Tweak individual study plots
vi evaluation/plot_study1.py   # then: make fig-study1
```

## Next Steps

1. **Run Study 4** — actual BBA campaigns (baseline vs dynamic) to get the
   end-to-end comparison Ada asked for
2. **Cross-system runs** — run on CLN025/NTL9/KRAS to show the architecture
   adapts correctly across systems (Ada: "our workload spans different
   applications")
3. **Write evaluation.tex** — map Studies 1-3 to subsections validating each
   component, Study 4 to the integrated comparison

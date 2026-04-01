# autoresearch: Dynamic Provisioning Signal Optimization

This is an experiment to have the LLM autonomously iterate on the signal
monitor and composite policy to find the best configuration for dynamic
GPU provisioning across multiple molecular systems.

## Problem Statement

The paper proposes a **pluggable architecture** that lets application
developers decide when to use the AI component and when to turn it off
and reclaim those resources for simulation. The architecture has four parts:

1. **Signal Monitor** — collects telemetry, applies a pluggable policy
2. **Policy** — developer-supplied logic that decides ACTIVE/DORMANT/RESUME
3. **Resource Broker** — handles GPU freeze/reclaim/resume safely
4. **Stateful Service** — checkpoints model state for fast resume

The key claim is: a developer can plug in a policy appropriate for their
system, and the architecture handles everything else (transitions, GPU
management, checkpointing). Different systems need different policies.

**Current issues to resolve:**
- KRAS (fast convergence) hasn't been tested — this is where freeze SHOULD fire
- CLN025 underperforms baseline (inference bottleneck on single ML GPU)
- GPU 7 sits idle after freeze instead of running simulations
- The full lifecycle (ACTIVE → DORMANT → GPU reclaimed → more sims) is untested

**The goal:** demonstrate that the architecture works end-to-end across
4 protein systems, that each system can use a policy appropriate for its
convergence behavior, and that the dynamic system beats or matches
baseline on scientific quality while improving system efficiency.

**What "better than baseline" means:**
- **Scientific quality**: equal or better RMSD / near-native % (the science
  must not get worse — this is the hard constraint)
- **System efficiency**: fewer wasted training GPU-hours, more simulations
  completed in the same wall-clock time, or both

A successful result is: same science + less training waste. Or: better
science (because reclaimed GPU runs more sims). Either wins.

**Document what signals were useful.** For each system, record which signals
the policy used and which ones drove the correct decision. This is a key
finding for the paper — showing that different systems need different signals
validates the pluggable architecture design.

### Why KRAS matters most

KRAS G12D converges fast. The paper shows freeze@5 gives 89.9% near-native
vs 69.0% for always-retrain — a 30% improvement from stopping training
early. KRAS is the only system where the **full dynamic provisioning
lifecycle** should fire: ACTIVE → loss plateaus → DORMANT → GPU reclaimed
→ more simulations. If the composite policy can detect KRAS convergence
and freeze correctly, that validates the entire architecture.

BBA/NTL9/CLN025 are systems where training should continue — the correct
behavior is staying ACTIVE, which our policy already does.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr1`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `study4-e2e-integration`.
3. **Read the in-scope files** for full context:
   - `results/PROGRESS.md` — current state, known issues, results so far
   - `deepdrivemd/signal_monitor/policy.py` — the policies you modify
   - `deepdrivemd/workflows/openmm_cvae_dynamic.py` — the dynamic workflow
   - `deepdrivemd/apps/cvae_train/app.py` — training app (final_loss, contact map loader)
   - `deepdrivemd/apps/cvae_inference/app.py` — inference app (outlier selection)
   - `evaluation/study4/configs/` — YAML configs for each system
4. **Verify infrastructure**: Redis running (`redis-cli ping`), GPUs available
   (`nvidia-smi`), no stale processes.
5. **Run baselines once** for each system (save run dirs). Baselines only need
   rerunning if baseline code or configs change.
6. **Initialize results.tsv**: Create `results.tsv` with header row.
7. **Confirm and go**.

## Experimentation

Each experiment runs the **dynamic workflow only** against a saved baseline.
Baselines are run once and reused — only rerun if baseline code changes.

**What you CAN modify:**
- `deepdrivemd/signal_monitor/policy.py` — policy logic, thresholds, new signals
- `deepdrivemd/workflows/openmm_cvae_dynamic.py` — signal collection, telemetry
  vector construction, GPU routing after freeze, when/where telemetry is submitted
- `evaluation/study4/configs/*_dynamic.yaml` — policy parameters, window sizes

**What you CANNOT modify:**
- `deepdrivemd/workflows/openmm_cvae.py` — the baseline workflow (control)
- `deepdrivemd/apps/openmm_simulation/` — simulation app
- `deepdrivemd/apps/cvae_inference/` — inference app (read-only, but you can
  extract more signals from its output)
- `deepdrivemd/api.py` — base workflow class

**Do NOT revert `deepdrivemd/apps/cvae_train/app.py`** — it contains the
sparse COO contact map normalization fix that all systems depend on, plus
the `final_loss` field needed for signal telemetry.

**The goal is simple: beat baseline on RMSD and near-native % for all 4 systems.**

Key metrics (lower RMSD is better, higher near-native % is better):
- `mean_rmsd`: average RMSD across all simulation frames
- `nn5`: percentage of frames with RMSD < 5 Å
- `last10_rmsd`: mean RMSD of last 10 simulations (convergence quality)
- `train_count`: number of successful training cycles
- `inference_count`: number of inference cycles (more = better steering)

## System-specific notes

| System | Residues | Solvent | Sim time | Sims/train | Timeout | Key behavior |
|--------|----------|---------|----------|------------|---------|-------------|
| BBA | 28 | implicit | ~10s/sim | 6 | 15 min | Needs continued training |
| CLN025 | 93 atoms | implicit | ~10s/sim | 6 | 15 min | Needs frequent inference |
| NTL9 | 39 | implicit | ~10s/sim | 6 | 15 min | Needs continued training |
| KRAS | 167 | explicit | ~40s/sim | 16 | 45 min | Converges fast, freeze should help |

KRAS needs longer timeouts because explicit solvent is slower and it needs
16 sims before training triggers. Use `timeout 2700` (45 min) for KRAS runs.

## Running an experiment

```bash
# Kill any stale processes
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3

# Pick a system: bba, cln025, ntl9, or kras
SYSTEM=bba
TIMEOUT=900  # 15 min for BBA/CLN025/NTL9, use 2700 for KRAS

# Start Redis (only needed for dynamic)
redis-server --daemonize yes --port 6379 --save "" 2>/dev/null

# Run dynamic
cd /shivam/deepdrivemd
PYTHONPATH=/shivam/deepdrivemd timeout $TIMEOUT python3 \
  deepdrivemd/workflows/openmm_cvae_dynamic.py \
  -c evaluation/study4/configs/${SYSTEM}_dynamic.yaml \
  > /tmp/${SYSTEM}_dynamic.log 2>&1

# Clean up after
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3
```

### Running a baseline (only needed once per system)

```bash
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3

SYSTEM=bba
TIMEOUT=900  # 15 min for BBA/CLN025/NTL9, use 2700 for KRAS

cd /shivam/deepdrivemd
PYTHONPATH=/shivam/deepdrivemd timeout $TIMEOUT python3 \
  deepdrivemd/workflows/openmm_cvae.py \
  -c evaluation/study4/configs/${SYSTEM}_baseline.yaml \
  > /tmp/${SYSTEM}_baseline.log 2>&1

# Save the run dir path
echo "$(ls -td runs/experiment-* | head -1)" > results/study4/${SYSTEM}_baseline_run_dir.txt
```

## Saved baseline results

These are the reference baselines. Only rerun if baseline code changes.

| System | Run dir | Sims | Trains | Infer | Mean RMSD | NN<5Å |
|--------|---------|------|--------|-------|-----------|-------|
| BBA | `runs/experiment-310326-015111` | 104 | 14 | 53 | 5.985 | 31.7% |
| CLN025 | `runs/experiment-310326-220720` | 112 | 12 | 47 | 5.679 | 20.9% |
| NTL9 | `runs/experiment-310326-205255` | 98 | 14 | 63 | 6.815 | 18.9% |
| KRAS | *not yet run (needs 45 min)* | — | — | — | — | — |

## Extracting results

```bash
# Set paths explicitly (don't rely on ls ordering)
BASELINE=$(cat results/study4/${SYSTEM}_baseline_run_dir.txt)
DYNAMIC=$(ls -td runs/experiment-* | head -1)

# Compare
python3 -c "
import numpy as np, glob, json

def stats(d):
    files = sorted(glob.glob(f'{d}/simulation/*/rmsd.npy'))
    if not files: return {}
    all_rmsd = np.concatenate([np.load(f) for f in files])
    means = [np.mean(np.load(f)) for f in files]
    trains = 0
    try:
        with open(f'{d}/result/train.json') as f:
            trains = sum(1 for l in f if json.loads(l).get('success'))
    except: pass
    infers = 0
    try:
        with open(f'{d}/result/inference.json') as f:
            infers = sum(1 for _ in f)
    except: pass
    return dict(sims=len(files), trains=trains, infers=infers,
                mean_rmsd=np.mean(all_rmsd), nn5=np.mean(all_rmsd<5)*100,
                last10=np.mean(means[-10:]))

b = stats('$BASELINE')
d = stats('$DYNAMIC')
for k in ['sims','trains','infers','mean_rmsd','nn5','last10']:
    bv, dv = b.get(k,0), d.get(k,0)
    if isinstance(bv, float):
        print(f'{k:>12}: base={bv:.3f}  dyn={dv:.3f}  delta={dv-bv:+.3f}')
    else:
        print(f'{k:>12}: base={bv}  dyn={dv}  delta={dv-bv:+d}')
"
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	system	base_rmsd	dyn_rmsd	base_nn5	dyn_nn5	base_trains	dyn_trains	base_infers	dyn_infers	status	description
```

Status: `improved` (dynamic beats baseline on science AND efficiency),
`neutral` (similar), `regressed` (baseline better), `crash` (dynamic failed).

In the description, always note **which signals drove the decision** and
whether a freeze event occurred. This builds the evidence for the paper
about which signals matter for which systems.

Example:
```
a1b2c3d	bba	5.985	5.606	31.7	38.3	14	15	53	65	improved	composite(loss+rmsd+stab); no freeze; more inferences helped
b2c3d4e	cln025	5.679	6.063	20.9	10.6	12	12	47	34	regressed	composite; no freeze; inference bottleneck on single ML GPU
c3d4e5f	kras	8.200	7.100	15.0	22.0	12	5	30	45	improved	composite; FREEZE@5 triggered by loss plateau; GPU7 reclaimed; +15 sims
```

## The experiment loop

LOOP FOREVER:

1. **Read the state**: Check `results.tsv`, `git log`, current policy code,
   and the available signals catalog above.
2. **Form a hypothesis**: What change to the signals, policy logic, or
   workflow could improve results? Look at which signals are available
   but unused, where the current policy makes wrong decisions, and what
   the data from previous runs tells you.
3. **Implement the change**: Edit policy.py and/or openmm_cvae_dynamic.py
   and/or the dynamic YAML configs.
4. **git commit** with a descriptive message.
5. **Run the experiment** on one system first (BBA is fastest for quick
   iteration). Extract and compare against saved baseline.
6. **If promising, test on ALL four systems** (BBA, CLN025, NTL9, KRAS).
   A change is only "kept" if it beats or matches baseline on every system.
   Log each system's results to `results.tsv`.
7. **Keep or revert**:
   - If dynamic beats or matches baseline on ALL systems → keep the commit
   - If it regresses on ANY system → `git reset --hard HEAD~1`
   - Exception: if it dramatically improves some systems and only slightly
     regresses on one, use judgment — but document the tradeoff
8. **Repeat** with the next hypothesis.

## Available signals

The system produces signals from three layers. A key part of this research
is figuring out **which signals matter for which systems** and whether
different systems need different signal combinations.

### ML Model signals (from `cvae_train/app.py`)
- **Training loss** (`trainer.loss_curve_["train_loss"]`) — currently used (normalized)
- **Validation loss** (`trainer.loss_curve_["valid_loss"]`) — NOT used, could detect overfitting
- **Reconstruction loss** (`trainer.loss_curve_["train_recon_loss"]`) — NOT used
- **KL divergence** (`trainer.loss_curve_["train_kld_loss"]`) — NOT used, measures latent space quality
- **Loss curve slope** — NOT computed, could detect plateau more robustly than half-window comparison

### Application/science signals (from simulation + inference)
- **Simulation RMSD** (`rmsd.npy` per sim) — currently used (rolling mean of last 10)
- **RMSD variance** — NOT used, high variance = still exploring, low = converged
- **RMSD trend slope** — NOT used, linear regression over recent sims
- **Near-native fraction** — NOT used at runtime, could track % of frames < threshold
- **LOF outlier scores** (`clf.negative_outlier_factor_`) — NOT used, available in inference app
- **Latent embedding spread** (`embeddings.npy`) — NOT used, variance of CVAE embeddings
  indicates how well the model distinguishes conformations

### System/infrastructure signals
- **Inference stability** (restart point overlap) — currently used
- **Simulation throughput** (sims completed per minute) — NOT tracked
- **Training time per cycle** — NOT tracked, increasing time = growing dataset overhead
- **GPU utilization** — NOT tracked, could detect idle GPUs
- **Queue depth** (pending tasks per executor) — available from Parsl but NOT tracked

Part of the experiment loop is discovering which signals matter for which
systems — try different signal combinations and see which ones produce
correct freeze/continue decisions.

## Ideas to try (ordered by expected impact)

1. **Run KRAS and validate freeze triggers** — the composite policy should
   detect KRAS convergence and fire DORMANT. If it doesn't, tune thresholds.
   This is the single most important experiment.

2. **Inference-only mode after freeze** — after freeze, keep running inference
   with the frozen model (no training). This gives CLN025 the steering it
   needs without the training overhead. Currently inference stops when
   training stops.

3. **Submit telemetry after inference too** — currently only after training.
   This would let the signal monitor react faster to quality changes.

4. **Prioritize inference over training on ML GPU** — inference is fast (~5s)
   and high-value for steering. Training takes ~60s. When both are queued,
   inference should go first.

5. **GPU 7 sim routing after freeze** — verify that after freeze, GPU 7
   actually runs simulations and throughput increases. The round-robin
   routing in `simulate()` is implemented but untested with real freeze.

6. **Add simulation throughput signal** — if sims/minute drops, the ML is
   hogging GPU time. Freeze to restore throughput.

7. **Adaptive freeze threshold** — instead of fixed `loss_plateau_threshold`,
   compute it relative to the loss variance in the window.

8. **Multi-cycle RMSD trend** — instead of comparing halves of the RMSD
   window, use a linear regression slope. More robust to noise.

## Critical constraints

- **Baselines are run once and saved.** Only rerun if baseline code changes.
- **Kill ALL stale processes** between runs — interchange and worker zombies
  cause port conflicts and GPU memory leaks. Always run:
  `pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3`
- **Timeouts**: 15 min for BBA/CLN025/NTL9, 45 min for KRAS.
- **Redis only for dynamic** — never start Redis for baseline runs.
- **Commit before running** — so you can revert cleanly if it fails.
- **Check training success** — always verify trains are OK, not FAIL.
  The contact map loader fix is critical; don't revert `cvae_train/app.py`.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you
should continue. The human might be asleep or away. Continue running
experiments indefinitely until manually stopped. If you run out of ideas,
re-read PROGRESS.md and the policy code, try combinations of previous
changes, or try more radical approaches (different policy entirely,
different signal sources, different freeze/resume logic).

Each dynamic-only experiment takes ~20 minutes (15 min run + 5 min overhead)
for BBA/CLN025/NTL9, or ~50 minutes for KRAS. You can run ~3 per hour on
fast systems, ~1 per hour on KRAS. Make each one count.

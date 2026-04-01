# autoresearch: Dynamic Provisioning Signal Optimization

This is an experiment to have the LLM autonomously iterate on the signal
monitor and composite policy to find the best configuration for dynamic
GPU provisioning across multiple molecular systems.

## Problem Statement

The dynamic provisioning system needs to decide when to freeze AI training
and reclaim GPUs for simulation. The current CompositePolicy uses three
signals but has known issues:
- CLN025 underperforms baseline (inference bottleneck)
- No system triggers an actual freeze (loss threshold never met for BBA/NTL9)
- GPU 7 sits idle after freeze instead of running simulations

The goal: find a signal configuration + policy logic that makes the dynamic
system **beat or match baseline on ALL three protein systems** (BBA, CLN025,
NTL9) on both scientific quality (RMSD, near-native %) and system efficiency
(GPU utilization, training waste reduction).

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr1`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `study4-e2e-integration`.
3. **Read the in-scope files** for full context:
   - `results/PROGRESS.md` — current state, known issues, results so far
   - `deepdrivemd/signal_monitor/policy.py` — the policies you modify
   - `deepdrivemd/workflows/openmm_cvae_dynamic.py` — the dynamic workflow (signal collection + dispatch)
   - `deepdrivemd/apps/cvae_train/app.py` — training app (where final_loss comes from)
   - `deepdrivemd/apps/cvae_inference/app.py` — inference app (outlier selection)
   - `evaluation/study4/configs/` — YAML configs for each system
4. **Verify infrastructure**: Redis running (`redis-cli ping`), GPUs available
   (`nvidia-smi`), no stale processes.
5. **Initialize results.tsv**: Create `results.tsv` with header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs a **paired comparison**: baseline vs dynamic for one
protein system, each with a 15-minute wall-clock timeout.

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

**The goal is simple: beat baseline on RMSD and near-native % for all 3 systems.**

Key metrics (lower RMSD is better, higher near-native % is better):
- `mean_rmsd`: average RMSD across all simulation frames
- `nn5`: percentage of frames with RMSD < 5 Å
- `last10_rmsd`: mean RMSD of last 10 simulations (convergence quality)
- `train_count`: number of successful training cycles
- `inference_count`: number of inference cycles (more = better steering)

## Running an experiment

```bash
# Kill any stale processes
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3

# Pick a system: bba, cln025, or ntl9
SYSTEM=bba

# Run baseline (15 min)
cd /shivam/deepdrivemd
PYTHONPATH=/shivam/deepdrivemd timeout 900 python3 \
  deepdrivemd/workflows/openmm_cvae.py \
  -c evaluation/study4/configs/${SYSTEM}_baseline.yaml \
  > /tmp/${SYSTEM}_baseline.log 2>&1

# Clean up between runs
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3

# Run dynamic (15 min, needs Redis)
redis-server --daemonize yes --port 6379 --save "" 2>/dev/null
PYTHONPATH=/shivam/deepdrivemd timeout 900 python3 \
  deepdrivemd/workflows/openmm_cvae_dynamic.py \
  -c evaluation/study4/configs/${SYSTEM}_dynamic.yaml \
  > /tmp/${SYSTEM}_dynamic.log 2>&1
```

## Extracting results

```bash
# Find the latest two run directories
BASELINE=$(ls -td /shivam/deepdrivemd/runs/experiment-* | head -2 | tail -1)
DYNAMIC=$(ls -td /shivam/deepdrivemd/runs/experiment-* | head -1)

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

Status: `improved` (dynamic beats baseline), `neutral` (similar), `regressed`
(baseline better), `crash` (dynamic failed).

Example:
```
a1b2c3d	bba	5.985	5.606	31.7	38.3	14	15	53	65	improved	composite policy with 3 signals
b2c3d4e	cln025	5.679	6.063	20.9	10.6	12	12	47	34	regressed	inference bottleneck on ML GPU
```

## The experiment loop

LOOP FOREVER:

1. **Read the state**: Check `results.tsv`, `git log`, current policy code.
2. **Form a hypothesis**: Based on known issues (see PROGRESS.md) or previous
   results. Examples:
   - "CLN025 needs more inference throughput → submit telemetry after inference too"
   - "Signal should include simulation throughput rate"
   - "Freeze threshold is too conservative → lower loss_plateau_threshold"
   - "Submit telemetry from handle_inference_output, not just handle_train_output"
   - "Add a 4th signal: simulation RMSD improvement rate"
   - "Try inference-only mode after freeze instead of stopping inference"
3. **Implement the change**: Edit policy.py and/or openmm_cvae_dynamic.py.
4. **git commit** with a descriptive message.
5. **Run the experiment** on the system most likely to show the effect.
   - CLN025 for inference-related changes
   - BBA for signal/freeze-related changes
   - NTL9 for general validation
6. **Extract and compare** results.
7. **Log to results.tsv**.
8. **Keep or revert**:
   - If dynamic beats baseline → keep the commit
   - If worse or neutral with added complexity → `git reset --hard HEAD~1`
9. **Repeat** with the next hypothesis.

## Ideas to try (ordered by expected impact)

1. **Submit telemetry after inference too** — currently only after training.
   This would let the signal monitor react faster to quality changes.

2. **Prioritize inference over training on ML GPU** — inference is fast (~5s)
   and high-value for steering. Training takes ~60s. When both are queued,
   inference should go first.

3. **Add simulation throughput signal** — if sims/minute drops, the ML is
   hogging GPU time. Freeze to restore throughput.

4. **Inference-only mode** — after freeze, keep running inference with the
   frozen model (no training). This gives CLN025 the steering it needs
   without the training overhead.

5. **Adaptive freeze threshold** — instead of fixed `loss_plateau_threshold`,
   compute it relative to the loss variance in the window.

6. **GPU 7 sim routing fix** — verify that after freeze, GPU 7 actually
   runs simulations and the throughput increases.

7. **Multi-cycle RMSD trend** — instead of comparing halves of the RMSD
   window, use a linear regression slope. More robust to noise.

## Critical constraints

- **Always run baseline first** for a fair comparison (same machine state).
- **Kill ALL stale processes** between runs — interchange and worker zombies
  will cause port conflicts and GPU memory leaks.
- **15-minute timeout** per run. If a run exceeds this, kill and treat as crash.
- **Redis only for dynamic** — never start Redis for baseline runs.
- **Commit before running** — so you can revert cleanly if it fails.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you
should continue. The human might be asleep or away. Continue running
experiments indefinitely until manually stopped. If you run out of ideas,
re-read PROGRESS.md and the policy code, try combinations of previous
changes, or try more radical approaches (different policy entirely,
different signal sources, different freeze/resume logic).

Each paired experiment takes ~35 minutes (15 min baseline + 15 min dynamic +
5 min overhead). You can run ~2 per hour, ~16 overnight. Make each one count.

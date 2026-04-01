# autoresearch: Dynamic Provisioning Signal Optimization

This is an experiment to have the LLM autonomously iterate on the system Kirin
which is a system purpose built to achieve better system performance for tightly
coupled HPC AI workflows by using dynamic GPU provisioning for AI to better steer
simulations.

## Problem Statement

The paper proposes a full system architecture with a **pluggable policy** that 
lets application developers decide when to use the AI component and when to turn 
it off and reclaim those resources for simulation. The architecture has four parts:

1. **Signal Monitor** — collects telemetry, applies a pluggable policy
2. **Policy** — developer-supplied logic that decides ACTIVE/DORMANT/RESUME
3. **Resource Broker** — handles GPU freeze/reclaim/resume safely
4. **Stateful Service** — checkpoints model state for fast resume

One claim is: a developer can plug in a policy appropriate for their
system, and the architecture handles everything else (transitions, GPU
management, checkpointing).

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
6. **Initialize metric tracking**: Set up three levels of metric logging that
   persist across the entire run. These give fine-grained visibility into what
   happened, enable rapid iteration, and surface observations for the paper.

   **a) Per-run time-series log** (`results/study4/<system>_<tag>_timeseries.tsv`):
   Append a row every time a significant event fires (sim complete, train
   complete, inference complete, freeze/resume, etc.). Columns:

   | Column | Source | Why |
   |--------|--------|-----|
   | `wall_clock_s` | `time.time() - t0` | Anchors everything to elapsed time |
   | `event` | workflow handler | What triggered this row (sim, train, infer, freeze, resume) |
   | `gpu_util_0..7` | `nvidia-smi` or `pynvml` | Per-GPU utilization snapshot — detects idle GPUs, proves reclaim works |
   | `gpu_mem_0..7` | `nvidia-smi` or `pynvml` | Per-GPU memory — catches OOM buildup, confirms GPU release after freeze |
   | `cpu_util` | `psutil.cpu_percent()` | Detects CPU bottlenecks in data loading |
   | `sim_throughput` | sims_completed / wall_clock_s | Running rate — drop means ML is starving sims |
   | `queue_depth_sim` | Parsl executor | Pending sim tasks — high = backpressure |
   | `queue_depth_train` | Parsl executor | Pending train tasks |
   | `queue_depth_infer` | Parsl executor | Pending inference tasks |
   | `staleness_pct` | time_since_last_model_update / wall_clock_s | Fraction of time the model is stale — key paper metric |
   | `train_loss` | `final_loss` from train app | Most recent training loss |
   | `valid_loss` | `trainer.loss_curve_["valid_loss"][-1]` | Overfitting detection |
   | `recon_loss` | `trainer.loss_curve_["train_recon_loss"][-1]` | Reconstruction quality |
   | `kld_loss` | `trainer.loss_curve_["train_kld_loss"][-1]` | Latent space quality |
   | `loss_slope` | linear regression over loss window | Plateau detection signal |
   | `train_time_s` | timer around train cycle | Growing = dataset overhead problem |
   | `infer_time_s` | timer around inference cycle | Inference latency |
   | `sim_rmsd` | `rmsd.npy` from latest sim | Per-sim RMSD |
   | `rolling_rmsd_mean` | mean of last 10 sims | Smoothed science quality |
   | `rolling_rmsd_std` | std of last 10 sims | High = still exploring, low = converged |
   | `rmsd_slope` | linear regression over recent sims | Trend direction |
   | `nn5_rolling` | % of last N frames with RMSD < 5Å | Running near-native fraction |
   | `outlier_score_mean` | `clf.negative_outlier_factor_` mean | Inference quality signal |
   | `embedding_spread` | variance of CVAE latent embeddings | How well model distinguishes conformations |
   | `data_novelty_pct` | fraction of new frames in unexplored regions | Low = redundant training data |
   | `policy_state` | ACTIVE / DORMANT / RESUME | Current policy decision |
   | `policy_signals` | JSON blob of signal values fed to policy | Exactly what the policy saw when it decided |

   **b) Experiment summary** (`results.tsv`): One row per experiment run,
   with the full header described in the Logging Results section below.
   This is the cross-experiment comparison table.

   **c) Signal audit log** (`results/study4/<system>_<tag>_signals.jsonl`):
   Every time the policy is evaluated, append a JSON line with all input
   signals, the policy decision, and the reason string. This is the primary
   data source for the paper's analysis of which signals drive correct
   decisions for which systems.

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

These are the primary success criteria. The full set of tracked system, ML,
and application metrics is defined in the time-series schema (Setup step 6a)
and the `results.tsv` header (Logging Results section).

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

### Pre-run validation

Before every run, verify the environment is clean. Previous runs leave behind
processes, GPU memory allocations, Redis state, and run directories that **will
silently corrupt results** if not cleaned up.

```bash
# 1. Kill ALL stale processes — zombies cause port conflicts and GPU leaks
pkill -9 -f "interchange\|process_worker" 2>/dev/null; sleep 3

# 2. Verify no leftover processes
ps aux | grep -E "interchange|process_worker|openmm_cvae" | grep -v grep
# ^ Must return NOTHING. If it does, kill those PIDs manually.

# 3. Check GPU memory is fully released
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
# ^ memory.used should be near 0 on all GPUs. If a GPU still has significant
#   memory allocated, a zombie process is holding it. Find and kill it:
#   fuser -v /dev/nvidia* 2>/dev/null

# 4. Flush Redis — stale telemetry from a previous run will pollute the
#    signal monitor's window and cause incorrect policy decisions
redis-cli FLUSHALL 2>/dev/null

# 5. Verify the run directory won't collide with a previous run
ls -td runs/experiment-* | head -3
# ^ Note the latest run dir. After your run completes, the NEW run dir
#   must be different (newer timestamp). If it's the same, you're reading
#   stale results.

# 6. Confirm the correct code is checked out
git status
git log --oneline -1
# ^ Must match the commit you intend to test. Uncommitted changes or
#   wrong branch = untraceable results.
```

### Run the experiment

```bash
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

### Post-run sanity checks

After every run, verify the results are valid before logging them. A run that
"completes" can still produce garbage if the environment was dirty or if
something silently failed.

```bash
# 1. Confirm a NEW run directory was created
DYNAMIC=$(ls -td runs/experiment-* | head -1)
echo "Run dir: $DYNAMIC"
# ^ Must be a new directory with a timestamp AFTER you started the run.

# 2. Check that simulations actually ran and produced data
SIM_COUNT=$(ls -d $DYNAMIC/simulation/*/ 2>/dev/null | wc -l)
echo "Simulations: $SIM_COUNT"
# ^ Must be > 0. If 0, the workflow crashed before any sims completed.

# 3. Verify RMSD files exist and are non-empty
RMSD_COUNT=$(find $DYNAMIC/simulation -name "rmsd.npy" | wc -l)
echo "RMSD files: $RMSD_COUNT"
# ^ Should equal SIM_COUNT. Missing rmsd.npy = simulation didn't finish.

# 4. Check training succeeded (not FAIL)
if [ -f "$DYNAMIC/result/train.json" ]; then
    TRAIN_OK=$(grep -c '"success": true' $DYNAMIC/result/train.json 2>/dev/null || echo 0)
    TRAIN_FAIL=$(grep -c '"success": false' $DYNAMIC/result/train.json 2>/dev/null || echo 0)
    echo "Training: $TRAIN_OK OK, $TRAIN_FAIL FAIL"
    # ^ FAIL > 0 means training crashed — check the log for errors.
    #   Common cause: reverted cvae_train/app.py contact map fix.
else
    echo "WARNING: No train.json — training never completed"
fi

# 5. Check inference ran
if [ -f "$DYNAMIC/result/inference.json" ]; then
    INFER_COUNT=$(wc -l < $DYNAMIC/result/inference.json)
    echo "Inference cycles: $INFER_COUNT"
else
    echo "WARNING: No inference.json — inference never ran"
fi

# 6. Sanity-check RMSD values — catch corrupted/stale data
python3 -c "
import numpy as np, glob
files = sorted(glob.glob('$DYNAMIC/simulation/*/rmsd.npy'))
if not files:
    print('ERROR: No RMSD files found')
else:
    all_rmsd = np.concatenate([np.load(f) for f in files])
    print(f'RMSD range: [{all_rmsd.min():.3f}, {all_rmsd.max():.3f}]')
    print(f'RMSD mean: {np.mean(all_rmsd):.3f}')
    if all_rmsd.max() > 50:
        print('WARNING: Extremely high RMSD — possible corrupt simulation')
    if all_rmsd.min() < 0:
        print('ERROR: Negative RMSD — data is corrupted')
    if np.isnan(all_rmsd).any():
        print('ERROR: NaN in RMSD — data is corrupted')
"

# 7. Check the log for errors/warnings
echo "=== Last 20 lines of log ==="
tail -20 /tmp/${SYSTEM}_dynamic.log
# ^ Look for: tracebacks, OOM errors, GPU errors, "FAIL", "Error"
grep -ci "error\|traceback\|exception\|oom\|killed" /tmp/${SYSTEM}_dynamic.log
# ^ Should be 0 or near-0. Investigate any hits.

# 8. Verify metric log files were written (if instrumented)
if [ -f "results/study4/${SYSTEM}_*_timeseries.tsv" ]; then
    TS_ROWS=$(wc -l < results/study4/${SYSTEM}_*_timeseries.tsv)
    echo "Time-series rows: $TS_ROWS"
fi
if [ -f "results/study4/${SYSTEM}_*_signals.jsonl" ]; then
    SIG_ROWS=$(wc -l < results/study4/${SYSTEM}_*_signals.jsonl)
    echo "Signal audit entries: $SIG_ROWS"
fi
```

**If any check fails, do NOT log the results.** Diagnose the issue first.
Common failure modes:
- **0 simulations**: workflow crashed at startup — check the full log
- **Training FAIL**: contact map loader issue — verify `cvae_train/app.py`
  has the sparse COO fix
- **Stale run dir**: you're reading a previous run's data — check timestamps
- **NaN/corrupt RMSD**: GPU memory was dirty from a previous run — flush
  and rerun
- **Extremely high RMSD**: wrong system config or simulation diverged

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

### Quick comparison (science metrics)

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

This gives the quick science comparison. For the full `results.tsv` summary row
(system metrics, freeze details, slopes, etc.), derive those columns from the
per-run time-series log — see "Deriving summary from time-series" in the
Logging Results section below.

## Logging results

### Per-run time-series (`results/study4/<system>_<tag>_timeseries.tsv`)

Append a row on every significant event (sim complete, train complete, inference
complete, freeze, resume). This is the fine-grained record of everything that
happened during the run. See the column table in Setup step 6a for the full
schema. The time-series lets you:

- Reconstruct GPU utilization over the entire run to prove reclaim works
- Pinpoint exactly when loss plateau / RMSD convergence / freeze occurred
- Correlate system backpressure (queue depth) with ML quality (loss, RMSD)
- Identify whether throughput improved after freeze (sims/min before vs after)
- Spot anomalies: OOM buildup, training time growth, inference latency spikes

### Signal audit log (`results/study4/<system>_<tag>_signals.jsonl`)

One JSON line per policy evaluation. Each line contains:
```json
{"wall_clock_s": 142.3, "cycle": 5, "signals": {"train_loss": 0.032, "loss_slope": -0.001, "rolling_rmsd": 5.2, "rmsd_slope": -0.04, "nn5_rolling": 35.1, "staleness_pct": 0.18, "sim_throughput": 3.2, ...}, "decision": "ACTIVE", "reason": "loss still decreasing; rmsd improving"}
```

This is the **primary evidence for the paper**: it shows exactly which signals
the policy used, what values they had, and what decision resulted. Across
many runs, this data reveals which signals are predictive for each system.

### Experiment summary (`results.tsv`)

One row per experiment run (tab-separated):

```
commit	system	base_rmsd	dyn_rmsd	base_nn5	dyn_nn5	base_trains	dyn_trains	base_infers	dyn_infers	base_sims	dyn_sims	freeze_fired	freeze_cycle	gpu_reclaimed	post_freeze_sims	avg_staleness_pct	avg_sim_throughput	avg_train_time_s	peak_gpu_util	final_loss_slope	final_rmsd_slope	status	signals_used	description
```

**Column definitions:**

| Column | Description |
|--------|-------------|
| `commit` | Git short SHA for reproducibility |
| `system` | bba / cln025 / ntl9 / kras |
| `base_rmsd`, `dyn_rmsd` | Mean RMSD for baseline vs dynamic |
| `base_nn5`, `dyn_nn5` | Near-native % (< 5Å) for baseline vs dynamic |
| `base_trains`, `dyn_trains` | Training cycle counts |
| `base_infers`, `dyn_infers` | Inference cycle counts |
| `base_sims`, `dyn_sims` | Total simulations completed |
| `freeze_fired` | yes/no — did the policy trigger DORMANT? |
| `freeze_cycle` | Which training cycle triggered freeze (blank if none) |
| `gpu_reclaimed` | yes/no — was GPU 7 actually reallocated to sims? |
| `post_freeze_sims` | Number of sims completed after freeze (0 if no freeze) |
| `avg_staleness_pct` | Average model staleness across the run |
| `avg_sim_throughput` | Average sims/min across the run |
| `avg_train_time_s` | Average training cycle duration |
| `peak_gpu_util` | Peak ML GPU utilization during training |
| `final_loss_slope` | Loss slope at end of run (negative = still improving) |
| `final_rmsd_slope` | RMSD slope at end of run (negative = still improving) |
| `status` | `improved` / `neutral` / `regressed` / `crash` |
| `signals_used` | Which signals the policy consumed (e.g., `loss+rmsd+stab+nn5`) |
| `description` | Free text: what changed, what happened, key observations |

Status: `improved` (dynamic beats baseline on science AND efficiency),
`neutral` (similar), `regressed` (baseline better), `crash` (dynamic failed).

In the description, always note **which signals drove the decision** and
whether a freeze event occurred. This builds the evidence for the paper
about which signals matter for which systems.

### Deriving summary from time-series

After each run, compute the summary row columns from the time-series log:
```python
import pandas as pd, json, numpy as np

ts = pd.read_csv(f'results/study4/{system}_{tag}_timeseries.tsv', sep='\t')

# System metrics
avg_staleness = ts['staleness_pct'].mean()
avg_throughput = ts[ts['event']=='sim']['sim_throughput'].mean()
avg_train_time = ts[ts['event']=='train']['train_time_s'].mean()
gpu_cols = [c for c in ts.columns if c.startswith('gpu_util')]
peak_gpu = ts[gpu_cols].max().max()

# Freeze details
freeze_rows = ts[ts['policy_state']=='DORMANT']
freeze_fired = len(freeze_rows) > 0
if freeze_fired:
    freeze_wall = freeze_rows.iloc[0]['wall_clock_s']
    # Extract cycle number from the signal audit log
    with open(f'results/study4/{system}_{tag}_signals.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            if entry['decision'] == 'DORMANT':
                freeze_cycle = entry['cycle']
                break
    gpu_reclaimed = 'yes'  # verify from GPU util drop after freeze
    post_freeze_sims = len(ts[(ts['event']=='sim') & (ts['wall_clock_s'] > freeze_wall)])
else:
    freeze_cycle = ''
    gpu_reclaimed = 'no'
    post_freeze_sims = 0

# Final slopes (from last entries in time-series)
train_rows = ts[ts['event']=='train'].tail(5)
final_loss_slope = np.polyfit(range(len(train_rows)), train_rows['train_loss'], 1)[0] if len(train_rows) >= 2 else 0.0
sim_rows = ts[ts['event']=='sim'].tail(10)
final_rmsd_slope = np.polyfit(range(len(sim_rows)), sim_rows['sim_rmsd'], 1)[0] if len(sim_rows) >= 2 else 0.0

# Signals used — read from the last policy evaluation
with open(f'results/study4/{system}_{tag}_signals.jsonl') as f:
    lines = f.readlines()
    last_eval = json.loads(lines[-1])
    signals_used = '+'.join(sorted(last_eval['signals'].keys()))
```

### Example summary rows
```
a1b2c3d	bba	5.985	5.606	31.7	38.3	14	15	53	65	104	118	no		no	0	0.22	3.1	58.2	95.3	-0.002	-0.04	improved	loss+rmsd+stab	composite(loss+rmsd+stab); no freeze; more inferences helped
b2c3d4e	cln025	5.679	6.063	20.9	10.6	12	12	47	34	112	98	no		no	0	0.31	2.8	62.1	94.1	-0.001	+0.02	regressed	loss+rmsd+stab	composite; no freeze; inference bottleneck on single ML GPU
c3d4e5f	kras	8.200	7.100	15.0	22.0	12	5	30	45	48	63	yes	5	yes	15	0.14	1.8	120.5	97.2	0.000	-0.08	improved	loss+rmsd+stab+nn5	composite; FREEZE@5 triggered by loss plateau; GPU7 reclaimed; +15 sims
```

## The experiment loop

### Phase 1: Instrument all signals

Before tuning any policy, **implement tracking for every signal listed
above that is marked "NOT tracked" or "NOT used."** The policy cannot
make good decisions with incomplete information. For each signal:

1. Add the tracking code to the appropriate handler in `openmm_cvae_dynamic.py`
   (e.g., compute staleness ratio in `handle_train_output`, compute data
   novelty in `handle_simulation_output`, etc.)
2. Log the signal value so it appears in the runtime log
3. Include it in the telemetry vector passed to the signal monitor
4. **Write to the three log files** defined in Setup step 6:
   - Append a row to the per-run time-series TSV on each event
   - Append a JSON line to the signal audit log on each policy evaluation
   - After the run completes, derive and append the summary row to `results.tsv`
5. Run a quick BBA test to verify nothing crashes and all three log files
   are populated correctly
6. Commit

Do NOT try to tune thresholds or change freeze logic during this phase.
The goal is to get all signals flowing, visible, and persisted to the log
files. Once you can see all the signals in the logs and time-series, you
have the information to design good policies.

### Phase 2: Experiment loop

Once all signals are instrumented:

LOOP FOREVER:

1. **Read the state**: Check `results.tsv` (experiment summary), the per-run
   time-series TSVs, the signal audit JSONLs, `git log`, and current policy
   code. The time-series and signal logs are the richest data sources — use
   them to understand exactly what happened and why.
2. **Form a hypothesis**: Based on what the signal data tells you about
   each system's behavior. Let the data guide you — don't guess.
3. **Implement the change**: Edit policy.py and/or openmm_cvae_dynamic.py
   and/or the dynamic YAML configs.
4. **git commit** with a descriptive message.
5. **Run the experiment** on one system first (BBA is fastest for quick
   iteration). Always run the pre-run validation, then the experiment, then
   the post-run sanity checks before extracting results. Do not skip the
   checks — stale processes and dirty GPU state from previous runs silently
   corrupt results.
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

### System signals
- **Training-to-simulation time ratio (staleness %)** — NOT tracked. Fraction of
  wall-clock time the model is stale (training in progress while sims run with
  old model). Key metric from the paper's AI Tax characterization.
- **Training cost factor (%)** — NOT tracked. Fraction of total GPU-hours consumed
  by training vs simulation. Indicates how much of the compute budget is "taxed."
- **Data movement costs (bytes)** — NOT tracked. Volume of data transferred between
  simulation, training, and inference stages. May conflict with data novelty.
- **Simulation throughput** (sims completed per minute) — NOT tracked
- **Inference stability** (restart point overlap) — currently used
- **Training time per cycle** — NOT tracked, increasing time = growing dataset overhead
- **GPU utilization** — NOT tracked, could detect idle GPUs
- **Queue depth** (pending tasks per executor) — available from Parsl but NOT tracked

### ML Model signals (from `cvae_train/app.py`)
- **Training loss** (`trainer.loss_curve_["train_loss"]`) — currently used (normalized)
- **Validation loss** (`trainer.loss_curve_["valid_loss"]`) — NOT used, could detect overfitting
- **Test loss** — NOT available (no held-out test set during online training)
- **Reconstruction loss** (`trainer.loss_curve_["train_recon_loss"]`) — NOT used
- **KL divergence** (`trainer.loss_curve_["train_kld_loss"]`) — NOT used, measures latent space quality
- **Loss curve slope** — NOT computed, could detect plateau more robustly than half-window comparison

### Application signals (from simulation + inference)
- **Simulation RMSD** (`rmsd.npy` per sim) — currently used (rolling mean of last 10)
- **Near-native fraction (%)** — NOT used at runtime, could track % of frames < threshold
  in a rolling window. Direct measure of scientific quality.
- **Data novelty fraction (%)** — NOT tracked. Fraction of new simulation frames that
  explore regions not seen before (e.g., new CVAE latent space clusters). If novelty
  is low, training on redundant data is wasteful.
- **RMSD variance** — NOT used, high variance = still exploring, low = converged
- **RMSD trend slope** — NOT used, linear regression over recent sims
- **LOF outlier scores** (`clf.negative_outlier_factor_`) — NOT used, available in inference app
- **Latent embedding spread** (`embeddings.npy`) — NOT used, variance of CVAE embeddings
  indicates how well the model distinguishes conformations

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
- **Flush Redis between runs** — stale telemetry from a previous run will
  pollute the signal monitor's window and cause the policy to make decisions
  based on a mix of old and new data. Always `redis-cli FLUSHALL` before
  starting a new dynamic run.
- **Verify GPU memory is released** — check `nvidia-smi` before each run.
  If a GPU still has memory allocated from a dead process, that memory won't
  be available and the run may silently OOM or produce corrupt output.
- **Confirm the run directory is new** — after a run, always verify the
  experiment directory timestamp is newer than when you started. Reading
  results from a stale run directory is the most common source of false
  "improved" or "regressed" conclusions.
- **Run post-run sanity checks every time** — see the checklist in "Running
  an experiment." Never log results to `results.tsv` without verifying the
  data is valid (sim count > 0, no NaN RMSD, no training FAIL, no tracebacks).
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

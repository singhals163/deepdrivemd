# Evaluation Experiment Progress

**Last updated:** 2026-04-01

## Paper Context

The paper ("The AI Tax") proposes a four-component dynamic provisioning
architecture (Section 4: Design):

1. **Signal Monitor** — lightweight, pluggable convergence detection
2. **AI Service Manager / Stateful Service** — model lifecycle via Redis
3. **Resource Broker** — dynamic GPU reallocation with safe transitions
4. **Workflow Engine Integration** — end-to-end orchestration

Ada's requirements from the 2026-03-30 meeting:
1. Show instrumentation overhead is negligible
2. Show stateful→stateless transition measurably improves system metrics
3. Show dynamic GPU reclaim works
4. Broader workload diversity (nice-to-have)

---

## Study 1-3: Component Benchmarks — COMPLETE

All component benchmarks pass. Results in `results/study{1,2,3}/`.
Figures generated in `results/figures/`. See previous PROGRESS entries
for detailed numbers.

---

## Study 4: End-to-End Integration — IN PROGRESS

### What works

- Dynamic workflow (`openmm_cvae_dynamic.py`) runs end-to-end
- Signal Monitor with CompositePolicy evaluates 3 signals per training cycle
- Resource Broker freeze/resume transitions work
- Contact map loader fixed for all 3 protein systems (sparse COO normalization)
- `final_loss` field added to `CVAETrainOutput` for real telemetry

### Cross-system results (15-min runs, 8x H100)

| System | Mode | Sims | Trains | Infer | Mean RMSD | NN<5Å | Last10 |
|--------|------|------|--------|-------|-----------|-------|--------|
| BBA | baseline | 104 | 14 | 53 | 5.985 | 31.7% | 4.838 |
| BBA | dynamic | 102 | 15 | 65 | **5.606** | **38.3%** | **4.628** |
| CLN025 | baseline | 112 | 12 | 47 | **5.679** | **20.9%** | 6.170 |
| CLN025 | dynamic | 110 | 12 | 34 | 6.063 | 10.6% | 6.206 |
| NTL9 | baseline | 98 | 14 | 63 | 6.815 | 18.9% | 8.134 |
| NTL9 | dynamic | 98 | 14 | 69 | **6.241** | 18.4% | **7.914** |

### Run directories

| System | Mode | Path |
|--------|------|------|
| BBA baseline | baseline | `runs/experiment-310326-015111` |
| BBA dynamic | dynamic | `runs/experiment-310326-024822` |
| CLN025 baseline | baseline | `runs/experiment-310326-220720` |
| CLN025 dynamic | dynamic | `runs/experiment-310326-225629` |
| NTL9 baseline | baseline | `runs/experiment-310326-205255` |
| NTL9 dynamic | dynamic | `runs/experiment-310326-211116` |

### Current signals in CompositePolicy

The telemetry vector `[normalized_loss, mean_rmsd, inference_stability]` is
submitted once per training cycle in `handle_train_output`:

1. **normalized_loss** (v[0]): `final_loss / first_loss`. Captures ML convergence.
2. **mean_rmsd** (v[1]): Rolling mean of last 10 simulation RMSDs. Captures scientific quality.
3. **inference_stability** (v[2]): Fraction of outlier restart dirs unchanged between
   consecutive inference calls. Captures whether the model's view has stabilized.

CompositePolicy decision:
- DORMANT: loss plateaued AND (RMSD not improving OR inference stable >0.8)
- RESUME: RMSD degrading (trending upward)
- ACTIVE: otherwise

### Known issues to solve

1. **CLN025 dynamic underperforms baseline.** The single ML GPU bottlenecks
   inference throughput (34 vs 47 inferences). CLN025 depends heavily on
   frequent inference steering. Options:
   - Allow inference on sim GPUs when ML GPU is busy
   - Prioritize inference over training on ML GPU
   - Share GPU time more efficiently between training and inference

2. **Signal monitor only evaluates after training completes.** Telemetry is
   submitted in `handle_train_output`, so decisions happen once per ~60s
   training cycle. Faster signals (e.g., from simulation RMSD trends or
   inference results) could enable more responsive decisions.

3. **GPU 7 goes idle after freeze.** The `simulate()` override with round-robin
   routing is implemented but untested in a scenario where freeze actually
   triggers. Need a system where freeze fires (e.g., KRAS-like convergence)
   to validate GPU reclaim throughput gain.

4. **No RESUME scenario tested.** The composite policy can emit RESUME but
   the workflow doesn't properly restart training after a freeze. Need to
   test: freeze → RMSD degrades → RESUME → training restarts.

### Next steps

1. Fix CLN025 inference bottleneck — most impactful for showing the system
   works across systems
2. Run longer campaigns (200+ sims) to see freeze actually trigger
3. Test with a system that converges (KRAS-like) to validate freeze + GPU reclaim
4. Generate Study 4 comparison figures
5. Write evaluation section of paper

---

## How to reproduce

```bash
cd /shivam/deepdrivemd

# Run a baseline campaign
PYTHONPATH=/shivam/deepdrivemd timeout 900 python3 \
  deepdrivemd/workflows/openmm_cvae.py \
  -c evaluation/study4/configs/bba_baseline.yaml

# Run a dynamic campaign (needs Redis)
redis-server --daemonize yes --port 6379 --save ""
PYTHONPATH=/shivam/deepdrivemd timeout 900 python3 \
  deepdrivemd/workflows/openmm_cvae_dynamic.py \
  -c evaluation/study4/configs/bba_dynamic.yaml

# Compare results
python3 -c "see analyze script in evaluation/study4/"

# Regenerate figures
make figures
```

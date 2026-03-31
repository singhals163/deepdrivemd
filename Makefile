.DEFAULT_GOAL := all
package_name = deepdrivemd
extra_folders = tests/ examples/ 
isort = isort $(package_name) $(extra_folders)
black = black --target-version py37 $(package_name) $(extra_folders)
flake8 = flake8 $(package_name)/ $(extra_folders)
pylint = pylint $(package_name)/ $(extra_folders)
pydocstyle = pydocstyle $(package_name)/
run_mypy = mypy --config-file setup.cfg

.PHONY: install
install:
	pip install -U pip setuptools wheel
	pip install -r requirements/dev.txt
	pip install -r requirements/requirements.txt
	pip install -e .

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	$(flake8)
	#$(pylint)
	#$(pydocstyle)

.PHONY: mypy
mypy:
	$(run_mypy) --package $(package_name)
	$(run_mypy) $(package_name)/
	$(run_mypy) $(extra_folders)

.PHONY: coverage
coverage:
	coverage run -m pytest tests
	coverage report
	coverage html
	open htmlcov/index.html

.PHONY: pygount
pygount:
	pygount --format=summary $(package_name)

.PHONY: radon
radon:
	radon cc $(package_name) -a

.PHONY: all
all: format lint # mypy

# ─────────────────────────────────────────────────────────
#  Evaluation: Run benchmarks and generate figures
# ─────────────────────────────────────────────────────────
PYPATH  := PYTHONPATH=/shivam/deepdrivemd
EVALDIR := evaluation
RESDIR  := results
RUNSDIR := /shivam/deepdrivemd/runs

.PHONY: eval benchmarks figures

eval: benchmarks figures

# ── Benchmarks (grouped by study) ────────────────────────
# `make benchmarks` runs real data only
# `make benchmarks-synthetic` runs synthetic workloads only
# `make benchmarks-all` runs both
benchmarks: bench-study1 bench-study2 bench-study3
benchmarks-synthetic: bench-study1-synthetic bench-study2-synthetic bench-study3-synthetic
benchmarks-all: benchmarks benchmarks-synthetic

.PHONY: bench-study1 bench-study2 bench-study3
.PHONY: bench-study1-synthetic bench-study2-synthetic bench-study3-synthetic

# ── Study 1: Real ─────────────────────────────────────────
bench-study1:
	@echo "=== Study 1: Signal Monitor (real) ==="
	@mkdir -p $(RESDIR)/study1/real
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_latency \
		--runs-dir $(RUNSDIR) --repeats 5000 --trials 5 \
		--output $(RESDIR)/study1/real/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_concurrency \
		--runs-dir $(RUNSDIR) --threads 50 --signals-per-thread 100 --trials 5 \
		--output $(RESDIR)/study1/real/results_concurrency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_pluggability \
		--runs-dir $(RUNSDIR) \
		--output $(RESDIR)/study1/real/results_pluggability.json

# ── Study 1: Synthetic ────────────────────────────────────
bench-study1-synthetic:
	@echo "=== Study 1: Signal Monitor (synthetic) ==="
	@mkdir -p $(RESDIR)/study1/synthetic
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_latency \
		--synthetic --repeats 5000 --trials 5 \
		--output $(RESDIR)/study1/synthetic/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_concurrency \
		--synthetic --threads 50 --signals-per-thread 100 --trials 5 \
		--output $(RESDIR)/study1/synthetic/results_concurrency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_pluggability \
		--synthetic \
		--output $(RESDIR)/study1/synthetic/results_pluggability.json

# ── Study 2: Real ─────────────────────────────────────────
bench-study2:
	@echo "=== Study 2: Stateful Service (real) ==="
	@mkdir -p $(RESDIR)/study2/real
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_latency \
		--runs-dir $(RUNSDIR) --repeats 10 \
		--output $(RESDIR)/study2/real/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_reproducibility \
		--runs-dir $(RUNSDIR) \
		--output $(RESDIR)/study2/real/results_reproducibility.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_memory \
		--runs-dir $(RUNSDIR) --trials 10 \
		--output $(RESDIR)/study2/real/results_memory.json

# ── Study 2: Synthetic ────────────────────────────────────
bench-study2-synthetic:
	@echo "=== Study 2: Stateful Service (synthetic) ==="
	@mkdir -p $(RESDIR)/study2/synthetic
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_latency \
		--synthetic --repeats 20 \
		--output $(RESDIR)/study2/synthetic/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_reproducibility \
		--synthetic \
		--output $(RESDIR)/study2/synthetic/results_reproducibility.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_memory \
		--synthetic --trials 5 \
		--output $(RESDIR)/study2/synthetic/results_memory.json

# ── Study 3: Real ─────────────────────────────────────────
bench-study3:
	@echo "=== Study 3: Resource Broker (real) ==="
	@mkdir -p $(RESDIR)/study3/real
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_reallocation \
		--runs-dir $(RUNSDIR) --repeats 20 \
		--output $(RESDIR)/study3/real/results_reallocation.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_throughput \
		--runs-dir $(RUNSDIR) \
		--output $(RESDIR)/study3/real/results_throughput.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_draining \
		--runs-dir $(RUNSDIR) \
		--output $(RESDIR)/study3/real/results_draining.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_cooldown \
		--output $(RESDIR)/study3/real/results_cooldown.json

# ── Study 3: Synthetic ────────────────────────────────────
bench-study3-synthetic:
	@echo "=== Study 3: Resource Broker (synthetic) ==="
	@mkdir -p $(RESDIR)/study3/synthetic
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_reallocation \
		--synthetic --repeats 20 \
		--output $(RESDIR)/study3/synthetic/results_reallocation.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_cooldown \
		--output $(RESDIR)/study3/synthetic/results_cooldown.json

# ── Figures (re-reads results/ and overwrites) ───────────
figures: fig-study1 fig-study2 fig-study3 fig-overview

.PHONY: fig-study1 fig-study2 fig-study3 fig-overview

fig-study1:
	$(PYPATH) python3 $(EVALDIR)/plot_study1.py

fig-study2:
	$(PYPATH) python3 $(EVALDIR)/plot_study2.py

fig-study3:
	$(PYPATH) python3 $(EVALDIR)/plot_study3.py

fig-overview:
	$(PYPATH) python3 $(EVALDIR)/plot_overview.py

# ── Cleanup ──────────────────────────────────────────────
.PHONY: clean-eval
clean-eval:
	rm -rf $(RESDIR)/study1/real/*.json $(RESDIR)/study1/synthetic/*.json \
	       $(RESDIR)/study2/real/*.json $(RESDIR)/study2/synthetic/*.json \
	       $(RESDIR)/study3/real/*.json $(RESDIR)/study3/synthetic/*.json \
	       $(RESDIR)/figures/*

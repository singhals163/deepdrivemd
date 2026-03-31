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

.PHONY: eval benchmarks figures

eval: benchmarks figures

# ── Benchmarks (grouped by study) ────────────────────────
benchmarks: bench-study1 bench-study2 bench-study3

.PHONY: bench-study1 bench-study2 bench-study3

bench-study1:
	@echo "=== Study 1: Signal Monitor ==="
	@mkdir -p $(RESDIR)/study1
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_latency \
		--repeats 5000 --output $(RESDIR)/study1/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_concurrency \
		--threads 50 --signals-per-thread 100 \
		--output $(RESDIR)/study1/results_concurrency.json
	$(PYPATH) python3 -m $(EVALDIR).study1.bench_pluggability \
		--output $(RESDIR)/study1/results_pluggability.json

bench-study2:
	@echo "=== Study 2: Stateful Service ==="
	@mkdir -p $(RESDIR)/study2
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_latency \
		--repeats 50 --output $(RESDIR)/study2/results_latency.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_reproducibility \
		--epochs 10 --output $(RESDIR)/study2/results_reproducibility.json
	$(PYPATH) python3 -m $(EVALDIR).study2.bench_memory \
		--output $(RESDIR)/study2/results_memory.json

bench-study3:
	@echo "=== Study 3: Resource Broker ==="
	@mkdir -p $(RESDIR)/study3
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_reallocation \
		--repeats 50 --output $(RESDIR)/study3/results_reallocation.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_throughput \
		--output $(RESDIR)/study3/results_throughput.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_draining \
		--output $(RESDIR)/study3/results_draining.json
	$(PYPATH) python3 -m $(EVALDIR).study3.bench_cooldown \
		--output $(RESDIR)/study3/results_cooldown.json

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
	rm -rf $(RESDIR)/study1/*.json $(RESDIR)/study2/*.json \
	       $(RESDIR)/study3/*.json $(RESDIR)/figures/*

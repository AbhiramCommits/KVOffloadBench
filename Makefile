PYTHON ?= python3
KVOB ?= kvoffloadbench
CONFIG ?= configs/sweeps/main.yaml

.PHONY: install lint typecheck test gen-traces bench sweep analyze smoke docker clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest --cov=kvoffloadbench --cov-report=term-missing

gen-traces:
	$(KVOB) gen-traces --out traces/ --seed 2024

bench:
	$(KVOB) bench-tiers --config configs/tiers_micro.yaml --out results/tiers/microbench.csv

sweep:
	$(KVOB) sweep --config $(CONFIG)

analyze:
	$(KVOB) analyze --results results/

smoke: gen-traces
	$(KVOB) sweep --config configs/sweeps/smoke.yaml --limit 2

docker:
	docker compose build

clean:
	rm -rf results/cache build dist .pytest_cache .mypy_cache .ruff_cache

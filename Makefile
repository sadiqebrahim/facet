# Override if your interpreter lives elsewhere:  make PY=/path/to/python exp001
PY ?= python
DATA ?= data/raw/SCUT-FBP5500_v2

.PHONY: help data features exp001 test lint clean summary

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

data: ## Download SCUT-FBP5500 (non-commercial research use only)
	$(PY) scripts/download_scut_fbp5500.py

summary: ## Print dataset facts (recomputed, so the docs stay honest)
	$(PY) scripts/dataset_summary.py

features: ## Extract and cache all frozen representations
	$(PY) scripts/extract_features.py --encoder geometry
	$(PY) scripts/extract_features.py --encoder arcface_buffalo_l
	$(PY) scripts/extract_features.py --encoder arcface_antelopev2
	$(PY) scripts/extract_features.py --encoder clip

exp001: ## Run experiment 001 (frozen representation + linear probe)
	$(PY) scripts/run_exp001.py

test: ## Run the test suite
	$(PY) -m pytest tests -q

lint:
	ruff check src scripts tests

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

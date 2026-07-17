# =============================================================================
# Credit Risk Analytics — Makefile
# =============================================================================
# Targets:
#   install       — Install Python dependencies
#   reproduce     — Re-run all notebooks (papermill)
#   test          — Run unit tests
#   lint          — Basic linting with flake8 / ruff
#   clean         — Remove cache, build artifacts, and git-ignored files
#   help          — Show this help
#
# Usage:
#   make install
#   make test
#   make help
# =============================================================================

.PHONY: help install test lint clean reproduce setup

SHELL := /bin/bash

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | 		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install papermill ruff pytest  # dev dependencies

test: ## Run unit tests (pytest)
	python -m pytest tests/ -v --tb=short

lint: ## Run ruff linter on src/ and tests/
	ruff check src/ tests/ --ignore E501 --quiet || true
	ruff format src/ tests/ --check --quiet || true

reproduce: ## Re-run all notebooks in sequence using papermill
	@echo "=== Reproducing full pipeline ==="
	mkdir -p notebooks/_output
	papermill notebooks/01_business_understanding.ipynb  notebooks/_output/01_business_understanding.ipynb
	papermill notebooks/02_exploratory_data_analysis.ipynb notebooks/_output/02_exploratory_data_analysis.ipynb
	papermill notebooks/03_feature_engineering.ipynb      notebooks/_output/03_feature_engineering.ipynb
	papermill notebooks/04_modeling.ipynb                 notebooks/_output/04_modeling.ipynb
	papermill notebooks/04_explainability.ipynb           notebooks/_output/04_explainability.ipynb
	papermill notebooks/05_business_simulation.ipynb      notebooks/_output/05_business_simulation.ipynb
	papermill notebooks/06_explainability_dashboard.ipynb notebooks/_output/06_explainability_dashboard.ipynb
	@echo "=== Pipeline reproduce complete ==="

setup: install test ## Install dependencies and run tests

infer: ## Score a CSV using the inference module
	python src/inference.py --help

clean: ## Remove cache, build artifacts, and node_modules
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf *.egg-info build dist
	rm -rf src/__pycache__ tests/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	@echo "Clean complete"

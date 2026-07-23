# Customer Churn Intelligence and Retention Decision-Support Platform
# Every target runs from the repository root.

SHELL := /bin/bash
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.DEFAULT_GOAL := help
.PHONY: help bootstrap validate eda train test app docker-build docker-run secret-scan verify notebook clean fairness calibration threshold drift track mlflow-ui analysis tune

help: ## Show the available targets
	@echo "Customer Churn Intelligence — available targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical first run:  make bootstrap && make validate && make eda && make train && make test"

bootstrap: ## Create .venv, install dependencies and validate the dataset
	bash scripts/bootstrap_macos.sh

validate: ## Validate the raw dataset against the documented contract
	$(PYTHON) -m src.data_validation

eda: ## Regenerate EDA figures, tables and observations
	$(PYTHON) -m src.eda

train: ## Train, compare, select and export the model pipeline
	$(PYTHON) -m src.train

notebook: ## Regenerate and execute the EDA/modelling notebook
	$(PYTHON) scripts/build_notebook.py

fairness: ## Fairness audit across protected attributes (H1-1)
	$(PYTHON) -m src.fairness

calibration: ## Probability calibration analysis (H1-2)
	$(PYTHON) -m src.calibration

threshold: ## Cost-sensitive decision-threshold analysis (H1-3)
	$(PYTHON) -m src.threshold

drift: ## Drift-detection apparatus and its demonstration (H2-2)
	$(PYTHON) -m src.drift --demo

tune: ## Feature engineering and hyperparameter search experiment (H1-6)
	$(PYTHON) -m src.tuning

track: ## Train and log the run to MLflow, then register it (H2-1)
	$(PYTHON) -m src.tracking --log-current

mlflow-ui: ## Browse tracked runs at http://127.0.0.1:5000
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

analysis: fairness calibration threshold drift ## Run every post-training analysis

test: ## Run the full pytest suite
	$(PYTHON) -m pytest -q

app: ## Run the Streamlit application locally on port 8501
	bash scripts/run_streamlit.sh

docker-build: ## Build the deployment Docker image
	bash scripts/build_docker.sh

docker-run: ## Build, run and health-check the image on port 7860
	bash scripts/build_docker.sh --run

secret-scan: ## Scan the project for credential patterns
	bash scripts/scan_secrets.sh

verify: ## Run every non-interactive local quality gate
	bash scripts/verify_release.sh

clean: ## Remove Python caches (never touches data/raw or artifacts)
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	@echo "Caches removed. data/raw and deploy/artifacts were not touched."

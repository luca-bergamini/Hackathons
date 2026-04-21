.PHONY: install test lint format run-ingestion run-processing run-features run-training run-serving run-all validate

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ --statistics
	ruff format --check src/

format:
	ruff format src/

validate:
	python scripts/validate_setup.py

run-ingestion:
	python -m src.ingestion.main

run-processing:
	python -m src.processing.main

run-features:
	python -m src.features.main

run-training:
	python -m src.models.main

run-serving:
	uvicorn src.serving.main:app --reload --port 8000

run-all: run-ingestion run-processing run-features run-training
	@echo "Pipeline completata."

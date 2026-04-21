.PHONY: setup test lint format validate

# --- Setup ---

setup:
	pip install -e ".[development]"

# --- Qualità del codice (identici alla pipeline CI/CD) ---

test:
	pytest tests/ -v --cov=src --cov-report=xml:coverage.xml --cov-report=term-missing

lint:
	ruff check src/ --output-format=json > ruff-report.json
	ruff check src/ --statistics

format:
	ruff format src/

validate:
	python scripts/validate_setup.py

# --- React Frontend ---

api:
	python -m uvicorn src.backend.api:app --reload --host 0.0.0.0 --port 8000

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

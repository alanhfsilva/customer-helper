.PHONY: dev serve test lint typecheck ingest eval install

install:
	pip install -e ".[dev]"

dev:
	docker compose up --build

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test: lint typecheck unittest

lint:
	ruff check app/ tests/ ingestion/ eval/

typecheck:
	mypy app/ tests/

unittest:
	pytest tests/ -v

ingest:
	python -m ingestion.run

eval:
	python -m eval.run --dataset golden

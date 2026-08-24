.PHONY: dev test lint typecheck ingest eval install

install:
	pip install -e ".[dev]"

dev:
	docker compose up --build

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

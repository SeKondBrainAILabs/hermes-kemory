.PHONY: check lint types test validate

check: lint types test validate

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy

test:
	uv run pytest -n 2

validate:
	uv run hermes plugins validate hermes_kemory

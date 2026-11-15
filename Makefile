.PHONY: check
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest -n 2
	uv run hermes plugins validate hermes_kemory

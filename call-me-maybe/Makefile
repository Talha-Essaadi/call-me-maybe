.PHONY: install run debug clean lint lint-strict help


all: install

install:
# 	uv venv --system-site-packages || true
# 	UV_CACHE_DIR="$(HOME)/goinfre/.uv-cache" uv sync
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true


lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

help:
	@echo "Available targets:"
	@echo "  install       - Install project dependencies using uv"
	@echo "  run           - Execute the main script"
	@echo "  run-custom    - Execute with custom file paths"
	@echo "  debug         - Run the main script in debug mode (pdb)"
	@echo "  clean         - Remove temporary files and caches"
	@echo "  lint          - Run flake8 and mypy with standard flags"
	@echo "  lint-strict   - Run flake8 and mypy with strict flags"
	@echo "  help          - Display this help message"

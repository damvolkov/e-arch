.PHONY: install sync test test-unit test-e2e demo demo-gemini lint check clean

# ── Setup ────────────────────────────────────────────────────────
install: ## Install all deps (dev included)
	@uv sync --dev

sync: ## Sync lockfile
	@uv sync --dev

# ── Tests ────────────────────────────────────────────────────────
test: ## Run all tests (unit + e2e)
	@uv run pytest tests/ -v --tb=short

test-unit: ## Run unit tests only (no network)
	@uv run pytest tests/unit/ -v --tb=short

test-e2e: ## Run e2e tests (hits Gemini API)
	@uv run pytest tests/e2e/ -v -s -m slow --log-cli-level=INFO

# ── Demos ────────────────────────────────────────────────────────
demo: ## Run mock agent demo (no API key needed)
	@uv run python scripts/demo_mock.py

demo-gemini: ## Run pydantic-ai + Gemini demo (needs GEMINI_API_KEY)
	@uv run python scripts/demo_gemini.py

# ── Quality ──────────────────────────────────────────────────────
lint: ## Lint + format with ruff
	@uv run ruff check --fix src/ tests/
	@uv run ruff format src/ tests/

check: ## Run all checks (lint + tests)
	@$(MAKE) lint
	@$(MAKE) test

# ── Cleanup ──────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	@rm -rf .pytest_cache .ruff_cache __pycache__ src/aeda/__pycache__ dist/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Help ─────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

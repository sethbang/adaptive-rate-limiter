# Makefile for adaptive-rate-limiter
# Uses uv for dependency management

.PHONY: help install install-dev install-redis install-all lock upgrade \
        test test-unit test-integration test-fast test-slow test-redis test-no-redis \
        test-quick test-verbose test-failed coverage coverage-xml \
        lint lint-fix format format-check typecheck check \
        security security-bandit security-audit pre-release \
        build publish publish-test docs docs-serve docs-deploy \
        clean clean-cache clean-build clean-coverage clean-all \
        ci-install ci-test ci-lint version info

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "$(BLUE)adaptive-rate-limiter$(NC) - Development Commands"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC) make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'

#------------------------------------------------------------------------------
# Installation & Dependencies
#------------------------------------------------------------------------------

install: ## Install production dependencies only
	uv sync --no-dev

install-dev: ## Install with dev dependencies
	uv sync --extra dev

install-redis: ## Install with redis support
	uv sync --extra redis

install-metrics: ## Install with metrics support
	uv sync --extra metrics

install-all: ## Install all dependencies (dev + redis)
	uv sync --extra dev --extra redis --extra metrics

lock: ## Update uv.lock from pyproject.toml
	uv lock

upgrade: ## Upgrade all dependencies
	uv lock --upgrade

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test: ## Run all tests with coverage
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration -v

test-fast: ## Run fast tests only
	uv run pytest -m fast

test-slow: ## Run slow tests only
	uv run pytest -m slow

test-redis: ## Run tests that require Redis
	uv run pytest -m requires_redis

test-no-redis: ## Run tests without Redis requirements
	uv run pytest -m "not requires_redis"

test-quick: ## Quick test run (no coverage, minimal output)
	uv run pytest --no-cov -q

test-verbose: ## Verbose test output with all logs
	uv run pytest -v --log-cli-level=DEBUG

test-failed: ## Re-run only failed tests
	uv run pytest --lf

coverage: ## Generate coverage report and open HTML
	uv run pytest --cov-report=html:tests/reports/coverage/html
	@echo "$(GREEN)Coverage report:$(NC) tests/reports/coverage/html/index.html"
	@open tests/reports/coverage/html/index.html 2>/dev/null || \
		xdg-open tests/reports/coverage/html/index.html 2>/dev/null || \
		echo "Open tests/reports/coverage/html/index.html in your browser"

coverage-xml: ## Generate XML coverage report (for CI)
	uv run pytest --cov-report=xml:tests/reports/coverage/coverage.xml

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint: ## Run ruff linter
	uv run ruff check src tests

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix src tests

format: ## Format code with ruff
	uv run ruff format src tests

format-check: ## Check code formatting
	uv run ruff format --check src tests

typecheck: ## Run type checking with mypy
	uv run mypy src

check: lint format-check typecheck ## Run all code quality checks

#------------------------------------------------------------------------------
# Security Scanning
#------------------------------------------------------------------------------

security: security-bandit security-audit ## Run all security checks

security-bandit: ## Run bandit security scanner on source code
	uv run bandit -r src/ -c pyproject.toml

security-audit: ## Run pip-audit to check for vulnerable dependencies
	uv run pip-audit

#------------------------------------------------------------------------------
# Pre-Release Checks
#------------------------------------------------------------------------------

pre-release: lint format-check typecheck security test ## Run all pre-release checks
	@echo "$(GREEN)All pre-release checks passed!$(NC)"

#------------------------------------------------------------------------------
# Building & Publishing
#------------------------------------------------------------------------------

build: clean-build ## Build distribution packages
	uv build

publish-test: build ## Publish to TestPyPI
	@echo "$(YELLOW)Ensure you have configured TestPyPI credentials$(NC)"
	uv publish --index testpypi

publish: build ## Publish to PyPI
	uv publish

#------------------------------------------------------------------------------
# Development Utilities
#------------------------------------------------------------------------------

shell: ## Open Python shell with package loaded
	uv run python -c "from adaptive_rate_limiter import *; import code; code.interact(local=locals())"

repl: ## Open IPython shell (if installed)
	uv run ipython

run: ## Run a Python script (usage: make run SCRIPT=path/to/script.py)
	uv run python $(SCRIPT)


#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------

clean-cache: ## Clean Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-build: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info

clean-coverage: ## Clean coverage reports
	rm -rf tests/reports/coverage/
	rm -f .coverage .coverage.*

clean: clean-cache clean-build ## Clean cache and build artifacts

clean-all: clean clean-coverage ## Clean everything including coverage
	rm -rf .venv

#------------------------------------------------------------------------------
# CI/CD Helpers
#------------------------------------------------------------------------------

ci-install: ## CI-optimized installation
	uv sync --extra dev --extra redis --no-progress

ci-test: ## Run tests for CI (with XML reports)
	uv run pytest \
		--junitxml=tests/reports/junit.xml \
		--cov-report=xml:tests/reports/coverage/coverage.xml \
		--cov-report=json:tests/reports/coverage/coverage.json

ci-lint: ## Lint check for CI
	uv run ruff check --output-format=github src tests
	uv run ruff format --check src tests

#------------------------------------------------------------------------------
# Project Info
#------------------------------------------------------------------------------

version: ## Show project version
	@grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2

info: ## Show project information
	@echo "$(BLUE)Project:$(NC) adaptive-rate-limiter"
	@echo "$(BLUE)Version:$(NC) $$(grep 'version = ' pyproject.toml | head -1 | cut -d'\"' -f2)"
	@echo "$(BLUE)Python:$(NC) $$(cat .python-version 2>/dev/null || echo 'not set')"
	@echo "$(BLUE)UV:$(NC) $$(uv --version 2>/dev/null || echo 'not installed')"

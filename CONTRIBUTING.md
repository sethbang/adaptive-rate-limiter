# Contributing to Adaptive Rate Limiter

Thank you for your interest in contributing! This guide will help you get started.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management
- (Optional) Redis for running integration tests

### Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sethbang/adaptive-rate-limiter.git
   cd adaptive-rate-limiter
   ```

2. **Install dependencies:**

   ```bash
   uv sync --extra dev --extra redis
   ```

3. **Verify your setup:**

   ```bash
   uv run pytest tests/unit -v
   ```

## Project Structure

```
adaptive-rate-limiter/
├── src/adaptive_rate_limiter/    # Main package source
│   ├── __init__.py               # Public API exports
│   ├── exceptions.py             # Exception hierarchy
│   ├── backends/                 # Storage backends (memory, redis)
│   ├── scheduler/                # Core scheduler and configuration
│   ├── streaming/                # Streaming response support
│   ├── observability/            # Metrics and monitoring
│   ├── providers/                # API provider interfaces
│   ├── protocols/                # Protocol definitions
│   ├── reservation/              # Capacity reservation tracking
│   ├── strategies/               # Scheduling strategies
│   └── types/                    # Type definitions
├── tests/
│   ├── unit/                     # Unit tests (run with pytest)
│   └── integration/              # Integration tests (require Redis)
├── docs/                         # MDX documentation files
├── benchmarks/                   # Performance benchmarks
└── .github/                      # GitHub workflows
```

## Development Workflow

### Running Tests

Run the full unit test suite:

```bash
uv run pytest tests/unit
```

Run integration tests (requires Redis or uses fakeredis):

```bash
uv run pytest tests/integration
```

Run all tests with coverage:

```bash
uv run pytest --cov=adaptive_rate_limiter --cov-report=term-missing
```

### Using Nox

We use [nox](https://nox.thea.codes/) for automated testing across Python versions:

```bash
uv run nox
```

### Code Quality

We use **ruff** for linting and formatting, and **mypy** for type checking:

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy
```

All checks must pass before a PR can be merged.

### Makefile Shortcuts

Common commands are available via the Makefile:

```bash
make test        # Run tests
make lint        # Run linting
make format      # Format code
make typecheck   # Run mypy
```

## Submitting Changes

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. Make your changes, ensuring tests pass locally.
3. Add tests for any new functionality.
4. Update documentation if needed.
5. Open a pull request with a clear description of your changes.

### What We Look For

- **Tests pass**: All existing and new tests must pass.
- **Code quality**: Linting and type checks must pass.
- **Clear description**: Explain what your PR does and why.
- **Focused changes**: Keep PRs focused on a single concern.

## Reporting Issues

Found a bug or have a feature request? Please [open an issue](https://github.com/sethbang/adaptive-rate-limiter/issues/new/choose) using the appropriate template.

## Questions?

Feel free to open a discussion or reach out via the issue tracker. We're happy to help!

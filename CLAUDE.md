# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tooling

Dependencies are managed with `uv` (not pip directly). The repo expects `uv sync --extra dev --extra redis` to be run for a normal dev install; for type-check parity with CI use `uv sync --extra dev --extra full` (mypy needs the optional extras for stubs). Test config lives in `pytest.toml` (pytest 9.0+ format) — not in `pyproject.toml`. Python 3.10–3.13 are supported.

## Common commands

Most workflow shortcuts live in the `Makefile` and wrap `uv run`. Prefer them over hand-rolling commands:

- `make test` — full suite with coverage (enforces `--cov-fail-under=80`)
- `make test-unit` / `make test-integration` — split by directory
- `make test-quick` — fast iteration (no coverage, quiet)
- `make test-failed` — re-run only previously failed tests (`pytest --lf`)
- `make lint` / `make lint-fix` / `make format` / `make typecheck`
- `make check` — lint + format-check + typecheck (run before committing)
- `make pre-release` — full gate: check + security + tests
- `make security` — bandit + pip-audit
- `uv run nox` — matrix tests across Python 3.10–3.13

Single test: `uv run pytest tests/unit/path/to/test_file.py::TestClass::test_name`.

Test markers (defined in `pytest.toml`) are gated with `--strict-markers`. Useful filters: `-m fast`, `-m "not requires_redis"`, `-m requires_redis_pool`. Integration tests use `fakeredis` by default; tests marked `requires_redis_pool` or `cluster` need a real Redis (see `docker-compose.redis-cluster.yml`).

## Architecture

The library is a provider-agnostic rate-limiting scheduler for AI/ML APIs. The shape to keep in mind:

**Scheduler (facade) → Mode Strategy (per-mode logic) → Backend (state storage)**, with a separate **Provider** for rate-limit discovery and a **Strategy** for queue selection.

- `scheduler/scheduler.py` — `Scheduler` is a thin facade. It owns a `BaseSchedulingModeStrategy` and forwards `submit_request` to it. Do not put mode-specific branching in the facade; it goes in the mode strategy. `create_scheduler(client, mode=...)` is the public factory.
- `strategies/modes/{basic,intelligent,account}.py` — the three `SchedulerMode` strategies. `BASIC` is pass-through with retry; `INTELLIGENT` is the production path (capacity reservation, queue management, streaming refunds); `ACCOUNT` is multi-tenant per-account tracking. Adding a mode means adding a strategy class and wiring it in `BaseScheduler._setup_mode_strategy`, not editing `Scheduler`.
- `strategies/scheduling.py` — separate concern: queue *selection* algorithms (WeightedRoundRobin, DRR, Adaptive, etc.) implementing `BaseSchedulingStrategy`. Don't conflate with mode strategies.
- `backends/{memory,redis}.py` — pluggable state storage behind `BaseBackend`. Redis backend uses Lua scripts in `backends/lua/*.lua` for atomic check-and-reserve / release / orphan-recovery operations; the Lua scripts are part of the wheel (see `[tool.hatch.build.targets.sdist]` include in `pyproject.toml`). When changing reservation semantics, the Lua script and the Python caller must move together.
- `providers/base.py` — `ProviderInterface` defines `discover_limits`, `parse_rate_limit_response`, `get_bucket_for_model`. Providers are how the library learns rate limits from response headers. The library ships only the abstract interface — concrete providers are user-supplied.
- `protocols/` — `ClientProtocol` (HTTP client shape) and `ClassifierProtocol` (request routing) are duck-typed via `typing.Protocol`. Users implement these, not subclass.
- `streaming/` — refund-based token accounting. `StreamingReservationContext` holds the optimistic reservation; `RateLimitedAsyncIterator` wraps the response stream and refunds unused tokens when the stream completes. Token accounting changes have to keep reservation and refund paths consistent.
- `reservation/` — `ReservationTracker` is the orphan-detection layer; expired reservations get reclaimed so a crashed caller can't permanently hold capacity.
- `scheduler/state/` — `StateManager` sits between mode strategies and backends, enforcing `CachePolicy` (WRITE_THROUGH default, WRITE_BACK opt-in with explicit production acknowledgement).

### Public API boundary

This is enforced, not advisory:

- Public exports come from seven `__init__.py` files: `adaptive_rate_limiter/`, `adaptive_rate_limiter/scheduler/`, `adaptive_rate_limiter/backends/`, `adaptive_rate_limiter/observability/`, `adaptive_rate_limiter/streaming/`, `adaptive_rate_limiter/reservation/`, `adaptive_rate_limiter/providers/`. Any non-underscore symbol exported from these files is part of the public API.
- Any `_`-prefixed name (module, class, or function) is explicitly not part of the public API and may change between any versions (this is in the README and is load-bearing for semver). New helpers default to a `_`-prefixed name unless they're being deliberately added to the public surface.
- `RedisBackend` (and `FallbackRateLimiter`, `InFlightRequest`, `ModelLimits`) are **lazy-imported** via `__getattr__` so the package imports cleanly without the `redis` extra installed. Don't add eager `from .backends import RedisBackend` at module top — it'll break users who haven't installed `[redis]`.

### Configuration

`RateLimiterConfig` (scheduler-wide) and `StateConfig` (state manager) are dataclasses with `__post_init__` validation. Tests for new config knobs should cover the validator branches, not just the happy path.

## Code quality settings worth knowing

- mypy is strict (`strict = true`, `warn_return_any`, `warn_unused_ignores`). New code must be fully typed; don't add blanket `# type: ignore` without a code.
- ruff lint set includes `S` (bandit), `ASYNC`, `PTH`, `UP`. Test files have `S101/S105/S106` exemptions; src files do not.
- Coverage gate is 80% (`--cov-fail-under=80`).
- Pre-commit runs ruff (with `--fix`), mypy `--strict`, and bandit on `src/`. Install hooks with `pre-commit install` after cloning.
- `pyproject.toml` is the version of record (`version = "..."`); `__init__.py` has its own `__version__` string and historically these have drifted (see commit `1df1aca`). Keep them in sync when bumping.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`ProviderError` exception**: custom providers can now raise
  `ProviderError` (importable as `from adaptive_rate_limiter import
  ProviderError`) when rate-limit discovery, response parsing, or
  bucket-lookup fails. It slots into the existing exception hierarchy under
  `RateLimiterError`.
- **`RedisBackend` explicit lifecycle methods**: `RedisBackend.start()` and
  `stop()` are now public. Orphan-reservation recovery starts automatically
  when the backend is activated through `StateManager` or `create_scheduler`,
  not only when used as an `async with` context manager.
- **Config validation**: eleven additional `RateLimiterConfig` numeric fields
  now validate their values in `__post_init__` and raise `ValueError` on bad
  input: `request_timeout`, `scheduler_interval`, `backoff_base`,
  `max_backoff`, `failure_window`, `max_failures`, `health_check_interval`,
  `max_consecutive_failures`, `metrics_export_interval`, `prometheus_port`,
  and `test_rate_multiplier`.

### Changed

- **Public API surface**: the library now documents seven public
  sub-packages — `observability`, `streaming`, `reservation`, and `providers`
  join the original three (`adaptive_rate_limiter`, `scheduler`, `backends`).
  Imports from those sub-packages are supported and stable.
- **Redis rate-limit updates**: the Lua update scripts now apply server-sent
  rate-limit *decreases* (e.g. tier downgrades) as well as increases.
  Previously, a lower limit reported by the server was silently ignored.
- **`RedisBackend` state storage**: `RateLimitState` key-value snapshots are
  stored under a distinct Redis key, separate from the atomic Lua reservation
  hash. This prevents the two write paths from corrupting each other.

### Removed

- **`StreamingInFlightTracker` (breaking)**: `StreamingInFlightTracker` has
  been removed from `adaptive_rate_limiter.streaming`. It was unused dead code
  and had no effect on runtime behaviour. Stale-streaming cleanup continues to
  work internally. `StreamingInFlightEntry` is **not** affected and remains
  available.

### Fixed

- **Streaming cancellation**: streaming reservations are now released
  immediately on task cancellation (`asyncio.CancelledError`) or generator
  close (`GeneratorExit`) instead of being leaked until background cleanup
  runs.
- **ACCOUNT mode concurrency**: the per-account concurrency cap is now
  enforced with a semaphore, closing a race where the limit could be exceeded
  under concurrent submissions.
- **Cold-start probe atomicity**: check-and-claim for cold-start probes is now
  atomic across resource types, preventing multiple concurrent callers from
  each believing they own the probe slot.
- **Signal-handler state flush**: the signal handler and the scheduler's
  normal write path now share a single lock. Fire-and-forget Redis background
  tasks are retained against garbage collection so they cannot be silently
  dropped mid-flight.
- **Backoff jitter**: jitter is now derived from `random.random()` rather than
  a time-derived seed, eliminating correlated backoffs across callers that
  retry at the same instant.

## [1.1.0] - 2026-05-19

This release resolves a full code audit of v1.0.2 — concurrency, lifecycle,
packaging, and API-boundary issues — and removes a class of intermittent CI
failures caused by un-awaited coroutines in the test suite.

### Added

- **Public API**: `ScheduleResult` and `QueuedRequest` are now exported from
  the top-level `adaptive_rate_limiter` package and from
  `adaptive_rate_limiter.scheduler`. They are the return type of
  `submit_request` in `INTELLIGENT` and `ACCOUNT` modes.
- **Config**: `RateLimiterConfig` gained `max_retries`, `batch_size`,
  `stale_entry_ttl`, and `max_tracking_entries` fields, each validated in
  `__post_init__`.
- **CI**: a packaging job builds the wheel and verifies all six bundled Lua
  scripts ship with it, guarding against a regression that would break the
  `[redis]` install.

### Changed

- **ACCOUNT mode**: the concurrency limit is now read from the correct
  config field, `max_concurrent_executions`. Previously the strategy looked
  up a non-existent `max_concurrent_requests` attribute and silently fell
  back to the hardcoded default of 10.
- **Reservations & streaming**: staleness and abandonment detection now use
  a monotonic clock (`time.monotonic()`), making it immune to wall-clock and
  NTP adjustments. The effective stale-reservation age is derived from
  `request_timeout`, so cleanup can no longer reclaim a still-running
  reservation.
- Removed the empty, unused `_internal/` package. `_`-prefixed names were
  never part of the public API, so this is not a breaking change.

### Fixed

- **Scheduler loop**: the INTELLIGENT mode loop no longer terminates
  permanently when an unexpected exception is raised inside it.
- **Queue race**: request enqueue and the `_queue_has_items` flag are now
  mutated under the per-queue lock, preventing a lost update that could
  leave a non-empty queue unscheduled.
- **Reset watcher**: no longer raises `RuntimeError: Set changed size during
  iteration` when a watcher fires during reset-time calculation.
- **Shutdown**: `stop()` now cancels and awaits in-flight execution tasks
  (INTELLIGENT and ACCOUNT modes) before the backend shuts down, so
  capacity-release cleanup completes and request coroutines are not left
  un-awaited.
- **Cancellation orphan**: a reservation orphaned by cancellation between
  capacity reservation and executor hand-off is now released immediately
  instead of being held until stale cleanup.
- **MemoryBackend**: streaming token refunds are reconciled against header
  syncs via a per-key state version, preventing capacity over-admission when
  a server header update superseded a reservation.
- **Streaming**: on a successful wrap, reservation ownership is handed to the
  iterator so stale-reservation cleanup cannot reclaim a live stream.

## [1.0.2] - 2026-04-25

### Fixed

- **Scheduler**: Initialize `circuit_breaker` and `_circuit_breaker_always_closed` on
  `BaseScheduler` so the INTELLIGENT mode strategy no longer raises
  `AttributeError: 'Scheduler' object has no attribute '_circuit_breaker_always_closed'`
  on the first `submit_request` when callers wire `Scheduler` + `Provider` +
  `Classifier` + `StateManager` themselves (e.g. the Venice AI SDK factory).
- **RedisBackend**: Sanitize state dicts before issuing `HSET mapping=...`.
  redis-py's encoder rejects both `NoneType` and `bool`, both of which
  appear in `RateLimitState` cold-start dumps (`request_limit`,
  `token_limit`, `bucket_id`, `last_request_time` default to `None`;
  `is_verified` is a `bool`). `set_state` now drops `None` entries and
  coerces `bool` → `int` (0/1). On read, missing fields are rehydrated to
  their declared Pydantic defaults and 0/1 is coerced back to `bool`.

## [1.0.1] - 2026-02-04

### Fixed

- **MemoryBackend**: Fixed `TypeError` when `remaining_requests` is explicitly `None` in state data. The `dict.get()` method returns `None` if the key exists with value `None`, not the default value. This caused failures on Windows with Python 3.10-3.12.

## [1.0.0] - 2026-01-28

Initial public release of Adaptive Rate Limiter.

### Added

#### Core Features
- **Provider-Agnostic Architecture**: Works with any OpenAI-compatible API (OpenAI, Anthropic, Venice, Groq, Together, etc.)
- **Adaptive Rate Limiting**: Intelligent rate limit discovery from response headers
- **Streaming Support**: Refund-based token accounting for streaming responses
- **Multi-Tenant Isolation**: Namespace-based isolation for multi-tenant applications

#### Scheduling Modes
- **Basic Mode**: Simple direct execution with retry logic for low-volume use cases
- **Intelligent Mode**: Advanced queuing with bucket-based scheduling and rate limit discovery
- **Account Mode**: Account-level request management for multi-tenant applications

#### Backends
- **MemoryBackend**: In-memory state storage for single-instance deployments
- **RedisBackend**: Distributed state storage with Lua scripts for atomic operations
  - `distributed_check_and_reserve.lua`: Atomic capacity reservation
  - `distributed_recover_orphan.lua`: Orphaned reservation recovery
  - `distributed_release_capacity.lua`: Capacity release operations
  - `distributed_release_streaming.lua`: Streaming response cleanup
  - `distributed_update_rate_limits.lua`: Rate limit state updates
  - `distributed_update_rate_limits_429.lua`: 429 response handling

#### Protocols & Interfaces
- `ClientProtocol`: Define how clients connect to APIs
- `ProviderInterface`: Extensible provider system for rate limit parsing
- `ClassifierProtocol`: Request classification for routing
- `StreamingResponseProtocol`: Streaming response handling

#### State Management
- `StateManager`: Centralized state management with configurable cache policies
- `CachePolicy.WRITE_THROUGH`: Immediate persistence for production safety
- `CachePolicy.WRITE_BACK`: Deferred writes for performance optimization
- `CachePolicy.WRITE_AROUND`: Direct backend writes for read-heavy workloads
- Bulk operations support for efficient state updates

#### Reservation System
- `ReservationTracker`: Token capacity reservation and tracking
- `ReservationContext`: Context manager for automatic reservation cleanup
- Heap-based cleanup for expired reservations
- Orphan recovery mechanisms

#### Streaming Support
- `StreamingInFlightTracker`: Track streaming response lifecycle
- `StreamingReservationContext`: Context manager for streaming operations
- `StreamingIterator`: Async iterator wrapper with token accounting
- `StreamingInFlightEntry`: Entry tracking for in-flight streaming requests
- Automatic token refunds on stream completion

#### Observability
- `UnifiedMetricsCollector`: Main collector for all rate limiter metrics
- 30+ named metric constants available for instrumentation
- Both dict and Prometheus output formats supported
- Built-in Prometheus metrics via optional `prometheus-client`
- Request latency histograms
- Queue depth gauges
- Rate limit state metrics

#### Exception Hierarchy
- `RateLimiterError`: Base exception for all rate limiter errors
- `CapacityExceededError`: Rate limit capacity exceeded with retry-after
- `BucketNotFoundError`: Unknown bucket identifier
- `ReservationCapacityError`: Reservation tracker at capacity
- `BackendConnectionError`: Backend connection failures
- `BackendOperationError`: Backend operation failures
- `ConfigurationError`: Invalid configuration
- `QueueOverflowError`: Request queue overflow with backpressure
- `TooManyFailedRequestsError`: Circuit breaker for failure rate protection

#### Type System
- `DiscoveredBucket`: Bucket configuration discovered from providers (bucket_id, RPM/TPM limits)
- `RateLimitInfo`: Parsed rate limit response data
- `RequestMetadata`: Request metadata for scheduling decisions
- `ResourceType`: Type-safe resource type constants (`TEXT`, `IMAGE`, `AUDIO`, `EMBEDDING`, `GENERIC`)
- `QueuedRequest`, `QueueInfo`, `ScheduleResult` for queue management
- `RateLimitType`, `RateLimitBucket`, `LimitCheckResult` for rate limit types

#### Documentation
- Comprehensive README with Quick Start guide
- API reference documentation
- Backend configuration guide
- Provider implementation guide
- Streaming support documentation

#### Testing Infrastructure
- Unit tests for all core components
- Integration tests for backend consistency
- Redis cluster integration tests
- Lua script integration tests
- End-to-end workflow tests
- Benchmark tests for concurrent scaling and scheduler overhead

### Technical Details

- **Python**: Requires Python 3.10+
- **Dependencies**: `pydantic`
- **Optional Dependencies**:
  - `[metrics]`: `prometheus-client` for Prometheus integration
  - `[redis]`: `redis` for distributed backends
  - `[full]`: All optional dependencies
- **License**: Apache-2.0

[Unreleased]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/sethbang/adaptive-rate-limiter/releases/tag/v1.0.0

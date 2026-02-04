# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

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

[Unreleased]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/sethbang/adaptive-rate-limiter/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/sethbang/adaptive-rate-limiter/releases/tag/v1.0.0

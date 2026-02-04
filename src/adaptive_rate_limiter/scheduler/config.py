# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Scheduler Configuration for Adaptive Rate Limiter

This module provides configuration classes for the rate limiter scheduler,
including rate limiting, metrics, and backoff settings.
"""

from dataclasses import dataclass
from enum import Enum


class SchedulerMode(Enum):
    """Scheduling mode that determines request handling behavior.

    - BASIC: Simple pass-through mode with minimal overhead. Use for low-volume
      or testing scenarios where adaptive behavior is not needed.
    - INTELLIGENT: Full adaptive scheduling with capacity reservation, queue
      management, and streaming support. Use for production workloads.
    - ACCOUNT: Multi-tenant mode with per-account rate limit tracking. Use
      when managing multiple API accounts through a single scheduler.
    """

    BASIC = "basic"
    INTELLIGENT = "intelligent"
    ACCOUNT = "account"


class CachePolicy(Enum):
    """
    Cache write policies for state management.

    WRITE_THROUGH:
        - Writes to both cache and backend storage simultaneously
        - Guarantees data durability - no data loss on crashes
        - Best for: Production systems, critical data

    WRITE_BACK:
        - Writes to cache immediately, backend writes are batched/delayed
        - Fastest write performance but risk of data loss on crashes
        - Best for: Development, testing

    WRITE_AROUND:
        - Writes directly to backend, bypassing cache
        - Best for: Large bulk writes, write-heavy workloads
    """

    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


@dataclass
class RateLimiterConfig:
    """
    Configuration for the rate limiter scheduler.

    This is a generic configuration that works with any provider.
    """

    # === Core Scheduling Configuration ===

    mode: SchedulerMode = SchedulerMode.INTELLIGENT
    """Scheduler operation mode."""

    max_concurrent_executions: int = 100
    """Maximum number of concurrent executions."""

    max_queue_size: int = 1000
    """Maximum size of each bucket's request queue."""

    overflow_policy: str = "reject"
    """Policy when queue is full: 'reject' or 'drop_oldest'."""

    scheduler_interval: float = 0.01
    """Interval between scheduler loops in seconds."""

    # === Request Processing ===

    request_timeout: float = 30.0
    """Request timeout duration in seconds."""

    enable_priority_scheduling: bool = True
    """Enable priority-based request scheduling."""

    # === Rate Limiting Integration ===

    enable_rate_limiting: bool = True
    """Enable rate limiting enforcement."""

    rate_limit_buffer_ratio: float = 0.9
    """Ratio of rate limit to use as buffer (0.9 = use 90% of limit)."""

    # === Failure Handling ===

    failure_window: float = 30.0
    """Time window for failure counting in seconds."""

    max_failures: int = 20
    """Maximum failures within window before circuit break."""

    backoff_base: float = 2.0
    """Base for exponential backoff calculation."""

    max_backoff: float = 60.0
    """Maximum backoff time in seconds."""

    # === Graceful Degradation ===

    enable_graceful_degradation: bool = True
    """Enable graceful degradation on failures."""

    health_check_interval: float = 30.0
    """Interval between health checks in seconds."""

    max_consecutive_failures: int = 3
    """Maximum consecutive failures before degradation."""

    conservative_multiplier: float = 0.6
    """Multiplier for conservative capacity during degradation."""

    # === Metrics and Monitoring ===

    metrics_enabled: bool = True
    """Enable metrics collection."""

    prometheus_host: str = "0.0.0.0"  # noqa: S104  # nosec B104  # Prometheus metrics server needs to bind to all interfaces for monitoring access
    """Prometheus metrics host."""

    prometheus_port: int = 9090
    """Prometheus metrics port."""

    metrics_export_interval: float = 60.0
    """Interval for metrics export in seconds."""

    enable_performance_tracking: bool = True
    """Enable detailed performance tracking."""

    # === State Management ===

    enable_state_persistence: bool = True
    """Enable state persistence to backend."""

    # === Testing Support ===

    test_mode: bool = False
    """Enable test mode with relaxed constraints."""

    test_rate_multiplier: float = 1.0
    """Rate limit multiplier for testing."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 < self.rate_limit_buffer_ratio <= 1.0:
            raise ValueError("rate_limit_buffer_ratio must be between 0 and 1.0")
        if not 0 < self.conservative_multiplier <= 1.0:
            raise ValueError("conservative_multiplier must be between 0 and 1.0")
        if self.max_concurrent_executions < 1:
            raise ValueError("max_concurrent_executions must be at least 1")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if self.overflow_policy not in ("reject", "drop_oldest"):
            raise ValueError("overflow_policy must be 'reject' or 'drop_oldest'")


@dataclass
class StateConfig:
    """
    Configuration for state management within the scheduler.

    Controls caching, TTL, cleanup, and persistence behavior.
    """

    # Cache configuration
    cache_ttl: float = 1.0
    """Cache TTL in seconds."""

    max_cache_size: int | None = 1000
    """Maximum number of cache entries before LRU eviction."""

    cache_policy: CachePolicy = CachePolicy.WRITE_THROUGH
    """Cache write policy. Defaults to WRITE_THROUGH for production safety.

    WRITE_BACK offers better performance but risks data loss on crashes.
    Only use WRITE_BACK in development/testing or when data loss is acceptable.
    """

    # Production safety
    warn_write_back_production: bool = True
    """Emit warning when WRITE_BACK is used in production."""

    is_production: bool = False
    """Set to True in production for safety checks."""

    acknowledge_write_back_risk: bool = False
    """Explicitly acknowledge WRITE_BACK risk in production.

    When is_production=True and cache_policy=WRITE_BACK, you must set this to True
    to acknowledge that data loss may occur on unexpected shutdown. Otherwise, a
    ValueError will be raised during StateManager initialization.
    """

    # Batch processing
    batch_size: int = 50
    """Batch size for backend writes."""

    batch_timeout: float = 0.1
    """Batch timeout in seconds."""

    # Flush retry settings
    flush_max_retries: int = 5
    """Maximum retry attempts for flushing pending updates before dropping.

    When backend writes fail, updates are re-queued with a retry counter.
    After this many attempts, updates are dropped and logged as errors.
    """

    flush_backoff_base: float = 1.0
    """Base delay in seconds for exponential backoff between flush retries."""

    flush_backoff_max: float = 60.0
    """Maximum delay in seconds for exponential backoff between flush retries."""

    # Cleanup and maintenance
    cleanup_interval: float = 30.0
    """Cleanup interval in seconds."""

    enable_background_cleanup: bool = True
    """Enable automatic background cleanup."""

    cleanup_task_cancel_timeout: float = 2.0
    """Timeout for cancelling cleanup tasks."""

    cleanup_task_wait_timeout: float = 1.0
    """Timeout for waiting on cleanup task cancellation."""

    # State persistence TTL
    state_ttl: int = 3600
    """TTL for state keys in backend storage (seconds)."""

    # Reservation cleanup
    reservation_cleanup_interval: float = 3600.0
    """Interval for cleaning up expired reservations in seconds."""

    reservation_ttl: float = 300.0
    """TTL for reservation entries in seconds."""

    # Account state cleanup
    account_state_ttl: float = 86400.0
    """TTL for account state entries in seconds."""

    account_state_max_size: int | None = 10000
    """Maximum number of account state entries."""

    # Versioning and recovery
    enable_versioning: bool = True
    """Enable state versioning for recovery."""

    max_versions: int = 10
    """Maximum versions to keep per state entry."""

    # Concurrency control
    lock_free_reads: bool = True
    """Enable lock-free read operations for better performance."""

    max_concurrent_operations: int = 100
    """Maximum concurrent state operations."""

    # Namespace isolation
    namespace: str = "default"
    """State namespace for multi-tenant isolation."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        if self.max_cache_size is not None and self.max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.state_ttl < 60:
            raise ValueError("state_ttl must be at least 60 seconds")


__all__ = [
    "CachePolicy",
    "RateLimiterConfig",
    "SchedulerMode",
    "StateConfig",
]

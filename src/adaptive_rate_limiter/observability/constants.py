# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Metric name constants following Prometheus naming conventions.

This module provides standardized metric names for all observability
in the adaptive-rate-limiter library. All metric names use the
`adaptive_rl_` prefix for Prometheus compatibility.

Naming Conventions:
    - Counter metrics end with `_total`
    - Histogram metrics for time end with `_seconds`
    - Gauges use present-tense descriptive names

Label Best Practices:
    To prevent label cardinality explosion, use only:
    - `bucket_id` - Rate limit bucket (categorical: tier:1, tier:2)
    - `model_id` - Model name (categorical: gpt-6, claude-haiku-4.5, venice-uncensored)
    - `reason` - Failure reason (enum: timeout, rate_limit, error)
    - `extraction_succeeded` - Boolean as string (true, false)

    NEVER use:
    - `request_id` - Unique per request (unbounded!)
    - `user_id` - Unique per user (unbounded!)
    - `timestamp` - Unique per second (unbounded!)

Usage:
    >>> from adaptive_rate_limiter.observability.constants import (
    ...     REQUESTS_SCHEDULED_TOTAL, METRIC_PREFIX
    ... )
    >>> print(REQUESTS_SCHEDULED_TOTAL)
    'adaptive_rl_requests_scheduled_total'
"""


# =============================================================================
# Global Prefix
# =============================================================================

METRIC_PREFIX = "adaptive_rl"
"""Prefix for all Prometheus metrics in this library."""


# =============================================================================
# Scheduling Metrics (scheduler/base.py, scheduler/scheduler.py)
# =============================================================================

REQUESTS_SCHEDULED_TOTAL = f"{METRIC_PREFIX}_requests_scheduled_total"
"""Total requests scheduled for execution."""

REQUESTS_COMPLETED_TOTAL = f"{METRIC_PREFIX}_requests_completed_total"
"""Total requests completed successfully."""

REQUESTS_FAILED_TOTAL = f"{METRIC_PREFIX}_requests_failed_total"
"""Total requests that failed (timeout, error, rate limit)."""

QUEUE_OVERFLOWS_TOTAL = f"{METRIC_PREFIX}_queue_overflows_total"
"""Total queue overflow events (request rejected due to full queue)."""

SCHEDULER_LOOPS_TOTAL = f"{METRIC_PREFIX}_scheduler_loops_total"
"""Total iterations of the scheduler main loop."""

CIRCUIT_BREAKER_REJECTIONS_TOTAL = f"{METRIC_PREFIX}_circuit_breaker_rejections_total"
"""Total requests rejected by circuit breaker."""

REQUEST_TIMEOUTS_TOTAL = f"{METRIC_PREFIX}_request_timeouts_total"
"""Total requests that timed out."""


# =============================================================================
# Active State Gauges (real-time operational state)
# =============================================================================

ACTIVE_REQUESTS = f"{METRIC_PREFIX}_active_requests"
"""Number of currently active (in-flight) requests."""

QUEUE_DEPTH = f"{METRIC_PREFIX}_queue_depth"
"""Current queue depth (requests waiting to be scheduled)."""

RESERVATIONS_ACTIVE = f"{METRIC_PREFIX}_reservations_active"
"""Number of currently active reservations."""


# =============================================================================
# Streaming Metrics (strategies/modes/streaming_handler.py)
# =============================================================================

STREAMING_COMPLETIONS_TOTAL = f"{METRIC_PREFIX}_streaming_completions_total"
"""Total streaming requests completed."""

STREAMING_ERRORS_TOTAL = f"{METRIC_PREFIX}_streaming_errors_total"
"""Total streaming requests that errored."""

STREAMING_TIMEOUTS_TOTAL = f"{METRIC_PREFIX}_streaming_timeouts_total"
"""Total streaming requests that timed out."""

STREAMING_STALE_CLEANUPS_TOTAL = f"{METRIC_PREFIX}_streaming_stale_cleanups_total"
"""Total stale streaming entries cleaned up by background task."""

STREAMING_TOKENS_REFUNDED_TOTAL = f"{METRIC_PREFIX}_streaming_tokens_refunded_total"
"""Total tokens refunded from streaming requests (reserved - actual)."""

STREAMING_DURATION_SECONDS = f"{METRIC_PREFIX}_streaming_duration_seconds"
"""Duration of streaming requests (histogram)."""


# =============================================================================
# State/Cache Metrics (scheduler/state.py)
# =============================================================================

CACHE_HITS_TOTAL = f"{METRIC_PREFIX}_cache_hits_total"
"""Total cache hits (state found in local cache)."""

CACHE_MISSES_TOTAL = f"{METRIC_PREFIX}_cache_misses_total"
"""Total cache misses (state not found or expired)."""

CACHE_EVICTIONS_TOTAL = f"{METRIC_PREFIX}_cache_evictions_total"
"""Total cache evictions (entries removed due to size limit)."""

BACKEND_WRITES_TOTAL = f"{METRIC_PREFIX}_backend_writes_total"
"""Total writes to the backend (Redis, memory, etc.)."""

BACKEND_READS_TOTAL = f"{METRIC_PREFIX}_backend_reads_total"
"""Total reads from the backend."""

VERSION_CONFLICTS_TOTAL = f"{METRIC_PREFIX}_version_conflicts_total"
"""Total version conflicts during optimistic locking updates."""


# =============================================================================
# Reservation Metrics (strategies/modes/intelligent.py)
# =============================================================================

RESERVATION_STALE_CLEANUPS_TOTAL = f"{METRIC_PREFIX}_reservation_stale_cleanups_total"
"""Total stale reservations cleaned up."""

RESERVATION_EMERGENCY_CLEANUPS_TOTAL = (
    f"{METRIC_PREFIX}_reservation_emergency_cleanups_total"
)
"""Total emergency reservation cleanups (high memory pressure)."""

RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL = (
    f"{METRIC_PREFIX}_reservation_backpressure_total"
)
"""Total reservations rejected due to backpressure."""


# =============================================================================
# Backend Metrics (backends/redis.py)
# =============================================================================

BACKEND_LUA_EXECUTIONS_TOTAL = f"{METRIC_PREFIX}_backend_lua_executions_total"
"""Total Lua script executions (Redis backend)."""

BACKEND_CONNECTION_ERRORS_TOTAL = f"{METRIC_PREFIX}_backend_connection_errors_total"
"""Total backend connection errors."""

BACKEND_LATENCY_SECONDS = f"{METRIC_PREFIX}_backend_latency_seconds"
"""Backend operation latency (histogram)."""


# =============================================================================
# Histogram Buckets
# =============================================================================

LATENCY_BUCKETS: list[float] = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
]
"""Default latency buckets for request duration histograms (in seconds)."""

STREAMING_DURATION_BUCKETS: list[float] = [
    1.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
]
"""Streaming duration buckets (in seconds, up to 20 minutes)."""

TOKEN_BUCKETS: list[float] = [
    100.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
    32000.0,
    64000.0,
    128000.0,
]
"""Token count buckets for token-related histograms."""


# =============================================================================
# Legacy Metric Name Mapping
# =============================================================================

LEGACY_METRIC_MAPPING = {
    "requests_scheduled": REQUESTS_SCHEDULED_TOTAL,
    "requests_completed": REQUESTS_COMPLETED_TOTAL,
    "requests_failed": REQUESTS_FAILED_TOTAL,
    "queue_overflows": QUEUE_OVERFLOWS_TOTAL,
    "scheduler_loops": SCHEDULER_LOOPS_TOTAL,
    "circuit_breaker_rejections": CIRCUIT_BREAKER_REJECTIONS_TOTAL,
    "request_timeouts": REQUEST_TIMEOUTS_TOTAL,
    "streaming_completions": STREAMING_COMPLETIONS_TOTAL,
    "streaming_errors": STREAMING_ERRORS_TOTAL,
    "streaming_timeouts": STREAMING_TIMEOUTS_TOTAL,
    "cache_hits": CACHE_HITS_TOTAL,
    "cache_misses": CACHE_MISSES_TOTAL,
}
"""Mapping from legacy metric names (dict-based) to Prometheus-style names."""


def legacy_to_prometheus_name(legacy_name: str) -> str:
    """
    Convert a legacy metric name to Prometheus-style name.

    Args:
        legacy_name: Legacy metric name (e.g., "requests_scheduled")

    Returns:
        Prometheus-style name (e.g., "adaptive_rl_requests_scheduled_total")

    Example:
        >>> legacy_to_prometheus_name("requests_completed")
        'adaptive_rl_requests_completed_total'
    """
    return LEGACY_METRIC_MAPPING.get(
        legacy_name, f"{METRIC_PREFIX}_{legacy_name}_total"
    )


__all__ = [
    # Gauges
    "ACTIVE_REQUESTS",
    "BACKEND_CONNECTION_ERRORS_TOTAL",
    "BACKEND_LATENCY_SECONDS",
    # Backend
    "BACKEND_LUA_EXECUTIONS_TOTAL",
    "BACKEND_READS_TOTAL",
    "BACKEND_WRITES_TOTAL",
    "CACHE_EVICTIONS_TOTAL",
    # Cache/State
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "CIRCUIT_BREAKER_REJECTIONS_TOTAL",
    # Buckets
    "LATENCY_BUCKETS",
    # Mapping
    "LEGACY_METRIC_MAPPING",
    # Prefix
    "METRIC_PREFIX",
    "QUEUE_DEPTH",
    "QUEUE_OVERFLOWS_TOTAL",
    "REQUESTS_COMPLETED_TOTAL",
    "REQUESTS_FAILED_TOTAL",
    # Scheduling
    "REQUESTS_SCHEDULED_TOTAL",
    "REQUEST_TIMEOUTS_TOTAL",
    "RESERVATIONS_ACTIVE",
    "RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL",
    "RESERVATION_EMERGENCY_CLEANUPS_TOTAL",
    # Reservation
    "RESERVATION_STALE_CLEANUPS_TOTAL",
    "SCHEDULER_LOOPS_TOTAL",
    # Streaming
    "STREAMING_COMPLETIONS_TOTAL",
    "STREAMING_DURATION_BUCKETS",
    "STREAMING_DURATION_SECONDS",
    "STREAMING_ERRORS_TOTAL",
    "STREAMING_STALE_CLEANUPS_TOTAL",
    "STREAMING_TIMEOUTS_TOTAL",
    "STREAMING_TOKENS_REFUNDED_TOTAL",
    "TOKEN_BUCKETS",
    "VERSION_CONFLICTS_TOTAL",
    "legacy_to_prometheus_name",
]

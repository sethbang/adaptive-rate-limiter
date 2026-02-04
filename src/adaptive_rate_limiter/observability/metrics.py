# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming metrics for the Adaptive Rate Limiter.

This module provides:
1. StreamingMetrics - Dataclass for tracking streaming-specific rate limit metrics
2. PrometheusStreamingMetrics - Optional Prometheus-style metrics for observability

The StreamingMetrics class provides observability into:
- Token refunds (reserved vs actual)
- Extraction success/failure rates
- Stale streaming cleanups
- Stream durations and completion patterns

Usage:
    metrics = StreamingMetrics()

    # Record a successful completion with token extraction
    metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)

    # Record an error
    metrics.record_error()

    # Record a stale cleanup
    metrics.record_stale_cleanup()

    # Get stats for JSON serialization
    stats = metrics.get_stats()

Important Notes on Bucket IDs:
    The per-bucket tracking dictionaries (_per_bucket_refunds, _per_bucket_cleanups)
    are designed for CATEGORICAL bucket IDs (e.g., "bucket:xs", "bucket:sm", "bucket:md").

    DO NOT use dynamic or unbounded identifiers (e.g., request IDs, user IDs, timestamps)
    as bucket IDs, as this will cause unbounded dictionary growth and memory leaks.

    If you have many buckets and are concerned about memory usage, you can configure
    max_tracked_buckets to enable LRU eviction of least-recently-used bucket entries.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Default maximum tracked buckets (0 = unlimited, for backward compatibility)
DEFAULT_MAX_TRACKED_BUCKETS = 0

# Type hint for the metrics callback
StreamingMetricsCallback = Callable[[int, int, bool, str | None, float | None], None]

# Type declarations for optional prometheus_client imports
if TYPE_CHECKING:
    from prometheus_client import Counter as CounterType, Histogram as HistogramType
else:
    CounterType = object
    HistogramType = object

# Try to import prometheus_client for optional Prometheus metrics
try:
    from prometheus_client import Counter as _Counter, Histogram as _Histogram

    Counter: type[CounterType] | None = _Counter
    Histogram: type[HistogramType] | None = _Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None
    Histogram = None
    PROMETHEUS_AVAILABLE = False


@dataclass
class StreamingMetrics:
    """
    Streaming-specific rate limit metrics.

    This dataclass provides comprehensive observability for streaming
    support, tracking:

    - Stream lifecycle (completions, timeouts, errors, fallbacks)
    - Token accounting (reserved, actual, refunded)
    - Extraction success rate
    - Background cleanup metrics

    Thread Safety:
        Simple counter increments use Python's GIL for atomicity.
        Per-bucket dictionary updates use a threading.Lock to ensure atomicity
        of the get + increment + set pattern in multi-threaded environments.

    Per-Bucket Tracking Warning:
        The per-bucket dictionaries (_per_bucket_refunds, _per_bucket_cleanups)
        are designed for CATEGORICAL bucket IDs (e.g., "bucket:xs", "bucket:sm").
        DO NOT use dynamic IDs (request IDs, user IDs) as this causes memory leaks.
        Set max_tracked_buckets > 0 to enable LRU eviction for many buckets.

    Example:
        >>> metrics = StreamingMetrics()
        >>> metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        >>> print(metrics.get_extraction_success_rate())
        1.0
        >>> print(metrics.total_refunded_tokens)
        3500
    """

    # Stream lifecycle counters
    streaming_completions: int = 0
    streaming_timeouts: int = 0
    streaming_errors: int = 0
    streaming_fallbacks: int = 0  # Token extraction failed, used reserved as actual

    # Token accounting
    total_reserved_tokens: int = 0
    total_actual_tokens: int = 0
    total_refunded_tokens: int = 0

    # Token extraction tracking
    extraction_successes: int = 0
    extraction_failures: int = 0

    # Background cleanup tracking
    stale_streaming_cleanups: int = 0

    # Per-bucket counters (optional detailed tracking)
    # Using OrderedDict for LRU eviction support when max_tracked_buckets > 0
    _per_bucket_refunds: OrderedDict[str, int] = field(
        default_factory=OrderedDict, repr=False
    )
    _per_bucket_cleanups: OrderedDict[str, int] = field(
        default_factory=OrderedDict, repr=False
    )

    # Maximum number of buckets to track (0 = unlimited, for backward compatibility)
    max_tracked_buckets: int = field(default=DEFAULT_MAX_TRACKED_BUCKETS, repr=False)

    # Lock for thread-safe per-bucket updates
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_extraction_success_rate(self) -> float:
        """
        Calculate the token extraction success rate.

        Returns:
            A float between 0.0 and 1.0 representing the proportion of
            successful token extractions. Returns 1.0 if no extractions
            have been attempted (optimistic default).

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.extraction_successes = 9
            >>> metrics.extraction_failures = 1
            >>> metrics.get_extraction_success_rate()
            0.9
        """
        total = self.extraction_successes + self.extraction_failures
        return self.extraction_successes / total if total > 0 else 1.0

    def get_refund_rate(self) -> float:
        """
        Calculate the average token refund rate.

        Returns:
            A float between 0.0 and 1.0 representing the average proportion
            of reserved tokens that were refunded. Returns 0.0 if no tokens
            have been reserved.

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.total_reserved_tokens = 4000
            >>> metrics.total_refunded_tokens = 3500
            >>> metrics.get_refund_rate()
            0.875
        """
        if self.total_reserved_tokens == 0:
            return 0.0
        return self.total_refunded_tokens / self.total_reserved_tokens

    def record_completion(
        self,
        reserved: int,
        actual: int,
        extraction_succeeded: bool,
        bucket_id: str | None = None,
    ) -> None:
        """
        Record a streaming completion.

        This method should be called when a stream completes successfully
        (iteration exhausted). It updates:
        - streaming_completions counter
        - Token accounting (reserved, actual, refunded)
        - Extraction success/failure counter
        - Per-bucket refund tracking (if bucket_id provided)

        Args:
            reserved: Number of tokens that were reserved at request start
            actual: Actual tokens consumed (from usage field or fallback)
            extraction_succeeded: True if tokens were extracted from usage field,
                                  False if fallback (reserved=actual) was used
            bucket_id: Optional bucket identifier for per-bucket tracking.
                       WARNING: Use categorical IDs only (e.g., "bucket:xs"),
                       NOT dynamic IDs (request IDs, timestamps).

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
            >>> print(metrics.streaming_completions)
            1
            >>> print(metrics.total_refunded_tokens)
            3500
        """
        self.streaming_completions += 1
        self.total_reserved_tokens += reserved
        self.total_actual_tokens += actual

        refund = reserved - actual
        self.total_refunded_tokens += refund

        if extraction_succeeded:
            self.extraction_successes += 1
        else:
            self.extraction_failures += 1
            self.streaming_fallbacks += 1

        # Per-bucket tracking with thread-safe update
        if bucket_id:
            self._update_per_bucket_counter(self._per_bucket_refunds, bucket_id, refund)

    def record_error(self) -> None:
        """
        Record a streaming error.

        This method should be called when a stream errors out during
        iteration (not timeout, but an exception during chunk processing).

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_error()
            >>> print(metrics.streaming_errors)
            1
        """
        self.streaming_errors += 1

    def record_timeout(self) -> None:
        """
        Record a streaming timeout.

        This method should be called when a stream times out waiting for
        the next chunk from the server.

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_timeout()
            >>> print(metrics.streaming_timeouts)
            1
        """
        self.streaming_timeouts += 1

    def record_stale_cleanup(self, bucket_id: str | None = None) -> None:
        """
        Record a stale streaming entry cleanup.

        This method should be called by the background cleanup task when
        it releases a stale streaming entry (inactive for > 5 minutes).

        Args:
            bucket_id: Optional bucket identifier for per-bucket tracking.
                       WARNING: Use categorical IDs only (e.g., "bucket:xs"),
                       NOT dynamic IDs (request IDs, timestamps).

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_stale_cleanup(bucket_id="bucket:xs")
            >>> print(metrics.stale_streaming_cleanups)
            1
        """
        self.stale_streaming_cleanups += 1

        # Per-bucket tracking with thread-safe update
        if bucket_id:
            self._update_per_bucket_counter(self._per_bucket_cleanups, bucket_id, 1)

    def record_fallback(self) -> None:
        """
        Record a fallback event where token extraction failed.

        This is called when a stream completes but we couldn't extract
        the actual token count, so we use reserved tokens (refund=0).

        Note: This is also tracked via record_completion() with
        extraction_succeeded=False, but this method can be used
        for explicit fallback tracking separate from completion.

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_fallback()
            >>> print(metrics.streaming_fallbacks)
            1
        """
        self.streaming_fallbacks += 1
        self.extraction_failures += 1

    def get_stats(self) -> dict[str, Any]:
        """
        Return metrics as a dictionary for JSON serialization.

        This method creates a snapshot of all metrics suitable for
        logging, monitoring, or API responses. All values are
        JSON-serializable basic types (int, float, dict).

        Returns:
            Dictionary containing all metrics with descriptive keys.

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_completion(4000, 500, True)
            >>> stats = metrics.get_stats()
            >>> print(stats["streaming_completions"])
            1
            >>> print(stats["extraction_success_rate"])
            1.0
        """
        return {
            # Lifecycle counters
            "streaming_completions": self.streaming_completions,
            "streaming_timeouts": self.streaming_timeouts,
            "streaming_errors": self.streaming_errors,
            "streaming_fallbacks": self.streaming_fallbacks,
            # Token accounting
            "total_reserved_tokens": self.total_reserved_tokens,
            "total_actual_tokens": self.total_actual_tokens,
            "total_refunded_tokens": self.total_refunded_tokens,
            # Derived metrics
            "extraction_success_rate": self.get_extraction_success_rate(),
            "refund_rate": self.get_refund_rate(),
            # Extraction tracking
            "extraction_successes": self.extraction_successes,
            "extraction_failures": self.extraction_failures,
            # Cleanup tracking
            "stale_streaming_cleanups": self.stale_streaming_cleanups,
            # Per-bucket breakdowns (only if there's data)
            "per_bucket_refunds": (
                dict(self._per_bucket_refunds) if self._per_bucket_refunds else {}
            ),
            "per_bucket_cleanups": (
                dict(self._per_bucket_cleanups) if self._per_bucket_cleanups else {}
            ),
        }

    def reset(self) -> None:
        """
        Reset all metrics to zero.

        Useful for testing or periodic metric collection where you want
        to track deltas between collection intervals.

        Example:
            >>> metrics = StreamingMetrics()
            >>> metrics.record_completion(4000, 500, True)
            >>> metrics.reset()
            >>> print(metrics.streaming_completions)
            0
        """
        self.streaming_completions = 0
        self.streaming_timeouts = 0
        self.streaming_errors = 0
        self.streaming_fallbacks = 0

        self.total_reserved_tokens = 0
        self.total_actual_tokens = 0
        self.total_refunded_tokens = 0

        self.extraction_successes = 0
        self.extraction_failures = 0

        self.stale_streaming_cleanups = 0

        with self._lock:
            self._per_bucket_refunds.clear()
            self._per_bucket_cleanups.clear()

    def _update_per_bucket_counter(
        self,
        counter_dict: OrderedDict[str, int],
        bucket_id: str,
        increment: int,
    ) -> None:
        """
        Thread-safe update of a per-bucket counter with LRU eviction.

        This method provides:
        1. Thread-safe get + increment + set using a lock
        2. LRU tracking by moving accessed buckets to the end
        3. Automatic eviction of oldest buckets when max_tracked_buckets is exceeded

        Args:
            counter_dict: The OrderedDict to update (_per_bucket_refunds or _per_bucket_cleanups)
            bucket_id: The bucket identifier (should be categorical, not dynamic)
            increment: The value to add to the counter

        Warning:
            Do NOT use dynamic bucket IDs (request IDs, user IDs, timestamps).
            Use categorical IDs only (e.g., "bucket:xs", "bucket:sm", "bucket:md").
        """
        with self._lock:
            # Get current value or 0
            current = counter_dict.get(bucket_id, 0)

            # Update the counter
            counter_dict[bucket_id] = current + increment

            # Move to end to mark as recently used (for LRU)
            counter_dict.move_to_end(bucket_id)

            # LRU eviction if max_tracked_buckets is set
            if self.max_tracked_buckets > 0:
                while len(counter_dict) > self.max_tracked_buckets:
                    # Remove oldest (first) entry
                    oldest_key, _ = counter_dict.popitem(last=False)
                    logger.debug(f"LRU evicted per-bucket metrics for: {oldest_key}")


class PrometheusStreamingMetrics:
    """
    Optional Prometheus-style metrics for streaming observability.

    This class provides Prometheus Counter and Histogram metrics for
    streaming requests. It integrates with the standard prometheus_client
    library usage pattern.

    Only instantiated if prometheus_client is available.

    Metrics:
        - adaptive_rl_streaming_tokens_refunded_total: Counter of refunded tokens
        - adaptive_rl_streaming_extraction_failures_total: Counter of extraction failures
        - adaptive_rl_streaming_stale_cleanups_total: Counter of stale cleanups
        - adaptive_rl_streaming_duration_seconds: Histogram of stream durations
        - adaptive_rl_streaming_completions_total: Counter of stream completions

    Usage:
        >>> if PROMETHEUS_AVAILABLE:
        ...     prom_metrics = PrometheusStreamingMetrics()
        ...     prom_metrics.observe_completion("bucket:xs", 4000, 500, 15.5)
    """

    def __init__(self, registry: Any | None = None) -> None:
        """
        Initialize Prometheus streaming metrics.

        Args:
            registry: Optional CollectorRegistry. If None, uses the default registry.

        Raises:
            ImportError: If prometheus_client is not available.
        """
        if not PROMETHEUS_AVAILABLE or Counter is None or Histogram is None:
            raise ImportError(
                "prometheus_client is not available. "
                "Install with: pip install prometheus-client"
            )

        # Token refund tracking
        self.streaming_tokens_refunded = Counter(
            "adaptive_rl_streaming_tokens_refunded_total",
            "Total tokens refunded by streaming requests",
            ["bucket_id"],
            registry=registry,
        )

        # Extraction failure tracking
        self.streaming_extraction_failures = Counter(
            "adaptive_rl_streaming_extraction_failures_total",
            "Streams where token extraction failed",
            ["reason"],  # Values: complete, timeout, error
            registry=registry,
        )

        # Stale cleanup tracking
        self.streaming_stale_cleanups = Counter(
            "adaptive_rl_streaming_stale_cleanups_total",
            "Stale streaming entries cleaned up",
            ["bucket_id"],
            registry=registry,
        )

        # Stream duration histogram
        self.streaming_duration_seconds = Histogram(
            "adaptive_rl_streaming_duration_seconds",
            "Duration of streaming requests",
            ["bucket_id"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],  # Up to 20 minutes
            registry=registry,
        )

        # Stream completion counter
        self.streaming_completions = Counter(
            "adaptive_rl_streaming_completions_total",
            "Total number of streaming completions",
            ["bucket_id", "extraction_succeeded"],
            registry=registry,
        )

        # Stream error counter
        self.streaming_errors = Counter(
            "adaptive_rl_streaming_errors_total",
            "Total number of streaming errors",
            ["bucket_id", "error_type"],  # Values: timeout, exception, unknown
            registry=registry,
        )

        logger.info("Prometheus streaming metrics initialized")

    def observe_completion(
        self,
        bucket_id: str,
        reserved: int,
        actual: int,
        duration_seconds: float,
        extraction_succeeded: bool = True,
    ) -> None:
        """
        Observe a streaming completion.

        Args:
            bucket_id: The rate limit bucket identifier
            reserved: Tokens reserved at request start
            actual: Actual tokens consumed
            duration_seconds: Stream duration in seconds
            extraction_succeeded: Whether token extraction succeeded
        """
        refund = reserved - actual

        self.streaming_tokens_refunded.labels(bucket_id=bucket_id).inc(refund)
        self.streaming_duration_seconds.labels(bucket_id=bucket_id).observe(
            duration_seconds
        )
        self.streaming_completions.labels(
            bucket_id=bucket_id,
            extraction_succeeded=str(extraction_succeeded).lower(),
        ).inc()

        if not extraction_succeeded:
            self.streaming_extraction_failures.labels(reason="complete").inc()

    def observe_error(
        self,
        bucket_id: str,
        error_type: str = "exception",
    ) -> None:
        """
        Observe a streaming error.

        Args:
            bucket_id: The rate limit bucket identifier
            error_type: Type of error (timeout, exception, unknown)
        """
        self.streaming_errors.labels(bucket_id=bucket_id, error_type=error_type).inc()
        self.streaming_extraction_failures.labels(reason=error_type).inc()

    def observe_stale_cleanup(self, bucket_id: str) -> None:
        """
        Observe a stale streaming cleanup.

        Args:
            bucket_id: The rate limit bucket identifier
        """
        self.streaming_stale_cleanups.labels(bucket_id=bucket_id).inc()


# Module-level singleton for Prometheus metrics (optional)
_prometheus_streaming_metrics: PrometheusStreamingMetrics | None = None
_prometheus_lock = threading.Lock()


def get_prometheus_streaming_metrics() -> PrometheusStreamingMetrics | None:
    """
    Get or create the Prometheus streaming metrics singleton.

    Thread-safe singleton initialization using double-checked locking pattern
    to prevent race conditions that could cause prometheus_client duplicate
    registration errors.

    Returns:
        PrometheusStreamingMetrics instance if prometheus_client is available,
        None otherwise.

    Example:
        >>> prom_metrics = get_prometheus_streaming_metrics()
        >>> if prom_metrics:
        ...     prom_metrics.observe_completion("bucket:xs", 4000, 500, 15.5)
    """
    global _prometheus_streaming_metrics

    if not PROMETHEUS_AVAILABLE:
        return None

    # Double-checked locking pattern for thread-safe singleton initialization
    if _prometheus_streaming_metrics is None:
        with _prometheus_lock:
            # Second check inside lock to prevent race conditions
            if _prometheus_streaming_metrics is None:
                try:
                    _prometheus_streaming_metrics = PrometheusStreamingMetrics()
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Prometheus streaming metrics: {e}"
                    )
                    return None

    return _prometheus_streaming_metrics


def reset_prometheus_streaming_metrics() -> None:
    """Reset the Prometheus streaming metrics singleton (mainly for testing)."""
    global _prometheus_streaming_metrics
    _prometheus_streaming_metrics = None


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Prometheus support (optional)
    "PrometheusStreamingMetrics",
    # Main classes
    "StreamingMetrics",
    # Type hints
    "StreamingMetricsCallback",
    "get_prometheus_streaming_metrics",
    "reset_prometheus_streaming_metrics",
]

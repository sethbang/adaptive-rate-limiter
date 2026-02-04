# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Unified metrics collector supporting both dict-based and Prometheus metrics.

This module provides the UnifiedMetricsCollector class that serves as the
single source of truth for all metrics in the adaptive-rate-limiter library.

Features:
    1. Thread-safe counter/gauge/histogram operations
    2. Automatic Prometheus metric registration when available
    3. Dict-based fallback for JSON export
    4. Label cardinality protection (max 1000 unique combinations per metric)
    5. Optional HTTP server for Prometheus scraping

Usage:
    >>> from adaptive_rate_limiter.observability.collector import get_metrics_collector
    >>> collector = get_metrics_collector()
    >>> collector.inc_counter('adaptive_rl_requests_scheduled_total',
    ...                       labels={'bucket_id': 'tier:1'})
    >>> metrics = collector.get_metrics()

Thread Safety:
    All operations are thread-safe. Uses RLock for reentrant locking.

Prometheus Integration:
    When prometheus_client is installed, metrics are automatically registered
    with the default Prometheus registry. The HTTP server can be started
    with collector.start_http_server().
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

from .constants import (
    # Gauge metrics
    ACTIVE_REQUESTS,
    BACKEND_CONNECTION_ERRORS_TOTAL,
    BACKEND_LATENCY_SECONDS,
    # Backend metrics
    BACKEND_LUA_EXECUTIONS_TOTAL,
    BACKEND_READS_TOTAL,
    BACKEND_WRITES_TOTAL,
    CACHE_EVICTIONS_TOTAL,
    # Cache metrics
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    CIRCUIT_BREAKER_REJECTIONS_TOTAL,
    LATENCY_BUCKETS,
    QUEUE_DEPTH,
    QUEUE_OVERFLOWS_TOTAL,
    REQUEST_TIMEOUTS_TOTAL,
    REQUESTS_COMPLETED_TOTAL,
    REQUESTS_FAILED_TOTAL,
    # Scheduling metrics
    REQUESTS_SCHEDULED_TOTAL,
    RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL,
    RESERVATION_EMERGENCY_CLEANUPS_TOTAL,
    # Reservation metrics
    RESERVATION_STALE_CLEANUPS_TOTAL,
    RESERVATIONS_ACTIVE,
    SCHEDULER_LOOPS_TOTAL,
    # Streaming metrics
    STREAMING_COMPLETIONS_TOTAL,
    STREAMING_DURATION_BUCKETS,
    STREAMING_DURATION_SECONDS,
    STREAMING_ERRORS_TOTAL,
    STREAMING_STALE_CLEANUPS_TOTAL,
    STREAMING_TIMEOUTS_TOTAL,
    STREAMING_TOKENS_REFUNDED_TOTAL,
    VERSION_CONFLICTS_TOTAL,
)

logger = logging.getLogger(__name__)

# Type declarations for optional prometheus_client imports
if TYPE_CHECKING:
    from prometheus_client import (
        CollectorRegistry as CollectorRegistryType,
        Counter as CounterType,
        Gauge as GaugeType,
        Histogram as HistogramType,
    )
else:
    CounterType = object
    GaugeType = object
    HistogramType = object
    CollectorRegistryType = object

# Check Prometheus availability with aliased imports to avoid no-redef
try:
    from prometheus_client import (
        REGISTRY as _REGISTRY,
        Counter as _Counter,
        Gauge as _Gauge,
        Histogram as _Histogram,
        start_http_server as _start_http_server,
    )

    Counter: type[CounterType] | None = _Counter
    Gauge: type[GaugeType] | None = _Gauge
    Histogram: type[HistogramType] | None = _Histogram
    CollectorRegistry: type[CollectorRegistryType] | None = None  # Not imported
    REGISTRY: CollectorRegistryType | None = _REGISTRY
    start_http_server: Callable[..., Any] | None = _start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None
    Gauge = None
    Histogram = None
    CollectorRegistry = None
    REGISTRY = None
    start_http_server = None
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricDefinition:
    """
    Definition for a metric that can be instantiated.

    This dataclass defines the schema for metrics, including their type,
    description, labels, and histogram buckets.
    """

    name: str
    metric_type: str  # 'counter', 'gauge', 'histogram'
    description: str
    label_names: tuple[str, ...] = ()
    buckets: list[float] | None = None


# Pre-defined metrics for the library
METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    # === Scheduling Counters ===
    REQUESTS_SCHEDULED_TOTAL: MetricDefinition(
        REQUESTS_SCHEDULED_TOTAL,
        "counter",
        "Total requests scheduled",
        ("bucket_id", "model_id"),
    ),
    REQUESTS_COMPLETED_TOTAL: MetricDefinition(
        REQUESTS_COMPLETED_TOTAL,
        "counter",
        "Total requests completed successfully",
        ("bucket_id", "model_id"),
    ),
    REQUESTS_FAILED_TOTAL: MetricDefinition(
        REQUESTS_FAILED_TOTAL,
        "counter",
        "Total requests failed",
        ("bucket_id", "model_id", "reason"),
    ),
    QUEUE_OVERFLOWS_TOTAL: MetricDefinition(
        QUEUE_OVERFLOWS_TOTAL,
        "counter",
        "Total queue overflow events",
        ("bucket_id",),
    ),
    SCHEDULER_LOOPS_TOTAL: MetricDefinition(
        SCHEDULER_LOOPS_TOTAL,
        "counter",
        "Total scheduler loop iterations",
        (),
    ),
    CIRCUIT_BREAKER_REJECTIONS_TOTAL: MetricDefinition(
        CIRCUIT_BREAKER_REJECTIONS_TOTAL,
        "counter",
        "Total circuit breaker rejections",
        ("bucket_id",),
    ),
    REQUEST_TIMEOUTS_TOTAL: MetricDefinition(
        REQUEST_TIMEOUTS_TOTAL,
        "counter",
        "Total request timeouts",
        ("bucket_id",),
    ),
    # === Gauges ===
    ACTIVE_REQUESTS: MetricDefinition(
        ACTIVE_REQUESTS,
        "gauge",
        "Currently active requests",
        ("bucket_id",),
    ),
    QUEUE_DEPTH: MetricDefinition(
        QUEUE_DEPTH,
        "gauge",
        "Current queue depth",
        ("bucket_id",),
    ),
    RESERVATIONS_ACTIVE: MetricDefinition(
        RESERVATIONS_ACTIVE,
        "gauge",
        "Currently active reservations",
        ("bucket_id",),
    ),
    # === Streaming Counters ===
    STREAMING_COMPLETIONS_TOTAL: MetricDefinition(
        STREAMING_COMPLETIONS_TOTAL,
        "counter",
        "Total streaming completions",
        ("bucket_id", "extraction_succeeded"),
    ),
    STREAMING_ERRORS_TOTAL: MetricDefinition(
        STREAMING_ERRORS_TOTAL,
        "counter",
        "Total streaming errors",
        ("bucket_id", "error_type"),
    ),
    STREAMING_TIMEOUTS_TOTAL: MetricDefinition(
        STREAMING_TIMEOUTS_TOTAL,
        "counter",
        "Total streaming timeouts",
        ("bucket_id",),
    ),
    STREAMING_STALE_CLEANUPS_TOTAL: MetricDefinition(
        STREAMING_STALE_CLEANUPS_TOTAL,
        "counter",
        "Total stale streaming cleanups",
        ("bucket_id",),
    ),
    STREAMING_TOKENS_REFUNDED_TOTAL: MetricDefinition(
        STREAMING_TOKENS_REFUNDED_TOTAL,
        "counter",
        "Total tokens refunded from streaming",
        ("bucket_id",),
    ),
    STREAMING_DURATION_SECONDS: MetricDefinition(
        STREAMING_DURATION_SECONDS,
        "histogram",
        "Duration of streaming requests",
        ("bucket_id",),
        buckets=STREAMING_DURATION_BUCKETS,
    ),
    # === Cache Counters ===
    CACHE_HITS_TOTAL: MetricDefinition(
        CACHE_HITS_TOTAL,
        "counter",
        "Total cache hits",
        (),
    ),
    CACHE_MISSES_TOTAL: MetricDefinition(
        CACHE_MISSES_TOTAL,
        "counter",
        "Total cache misses",
        (),
    ),
    CACHE_EVICTIONS_TOTAL: MetricDefinition(
        CACHE_EVICTIONS_TOTAL,
        "counter",
        "Total cache evictions",
        (),
    ),
    BACKEND_WRITES_TOTAL: MetricDefinition(
        BACKEND_WRITES_TOTAL,
        "counter",
        "Total backend writes",
        (),
    ),
    BACKEND_READS_TOTAL: MetricDefinition(
        BACKEND_READS_TOTAL,
        "counter",
        "Total backend reads",
        (),
    ),
    VERSION_CONFLICTS_TOTAL: MetricDefinition(
        VERSION_CONFLICTS_TOTAL,
        "counter",
        "Total version conflicts",
        (),
    ),
    # === Reservation Counters ===
    RESERVATION_STALE_CLEANUPS_TOTAL: MetricDefinition(
        RESERVATION_STALE_CLEANUPS_TOTAL,
        "counter",
        "Total stale reservation cleanups",
        ("bucket_id",),
    ),
    RESERVATION_EMERGENCY_CLEANUPS_TOTAL: MetricDefinition(
        RESERVATION_EMERGENCY_CLEANUPS_TOTAL,
        "counter",
        "Total emergency reservation cleanups",
        (),
    ),
    RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL: MetricDefinition(
        RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL,
        "counter",
        "Total backpressure rejections",
        ("bucket_id",),
    ),
    # === Backend Metrics ===
    BACKEND_LUA_EXECUTIONS_TOTAL: MetricDefinition(
        BACKEND_LUA_EXECUTIONS_TOTAL,
        "counter",
        "Total Lua script executions",
        ("script_name",),
    ),
    BACKEND_CONNECTION_ERRORS_TOTAL: MetricDefinition(
        BACKEND_CONNECTION_ERRORS_TOTAL,
        "counter",
        "Total backend connection errors",
        ("error_type",),
    ),
    BACKEND_LATENCY_SECONDS: MetricDefinition(
        BACKEND_LATENCY_SECONDS,
        "histogram",
        "Backend operation latency",
        ("operation",),
        buckets=LATENCY_BUCKETS,
    ),
}


class UnifiedMetricsCollector:
    """
    Unified metrics collector supporting both dict-based and Prometheus metrics.

    This class provides:
    1. Thread-safe counter/gauge/histogram operations
    2. Automatic Prometheus metric registration when available
    3. Dict-based fallback for JSON export
    4. Label cardinality protection
    5. Optional HTTP server for Prometheus scraping

    Thread Safety:
        All operations use RLock for thread-safe access. The lock is reentrant
        to allow nested calls from callbacks.

    Cardinality Protection:
        To prevent unbounded memory growth, a maximum of MAX_LABEL_COMBINATIONS
        unique label combinations are tracked per metric.

    Example:
        >>> collector = UnifiedMetricsCollector()
        >>> collector.inc_counter('adaptive_rl_requests_scheduled_total',
        ...                       labels={'bucket_id': 'tier:1'})
        >>> metrics = collector.get_metrics()
    """

    # Maximum unique label combinations per metric to prevent cardinality explosion
    MAX_LABEL_COMBINATIONS: ClassVar[int] = 1000

    def __init__(
        self,
        enable_prometheus: bool = True,
        registry: Any | None = None,
    ) -> None:
        """
        Initialize the metrics collector.

        Args:
            enable_prometheus: Whether to enable Prometheus metrics (if available)
            registry: Optional Prometheus CollectorRegistry for testing
        """
        self._enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self._registry = (
            registry if registry else (REGISTRY if PROMETHEUS_AVAILABLE else None)
        )

        # Dict-based metrics (always available)
        self._counters: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Thread safety
        self._lock = threading.RLock()

        # Prometheus metric instances (lazy initialized)
        self._prom_counters: dict[str, Any] = {}
        self._prom_gauges: dict[str, Any] = {}
        self._prom_histograms: dict[str, Any] = {}

        # Label cardinality tracking
        self._label_combinations: dict[str, set[str]] = defaultdict(set)

        # HTTP server state
        self._server_running = False

        logger.debug(
            f"UnifiedMetricsCollector initialized "
            f"(prometheus={'enabled' if self._enable_prometheus else 'disabled'})"
        )

    def _labels_to_key(self, labels: dict[str, str] | None) -> str:
        """Convert labels dict to a stable string key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _check_cardinality(self, name: str, label_key: str) -> bool:
        """
        Check if adding this label combination would exceed cardinality limit.

        Returns:
            True if the label combination is allowed, False otherwise
        """
        if label_key in self._label_combinations[name]:
            return True
        if len(self._label_combinations[name]) >= self.MAX_LABEL_COMBINATIONS:
            logger.warning(
                f"Cardinality limit ({self.MAX_LABEL_COMBINATIONS}) reached "
                f"for metric {name}. Dropping label combination: {label_key}"
            )
            return False
        self._label_combinations[name].add(label_key)
        return True

    def _get_or_create_prom_counter(self, name: str) -> Any | None:
        """Get or create a Prometheus counter."""
        if not self._enable_prometheus or Counter is None:
            return None

        if name not in self._prom_counters:
            defn = METRIC_DEFINITIONS.get(name)
            if defn and defn.metric_type == "counter":
                try:
                    self._prom_counters[name] = Counter(
                        name,
                        defn.description,
                        list(defn.label_names),
                        registry=self._registry,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Prometheus counter {name}: {e}")
                    return None
            else:
                # Dynamic counter (not pre-defined)
                try:
                    self._prom_counters[name] = Counter(
                        name,
                        f"Dynamic counter: {name}",
                        [],
                        registry=self._registry,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create dynamic counter {name}: {e}")
                    return None

        return self._prom_counters.get(name)

    def _get_or_create_prom_gauge(self, name: str) -> Any | None:
        """Get or create a Prometheus gauge."""
        if not self._enable_prometheus or Gauge is None:
            return None

        if name not in self._prom_gauges:
            defn = METRIC_DEFINITIONS.get(name)
            if defn and defn.metric_type == "gauge":
                try:
                    self._prom_gauges[name] = Gauge(
                        name,
                        defn.description,
                        list(defn.label_names),
                        registry=self._registry,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Prometheus gauge {name}: {e}")
                    return None
            else:
                try:
                    self._prom_gauges[name] = Gauge(
                        name,
                        f"Dynamic gauge: {name}",
                        [],
                        registry=self._registry,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create dynamic gauge {name}: {e}")
                    return None

        return self._prom_gauges.get(name)

    def _get_or_create_prom_histogram(self, name: str) -> Any | None:
        """Get or create a Prometheus histogram."""
        if not self._enable_prometheus or Histogram is None:
            return None

        if name not in self._prom_histograms:
            defn = METRIC_DEFINITIONS.get(name)
            buckets = defn.buckets if defn and defn.buckets else LATENCY_BUCKETS
            label_names = defn.label_names if defn else ()
            description = defn.description if defn else f"Dynamic histogram: {name}"

            try:
                self._prom_histograms[name] = Histogram(
                    name,
                    description,
                    list(label_names),
                    buckets=buckets,
                    registry=self._registry,
                )
            except Exception as e:
                logger.warning(f"Failed to create Prometheus histogram {name}: {e}")
                return None

        return self._prom_histograms.get(name)

    # === Counter Operations ===

    def inc_counter(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name (should follow Prometheus naming convention)
            value: Value to increment by (must be positive)
            labels: Optional labels dict

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Counter increment must be non-negative")

        label_key = self._labels_to_key(labels)

        with self._lock:
            if not self._check_cardinality(name, label_key):
                return
            self._counters[name][label_key] += value

        # Update Prometheus counter
        prom_counter = self._get_or_create_prom_counter(name)
        if prom_counter:
            try:
                if labels:
                    prom_counter.labels(**labels).inc(value)
                else:
                    prom_counter.inc(value)
            except Exception as e:
                logger.debug(f"Prometheus counter update failed for {name}: {e}")

    # === Gauge Operations ===

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric to a specific value."""
        label_key = self._labels_to_key(labels)

        with self._lock:
            if not self._check_cardinality(name, label_key):
                return
            self._gauges[name][label_key] = value

        prom_gauge = self._get_or_create_prom_gauge(name)
        if prom_gauge:
            try:
                if labels:
                    prom_gauge.labels(**labels).set(value)
                else:
                    prom_gauge.set(value)
            except Exception as e:
                logger.debug(f"Prometheus gauge update failed for {name}: {e}")

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a gauge metric."""
        label_key = self._labels_to_key(labels)

        with self._lock:
            if not self._check_cardinality(name, label_key):
                return
            self._gauges[name][label_key] += value

        prom_gauge = self._get_or_create_prom_gauge(name)
        if prom_gauge:
            try:
                if labels:
                    prom_gauge.labels(**labels).inc(value)
                else:
                    prom_gauge.inc(value)
            except Exception as e:
                logger.debug(f"Prometheus gauge inc failed for {name}: {e}")

    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Decrement a gauge metric."""
        label_key = self._labels_to_key(labels)

        with self._lock:
            if not self._check_cardinality(name, label_key):
                return
            self._gauges[name][label_key] -= value

        prom_gauge = self._get_or_create_prom_gauge(name)
        if prom_gauge:
            try:
                if labels:
                    prom_gauge.labels(**labels).dec(value)
                else:
                    prom_gauge.dec(value)
            except Exception as e:
                logger.debug(f"Prometheus gauge dec failed for {name}: {e}")

    # === Histogram Operations ===

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram."""
        label_key = self._labels_to_key(labels)

        with self._lock:
            if not self._check_cardinality(name, label_key):
                return
            # Store observations for dict-based export
            self._histograms[name][label_key].append(value)
            # Keep only recent observations to prevent memory growth
            if len(self._histograms[name][label_key]) > 10000:
                self._histograms[name][label_key] = self._histograms[name][label_key][
                    -5000:
                ]

        prom_histogram = self._get_or_create_prom_histogram(name)
        if prom_histogram:
            try:
                if labels:
                    prom_histogram.labels(**labels).observe(value)
                else:
                    prom_histogram.observe(value)
            except Exception as e:
                logger.debug(f"Prometheus histogram observe failed for {name}: {e}")

    # === Snapshot Operations ===

    def get_metrics(self) -> dict[str, Any]:
        """
        Get a snapshot of all metrics.

        Returns a dict suitable for JSON serialization with structure:
        {
            "counters": {"metric_name": {"label_key": value, ...}, ...},
            "gauges": {"metric_name": {"label_key": value, ...}, ...},
            "histograms": {"metric_name": {"label_key": {...}, ...}, ...}
        }
        """
        with self._lock:
            # Deep copy counters and gauges
            counters = {
                name: dict(label_values)
                for name, label_values in self._counters.items()
            }
            gauges = {
                name: dict(label_values) for name, label_values in self._gauges.items()
            }

            # Compute histogram summaries
            histograms: dict[str, dict[str, dict[str, Any]]] = {}
            for name, label_values in self._histograms.items():
                histograms[name] = {}
                for label_key, observations in label_values.items():
                    if observations:
                        histograms[name][label_key] = {
                            "count": len(observations),
                            "sum": sum(observations),
                            "avg": sum(observations) / len(observations),
                            "min": min(observations),
                            "max": max(observations),
                        }

        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def get_flat_metrics(self) -> dict[str, Any]:
        """
        Get metrics in a flat dict format for backward compatibility.

        Returns:
            Dict with metric names as keys and values as int/float.
            For labeled metrics, uses format "metric_name{label=value,...}".
        """
        result: dict[str, Any] = {}

        with self._lock:
            for name, label_values in self._counters.items():
                for label_key, value in label_values.items():
                    if label_key:
                        result[f"{name}{{{label_key}}}"] = value
                    else:
                        result[name] = value

            for gauge_name, gauge_label_values in self._gauges.items():
                for gauge_label_key, gauge_value in gauge_label_values.items():
                    if gauge_label_key:
                        result[f"{gauge_name}{{{gauge_label_key}}}"] = gauge_value
                    else:
                        result[gauge_name] = gauge_value

        return result

    # === Lifecycle ===

    def reset(self) -> None:
        """Reset all metrics to zero."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._label_combinations.clear()

        logger.debug("Metrics collector reset")

    # === Prometheus HTTP Server ===

    def start_http_server(self, host: str = "127.0.0.1", port: int = 9090) -> bool:
        """
        Start the Prometheus HTTP server for metrics scraping.

        Args:
            host: Host to bind to (default: 127.0.0.1 for localhost only).
                  Use "0.0.0.0" for external access in containerized environments.
            port: Port to bind to

        Returns:
            True if server started successfully, False otherwise

        Security Note:
            By default, binds to localhost (127.0.0.1) only for security.
            If you need external access (e.g., in Kubernetes/Docker), explicitly
            pass host="0.0.0.0" and ensure network-level security controls are in place.
        """
        if not PROMETHEUS_AVAILABLE or start_http_server is None:
            logger.warning(
                "Cannot start Prometheus server: prometheus_client not installed"
            )
            return False

        if self._server_running:
            logger.warning("Prometheus server already running")
            return True

        try:
            # start_http_server runs in a daemon thread
            if self._registry is not None:
                start_http_server(port, addr=host, registry=self._registry)
            else:
                start_http_server(port, addr=host)
            self._server_running = True
            logger.info(f"Prometheus metrics server started on {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            return False

    @property
    def prometheus_available(self) -> bool:
        """Check if Prometheus is available."""
        return PROMETHEUS_AVAILABLE

    @property
    def prometheus_enabled(self) -> bool:
        """Check if Prometheus metrics are enabled."""
        return self._enable_prometheus

    @property
    def server_running(self) -> bool:
        """Check if the Prometheus HTTP server is running."""
        return self._server_running


# =============================================================================
# Singleton Pattern
# =============================================================================

_global_collector: UnifiedMetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector(
    enable_prometheus: bool = True,
) -> UnifiedMetricsCollector:
    """
    Get or create the global metrics collector singleton.

    Thread-safe singleton initialization.

    Args:
        enable_prometheus: Whether to enable Prometheus metrics
            (only used on first call)

    Returns:
        The UnifiedMetricsCollector singleton
    """
    global _global_collector

    if _global_collector is None:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = UnifiedMetricsCollector(
                    enable_prometheus=enable_prometheus
                )

    return _global_collector


def reset_metrics_collector() -> None:
    """
    Reset the global metrics collector singleton (mainly for testing).

    This clears the singleton instance, allowing a new one to be created
    on the next call to get_metrics_collector().

    Warning:
        This is primarily for testing. In production, the collector should
        persist for the lifetime of the application.
    """
    global _global_collector
    with _collector_lock:
        if _global_collector:
            _global_collector.reset()
        _global_collector = None


__all__ = [
    "METRIC_DEFINITIONS",
    "PROMETHEUS_AVAILABLE",
    "MetricDefinition",
    "UnifiedMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
]

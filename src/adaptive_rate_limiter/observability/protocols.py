# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Protocol definitions for metrics collection backends.

This module defines the protocols that metrics collectors must implement,
allowing for duck typing and custom implementations (in-memory, Prometheus,
StatsD, OpenTelemetry, etc.).

Design Goals:
    1. Protocol-based - Allow duck typing and custom implementations
    2. Metric Type Agnostic - Support counters, gauges, histograms
    3. Thread-safe - Library may be used in threaded contexts
    4. Async-compatible - Scheduler is async
    5. Minimal footprint - No heavy operations in hot paths
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """
    Protocol for metrics collection backends.

    Implementations can use in-memory dicts, Prometheus, StatsD,
    OpenTelemetry, or any other metrics backend.

    This protocol defines the core operations for:
    - Counters: Monotonically increasing values (requests, errors)
    - Gauges: Values that can increase or decrease (queue depth)
    - Histograms: Distribution of values (latencies)

    Example:
        >>> class MyCollector:
        ...     def inc_counter(self, name, value=1, labels=None): pass
        ...     def set_gauge(self, name, value, labels=None): pass
        ...     def inc_gauge(self, name, value=1.0, labels=None): pass
        ...     def dec_gauge(self, name, value=1.0, labels=None): pass
        ...     def observe_histogram(self, name, value, labels=None): pass
        ...     def get_metrics(self): return {}
        ...     def reset(self): pass
        >>>
        >>> isinstance(MyCollector(), MetricsCollectorProtocol)
        True
    """

    # === Counter Operations ===

    def inc_counter(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Counters are monotonically increasing values, typically used for
        tracking totals (requests, errors, etc.). The value must be non-negative.

        Args:
            name: Metric name (should follow Prometheus naming convention)
            value: Value to increment by (must be >= 0)
            labels: Optional labels dict for dimensional metrics

        Raises:
            ValueError: If value is negative
        """
        ...

    # === Gauge Operations ===

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Set a gauge metric to a specific value.

        Gauges represent a snapshot value that can arbitrarily go up or down,
        like queue depth or active connections.

        Args:
            name: Metric name
            value: Value to set
            labels: Optional labels dict
        """
        ...

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a gauge metric.

        Args:
            name: Metric name
            value: Value to increment by (can be negative)
            labels: Optional labels dict
        """
        ...

    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Decrement a gauge metric.

        Args:
            name: Metric name
            value: Value to decrement by
            labels: Optional labels dict
        """
        ...

    # === Histogram Operations ===

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record an observation in a histogram.

        Histograms track the distribution of values, allowing for percentile
        calculations (p50, p95, p99, etc.).

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels dict
        """
        ...

    # === Snapshot Operations ===

    def get_metrics(self) -> dict[str, Any]:
        """
        Get a snapshot of all metrics.

        Returns dict-based format for JSON serialization and
        backward compatibility with existing get_metrics() API.

        Returns:
            Dictionary with structure:
            {
                "counters": {"metric_name": {"label_key": value, ...}, ...},
                "gauges": {"metric_name": {"label_key": value, ...}, ...},
                "histograms": {"metric_name": {"label_key": {...}, ...}, ...}
            }
        """
        ...

    # === Lifecycle ===

    def reset(self) -> None:
        """
        Reset all metrics to zero.

        Useful for testing or periodic metric collection where you want
        to track deltas between collection intervals.
        """
        ...


@runtime_checkable
class PrometheusExporterProtocol(Protocol):
    """
    Protocol for Prometheus HTTP server management.

    Implementations provide methods to start and stop the HTTP server
    that Prometheus scrapes for metrics.
    """

    def start_server(self, host: str, port: int) -> None:
        """
        Start the Prometheus HTTP server.

        Args:
            host: Host to bind to (e.g., "0.0.0.0")
            port: Port to bind to (e.g., 9090)
        """
        ...

    def stop_server(self) -> None:
        """Stop the Prometheus HTTP server."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        ...


__all__ = [
    "MetricsCollectorProtocol",
    "PrometheusExporterProtocol",
]

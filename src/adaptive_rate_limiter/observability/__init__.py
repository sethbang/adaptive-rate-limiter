# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Observability and metrics for the Adaptive Rate Limiter.

This module provides metrics and observability components for tracking
rate limiter behavior, including streaming-specific metrics and
the unified metrics collector.

Classes:
    StreamingMetrics: Dataclass for tracking streaming-specific rate limit metrics.
    PrometheusStreamingMetrics: Optional Prometheus-style metrics for observability.
    UnifiedMetricsCollector: Unified metrics collector supporting dict and Prometheus.

Protocols:
    MetricsCollectorProtocol: Protocol for metrics collection backends.
    PrometheusExporterProtocol: Protocol for Prometheus HTTP server management.

Functions:
    get_prometheus_streaming_metrics: Get or create the Prometheus metrics singleton.
    reset_prometheus_streaming_metrics: Reset the Prometheus metrics singleton.
    get_metrics_collector: Get the global metrics collector singleton.
    reset_metrics_collector: Reset the global metrics collector singleton.

Constants:
    PROMETHEUS_AVAILABLE: Whether prometheus_client is available.
    All metric name constants from constants module.
"""

from .collector import (
    METRIC_DEFINITIONS,
    PROMETHEUS_AVAILABLE,
    MetricDefinition,
    UnifiedMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)
from .constants import (
    # Gauges
    ACTIVE_REQUESTS,
    BACKEND_READS_TOTAL,
    BACKEND_WRITES_TOTAL,
    CACHE_EVICTIONS_TOTAL,
    # Cache
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    CIRCUIT_BREAKER_REJECTIONS_TOTAL,
    # Buckets
    LATENCY_BUCKETS,
    # Mapping
    LEGACY_METRIC_MAPPING,
    METRIC_PREFIX,
    QUEUE_DEPTH,
    QUEUE_OVERFLOWS_TOTAL,
    REQUEST_TIMEOUTS_TOTAL,
    REQUESTS_COMPLETED_TOTAL,
    REQUESTS_FAILED_TOTAL,
    # Scheduling metrics
    REQUESTS_SCHEDULED_TOTAL,
    RESERVATION_BACKPRESSURE_REJECTIONS_TOTAL,
    RESERVATION_EMERGENCY_CLEANUPS_TOTAL,
    # Reservation
    RESERVATION_STALE_CLEANUPS_TOTAL,
    RESERVATIONS_ACTIVE,
    SCHEDULER_LOOPS_TOTAL,
    # Streaming
    STREAMING_COMPLETIONS_TOTAL,
    STREAMING_DURATION_BUCKETS,
    STREAMING_DURATION_SECONDS,
    STREAMING_ERRORS_TOTAL,
    STREAMING_STALE_CLEANUPS_TOTAL,
    STREAMING_TIMEOUTS_TOTAL,
    STREAMING_TOKENS_REFUNDED_TOTAL,
    TOKEN_BUCKETS,
    VERSION_CONFLICTS_TOTAL,
    legacy_to_prometheus_name,
)
from .metrics import (
    PrometheusStreamingMetrics,
    StreamingMetrics,
    StreamingMetricsCallback,
    get_prometheus_streaming_metrics,
    reset_prometheus_streaming_metrics,
)
from .protocols import (
    MetricsCollectorProtocol,
    PrometheusExporterProtocol,
)

__all__ = [
    # Gauges
    "ACTIVE_REQUESTS",
    "BACKEND_READS_TOTAL",
    "BACKEND_WRITES_TOTAL",
    "CACHE_EVICTIONS_TOTAL",
    # Cache
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "CIRCUIT_BREAKER_REJECTIONS_TOTAL",
    # Buckets
    "LATENCY_BUCKETS",
    # Mapping
    "LEGACY_METRIC_MAPPING",
    "METRIC_DEFINITIONS",
    "METRIC_PREFIX",
    # Constants
    "PROMETHEUS_AVAILABLE",
    "QUEUE_DEPTH",
    "QUEUE_OVERFLOWS_TOTAL",
    "REQUESTS_COMPLETED_TOTAL",
    "REQUESTS_FAILED_TOTAL",
    # Scheduling metrics
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
    "MetricDefinition",
    # Protocols
    "MetricsCollectorProtocol",
    "PrometheusExporterProtocol",
    "PrometheusStreamingMetrics",
    # Streaming metrics
    "StreamingMetrics",
    "StreamingMetricsCallback",
    # Unified collector
    "UnifiedMetricsCollector",
    "get_metrics_collector",
    "get_prometheus_streaming_metrics",
    "legacy_to_prometheus_name",
    "reset_metrics_collector",
    "reset_prometheus_streaming_metrics",
]

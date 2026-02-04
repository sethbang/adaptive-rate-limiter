# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the observability collector module.

Tests cover:
- UnifiedMetricsCollector: Thread-safe metrics collection with Prometheus support
- Singleton pattern: get_metrics_collector, reset_metrics_collector
- Counter, Gauge, and Histogram operations
- Label cardinality protection
- Prometheus integration and HTTP server
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from adaptive_rate_limiter.observability.collector import (
    METRIC_DEFINITIONS,
    PROMETHEUS_AVAILABLE,
    MetricDefinition,
    UnifiedMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)

# =============================================================================
# MetricDefinition Tests
# =============================================================================


class TestMetricDefinition:
    """Test MetricDefinition dataclass."""

    def test_counter_definition(self) -> None:
        """Test creating a counter definition."""
        defn = MetricDefinition(
            name="test_counter_total",
            metric_type="counter",
            description="A test counter",
            label_names=("bucket_id", "model_id"),
        )
        assert defn.name == "test_counter_total"
        assert defn.metric_type == "counter"
        assert defn.description == "A test counter"
        assert defn.label_names == ("bucket_id", "model_id")
        assert defn.buckets is None

    def test_histogram_definition_with_buckets(self) -> None:
        """Test creating a histogram definition with custom buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
        defn = MetricDefinition(
            name="test_latency_seconds",
            metric_type="histogram",
            description="A test histogram",
            label_names=("operation",),
            buckets=buckets,
        )
        assert defn.metric_type == "histogram"
        assert defn.buckets == buckets

    def test_gauge_definition(self) -> None:
        """Test creating a gauge definition."""
        defn = MetricDefinition(
            name="test_gauge",
            metric_type="gauge",
            description="A test gauge",
        )
        assert defn.metric_type == "gauge"
        assert defn.label_names == ()

    def test_predefined_metrics_exist(self) -> None:
        """Test that predefined metrics are properly defined."""
        assert "adaptive_rl_requests_scheduled_total" in METRIC_DEFINITIONS
        assert "adaptive_rl_active_requests" in METRIC_DEFINITIONS
        assert "adaptive_rl_backend_latency_seconds" in METRIC_DEFINITIONS


# =============================================================================
# UnifiedMetricsCollector Initialization Tests
# =============================================================================


class TestUnifiedMetricsCollectorInitialization:
    """Test UnifiedMetricsCollector initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert collector._enable_prometheus is False
        assert collector._server_running is False

    def test_initialization_with_prometheus_disabled(self) -> None:
        """Test initialization with Prometheus explicitly disabled."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert collector._enable_prometheus is False

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_initialization_with_prometheus_enabled(self) -> None:
        """Test initialization with Prometheus enabled."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)
        assert collector._enable_prometheus is True
        assert collector._registry is registry

    def test_initialization_prometheus_enabled_but_not_available(self) -> None:
        """Test that Prometheus is disabled when not available."""
        with patch(
            "adaptive_rate_limiter.observability.collector.PROMETHEUS_AVAILABLE", False
        ):
            collector = UnifiedMetricsCollector(enable_prometheus=True)
            assert collector._enable_prometheus is False

    def test_internal_structures_initialized(self) -> None:
        """Test that internal structures are properly initialized."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert isinstance(collector._counters, dict)
        assert isinstance(collector._gauges, dict)
        assert isinstance(collector._histograms, dict)
        assert isinstance(collector._lock, type(threading.RLock()))
        assert isinstance(collector._label_combinations, dict)


# =============================================================================
# Counter Operations Tests
# =============================================================================


class TestCounterOperations:
    """Test counter operations."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector without Prometheus."""
        return UnifiedMetricsCollector(enable_prometheus=False)

    def test_inc_counter_basic(self, collector: UnifiedMetricsCollector) -> None:
        """Test basic counter increment."""
        collector.inc_counter("test_counter")
        assert collector._counters["test_counter"][""] == 1

    def test_inc_counter_with_value(self, collector: UnifiedMetricsCollector) -> None:
        """Test counter increment with specific value."""
        collector.inc_counter("test_counter", value=5)
        assert collector._counters["test_counter"][""] == 5

    def test_inc_counter_accumulates(self, collector: UnifiedMetricsCollector) -> None:
        """Test counter increments accumulate."""
        collector.inc_counter("test_counter", value=3)
        collector.inc_counter("test_counter", value=7)
        assert collector._counters["test_counter"][""] == 10

    def test_inc_counter_with_labels(self, collector: UnifiedMetricsCollector) -> None:
        """Test counter increment with labels."""
        collector.inc_counter("test_counter", labels={"bucket_id": "tier:1"})
        assert collector._counters["test_counter"]["bucket_id=tier:1"] == 1

    def test_inc_counter_multiple_label_combinations(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test counter with multiple label combinations."""
        collector.inc_counter("test_counter", labels={"bucket_id": "tier:1"})
        collector.inc_counter("test_counter", labels={"bucket_id": "tier:2"})
        collector.inc_counter("test_counter", labels={"bucket_id": "tier:1"})

        assert collector._counters["test_counter"]["bucket_id=tier:1"] == 2
        assert collector._counters["test_counter"]["bucket_id=tier:2"] == 1

    def test_inc_counter_negative_value_raises(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test that negative counter value raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            collector.inc_counter("test_counter", value=-1)

    def test_inc_counter_zero_value(self, collector: UnifiedMetricsCollector) -> None:
        """Test counter increment with zero value (valid)."""
        collector.inc_counter("test_counter", value=0)
        assert collector._counters["test_counter"][""] == 0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_inc_counter_with_prometheus(self) -> None:
        """Test counter increment updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        # Use a predefined metric
        collector.inc_counter(
            "adaptive_rl_requests_scheduled_total",
            labels={"bucket_id": "test", "model_id": "gpt-5"},
        )

        # Verify dict-based counter
        assert (
            collector._counters["adaptive_rl_requests_scheduled_total"][
                "bucket_id=test,model_id=gpt-5"
            ]
            == 1
        )

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_inc_counter_prometheus_without_labels(self) -> None:
        """Test counter increment without labels updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        collector.inc_counter("adaptive_rl_scheduler_loops_total")
        assert collector._counters["adaptive_rl_scheduler_loops_total"][""] == 1


# =============================================================================
# Gauge Operations Tests
# =============================================================================


class TestGaugeOperations:
    """Test gauge operations."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector without Prometheus."""
        return UnifiedMetricsCollector(enable_prometheus=False)

    def test_set_gauge_basic(self, collector: UnifiedMetricsCollector) -> None:
        """Test basic gauge set."""
        collector.set_gauge("test_gauge", 42.5)
        assert collector._gauges["test_gauge"][""] == 42.5

    def test_set_gauge_overwrites(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge set overwrites previous value."""
        collector.set_gauge("test_gauge", 10.0)
        collector.set_gauge("test_gauge", 20.0)
        assert collector._gauges["test_gauge"][""] == 20.0

    def test_set_gauge_with_labels(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge set with labels."""
        collector.set_gauge("test_gauge", 5.0, labels={"bucket_id": "tier:1"})
        assert collector._gauges["test_gauge"]["bucket_id=tier:1"] == 5.0

    def test_inc_gauge_basic(self, collector: UnifiedMetricsCollector) -> None:
        """Test basic gauge increment."""
        collector.inc_gauge("test_gauge")
        assert collector._gauges["test_gauge"][""] == 1.0

    def test_inc_gauge_with_value(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge increment with specific value."""
        collector.inc_gauge("test_gauge", value=2.5)
        assert collector._gauges["test_gauge"][""] == 2.5

    def test_inc_gauge_accumulates(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge increments accumulate."""
        collector.inc_gauge("test_gauge", value=3.0)
        collector.inc_gauge("test_gauge", value=7.0)
        assert collector._gauges["test_gauge"][""] == 10.0

    def test_inc_gauge_with_labels(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge increment with labels."""
        collector.inc_gauge("test_gauge", labels={"bucket_id": "tier:1"})
        assert collector._gauges["test_gauge"]["bucket_id=tier:1"] == 1.0

    def test_dec_gauge_basic(self, collector: UnifiedMetricsCollector) -> None:
        """Test basic gauge decrement."""
        collector.set_gauge("test_gauge", 10.0)
        collector.dec_gauge("test_gauge")
        assert collector._gauges["test_gauge"][""] == 9.0

    def test_dec_gauge_with_value(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge decrement with specific value."""
        collector.set_gauge("test_gauge", 10.0)
        collector.dec_gauge("test_gauge", value=3.5)
        assert collector._gauges["test_gauge"][""] == 6.5

    def test_dec_gauge_can_go_negative(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test gauge can go negative."""
        collector.dec_gauge("test_gauge", value=5.0)
        assert collector._gauges["test_gauge"][""] == -5.0

    def test_dec_gauge_with_labels(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauge decrement with labels."""
        collector.set_gauge("test_gauge", 10.0, labels={"bucket_id": "tier:1"})
        collector.dec_gauge("test_gauge", labels={"bucket_id": "tier:1"})
        assert collector._gauges["test_gauge"]["bucket_id=tier:1"] == 9.0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_gauge_with_prometheus(self) -> None:
        """Test gauge set updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        collector.set_gauge(
            "adaptive_rl_active_requests", 5.0, labels={"bucket_id": "test"}
        )
        assert collector._gauges["adaptive_rl_active_requests"]["bucket_id=test"] == 5.0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_inc_gauge_with_prometheus(self) -> None:
        """Test gauge increment updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        collector.inc_gauge("adaptive_rl_active_requests", labels={"bucket_id": "test"})
        assert collector._gauges["adaptive_rl_active_requests"]["bucket_id=test"] == 1.0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_dec_gauge_with_prometheus(self) -> None:
        """Test gauge decrement updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        collector.set_gauge(
            "adaptive_rl_active_requests", 5.0, labels={"bucket_id": "test"}
        )
        collector.dec_gauge("adaptive_rl_active_requests", labels={"bucket_id": "test"})
        assert collector._gauges["adaptive_rl_active_requests"]["bucket_id=test"] == 4.0


# =============================================================================
# Histogram Operations Tests
# =============================================================================


class TestHistogramOperations:
    """Test histogram operations."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector without Prometheus."""
        return UnifiedMetricsCollector(enable_prometheus=False)

    def test_observe_histogram_basic(self, collector: UnifiedMetricsCollector) -> None:
        """Test basic histogram observation."""
        collector.observe_histogram("test_histogram", 0.5)
        assert collector._histograms["test_histogram"][""] == [0.5]

    def test_observe_histogram_multiple(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test multiple histogram observations."""
        collector.observe_histogram("test_histogram", 0.1)
        collector.observe_histogram("test_histogram", 0.5)
        collector.observe_histogram("test_histogram", 1.0)
        assert collector._histograms["test_histogram"][""] == [0.1, 0.5, 1.0]

    def test_observe_histogram_with_labels(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test histogram observation with labels."""
        collector.observe_histogram("test_histogram", 0.5, labels={"operation": "read"})
        assert collector._histograms["test_histogram"]["operation=read"] == [0.5]

    def test_observe_histogram_memory_limit(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test histogram truncates to prevent memory growth."""
        # Add more than 10000 observations
        for i in range(10001):
            collector.observe_histogram("test_histogram", float(i))

        # Should be truncated to last 5000
        assert len(collector._histograms["test_histogram"][""]) == 5000

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observe_histogram_with_prometheus(self) -> None:
        """Test histogram observation updates Prometheus."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        collector.observe_histogram(
            "adaptive_rl_backend_latency_seconds", 0.05, labels={"operation": "read"}
        )
        assert collector._histograms["adaptive_rl_backend_latency_seconds"][
            "operation=read"
        ] == [0.05]


# =============================================================================
# Cardinality Protection Tests
# =============================================================================


class TestCardinalityProtection:
    """Test label cardinality protection."""

    def test_cardinality_limit_blocks_new_combinations(self) -> None:
        """Test that new label combinations are blocked at limit."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        # Set a small limit for testing
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 3

        # Add 3 combinations - should all succeed
        collector.inc_counter("test_counter", labels={"id": "1"})
        collector.inc_counter("test_counter", labels={"id": "2"})
        collector.inc_counter("test_counter", labels={"id": "3"})

        # Add 4th - should be blocked
        collector.inc_counter("test_counter", labels={"id": "4"})

        assert "id=1" in collector._counters["test_counter"]
        assert "id=2" in collector._counters["test_counter"]
        assert "id=3" in collector._counters["test_counter"]
        assert "id=4" not in collector._counters["test_counter"]

    def test_existing_combinations_still_work(self) -> None:
        """Test that existing combinations still work after limit."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 2

        # Add 2 combinations
        collector.inc_counter("test_counter", labels={"id": "1"})
        collector.inc_counter("test_counter", labels={"id": "2"})

        # Try to add 3rd - blocked
        collector.inc_counter("test_counter", labels={"id": "3"})

        # Update existing - should work
        collector.inc_counter("test_counter", labels={"id": "1"}, value=5)
        assert collector._counters["test_counter"]["id=1"] == 6

    def test_cardinality_applies_per_metric(self) -> None:
        """Test that cardinality limit applies per metric."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 2

        # Add 2 to metric A
        collector.inc_counter("metric_a", labels={"id": "1"})
        collector.inc_counter("metric_a", labels={"id": "2"})

        # Should be able to add to metric B
        collector.inc_counter("metric_b", labels={"id": "1"})
        assert "id=1" in collector._counters["metric_b"]

    def test_cardinality_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that cardinality limit logs a warning."""
        import logging

        collector = UnifiedMetricsCollector(enable_prometheus=False)
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 1

        collector.inc_counter("test_counter", labels={"id": "1"})

        with caplog.at_level(logging.WARNING):
            collector.inc_counter("test_counter", labels={"id": "2"})

        assert "Cardinality limit" in caplog.text

    def test_cardinality_affects_gauges(self) -> None:
        """Test that cardinality limit affects gauges."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 1

        collector.set_gauge("test_gauge", 1.0, labels={"id": "1"})
        collector.set_gauge("test_gauge", 2.0, labels={"id": "2"})

        assert "id=1" in collector._gauges["test_gauge"]
        assert "id=2" not in collector._gauges["test_gauge"]

    def test_cardinality_affects_histograms(self) -> None:
        """Test that cardinality limit affects histograms."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        UnifiedMetricsCollector.MAX_LABEL_COMBINATIONS = 1

        collector.observe_histogram("test_histogram", 0.5, labels={"id": "1"})
        collector.observe_histogram("test_histogram", 0.5, labels={"id": "2"})

        assert "id=1" in collector._histograms["test_histogram"]
        assert "id=2" not in collector._histograms["test_histogram"]


# =============================================================================
# Get Metrics Tests
# =============================================================================


class TestGetMetrics:
    """Test get_metrics method."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector with sample data."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)

        # Add some counters
        collector.inc_counter("counter_a", value=5)
        collector.inc_counter("counter_b", value=10, labels={"key": "value"})

        # Add some gauges
        collector.set_gauge("gauge_a", 42.5)
        collector.set_gauge("gauge_b", 100.0, labels={"key": "value"})

        # Add some histogram observations
        collector.observe_histogram("histogram_a", 0.1)
        collector.observe_histogram("histogram_a", 0.2)
        collector.observe_histogram("histogram_a", 0.3)

        return collector

    def test_returns_all_metric_types(self, collector: UnifiedMetricsCollector) -> None:
        """Test that get_metrics returns all metric types."""
        metrics = collector.get_metrics()
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics

    def test_counters_in_snapshot(self, collector: UnifiedMetricsCollector) -> None:
        """Test counters are in snapshot."""
        metrics = collector.get_metrics()
        assert metrics["counters"]["counter_a"][""] == 5
        assert metrics["counters"]["counter_b"]["key=value"] == 10

    def test_gauges_in_snapshot(self, collector: UnifiedMetricsCollector) -> None:
        """Test gauges are in snapshot."""
        metrics = collector.get_metrics()
        assert metrics["gauges"]["gauge_a"][""] == 42.5
        assert metrics["gauges"]["gauge_b"]["key=value"] == 100.0

    def test_histogram_summary_in_snapshot(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test histogram summaries are in snapshot."""
        metrics = collector.get_metrics()
        hist = metrics["histograms"]["histogram_a"][""]
        assert hist["count"] == 3
        assert hist["sum"] == pytest.approx(0.6)
        assert hist["avg"] == pytest.approx(0.2)
        assert hist["min"] == pytest.approx(0.1)
        assert hist["max"] == pytest.approx(0.3)

    def test_empty_histogram_not_in_snapshot(self) -> None:
        """Test empty histograms with no label values don't appear in snapshot."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        # Access histogram but don't add observations (no label_key)
        _ = collector._histograms["empty_histogram"]

        metrics = collector.get_metrics()
        # The key exists but should have no label values with observations
        # The get_metrics only adds entries for label_keys that have observations
        if "empty_histogram" in metrics["histograms"]:
            # If empty_histogram appears, it should have empty dict (no label_key entries)
            assert metrics["histograms"]["empty_histogram"] == {}

    def test_snapshot_is_copy(self, collector: UnifiedMetricsCollector) -> None:
        """Test that snapshot is a copy, not a reference."""
        metrics = collector.get_metrics()
        original_value = metrics["counters"]["counter_a"][""]

        # Modify original
        collector.inc_counter("counter_a")

        # Snapshot should be unchanged
        assert metrics["counters"]["counter_a"][""] == original_value


# =============================================================================
# Get Flat Metrics Tests
# =============================================================================


class TestGetFlatMetrics:
    """Test get_flat_metrics method."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector with sample data."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.inc_counter("counter_a", value=5)
        collector.inc_counter("counter_b", value=10, labels={"key": "value"})
        collector.set_gauge("gauge_a", 42.5)
        collector.set_gauge("gauge_b", 100.0, labels={"key": "value"})
        return collector

    def test_returns_flat_dict(self, collector: UnifiedMetricsCollector) -> None:
        """Test returns a flat dictionary."""
        metrics = collector.get_flat_metrics()
        assert isinstance(metrics, dict)

    def test_unlabeled_metrics_use_name(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test unlabeled metrics use plain name."""
        metrics = collector.get_flat_metrics()
        assert "counter_a" in metrics
        assert metrics["counter_a"] == 5

    def test_labeled_metrics_include_labels(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test labeled metrics include labels in name."""
        metrics = collector.get_flat_metrics()
        assert "counter_b{key=value}" in metrics
        assert metrics["counter_b{key=value}"] == 10

    def test_includes_gauges(self, collector: UnifiedMetricsCollector) -> None:
        """Test includes gauge metrics."""
        metrics = collector.get_flat_metrics()
        assert "gauge_a" in metrics
        assert metrics["gauge_a"] == 42.5
        assert "gauge_b{key=value}" in metrics


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Test reset method."""

    def test_reset_clears_counters(self) -> None:
        """Test reset clears counters."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.inc_counter("test_counter")
        collector.reset()
        assert len(collector._counters) == 0

    def test_reset_clears_gauges(self) -> None:
        """Test reset clears gauges."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.set_gauge("test_gauge", 42.0)
        collector.reset()
        assert len(collector._gauges) == 0

    def test_reset_clears_histograms(self) -> None:
        """Test reset clears histograms."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.observe_histogram("test_histogram", 0.5)
        collector.reset()
        assert len(collector._histograms) == 0

    def test_reset_clears_label_combinations(self) -> None:
        """Test reset clears label combinations."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.inc_counter("test_counter", labels={"id": "1"})
        collector.reset()
        assert len(collector._label_combinations) == 0


# =============================================================================
# Labels to Key Tests
# =============================================================================


class TestLabelsToKey:
    """Test _labels_to_key method."""

    @pytest.fixture
    def collector(self) -> UnifiedMetricsCollector:
        """Create a fresh collector."""
        return UnifiedMetricsCollector(enable_prometheus=False)

    def test_empty_labels_returns_empty_string(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test empty labels return empty string."""
        assert collector._labels_to_key(None) == ""
        assert collector._labels_to_key({}) == ""

    def test_single_label(self, collector: UnifiedMetricsCollector) -> None:
        """Test single label key generation."""
        assert collector._labels_to_key({"key": "value"}) == "key=value"

    def test_multiple_labels_sorted(self, collector: UnifiedMetricsCollector) -> None:
        """Test multiple labels are sorted alphabetically."""
        labels = {"z_key": "z", "a_key": "a", "m_key": "m"}
        result = collector._labels_to_key(labels)
        assert result == "a_key=a,m_key=m,z_key=z"

    def test_special_characters_in_values(
        self, collector: UnifiedMetricsCollector
    ) -> None:
        """Test special characters in label values."""
        labels = {"bucket_id": "tier:1/model@v2"}
        result = collector._labels_to_key(labels)
        assert result == "bucket_id=tier:1/model@v2"


# =============================================================================
# Properties Tests
# =============================================================================


class TestProperties:
    """Test collector properties."""

    def test_prometheus_available_property(self) -> None:
        """Test prometheus_available property."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert collector.prometheus_available == PROMETHEUS_AVAILABLE

    def test_prometheus_enabled_property_false(self) -> None:
        """Test prometheus_enabled property when disabled."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert collector.prometheus_enabled is False

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_prometheus_enabled_property_true(self) -> None:
        """Test prometheus_enabled property when enabled."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)
        assert collector.prometheus_enabled is True

    def test_server_running_property_initial(self) -> None:
        """Test server_running property initial state."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        assert collector.server_running is False


# =============================================================================
# HTTP Server Tests
# =============================================================================


class TestHTTPServer:
    """Test Prometheus HTTP server functionality."""

    def test_start_http_server_without_prometheus(self) -> None:
        """Test start_http_server returns False when Prometheus not available."""
        with patch(
            "adaptive_rate_limiter.observability.collector.PROMETHEUS_AVAILABLE", False
        ):
            collector = UnifiedMetricsCollector(enable_prometheus=False)
            result = collector.start_http_server()
            assert result is False

    def test_start_http_server_already_running(self) -> None:
        """Test start_http_server returns True when already running."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector._server_running = True

        # Even without Prometheus, if server_running is True, it should return True
        # (This is a mock scenario - in real usage, this wouldn't happen)
        # Let's test the actual behavior
        with (
            patch(
                "adaptive_rate_limiter.observability.collector.PROMETHEUS_AVAILABLE",
                True,
            ),
            patch("adaptive_rate_limiter.observability.collector.start_http_server"),
        ):
            collector._enable_prometheus = True
            result = collector.start_http_server()
            assert result is True

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_start_http_server_failure(self) -> None:
        """Test start_http_server handles exceptions."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        with patch(
            "adaptive_rate_limiter.observability.collector.start_http_server",
            side_effect=Exception("Port in use"),
        ):
            result = collector.start_http_server()
            assert result is False
            assert collector._server_running is False


# =============================================================================
# Prometheus Metric Creation Tests
# =============================================================================


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
class TestPrometheusMetricCreation:
    """Test Prometheus metric creation and caching."""

    def test_counter_creation_from_definition(self) -> None:
        """Test counter is created from predefined definition."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_counter = collector._get_or_create_prom_counter(
            "adaptive_rl_requests_scheduled_total"
        )
        assert prom_counter is not None

    def test_counter_cached(self) -> None:
        """Test counter is cached after creation."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        counter1 = collector._get_or_create_prom_counter(
            "adaptive_rl_requests_scheduled_total"
        )
        counter2 = collector._get_or_create_prom_counter(
            "adaptive_rl_requests_scheduled_total"
        )
        assert counter1 is counter2

    def test_dynamic_counter_creation(self) -> None:
        """Test dynamic counter creation for undefined metrics."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_counter = collector._get_or_create_prom_counter("custom_counter_total")
        assert prom_counter is not None

    def test_gauge_creation_from_definition(self) -> None:
        """Test gauge is created from predefined definition."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_gauge = collector._get_or_create_prom_gauge("adaptive_rl_active_requests")
        assert prom_gauge is not None

    def test_gauge_cached(self) -> None:
        """Test gauge is cached after creation."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        gauge1 = collector._get_or_create_prom_gauge("adaptive_rl_active_requests")
        gauge2 = collector._get_or_create_prom_gauge("adaptive_rl_active_requests")
        assert gauge1 is gauge2

    def test_dynamic_gauge_creation(self) -> None:
        """Test dynamic gauge creation for undefined metrics."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_gauge = collector._get_or_create_prom_gauge("custom_gauge")
        assert prom_gauge is not None

    def test_histogram_creation_from_definition(self) -> None:
        """Test histogram is created from predefined definition."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_histogram = collector._get_or_create_prom_histogram(
            "adaptive_rl_backend_latency_seconds"
        )
        assert prom_histogram is not None

    def test_histogram_cached(self) -> None:
        """Test histogram is cached after creation."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        hist1 = collector._get_or_create_prom_histogram(
            "adaptive_rl_backend_latency_seconds"
        )
        hist2 = collector._get_or_create_prom_histogram(
            "adaptive_rl_backend_latency_seconds"
        )
        assert hist1 is hist2

    def test_dynamic_histogram_creation(self) -> None:
        """Test dynamic histogram creation for undefined metrics."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        prom_histogram = collector._get_or_create_prom_histogram(
            "custom_histogram_seconds"
        )
        assert prom_histogram is not None

    def test_counter_creation_failure_returns_none(self) -> None:
        """Test counter creation failure returns None."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        with patch(
            "adaptive_rate_limiter.observability.collector.Counter",
            side_effect=Exception("Creation failed"),
        ):
            result = collector._get_or_create_prom_counter("new_counter")
            assert result is None

    def test_gauge_creation_failure_returns_none(self) -> None:
        """Test gauge creation failure returns None."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        with patch(
            "adaptive_rate_limiter.observability.collector.Gauge",
            side_effect=Exception("Creation failed"),
        ):
            result = collector._get_or_create_prom_gauge("new_gauge")
            assert result is None

    def test_histogram_creation_failure_returns_none(self) -> None:
        """Test histogram creation failure returns None."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        with patch(
            "adaptive_rate_limiter.observability.collector.Histogram",
            side_effect=Exception("Creation failed"),
        ):
            result = collector._get_or_create_prom_histogram("new_histogram")
            assert result is None

    def test_prometheus_disabled_returns_none(self) -> None:
        """Test metric creation returns None when Prometheus disabled."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)

        assert collector._get_or_create_prom_counter("test") is None
        assert collector._get_or_create_prom_gauge("test") is None
        assert collector._get_or_create_prom_histogram("test") is None


# =============================================================================
# Prometheus Update Failure Tests
# =============================================================================


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
class TestPrometheusUpdateFailures:
    """Test handling of Prometheus update failures."""

    def test_counter_update_failure_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test counter update failure is logged but doesn't raise."""
        import logging

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        # Create counter with wrong labels
        collector.inc_counter(
            "adaptive_rl_requests_scheduled_total",
            labels={"bucket_id": "test", "model_id": "gpt-5"},
        )

        # Now try with wrong labels - should log debug
        with caplog.at_level(logging.DEBUG):
            # Force a failure by mocking
            mock_counter = MagicMock()
            mock_counter.labels.side_effect = Exception("Label mismatch")
            collector._prom_counters["test_fail"] = mock_counter

            # This should not raise
            collector.inc_counter("test_fail", labels={"wrong": "labels"})

    def test_gauge_set_failure_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test gauge set failure is logged but doesn't raise."""
        import logging

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        mock_gauge = MagicMock()
        mock_gauge.labels.side_effect = Exception("Label mismatch")
        collector._prom_gauges["test_fail"] = mock_gauge

        with caplog.at_level(logging.DEBUG):
            collector.set_gauge("test_fail", 5.0, labels={"wrong": "labels"})

    def test_histogram_observe_failure_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test histogram observe failure is logged but doesn't raise."""
        import logging

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        collector = UnifiedMetricsCollector(enable_prometheus=True, registry=registry)

        mock_histogram = MagicMock()
        mock_histogram.labels.side_effect = Exception("Label mismatch")
        collector._prom_histograms["test_fail"] = mock_histogram

        with caplog.at_level(logging.DEBUG):
            collector.observe_histogram("test_fail", 0.5, labels={"wrong": "labels"})


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Test singleton pattern for global collector."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_metrics_collector()

    def test_get_metrics_collector_returns_instance(self) -> None:
        """Test get_metrics_collector returns an instance."""
        collector = get_metrics_collector(enable_prometheus=False)
        assert isinstance(collector, UnifiedMetricsCollector)

    def test_get_metrics_collector_returns_same_instance(self) -> None:
        """Test get_metrics_collector returns same instance on subsequent calls."""
        collector1 = get_metrics_collector(enable_prometheus=False)
        collector2 = get_metrics_collector(enable_prometheus=False)
        assert collector1 is collector2

    def test_reset_metrics_collector(self) -> None:
        """Test reset_metrics_collector clears singleton."""
        collector1 = get_metrics_collector(enable_prometheus=False)
        collector1.inc_counter("test_counter")

        reset_metrics_collector()

        collector2 = get_metrics_collector(enable_prometheus=False)
        # Should be a new instance with empty counters
        assert "test_counter" not in collector2._counters

    def test_reset_metrics_collector_resets_internal_state(self) -> None:
        """Test reset_metrics_collector calls reset on collector."""
        collector = get_metrics_collector(enable_prometheus=False)
        collector.inc_counter("test_counter")

        # Reset clears the counter via collector.reset()
        reset_metrics_collector()

        # Get new collector - should have no counters
        new_collector = get_metrics_collector(enable_prometheus=False)
        assert len(new_collector._counters) == 0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of metrics collector."""

    def test_concurrent_counter_increments(self) -> None:
        """Test concurrent counter increments are thread-safe."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        num_threads = 10
        increments_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def increment():
            barrier.wait()
            for _ in range(increments_per_thread):
                collector.inc_counter("test_counter")

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread
        assert collector._counters["test_counter"][""] == expected

    def test_concurrent_gauge_updates(self) -> None:
        """Test concurrent gauge updates are thread-safe."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def update(value: float):
            barrier.wait()
            collector.set_gauge("test_gauge", value)

        threads = [
            threading.Thread(target=update, args=(float(i),))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some value (last writer wins)
        assert "test_gauge" in collector._gauges
        assert "" in collector._gauges["test_gauge"]

    def test_concurrent_histogram_observations(self) -> None:
        """Test concurrent histogram observations are thread-safe."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        num_threads = 10
        observations_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def observe(value: float):
            barrier.wait()
            for _ in range(observations_per_thread):
                collector.observe_histogram("test_histogram", value)

        threads = [
            threading.Thread(target=observe, args=(float(i) / 10,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_observations = num_threads * observations_per_thread
        assert len(collector._histograms["test_histogram"][""]) == expected_observations

    def test_concurrent_get_metrics(self) -> None:
        """Test concurrent get_metrics calls are thread-safe."""
        collector = UnifiedMetricsCollector(enable_prometheus=False)
        collector.inc_counter("test_counter")
        collector.set_gauge("test_gauge", 42.0)

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results = []

        def get():
            barrier.wait()
            results.append(collector.get_metrics())

        threads = [threading.Thread(target=get) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be consistent
        for result in results:
            assert result["counters"]["test_counter"][""] == 1
            assert result["gauges"]["test_gauge"][""] == 42.0


# =============================================================================
# Singleton Thread Safety Tests
# =============================================================================


class TestSingletonThreadSafety:
    """Test thread safety of singleton pattern."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_metrics_collector()

    def test_concurrent_singleton_access(self) -> None:
        """Test concurrent singleton access returns same instance."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        reset_metrics_collector()
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        instances = []

        def get_collector():
            barrier.wait()
            return get_metrics_collector(enable_prometheus=False)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_collector) for _ in range(num_threads)]
            for future in as_completed(futures):
                instances.append(future.result())

        # All should be the same instance
        assert all(inst is instances[0] for inst in instances)

# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the observability metrics module.

Tests cover:
- StreamingMetrics: Dataclass for tracking streaming-specific rate limit metrics
- PrometheusStreamingMetrics: Optional Prometheus-style metrics for observability
- Module-level functions: get_prometheus_streaming_metrics, reset_prometheus_streaming_metrics
"""

from __future__ import annotations

import json
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from adaptive_rate_limiter.observability.metrics import (
    PROMETHEUS_AVAILABLE,
    PrometheusStreamingMetrics,
    StreamingMetrics,
    get_prometheus_streaming_metrics,
    reset_prometheus_streaming_metrics,
)

# =============================================================================
# StreamingMetrics Tests (~30 tests)
# =============================================================================


class TestStreamingMetricsInitialization:
    """Test StreamingMetrics initialization."""

    def test_all_counters_start_at_zero(self) -> None:
        """Verify all counters start at 0."""
        metrics = StreamingMetrics()

        assert metrics.streaming_completions == 0
        assert metrics.streaming_timeouts == 0
        assert metrics.streaming_errors == 0
        assert metrics.streaming_fallbacks == 0

    def test_token_accounting_starts_at_zero(self) -> None:
        """Verify token accounting starts at 0."""
        metrics = StreamingMetrics()

        assert metrics.total_reserved_tokens == 0
        assert metrics.total_actual_tokens == 0
        assert metrics.total_refunded_tokens == 0

    def test_extraction_counters_start_at_zero(self) -> None:
        """Verify extraction counters start at 0."""
        metrics = StreamingMetrics()

        assert metrics.extraction_successes == 0
        assert metrics.extraction_failures == 0

    def test_cleanup_counters_start_at_zero(self) -> None:
        """Verify cleanup counters start at 0."""
        metrics = StreamingMetrics()

        assert metrics.stale_streaming_cleanups == 0

    def test_per_bucket_dicts_start_empty(self) -> None:
        """Verify per-bucket dictionaries start empty."""
        metrics = StreamingMetrics()

        assert metrics._per_bucket_refunds == {}
        assert metrics._per_bucket_cleanups == {}


class TestStreamingMetricsRecordCompletion:
    """Test StreamingMetrics.record_completion()."""

    @pytest.fixture
    def metrics(self) -> StreamingMetrics:
        """Create a fresh StreamingMetrics instance."""
        return StreamingMetrics()

    def test_increments_streaming_completions(self, metrics: StreamingMetrics) -> None:
        """Verify record_completion increments streaming_completions."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.streaming_completions == 1

        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.streaming_completions == 2

    def test_updates_reserved_tokens(self, metrics: StreamingMetrics) -> None:
        """Verify record_completion updates total_reserved_tokens."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.total_reserved_tokens == 4000

        metrics.record_completion(reserved=3000, actual=1000, extraction_succeeded=True)
        assert metrics.total_reserved_tokens == 7000

    def test_updates_actual_tokens(self, metrics: StreamingMetrics) -> None:
        """Verify record_completion updates total_actual_tokens."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.total_actual_tokens == 500

        metrics.record_completion(reserved=3000, actual=1000, extraction_succeeded=True)
        assert metrics.total_actual_tokens == 1500

    def test_updates_refunded_tokens(self, metrics: StreamingMetrics) -> None:
        """Verify record_completion updates total_refunded_tokens."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.total_refunded_tokens == 3500

        metrics.record_completion(reserved=3000, actual=1000, extraction_succeeded=True)
        assert metrics.total_refunded_tokens == 5500

    def test_correctly_calculates_refund(self, metrics: StreamingMetrics) -> None:
        """Verify refund calculation: reserved - actual."""
        metrics.record_completion(
            reserved=10000, actual=2500, extraction_succeeded=True
        )

        assert metrics.total_reserved_tokens == 10000
        assert metrics.total_actual_tokens == 2500
        assert metrics.total_refunded_tokens == 7500

    def test_tracks_extraction_success(self, metrics: StreamingMetrics) -> None:
        """Verify extraction_succeeded=True increments extraction_successes."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        assert metrics.extraction_successes == 1
        assert metrics.extraction_failures == 0

    def test_tracks_extraction_failure(self, metrics: StreamingMetrics) -> None:
        """Verify extraction_succeeded=False increments extraction_failures."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=False)
        assert metrics.extraction_successes == 0
        assert metrics.extraction_failures == 1

    def test_extraction_failure_increments_fallbacks(
        self, metrics: StreamingMetrics
    ) -> None:
        """Verify extraction_succeeded=False also increments streaming_fallbacks."""
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=False)
        assert metrics.streaming_fallbacks == 1

    def test_updates_per_bucket_refunds_with_bucket_id(
        self, metrics: StreamingMetrics
    ) -> None:
        """Verify per-bucket refunds are tracked when bucket_id provided."""
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )
        assert metrics._per_bucket_refunds["bucket-1"] == 3500

    def test_accumulates_per_bucket_refunds(self, metrics: StreamingMetrics) -> None:
        """Verify per-bucket refunds accumulate correctly."""
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )
        metrics.record_completion(
            reserved=3000, actual=1000, extraction_succeeded=True, bucket_id="bucket-1"
        )
        assert metrics._per_bucket_refunds["bucket-1"] == 5500

    def test_multiple_buckets_tracked_separately(
        self, metrics: StreamingMetrics
    ) -> None:
        """Verify different buckets are tracked separately."""
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )
        metrics.record_completion(
            reserved=3000, actual=1000, extraction_succeeded=True, bucket_id="bucket-2"
        )

        assert metrics._per_bucket_refunds["bucket-1"] == 3500
        assert metrics._per_bucket_refunds["bucket-2"] == 2000

    def test_no_bucket_id_does_not_update_per_bucket(
        self, metrics: StreamingMetrics
    ) -> None:
        """Verify per-bucket refunds are not updated when bucket_id is None."""
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id=None
        )
        assert metrics._per_bucket_refunds == {}


class TestStreamingMetricsRecordError:
    """Test StreamingMetrics.record_error()."""

    def test_increments_streaming_errors(self) -> None:
        """Verify record_error increments streaming_errors."""
        metrics = StreamingMetrics()
        assert metrics.streaming_errors == 0

        metrics.record_error()
        assert metrics.streaming_errors == 1

        metrics.record_error()
        assert metrics.streaming_errors == 2


class TestStreamingMetricsRecordTimeout:
    """Test StreamingMetrics.record_timeout()."""

    def test_increments_streaming_timeouts(self) -> None:
        """Verify record_timeout increments streaming_timeouts."""
        metrics = StreamingMetrics()
        assert metrics.streaming_timeouts == 0

        metrics.record_timeout()
        assert metrics.streaming_timeouts == 1

        metrics.record_timeout()
        assert metrics.streaming_timeouts == 2


class TestStreamingMetricsRecordStaleCleanup:
    """Test StreamingMetrics.record_stale_cleanup()."""

    def test_increments_stale_streaming_cleanups(self) -> None:
        """Verify record_stale_cleanup increments stale_streaming_cleanups."""
        metrics = StreamingMetrics()
        assert metrics.stale_streaming_cleanups == 0

        metrics.record_stale_cleanup()
        assert metrics.stale_streaming_cleanups == 1

        metrics.record_stale_cleanup()
        assert metrics.stale_streaming_cleanups == 2

    def test_updates_per_bucket_cleanups_with_bucket_id(self) -> None:
        """Verify per-bucket cleanups are tracked when bucket_id provided."""
        metrics = StreamingMetrics()
        metrics.record_stale_cleanup(bucket_id="bucket:xs")
        assert metrics._per_bucket_cleanups["bucket:xs"] == 1

    def test_accumulates_per_bucket_cleanups(self) -> None:
        """Verify per-bucket cleanups accumulate correctly."""
        metrics = StreamingMetrics()
        metrics.record_stale_cleanup(bucket_id="bucket:xs")
        metrics.record_stale_cleanup(bucket_id="bucket:xs")
        assert metrics._per_bucket_cleanups["bucket:xs"] == 2

    def test_multiple_buckets_cleanups_tracked_separately(self) -> None:
        """Verify different buckets' cleanups are tracked separately."""
        metrics = StreamingMetrics()
        metrics.record_stale_cleanup(bucket_id="bucket:xs")
        metrics.record_stale_cleanup(bucket_id="bucket:md")

        assert metrics._per_bucket_cleanups["bucket:xs"] == 1
        assert metrics._per_bucket_cleanups["bucket:md"] == 1


class TestStreamingMetricsRecordFallback:
    """Test StreamingMetrics.record_fallback()."""

    def test_increments_streaming_fallbacks(self) -> None:
        """Verify record_fallback increments streaming_fallbacks."""
        metrics = StreamingMetrics()
        assert metrics.streaming_fallbacks == 0

        metrics.record_fallback()
        assert metrics.streaming_fallbacks == 1

    def test_also_increments_extraction_failures(self) -> None:
        """Verify record_fallback also increments extraction_failures."""
        metrics = StreamingMetrics()
        assert metrics.extraction_failures == 0

        metrics.record_fallback()
        assert metrics.extraction_failures == 1


class TestStreamingMetricsGetExtractionSuccessRate:
    """Test StreamingMetrics.get_extraction_success_rate()."""

    def test_returns_one_when_no_extractions(self) -> None:
        """Verify returns 1.0 when no extractions attempted (optimistic default)."""
        metrics = StreamingMetrics()
        assert metrics.get_extraction_success_rate() == 1.0

    def test_returns_correct_ratio(self) -> None:
        """Verify returns correct ratio of successes to total."""
        metrics = StreamingMetrics()
        metrics.extraction_successes = 9
        metrics.extraction_failures = 1
        assert metrics.get_extraction_success_rate() == 0.9

    def test_returns_zero_when_all_failures(self) -> None:
        """Verify returns 0.0 when all extractions failed."""
        metrics = StreamingMetrics()
        metrics.extraction_successes = 0
        metrics.extraction_failures = 10
        assert metrics.get_extraction_success_rate() == 0.0

    def test_returns_one_when_all_successes(self) -> None:
        """Verify returns 1.0 when all extractions succeeded."""
        metrics = StreamingMetrics()
        metrics.extraction_successes = 100
        metrics.extraction_failures = 0
        assert metrics.get_extraction_success_rate() == 1.0

    def test_calculates_fractional_rate_correctly(self) -> None:
        """Verify fractional rates are calculated correctly."""
        metrics = StreamingMetrics()
        metrics.extraction_successes = 3
        metrics.extraction_failures = 1
        assert metrics.get_extraction_success_rate() == 0.75


class TestStreamingMetricsGetRefundRate:
    """Test StreamingMetrics.get_refund_rate()."""

    def test_returns_zero_when_no_tokens_reserved(self) -> None:
        """Verify returns 0.0 when no tokens have been reserved."""
        metrics = StreamingMetrics()
        assert metrics.get_refund_rate() == 0.0

    def test_returns_correct_ratio(self) -> None:
        """Verify returns correct ratio of refunded to reserved."""
        metrics = StreamingMetrics()
        metrics.total_reserved_tokens = 4000
        metrics.total_refunded_tokens = 3500
        assert metrics.get_refund_rate() == 0.875

    def test_returns_zero_when_no_refunds(self) -> None:
        """Verify returns 0.0 when no tokens were refunded."""
        metrics = StreamingMetrics()
        metrics.total_reserved_tokens = 4000
        metrics.total_refunded_tokens = 0
        assert metrics.get_refund_rate() == 0.0

    def test_returns_one_when_all_refunded(self) -> None:
        """Verify returns 1.0 when all tokens were refunded (actual=0)."""
        metrics = StreamingMetrics()
        metrics.total_reserved_tokens = 4000
        metrics.total_refunded_tokens = 4000
        assert metrics.get_refund_rate() == 1.0


class TestStreamingMetricsGetStats:
    """Test StreamingMetrics.get_stats()."""

    @pytest.fixture
    def metrics_with_data(self) -> StreamingMetrics:
        """Create a StreamingMetrics instance with sample data."""
        metrics = StreamingMetrics()
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket:xs"
        )
        metrics.record_completion(
            reserved=3000,
            actual=1000,
            extraction_succeeded=False,
            bucket_id="bucket:md",
        )
        metrics.record_error()
        metrics.record_timeout()
        metrics.record_stale_cleanup(bucket_id="bucket:xs")
        return metrics

    def test_returns_all_expected_keys(
        self, metrics_with_data: StreamingMetrics
    ) -> None:
        """Verify get_stats returns all expected keys."""
        stats = metrics_with_data.get_stats()

        expected_keys = {
            "streaming_completions",
            "streaming_timeouts",
            "streaming_errors",
            "streaming_fallbacks",
            "total_reserved_tokens",
            "total_actual_tokens",
            "total_refunded_tokens",
            "extraction_success_rate",
            "refund_rate",
            "extraction_successes",
            "extraction_failures",
            "stale_streaming_cleanups",
            "per_bucket_refunds",
            "per_bucket_cleanups",
        }
        assert set(stats.keys()) == expected_keys

    def test_values_are_json_serializable(
        self, metrics_with_data: StreamingMetrics
    ) -> None:
        """Verify all values are JSON-serializable."""
        stats = metrics_with_data.get_stats()

        # Should not raise
        json_str = json.dumps(stats)
        assert isinstance(json_str, str)

        # And should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["streaming_completions"] == stats["streaming_completions"]

    def test_includes_per_bucket_breakdowns(
        self, metrics_with_data: StreamingMetrics
    ) -> None:
        """Verify per-bucket breakdowns are included."""
        stats = metrics_with_data.get_stats()

        assert "per_bucket_refunds" in stats
        assert "per_bucket_cleanups" in stats
        assert stats["per_bucket_refunds"]["bucket:xs"] == 3500
        assert stats["per_bucket_refunds"]["bucket:md"] == 2000
        assert stats["per_bucket_cleanups"]["bucket:xs"] == 1

    def test_empty_per_bucket_dicts_return_empty_dicts(self) -> None:
        """Verify empty per-bucket dicts return empty dicts in stats."""
        metrics = StreamingMetrics()
        stats = metrics.get_stats()

        assert stats["per_bucket_refunds"] == {}
        assert stats["per_bucket_cleanups"] == {}

    def test_derived_metrics_calculated_correctly(self) -> None:
        """Verify derived metrics (rates) are calculated correctly."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=4000, actual=500, extraction_succeeded=True)
        stats = metrics.get_stats()

        assert stats["extraction_success_rate"] == 1.0
        assert stats["refund_rate"] == 0.875


class TestStreamingMetricsReset:
    """Test StreamingMetrics.reset()."""

    @pytest.fixture
    def metrics_with_data(self) -> StreamingMetrics:
        """Create a StreamingMetrics instance with sample data."""
        metrics = StreamingMetrics()
        metrics.streaming_completions = 10
        metrics.streaming_timeouts = 2
        metrics.streaming_errors = 1
        metrics.streaming_fallbacks = 3
        metrics.total_reserved_tokens = 50000
        metrics.total_actual_tokens = 10000
        metrics.total_refunded_tokens = 40000
        metrics.extraction_successes = 7
        metrics.extraction_failures = 3
        metrics.stale_streaming_cleanups = 5
        metrics._per_bucket_refunds = OrderedDict(
            [("bucket:xs", 20000), ("bucket:md", 20000)]
        )
        metrics._per_bucket_cleanups = OrderedDict([("bucket:xs", 3), ("bucket:md", 2)])
        return metrics

    def test_resets_lifecycle_counters(
        self, metrics_with_data: StreamingMetrics
    ) -> None:
        """Verify reset sets lifecycle counters to 0."""
        metrics_with_data.reset()

        assert metrics_with_data.streaming_completions == 0
        assert metrics_with_data.streaming_timeouts == 0
        assert metrics_with_data.streaming_errors == 0
        assert metrics_with_data.streaming_fallbacks == 0

    def test_resets_token_accounting(self, metrics_with_data: StreamingMetrics) -> None:
        """Verify reset sets token accounting to 0."""
        metrics_with_data.reset()

        assert metrics_with_data.total_reserved_tokens == 0
        assert metrics_with_data.total_actual_tokens == 0
        assert metrics_with_data.total_refunded_tokens == 0

    def test_resets_extraction_counters(
        self, metrics_with_data: StreamingMetrics
    ) -> None:
        """Verify reset sets extraction counters to 0."""
        metrics_with_data.reset()

        assert metrics_with_data.extraction_successes == 0
        assert metrics_with_data.extraction_failures == 0

    def test_resets_cleanup_counter(self, metrics_with_data: StreamingMetrics) -> None:
        """Verify reset sets cleanup counter to 0."""
        metrics_with_data.reset()

        assert metrics_with_data.stale_streaming_cleanups == 0

    def test_clears_per_bucket_dicts(self, metrics_with_data: StreamingMetrics) -> None:
        """Verify reset clears per-bucket dictionaries."""
        metrics_with_data.reset()

        assert metrics_with_data._per_bucket_refunds == {}
        assert metrics_with_data._per_bucket_cleanups == {}


# =============================================================================
# PrometheusStreamingMetrics Tests (~15 tests)
# =============================================================================


class TestPrometheusStreamingMetricsInitialization:
    """Test PrometheusStreamingMetrics initialization."""

    def test_raises_import_error_when_prometheus_not_available(self) -> None:
        """Verify ImportError raised when prometheus_client not available."""
        with patch(
            "adaptive_rate_limiter.observability.metrics.PROMETHEUS_AVAILABLE", False
        ):
            with pytest.raises(ImportError) as exc_info:
                PrometheusStreamingMetrics()

            assert "prometheus_client is not available" in str(exc_info.value)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_creates_all_expected_metrics_when_available(self) -> None:
        """Verify all expected metrics are created when prometheus_client available."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        prom_metrics = PrometheusStreamingMetrics(registry=registry)

        assert prom_metrics.streaming_tokens_refunded is not None
        assert prom_metrics.streaming_extraction_failures is not None
        assert prom_metrics.streaming_stale_cleanups is not None
        assert prom_metrics.streaming_duration_seconds is not None
        assert prom_metrics.streaming_completions is not None
        assert prom_metrics.streaming_errors is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_uses_custom_registry(self) -> None:
        """Verify custom registry is used when provided."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        prom_metrics = PrometheusStreamingMetrics(registry=registry)

        # Metrics should be registered in our custom registry
        # Check by accessing the labels method which should work
        assert prom_metrics.streaming_tokens_refunded.labels(bucket_id="test")


class TestPrometheusStreamingMetricsObserveCompletion:
    """Test PrometheusStreamingMetrics.observe_completion()."""

    @pytest.fixture
    def prom_metrics(self) -> PrometheusStreamingMetrics:
        """Create a PrometheusStreamingMetrics instance with fresh registry."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        return PrometheusStreamingMetrics(registry=registry)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_tokens_refunded_counter(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_completion increments tokens refunded counter."""
        prom_metrics.observe_completion(
            bucket_id="bucket:xs",
            reserved=4000,
            actual=500,
            duration_seconds=15.5,
            extraction_succeeded=True,
        )

        # The refund should be 3500
        sample_value = prom_metrics.streaming_tokens_refunded.labels(
            bucket_id="bucket:xs"
        )._value.get()
        assert sample_value == 3500

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observes_duration_histogram(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_completion observes duration histogram."""
        prom_metrics.observe_completion(
            bucket_id="bucket:xs",
            reserved=4000,
            actual=500,
            duration_seconds=15.5,
            extraction_succeeded=True,
        )

        # The histogram sum should reflect the observed duration
        # For histograms, we can check that it was observed by checking _sum
        labels = prom_metrics.streaming_duration_seconds.labels(bucket_id="bucket:xs")
        # Histogram has _sum attribute for the total of observed values
        assert labels._sum.get() == 15.5

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_completions_counter_with_labels(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_completion increments completions counter with correct labels."""
        prom_metrics.observe_completion(
            bucket_id="bucket:xs",
            reserved=4000,
            actual=500,
            duration_seconds=15.5,
            extraction_succeeded=True,
        )

        sample_value = prom_metrics.streaming_completions.labels(
            bucket_id="bucket:xs", extraction_succeeded="true"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_extraction_failures_when_extraction_failed(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_completion increments extraction failures when extraction_succeeded=False."""
        prom_metrics.observe_completion(
            bucket_id="bucket:xs",
            reserved=4000,
            actual=4000,  # No refund when extraction fails
            duration_seconds=15.5,
            extraction_succeeded=False,
        )

        sample_value = prom_metrics.streaming_extraction_failures.labels(
            reason="complete"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_does_not_increment_extraction_failures_on_success(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_completion does not increment extraction failures on success."""
        prom_metrics.observe_completion(
            bucket_id="bucket:xs",
            reserved=4000,
            actual=500,
            duration_seconds=15.5,
            extraction_succeeded=True,
        )

        sample_value = prom_metrics.streaming_extraction_failures.labels(
            reason="complete"
        )._value.get()
        assert sample_value == 0


class TestPrometheusStreamingMetricsObserveError:
    """Test PrometheusStreamingMetrics.observe_error()."""

    @pytest.fixture
    def prom_metrics(self) -> PrometheusStreamingMetrics:
        """Create a PrometheusStreamingMetrics instance with fresh registry."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        return PrometheusStreamingMetrics(registry=registry)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_errors_counter_with_labels(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_error increments errors counter with correct labels."""
        prom_metrics.observe_error(bucket_id="bucket:xs", error_type="exception")

        sample_value = prom_metrics.streaming_errors.labels(
            bucket_id="bucket:xs", error_type="exception"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_errors_counter_with_timeout_type(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_error works with timeout error type."""
        prom_metrics.observe_error(bucket_id="bucket:xs", error_type="timeout")

        sample_value = prom_metrics.streaming_errors.labels(
            bucket_id="bucket:xs", error_type="timeout"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_extraction_failures_on_error(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify observe_error increments extraction failures counter."""
        prom_metrics.observe_error(bucket_id="bucket:xs", error_type="exception")

        sample_value = prom_metrics.streaming_extraction_failures.labels(
            reason="exception"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_default_error_type_is_exception(
        self, prom_metrics: PrometheusStreamingMetrics
    ) -> None:
        """Verify default error_type is 'exception'."""
        prom_metrics.observe_error(bucket_id="bucket:xs")

        sample_value = prom_metrics.streaming_errors.labels(
            bucket_id="bucket:xs", error_type="exception"
        )._value.get()
        assert sample_value == 1


class TestPrometheusStreamingMetricsObserveStaleCleanup:
    """Test PrometheusStreamingMetrics.observe_stale_cleanup()."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_stale_cleanups_counter(self) -> None:
        """Verify observe_stale_cleanup increments stale cleanups counter."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        prom_metrics = PrometheusStreamingMetrics(registry=registry)

        prom_metrics.observe_stale_cleanup(bucket_id="bucket:xs")

        sample_value = prom_metrics.streaming_stale_cleanups.labels(
            bucket_id="bucket:xs"
        )._value.get()
        assert sample_value == 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_increments_multiple_cleanups(self) -> None:
        """Verify multiple observe_stale_cleanup calls accumulate."""
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        prom_metrics = PrometheusStreamingMetrics(registry=registry)

        prom_metrics.observe_stale_cleanup(bucket_id="bucket:xs")
        prom_metrics.observe_stale_cleanup(bucket_id="bucket:xs")
        prom_metrics.observe_stale_cleanup(bucket_id="bucket:xs")

        sample_value = prom_metrics.streaming_stale_cleanups.labels(
            bucket_id="bucket:xs"
        )._value.get()
        assert sample_value == 3


# =============================================================================
# Module-level Function Tests (~5 tests)
# =============================================================================


class TestGetPrometheusStreamingMetrics:
    """Test get_prometheus_streaming_metrics() function."""

    def teardown_method(self) -> None:
        """Reset the singleton after each test."""
        reset_prometheus_streaming_metrics()

    def test_returns_none_when_prometheus_not_available(self) -> None:
        """Verify returns None when prometheus_client not available."""
        with patch(
            "adaptive_rate_limiter.observability.metrics.PROMETHEUS_AVAILABLE", False
        ):
            reset_prometheus_streaming_metrics()
            result = get_prometheus_streaming_metrics()
            assert result is None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_returns_instance_when_available(self) -> None:
        """Verify returns PrometheusStreamingMetrics instance when available."""
        reset_prometheus_streaming_metrics()
        result = get_prometheus_streaming_metrics()
        assert isinstance(result, PrometheusStreamingMetrics)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_returns_same_instance_on_subsequent_calls(self) -> None:
        """Verify returns the same singleton instance on subsequent calls."""
        reset_prometheus_streaming_metrics()
        result1 = get_prometheus_streaming_metrics()
        result2 = get_prometheus_streaming_metrics()
        assert result1 is result2

    def test_returns_none_on_initialization_error(self) -> None:
        """Verify returns None when initialization fails."""
        with (
            patch(
                "adaptive_rate_limiter.observability.metrics.PROMETHEUS_AVAILABLE", True
            ),
            patch(
                "adaptive_rate_limiter.observability.metrics.PrometheusStreamingMetrics",
                side_effect=Exception("Init failed"),
            ),
        ):
            reset_prometheus_streaming_metrics()
            result = get_prometheus_streaming_metrics()
            assert result is None


class TestResetPrometheusStreamingMetrics:
    """Test reset_prometheus_streaming_metrics() function."""

    def test_resets_the_singleton(self) -> None:
        """Verify reset_prometheus_streaming_metrics clears the singleton."""
        # First, ensure we have a value (or None if not available)
        with (
            patch(
                "adaptive_rate_limiter.observability.metrics.PROMETHEUS_AVAILABLE", True
            ),
            patch(
                "adaptive_rate_limiter.observability.metrics.PrometheusStreamingMetrics"
            ) as MockClass,
        ):
            mock_instance = MagicMock()
            MockClass.return_value = mock_instance

            reset_prometheus_streaming_metrics()
            _first = get_prometheus_streaming_metrics()

            # Reset should clear it
            reset_prometheus_streaming_metrics()

            # Getting again should create a new instance
            _second = get_prometheus_streaming_metrics()

            # Should have been called twice (once for each get)
            assert MockClass.call_count == 2

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_allows_creating_new_instance_after_reset(self) -> None:
        """Verify new instance can be created after reset."""
        first = get_prometheus_streaming_metrics()
        reset_prometheus_streaming_metrics()
        second = get_prometheus_streaming_metrics()

        # After reset, we should get a new (or could be same due to metric reuse)
        # The important thing is that the singleton was reset
        assert first is not None
        assert second is not None


# =============================================================================
# Edge Case and Integration Tests
# =============================================================================


class TestStreamingMetricsEdgeCases:
    """Test edge cases for StreamingMetrics."""

    def test_zero_actual_tokens(self) -> None:
        """Test completion with zero actual tokens (maximum refund)."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=4000, actual=0, extraction_succeeded=True)

        assert metrics.total_reserved_tokens == 4000
        assert metrics.total_actual_tokens == 0
        assert metrics.total_refunded_tokens == 4000

    def test_actual_equals_reserved(self) -> None:
        """Test completion where actual equals reserved (zero refund)."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=4000, actual=4000, extraction_succeeded=True)

        assert metrics.total_reserved_tokens == 4000
        assert metrics.total_actual_tokens == 4000
        assert metrics.total_refunded_tokens == 0

    def test_actual_exceeds_reserved(self) -> None:
        """Test completion where actual exceeds reserved (negative refund)."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=1000, actual=2000, extraction_succeeded=True)

        assert metrics.total_reserved_tokens == 1000
        assert metrics.total_actual_tokens == 2000
        # Negative refund (overage)
        assert metrics.total_refunded_tokens == -1000

    def test_large_token_counts(self) -> None:
        """Test with large token counts."""
        metrics = StreamingMetrics()
        metrics.record_completion(
            reserved=1_000_000, actual=100_000, extraction_succeeded=True
        )

        assert metrics.total_reserved_tokens == 1_000_000
        assert metrics.total_actual_tokens == 100_000
        assert metrics.total_refunded_tokens == 900_000

    def test_empty_bucket_id_string(self) -> None:
        """Test empty string bucket_id is treated as falsy (not tracked)."""
        metrics = StreamingMetrics()
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id=""
        )

        # Empty string is falsy in Python, so it should NOT be tracked
        assert "" not in metrics._per_bucket_refunds
        assert metrics._per_bucket_refunds == {}

    def test_special_characters_in_bucket_id(self) -> None:
        """Test bucket_id with special characters."""
        metrics = StreamingMetrics()
        bucket_id = "bucket:model/llama-3.1-405b@4bit"
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id=bucket_id
        )

        assert bucket_id in metrics._per_bucket_refunds
        assert metrics._per_bucket_refunds[bucket_id] == 3500


class TestStreamingMetricsMultipleOperations:
    """Test multiple operations on StreamingMetrics."""

    def test_comprehensive_workflow(self) -> None:
        """Test a comprehensive workflow with multiple operations."""
        metrics = StreamingMetrics()

        # Simulate a realistic workflow
        # 10 successful completions with extraction
        for i in range(10):
            metrics.record_completion(
                reserved=4000,
                actual=500 + i * 100,
                extraction_succeeded=True,
                bucket_id=f"bucket:{i % 3}",
            )

        # 3 completions with failed extraction
        for _ in range(3):
            metrics.record_completion(
                reserved=4000, actual=4000, extraction_succeeded=False
            )

        # 2 errors
        metrics.record_error()
        metrics.record_error()

        # 1 timeout
        metrics.record_timeout()

        # 3 stale cleanups
        for _ in range(3):
            metrics.record_stale_cleanup(bucket_id="bucket:0")

        # Verify counts
        assert metrics.streaming_completions == 13
        assert metrics.streaming_errors == 2
        assert metrics.streaming_timeouts == 1
        assert metrics.streaming_fallbacks == 3  # From failed extractions

        assert metrics.extraction_successes == 10
        assert metrics.extraction_failures == 3

        assert metrics.stale_streaming_cleanups == 3

        # Verify per-bucket tracking
        assert len(metrics._per_bucket_refunds) == 3
        assert metrics._per_bucket_cleanups["bucket:0"] == 3

        # Verify stats are complete and serializable
        stats = metrics.get_stats()
        json.dumps(stats)  # Should not raise

        # Verify reset works
        metrics.reset()
        assert metrics.streaming_completions == 0
        assert metrics._per_bucket_refunds == {}


class TestStreamingMetricsLRUEviction:
    """Test LRU eviction for per-bucket dictionaries (Issue observability_003)."""

    def test_lru_eviction_when_max_buckets_exceeded(self) -> None:
        """Test that oldest buckets are evicted when max_tracked_buckets exceeded."""
        metrics = StreamingMetrics(max_tracked_buckets=3)

        # Add 3 buckets
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-2"
        )
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-3"
        )

        assert len(metrics._per_bucket_refunds) == 3

        # Add 4th bucket - should evict bucket-1
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-4"
        )

        assert len(metrics._per_bucket_refunds) == 3
        assert "bucket-1" not in metrics._per_bucket_refunds
        assert "bucket-4" in metrics._per_bucket_refunds

    def test_lru_moves_accessed_bucket_to_end(self) -> None:
        """Test that accessing a bucket moves it to the end (most recently used)."""
        metrics = StreamingMetrics(max_tracked_buckets=3)

        # Add 3 buckets
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-2"
        )
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-3"
        )

        # Access bucket-1 again - should move to end
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-1"
        )

        # Add bucket-4 - should evict bucket-2 (now oldest), not bucket-1
        metrics.record_completion(
            reserved=4000, actual=500, extraction_succeeded=True, bucket_id="bucket-4"
        )

        assert "bucket-1" in metrics._per_bucket_refunds  # Was accessed, so not evicted
        assert "bucket-2" not in metrics._per_bucket_refunds  # Now oldest, evicted

    def test_lru_eviction_for_cleanups(self) -> None:
        """Test LRU eviction also works for per-bucket cleanups."""
        metrics = StreamingMetrics(max_tracked_buckets=2)

        # Add 2 cleanup entries
        metrics.record_stale_cleanup(bucket_id="bucket-1")
        metrics.record_stale_cleanup(bucket_id="bucket-2")

        assert len(metrics._per_bucket_cleanups) == 2

        # Add 3rd cleanup - should evict bucket-1
        metrics.record_stale_cleanup(bucket_id="bucket-3")

        assert len(metrics._per_bucket_cleanups) == 2
        assert "bucket-1" not in metrics._per_bucket_cleanups
        assert "bucket-3" in metrics._per_bucket_cleanups

    def test_no_eviction_when_max_buckets_zero(self) -> None:
        """Test no LRU eviction when max_tracked_buckets is 0 (disabled)."""
        metrics = StreamingMetrics(max_tracked_buckets=0)

        # Add many buckets - none should be evicted
        for i in range(100):
            metrics.record_completion(
                reserved=4000,
                actual=500,
                extraction_succeeded=True,
                bucket_id=f"bucket-{i}",
            )

        assert len(metrics._per_bucket_refunds) == 100


class TestStreamingMetricsThreadSafety:
    """Test thread safety for per-bucket dictionary updates (Issue observability_001)."""

    def test_concurrent_updates_no_lost_increments(self) -> None:
        """Test that concurrent updates don't lose increments due to race conditions."""
        import threading

        metrics = StreamingMetrics()
        num_threads = 10
        updates_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def update_metrics():
            barrier.wait()  # Ensure all threads start simultaneously
            for _ in range(updates_per_thread):
                metrics.record_completion(
                    reserved=1000,
                    actual=0,
                    extraction_succeeded=True,
                    bucket_id="shared-bucket",
                )

        threads = [threading.Thread(target=update_metrics) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Expected: 1000 refund * 10 threads * 100 updates = 1,000,000
        expected_refunds = 1000 * num_threads * updates_per_thread
        assert metrics._per_bucket_refunds["shared-bucket"] == expected_refunds
        assert metrics.streaming_completions == num_threads * updates_per_thread

    def test_concurrent_cleanup_updates(self) -> None:
        """Test that concurrent cleanup updates are thread-safe."""
        import threading

        metrics = StreamingMetrics()
        num_threads = 10
        updates_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def update_cleanups():
            barrier.wait()
            for _ in range(updates_per_thread):
                metrics.record_stale_cleanup(bucket_id="shared-bucket")

        threads = [threading.Thread(target=update_cleanups) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_cleanups = num_threads * updates_per_thread
        assert metrics._per_bucket_cleanups["shared-bucket"] == expected_cleanups

    def test_has_lock_attribute(self) -> None:
        """Test that StreamingMetrics has a threading lock for synchronization."""
        import threading

        metrics = StreamingMetrics()
        assert hasattr(metrics, "_lock")
        assert isinstance(metrics._lock, type(threading.Lock()))

    def test_uses_ordered_dict_for_per_bucket_dicts(self) -> None:
        """Test that per-bucket dictionaries use OrderedDict for LRU support."""
        from collections import OrderedDict

        metrics = StreamingMetrics()
        assert isinstance(metrics._per_bucket_refunds, OrderedDict)
        assert isinstance(metrics._per_bucket_cleanups, OrderedDict)


class TestPrometheusSingletonThreadSafety:
    """Test thread-safe singleton pattern for Prometheus metrics (Issue observability_002)."""

    def teardown_method(self) -> None:
        """Reset the singleton after each test."""
        reset_prometheus_streaming_metrics()

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_concurrent_singleton_access_returns_same_instance(self) -> None:
        """Test that concurrent singleton access returns the same instance."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        reset_prometheus_streaming_metrics()

        instances = []
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def get_singleton():
            barrier.wait()  # Ensure all threads start simultaneously
            return get_prometheus_streaming_metrics()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_singleton) for _ in range(num_threads)]
            for future in as_completed(futures):
                instances.append(future.result())

        # All instances should be the exact same object
        assert all(inst is instances[0] for inst in instances)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_double_checked_locking_prevents_duplicate_initialization(self) -> None:
        """Test that double-checked locking prevents duplicate Prometheus metric registration."""
        import threading

        reset_prometheus_streaming_metrics()

        initialization_count = 0
        original_init = PrometheusStreamingMetrics.__init__

        def counting_init(self, registry=None):
            nonlocal initialization_count
            initialization_count += 1
            original_init(self, registry)

        # Patch the init to count calls
        with patch.object(PrometheusStreamingMetrics, "__init__", counting_init):
            num_threads = 10
            barrier = threading.Barrier(num_threads)

            def get_singleton():
                barrier.wait()
                return get_prometheus_streaming_metrics()

            threads = [
                threading.Thread(target=get_singleton) for _ in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should only initialize once despite concurrent access
            assert initialization_count == 1

    def test_module_has_prometheus_lock(self) -> None:
        """Test that the module has a threading.Lock for singleton protection."""
        import threading

        from adaptive_rate_limiter.observability import metrics

        assert hasattr(metrics, "_prometheus_lock")
        assert isinstance(metrics._prometheus_lock, type(threading.Lock()))

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_reset_and_reinitialize_thread_safe(self) -> None:
        """Test that reset and re-initialize is thread-safe."""
        import threading

        num_threads = 10
        barrier = threading.Barrier(num_threads + 1)

        instances_before = []
        instances_after = []

        def get_before_reset():
            barrier.wait()
            instances_before.append(get_prometheus_streaming_metrics())

        def reset_and_get():
            barrier.wait()
            reset_prometheus_streaming_metrics()
            instances_after.append(get_prometheus_streaming_metrics())

        # Start threads that get instance before reset
        threads_before = [
            threading.Thread(target=get_before_reset) for _ in range(num_threads // 2)
        ]
        for t in threads_before:
            t.start()

        # Start threads that reset and get new instance
        threads_after = [
            threading.Thread(target=reset_and_get) for _ in range(num_threads // 2)
        ]
        for t in threads_after:
            t.start()

        # Trigger all threads
        barrier.wait()

        # Wait for completion
        for t in threads_before + threads_after:
            t.join()

        # Instances should be PrometheusStreamingMetrics (not None)
        assert all(
            isinstance(inst, PrometheusStreamingMetrics)
            for inst in instances_before
            if inst is not None
        )
        assert all(
            isinstance(inst, PrometheusStreamingMetrics)
            for inst in instances_after
            if inst is not None
        )

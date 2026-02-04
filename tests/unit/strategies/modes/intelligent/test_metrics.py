"""
Unit tests for ReservationMetrics and StreamingMetrics.

Tests the metrics dataclasses used by IntelligentModeStrategy.
"""

from adaptive_rate_limiter.observability.metrics import StreamingMetrics
from adaptive_rate_limiter.strategies.modes.intelligent import ReservationMetrics

# ============================================================================
# ReservationMetrics Tests
# ============================================================================


class TestReservationMetrics:
    """Tests for ReservationMetrics dataclass."""

    def test_record_stale_cleanup(self):
        """Test stale cleanup recording."""
        metrics = ReservationMetrics()
        metrics.record_stale_cleanup("bucket-1")
        metrics.record_stale_cleanup("bucket-1")
        metrics.record_stale_cleanup("bucket-2")

        assert metrics.stale_cleanups["bucket-1"] == 2
        assert metrics.stale_cleanups["bucket-2"] == 1

    def test_record_emergency_cleanup(self):
        """Test emergency cleanup recording."""
        metrics = ReservationMetrics()
        metrics.record_emergency_cleanup()
        metrics.record_emergency_cleanup()

        assert metrics.emergency_cleanups == 2

    def test_record_backpressure_rejection(self):
        """Test backpressure rejection recording."""
        metrics = ReservationMetrics()
        metrics.record_backpressure_rejection("bucket-1")

        assert metrics.backpressure_rejections["bucket-1"] == 1

    def test_record_update_fallback(self):
        """Test update fallback recording."""
        metrics = ReservationMetrics()
        metrics.record_update_fallback("bucket-1", "no_headers")
        metrics.record_update_fallback("bucket-1", "partial_headers")
        metrics.record_update_fallback("bucket-1", "no_headers")

        assert metrics.update_fallbacks["bucket-1"]["no_headers"] == 2
        assert metrics.update_fallbacks["bucket-1"]["partial_headers"] == 1

    def test_record_update_fallback_unknown_bucket(self):
        """Test update fallback with None bucket."""
        metrics = ReservationMetrics()
        metrics.record_update_fallback(None, "reason")

        assert metrics.update_fallbacks["unknown"]["reason"] == 1

    def test_record_request_type_streaming(self):
        """Test streaming request type recording."""
        metrics = ReservationMetrics()
        metrics.record_request_type(is_streaming=True)

        assert metrics.streaming_requests == 1
        assert metrics.non_streaming_requests == 0

    def test_record_request_type_non_streaming(self):
        """Test non-streaming request type recording."""
        metrics = ReservationMetrics()
        metrics.record_request_type(is_streaming=False)

        assert metrics.streaming_requests == 0
        assert metrics.non_streaming_requests == 1

    def test_get_streaming_ratio_no_requests(self):
        """Test streaming ratio with no requests."""
        metrics = ReservationMetrics()
        assert metrics.get_streaming_ratio() == 0.0

    def test_get_streaming_ratio(self):
        """Test streaming ratio calculation."""
        metrics = ReservationMetrics()
        metrics.streaming_requests = 3
        metrics.non_streaming_requests = 7

        assert metrics.get_streaming_ratio() == 0.3

    def test_get_stats(self):
        """Test get_stats returns complete stats."""
        metrics = ReservationMetrics()
        metrics.record_stale_cleanup("bucket-1")
        metrics.record_request_type(is_streaming=True)

        stats = metrics.get_stats()

        assert "stale_cleanups" in stats
        assert "emergency_cleanups" in stats
        assert "backpressure_rejections" in stats
        assert "update_fallbacks" in stats
        assert "streaming_requests" in stats
        assert "non_streaming_requests" in stats
        assert "streaming_ratio" in stats


# ============================================================================
# StreamingMetrics Tests
# ============================================================================


class TestStreamingMetrics:
    """Tests for StreamingMetrics dataclass."""

    def test_record_completion(self):
        """Test completion recording."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=1000, actual=800, extraction_succeeded=True)

        assert metrics.streaming_completions == 1
        assert metrics.total_reserved_tokens == 1000
        assert metrics.total_actual_tokens == 800
        assert metrics.extraction_successes == 1
        assert metrics.extraction_failures == 0

    def test_record_completion_extraction_failed(self):
        """Test completion with failed extraction."""
        metrics = StreamingMetrics()
        metrics.record_completion(reserved=500, actual=500, extraction_succeeded=False)

        assert metrics.extraction_successes == 0
        assert metrics.extraction_failures == 1

    def test_record_error(self):
        """Test error recording."""
        metrics = StreamingMetrics()
        metrics.record_error()
        metrics.record_error()

        assert metrics.streaming_errors == 2

    def test_record_stale_cleanup(self):
        """Test stale cleanup recording."""
        metrics = StreamingMetrics()
        metrics.record_stale_cleanup("bucket-1")

        assert metrics.stale_streaming_cleanups == 1

    def test_get_stats(self):
        """Test get_stats returns complete stats."""
        metrics = StreamingMetrics()
        metrics.record_completion(1000, 800, True)
        metrics.record_error()

        stats = metrics.get_stats()

        assert stats["streaming_completions"] == 1
        assert stats["streaming_errors"] == 1
        # The canonical StreamingMetrics uses total_refunded_tokens instead of refund_total
        assert stats["total_refunded_tokens"] == 200

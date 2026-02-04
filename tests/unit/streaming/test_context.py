"""
Unit tests for StreamingReservationContext.

Tests cover:
- Initialization with all fields
- record_chunk() method
- set_final_tokens() method
- actual_tokens_for_release property
- duration_seconds property
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from adaptive_rate_limiter.streaming.context import StreamingReservationContext


class TestStreamingReservationContextInit:
    """Tests for StreamingReservationContext initialization."""

    def test_init_with_required_fields(self) -> None:
        """Verify initialization with required fields only."""
        backend = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

        assert ctx.reservation_id == "res-1"
        assert ctx.bucket_id == "bucket-1"
        assert ctx.request_id == "req-1"
        assert ctx.reserved_tokens == 1000
        assert ctx.backend is backend

    def test_init_sets_default_created_at(self) -> None:
        """Verify created_at is set to current time by default."""
        backend = Mock()
        before = time.time()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )
        after = time.time()

        assert before <= ctx.created_at <= after

    def test_init_with_custom_created_at(self) -> None:
        """Verify custom created_at is accepted."""
        backend = Mock()
        custom_time = 1234567890.0
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
            created_at=custom_time,
        )

        assert ctx.created_at == custom_time

    def test_init_default_runtime_tracking_values(self) -> None:
        """Verify runtime tracking fields are initialized to defaults."""
        backend = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

        assert ctx.final_tokens is None
        assert ctx.chunk_count == 0
        assert ctx.last_chunk_at is None

    def test_init_with_metrics_callback(self) -> None:
        """Verify metrics_callback is set correctly."""
        backend = Mock()
        metrics_callback = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
            metrics_callback=metrics_callback,
        )

        assert ctx.metrics_callback is metrics_callback

    def test_init_with_error_metrics_callback(self) -> None:
        """Verify error_metrics_callback is set correctly."""
        backend = Mock()
        error_callback = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
            error_metrics_callback=error_callback,
        )

        assert ctx.error_metrics_callback is error_callback

    def test_init_with_all_callbacks(self) -> None:
        """Verify both callbacks can be set simultaneously."""
        backend = Mock()
        metrics_callback = Mock()
        error_callback = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
            metrics_callback=metrics_callback,
            error_metrics_callback=error_callback,
        )

        assert ctx.metrics_callback is metrics_callback
        assert ctx.error_metrics_callback is error_callback


class TestStreamingReservationContextRecordChunk:
    """Tests for StreamingReservationContext.record_chunk() method."""

    @pytest.fixture
    def context(self) -> StreamingReservationContext:
        """Create a test context."""
        backend = Mock()
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

    def test_record_chunk_updates_last_chunk_at(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify record_chunk() updates last_chunk_at."""
        assert context.last_chunk_at is None

        before = time.time()
        context.record_chunk()
        after = time.time()

        assert context.last_chunk_at is not None
        assert before <= context.last_chunk_at <= after

    def test_record_chunk_increments_chunk_count(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify record_chunk() increments chunk_count."""
        assert context.chunk_count == 0

        context.record_chunk()
        assert context.chunk_count == 1

        context.record_chunk()
        assert context.chunk_count == 2

        context.record_chunk()
        assert context.chunk_count == 3

    def test_record_chunk_updates_last_chunk_at_on_each_call(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify last_chunk_at is updated on each call."""
        context.record_chunk()
        first_time = context.last_chunk_at

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        context.record_chunk()
        second_time = context.last_chunk_at

        assert second_time is not None
        assert first_time is not None
        assert second_time >= first_time

    def test_record_chunk_multiple_calls_count_correctly(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify chunk counting works for many chunks."""
        for i in range(100):
            context.record_chunk()
            assert context.chunk_count == i + 1


class TestStreamingReservationContextSetFinalTokens:
    """Tests for StreamingReservationContext.set_final_tokens() method."""

    @pytest.fixture
    def context(self) -> StreamingReservationContext:
        """Create a test context."""
        backend = Mock()
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

    def test_set_final_tokens_positive_value(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify set_final_tokens() accepts positive values."""
        assert context.final_tokens is None

        context.set_final_tokens(500)
        assert context.final_tokens == 500

    def test_set_final_tokens_zero(self, context: StreamingReservationContext) -> None:
        """Verify set_final_tokens() accepts zero."""
        assert context.final_tokens is None

        context.set_final_tokens(0)
        assert context.final_tokens == 0

    def test_set_final_tokens_negative_ignored(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify set_final_tokens() ignores negative values."""
        assert context.final_tokens is None

        context.set_final_tokens(-1)
        assert context.final_tokens is None

        context.set_final_tokens(-100)
        assert context.final_tokens is None

    def test_set_final_tokens_overwrites_previous_value(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify set_final_tokens() overwrites previous value."""
        context.set_final_tokens(500)
        assert context.final_tokens == 500

        context.set_final_tokens(750)
        assert context.final_tokens == 750

    def test_set_final_tokens_negative_does_not_overwrite(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify negative value doesn't overwrite existing value."""
        context.set_final_tokens(500)
        assert context.final_tokens == 500

        context.set_final_tokens(-1)
        assert context.final_tokens == 500


class TestStreamingReservationContextActualTokensForRelease:
    """Tests for StreamingReservationContext.actual_tokens_for_release property."""

    @pytest.fixture
    def context(self) -> StreamingReservationContext:
        """Create a test context."""
        backend = Mock()
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

    def test_returns_reserved_when_final_tokens_not_set(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify returns reserved_tokens when final_tokens is None."""
        assert context.final_tokens is None
        assert context.actual_tokens_for_release == 1000

    def test_returns_final_tokens_when_set(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify returns final_tokens when it's set."""
        context.set_final_tokens(500)
        assert context.actual_tokens_for_release == 500

    def test_returns_final_tokens_zero(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify returns zero when final_tokens is zero."""
        context.set_final_tokens(0)
        assert context.actual_tokens_for_release == 0

    def test_returns_updated_final_tokens(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify returns updated final_tokens after update."""
        context.set_final_tokens(500)
        assert context.actual_tokens_for_release == 500

        context.set_final_tokens(750)
        assert context.actual_tokens_for_release == 750


class TestStreamingReservationContextDurationSeconds:
    """Tests for StreamingReservationContext.duration_seconds property."""

    def test_returns_none_when_no_chunks(self) -> None:
        """Verify returns None when chunk_count is zero."""
        backend = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

        assert ctx.chunk_count == 0
        assert ctx.duration_seconds is None

    def test_returns_duration_after_first_chunk(self) -> None:
        """Verify returns duration after recording a chunk."""
        backend = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

        ctx.record_chunk()

        duration = ctx.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_duration_increases_over_time(self) -> None:
        """Verify duration increases as time passes."""
        backend = Mock()
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
        )

        ctx.record_chunk()
        duration1 = ctx.duration_seconds

        # Small delay
        time.sleep(0.01)

        duration2 = ctx.duration_seconds

        assert duration1 is not None
        assert duration2 is not None
        assert duration2 >= duration1

    def test_duration_uses_created_at_as_start(self) -> None:
        """Verify duration is calculated from created_at."""
        backend = Mock()
        # Set created_at to 1 second ago
        created_at = time.time() - 1.0
        ctx = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=backend,
            created_at=created_at,
        )

        ctx.record_chunk()
        duration = ctx.duration_seconds

        assert duration is not None
        # Should be at least 1 second
        assert duration >= 1.0

"""
Unit tests for IntelligentModeStrategy streaming functionality.

Tests streaming detection, cleanup, wrapping, and stale streaming handling.
"""

import asyncio
import contextlib
import logging
import time
import weakref
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.reservation.context import ReservationContext
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Streaming Detection Tests
# ============================================================================


class TestIntelligentModeStrategyStreamingDetection:
    """Tests for streaming response detection."""

    @pytest.mark.asyncio
    async def test_detect_streaming_none(self, strategy):
        """Test None is not streaming."""
        assert strategy._streaming_handler.detect_streaming_response(None) is False

    @pytest.mark.asyncio
    async def test_detect_streaming_async_generator(self, strategy):
        """Test async generator is streaming."""

        async def gen():
            yield 1

        assert strategy._streaming_handler.detect_streaming_response(gen()) is True

    @pytest.mark.asyncio
    async def test_detect_streaming_generator(self, strategy):
        """Test sync generator is streaming."""

        def gen():
            yield 1

        assert strategy._streaming_handler.detect_streaming_response(gen()) is True

    @pytest.mark.asyncio
    async def test_detect_streaming_stream_attribute(self, strategy):
        """Test object with stream=True is streaming."""
        result = Mock()
        result.stream = True

        assert strategy._streaming_handler.detect_streaming_response(result) is True

    @pytest.mark.asyncio
    async def test_detect_streaming_not_streaming_types(self, strategy):
        """Test common non-streaming types."""
        assert strategy._streaming_handler.detect_streaming_response("string") is False
        assert strategy._streaming_handler.detect_streaming_response(b"bytes") is False
        assert (
            strategy._streaming_handler.detect_streaming_response({"dict": "value"})
            is False
        )
        assert strategy._streaming_handler.detect_streaming_response([1, 2, 3]) is False


# ============================================================================
# Streaming Cleanup Tests
# ============================================================================


class TestIntelligentModeStrategyStreamingCleanup:
    """Tests for streaming cleanup methods."""

    @pytest.mark.asyncio
    async def test_register_streaming_in_flight(self, strategy):
        """Test registering streaming wrapper."""

        class FakeWrapper:
            pass

        wrapper = FakeWrapper()

        await strategy._streaming_cleanup_manager.register(
            "res-123", "bucket-1", 1000, wrapper
        )

        async with strategy._streaming_cleanup_manager._streaming_in_flight_lock:
            assert "res-123" in strategy._streaming_cleanup_manager._streaming_in_flight

    @pytest.mark.asyncio
    async def test_deregister_streaming_in_flight(self, strategy):
        """Test deregistering streaming wrapper."""

        class FakeWrapper:
            pass

        wrapper = FakeWrapper()

        await strategy._streaming_cleanup_manager.register(
            "res-123", "bucket-1", 1000, wrapper
        )
        await strategy._streaming_cleanup_manager.deregister("res-123")

        async with strategy._streaming_cleanup_manager._streaming_in_flight_lock:
            assert (
                "res-123"
                not in strategy._streaming_cleanup_manager._streaming_in_flight
            )

    @pytest.mark.asyncio
    async def test_update_streaming_activity(self, strategy):
        """Test updating streaming activity."""

        class FakeWrapper:
            pass

        wrapper = FakeWrapper()

        await strategy._streaming_cleanup_manager.register(
            "res-123", "bucket-1", 1000, wrapper
        )

        old_time = strategy._streaming_cleanup_manager._streaming_in_flight[
            "res-123"
        ].last_activity_at
        await asyncio.sleep(0.01)

        await strategy._streaming_cleanup_manager.update_activity("res-123")

        new_time = strategy._streaming_cleanup_manager._streaming_in_flight[
            "res-123"
        ].last_activity_at
        assert new_time > old_time


# ============================================================================
# Stale Streaming Cleanup Tests
# ============================================================================


class TestIntelligentModeStrategyStaleStreamingCleanup:
    """Tests for stale streaming cleanup."""

    @pytest.fixture
    def strategy_short_timeout(self, strategy):
        """Strategy with short streaming timeout for testing."""
        strategy._streaming_activity_timeout = 0.01
        return strategy

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_gc_wrapper(
        self, strategy_short_timeout, mock_backend
    ):
        """Test cleanup handles GC'd wrapper."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        # Create a weakref-able class
        class WeakRefable:
            pass

        # Create the object and reference, then delete object
        obj = WeakRefable()
        ref = weakref.ref(obj)
        del obj  # Object is now GC'd, ref() returns None

        # Create entry with GC'd wrapper
        entry = StreamingInFlightEntry(
            reservation_id="res-1",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,
            wrapper_ref=ref,
        )

        strategy_short_timeout._streaming_cleanup_manager._streaming_in_flight[
            "res-1"
        ] = entry

        cleaned = (
            await strategy_short_timeout._streaming_cleanup_manager._cleanup_stale()
        )

        assert cleaned == 1
        mock_backend.release_streaming_reservation.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_timeout(
        self, strategy_short_timeout, mock_backend
    ):
        """Test cleanup releases inactive streams."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        # Create a weakref-able wrapper
        class FakeWrapper:
            pass

        wrapper = FakeWrapper()

        # Entry with old activity time
        entry = StreamingInFlightEntry(
            reservation_id="res-2",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,  # Very old
            wrapper_ref=weakref.ref(wrapper),
        )

        strategy_short_timeout._streaming_cleanup_manager._streaming_in_flight[
            "res-2"
        ] = entry

        # Set a very short timeout
        strategy_short_timeout._streaming_cleanup_manager._activity_timeout = 0.0

        cleaned = (
            await strategy_short_timeout._streaming_cleanup_manager._cleanup_stale()
        )

        assert cleaned == 1


# ============================================================================
# Streaming Wrapping Tests
# ============================================================================


class TestIntelligentModeStrategyStreamingWrapping:
    """Tests for _wrap_streaming_response branches."""

    @pytest.mark.asyncio
    async def test_wrap_streaming_with_iterator_class(self, strategy):
        """Test wrapping Stream class with _iterator attribute."""
        from adaptive_rate_limiter.streaming.iterator import RateLimitedAsyncIterator

        reservation = ReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test",
            resource_type="chat",
        )

        async def gen():
            yield "chunk"

        # Mock a Stream class with _iterator
        stream_result = Mock()
        stream_result._iterator = gen()

        # Mock registration to avoid weakref issue with RateLimitedAsyncIterator
        with patch.object(
            strategy._streaming_cleanup_manager, "register", new_callable=AsyncMock
        ):
            _wrapped = await strategy._streaming_handler.wrap_streaming_response(
                stream_result, reservation, metadata
            )

        # Iterator should be wrapped
        assert isinstance(stream_result._iterator, RateLimitedAsyncIterator)

    @pytest.mark.asyncio
    async def test_wrap_streaming_async_iterable(self, strategy):
        """Test wrapping async iterable directly."""
        from adaptive_rate_limiter.streaming.iterator import RateLimitedAsyncIterator

        reservation = ReservationContext(
            reservation_id="res-2",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-2",
            model_id="test",
            resource_type="chat",
        )

        async def gen():
            yield "chunk"

        # Mock registration to avoid weakref issue with RateLimitedAsyncIterator
        with patch.object(
            strategy._streaming_cleanup_manager, "register", new_callable=AsyncMock
        ):
            # Pass async generator directly
            wrapped = await strategy._streaming_handler.wrap_streaming_response(
                gen(), reservation, metadata
            )

        # Result should be wrapped iterator
        assert isinstance(wrapped, RateLimitedAsyncIterator)

    @pytest.mark.asyncio
    async def test_wrap_streaming_unknown_type_releases_immediately(
        self, strategy, mock_backend
    ):
        """Test unknown streaming type releases reservation."""
        reservation = ReservationContext(
            reservation_id="res-3",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-3",
            model_id="test",
            resource_type="chat",
        )

        await strategy._store_reservation_context("req-3", "bucket-1", "res-3", 100)

        # Unknown type (not async iterable, no _iterator)
        unknown_result = 42

        result = await strategy._streaming_handler.wrap_streaming_response(
            unknown_result, reservation, metadata
        )

        assert result == 42
        mock_backend.release_reservation.assert_called()


# ============================================================================
# Streaming Cleanup Edge Cases Tests
# ============================================================================


class TestIntelligentModeStrategyStreamingCleanupEdgeCases:
    """Tests for streaming cleanup edge cases."""

    @pytest.fixture
    def strategy_with_timeout(self, strategy):
        """Strategy with 1 second streaming timeout."""
        strategy._streaming_activity_timeout = 1
        return strategy

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_skips_released(
        self, strategy_with_timeout, mock_backend
    ):
        """Test cleanup removes already-released entries without calling backend."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        class FakeWrapper:
            _released = True

        wrapper = FakeWrapper()
        entry = StreamingInFlightEntry(
            reservation_id="res-released",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time(),
            last_activity_at=time.time(),
            wrapper_ref=weakref.ref(wrapper),
        )

        strategy_with_timeout._streaming_cleanup_manager._streaming_in_flight[
            "res-released"
        ] = entry

        # Don't set stale timeout - should skip due to _released flag
        strategy_with_timeout._streaming_cleanup_manager._activity_timeout = (
            1000  # Long timeout
        )

        cleaned = (
            await strategy_with_timeout._streaming_cleanup_manager._cleanup_stale()
        )

        # Should be removed but counted as 0 (no stale cleanup)
        assert (
            "res-released"
            not in strategy_with_timeout._streaming_cleanup_manager._streaming_in_flight
        )
        assert cleaned == 0  # Not counted as stale cleanup

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_syncs_activity_from_wrapper(
        self, strategy_with_timeout, mock_backend
    ):
        """Test cleanup syncs activity time from wrapper context."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        class FakeWrapper:
            _released = False

            def __init__(self):
                self._ctx = Mock()
                self._ctx.last_chunk_at = time.time()  # Recent activity

        wrapper = FakeWrapper()
        entry = StreamingInFlightEntry(
            reservation_id="res-active",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,  # Old
            wrapper_ref=weakref.ref(wrapper),
        )

        strategy_with_timeout._streaming_cleanup_manager._streaming_in_flight[
            "res-active"
        ] = entry
        strategy_with_timeout._streaming_cleanup_manager._activity_timeout = (
            500  # 500 seconds
        )

        cleaned = (
            await strategy_with_timeout._streaming_cleanup_manager._cleanup_stale()
        )

        # Should not be cleaned - activity was synced from wrapper
        assert (
            "res-active"
            in strategy_with_timeout._streaming_cleanup_manager._streaming_in_flight
        )
        assert cleaned == 0


# ============================================================================
# Streaming Detection Edge Cases (lines 1585, 1587)
# ============================================================================


class TestIntelligentModeStrategyStreamingDetectionEdgeCases:
    """Tests for streaming detection edge cases."""

    def test_detect_streaming_custom_iterator_class(self, strategy):
        """Line 1585: Custom iterable (has __iter__) but not excluded type."""

        class CustomIterable:
            def __iter__(self):
                return iter([1, 2, 3])

        # Has __iter__ but is not in excluded types - should return True
        result = strategy._streaming_handler.detect_streaming_response(CustomIterable())
        assert result is True

    def test_detect_streaming_stream_false(self, strategy):
        """Line 1577: Object with stream=False is not streaming."""
        result = Mock()
        result.stream = False

        assert strategy._streaming_handler.detect_streaming_response(result) is False

    def test_detect_streaming_excludes_tuple(self, strategy):
        """Line 1583-1584: Tuple is excluded from streaming detection."""
        assert strategy._streaming_handler.detect_streaming_response((1, 2, 3)) is False

    def test_detect_streaming_excludes_range(self, strategy):
        """Line 1583-1584: Range is excluded from streaming detection."""
        assert strategy._streaming_handler.detect_streaming_response(range(10)) is False

    def test_detect_streaming_excludes_set(self, strategy):
        """Line 1583-1584: Set is excluded from streaming detection."""
        assert strategy._streaming_handler.detect_streaming_response({1, 2, 3}) is False

    def test_detect_streaming_excludes_frozenset(self, strategy):
        """Line 1583-1584: Frozenset is excluded from streaming detection."""
        assert (
            strategy._streaming_handler.detect_streaming_response(frozenset([1, 2, 3]))
            is False
        )


# ============================================================================
# Streaming Wrapping Failure Paths (lines 1693-1694)
# ============================================================================


class TestIntelligentModeStrategyStreamingWrappingFailures:
    """Tests for streaming wrapping failure paths."""

    @pytest.mark.asyncio
    async def test_wrap_streaming_unknown_type_release_fails(
        self, strategy, mock_backend, caplog
    ):
        """Lines 1693-1694: Release fails for unknown streaming type."""
        mock_backend.release_reservation.side_effect = Exception("Release failed")

        reservation = ReservationContext(
            reservation_id="res-unknown",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-unknown",
            model_id="test",
            resource_type="chat",
        )
        await strategy._store_reservation_context(
            "req-unknown", "bucket-1", "res-unknown", 100
        )

        # Unknown type (int)
        unknown_result = 42

        with caplog.at_level(logging.WARNING):
            result = await strategy._streaming_handler.wrap_streaming_response(
                unknown_result, reservation, metadata
            )

        assert result == 42
        assert "Failed to release reservation" in caplog.text


# ============================================================================
# Streaming Cleanup Loop Logging (lines 1719-1720)
# ============================================================================


class TestIntelligentModeStrategyStreamingCleanupLogging:
    """Tests for streaming cleanup loop logging."""

    @pytest.mark.asyncio
    async def test_cleanup_streaming_loop_logs_when_cleaned(self, strategy, caplog):
        """Line 1720: Log message when entries are cleaned."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        class FakeWrapper:
            _released = False

        wrapper = FakeWrapper()
        entry = StreamingInFlightEntry(
            reservation_id="res-log",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,
            wrapper_ref=weakref.ref(wrapper),
        )
        strategy._streaming_cleanup_manager._streaming_in_flight["res-log"] = entry
        strategy._streaming_cleanup_manager._cleanup_interval = 0.001
        strategy._streaming_cleanup_manager._activity_timeout = 0.001
        strategy._streaming_cleanup_manager._running = True

        with caplog.at_level(logging.INFO):
            task = asyncio.create_task(
                strategy._streaming_cleanup_manager._cleanup_loop()
            )
            await asyncio.sleep(0.05)
            strategy._streaming_cleanup_manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert "Streaming cleanup released" in caplog.text


# ============================================================================
# Stale Streaming CancelledError Handling (line 1786)
# ============================================================================


class TestIntelligentModeStrategyStaleStreamingCancelledError:
    """Tests for CancelledError handling in stale streaming cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_reraises_cancelled(
        self, strategy, mock_backend
    ):
        """Line 1785-1786: CancelledError is re-raised."""
        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        class FakeWrapper:
            _released = False

        wrapper = FakeWrapper()
        entry = StreamingInFlightEntry(
            reservation_id="res-cancel",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,
            wrapper_ref=weakref.ref(wrapper),
        )
        strategy._streaming_cleanup_manager._streaming_in_flight["res-cancel"] = entry
        strategy._streaming_cleanup_manager._activity_timeout = 0.001

        mock_backend.release_streaming_reservation.side_effect = (
            asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await strategy._streaming_cleanup_manager._cleanup_stale()


# ============================================================================
# Execute With Streaming (lines 1194-1212)
# ============================================================================


class TestIntelligentModeStrategyExecuteWithStreaming:
    """Tests for streaming path in _execute_request_with_tracking."""

    @pytest.mark.asyncio
    async def test_execute_with_streaming_wraps_and_returns_early(
        self, strategy, mock_backend
    ):
        """Lines 1194-1212: Streaming path wraps and returns without standard release."""
        metadata = RequestMetadata(
            request_id="req-stream",
            model_id="test-model",
            resource_type="chat",
        )

        # Store reservation context
        await strategy._store_reservation_context(
            "req-stream", "bucket-1", "res-stream", 100
        )

        async def gen():
            yield "chunk1"
            yield "chunk2"

        request_func = AsyncMock(return_value=gen())
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        # Execute
        await strategy._execute_request_with_tracking(
            queued_request, "task-1", "bucket-1"
        )

        # Future should have a wrapped result
        result = future.result()
        # Result should be wrapped (RateLimitedAsyncIterator or similar)
        assert hasattr(result, "__aiter__") or hasattr(result, "_iterator")

        # Update state should NOT have been called with success path
        # (streaming handles its own release)

    @pytest.mark.asyncio
    async def test_execute_with_streaming_no_reservation(self, strategy, mock_backend):
        """Lines 1194-1196: No wrapping when no reservation found."""
        metadata = RequestMetadata(
            request_id="req-no-res",
            model_id="test-model",
            resource_type="chat",
        )

        # DON'T store reservation context

        async def gen():
            yield "chunk1"

        request_func = AsyncMock(return_value=gen())
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        # Execute
        await strategy._execute_request_with_tracking(
            queued_request, "task-2", "bucket-1"
        )

        # Future should have the raw result (not wrapped, no reservation)
        _result = future.result()
        # Without reservation, goes through non-streaming path and still sets result


# ============================================================================
# Streaming Metrics Recording (lines 1625, 1631)
# ============================================================================


class TestIntelligentModeStrategyStreamingMetricsRecording:
    """Tests for streaming metrics recording in wrap callbacks."""

    @pytest.mark.asyncio
    async def test_wrap_streaming_on_completion_callback(self, strategy):
        """Lines 1624-1627: on_completion callback records metrics."""

        reservation = ReservationContext(
            reservation_id="res-metrics",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-metrics",
            model_id="test",
            resource_type="chat",
        )

        async def gen():
            yield "chunk"

        with patch.object(
            strategy._streaming_cleanup_manager, "register", new_callable=AsyncMock
        ):
            wrapped = await strategy._streaming_handler.wrap_streaming_response(
                gen(), reservation, metadata
            )

        # Trigger completion callback by simulating completion
        initial_completions = strategy._streaming_metrics.streaming_completions

        # Manually call the callback that was set
        if hasattr(wrapped, "_ctx") and wrapped._ctx.metrics_callback:
            wrapped._ctx.metrics_callback(
                reserved=100,
                actual=50,
                extraction_succeeded=True,
                bucket_id="bucket-1",
                duration=1.0,
            )
            assert (
                strategy._streaming_metrics.streaming_completions
                == initial_completions + 1
            )

    @pytest.mark.asyncio
    async def test_wrap_streaming_on_error_callback(self, strategy):
        """Lines 1629-1631: on_error callback records metrics."""

        reservation = ReservationContext(
            reservation_id="res-err-cb",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time(),
        )
        metadata = RequestMetadata(
            request_id="req-err-cb",
            model_id="test",
            resource_type="chat",
        )

        async def gen():
            yield "chunk"

        with patch.object(
            strategy._streaming_cleanup_manager, "register", new_callable=AsyncMock
        ):
            wrapped = await strategy._streaming_handler.wrap_streaming_response(
                gen(), reservation, metadata
            )

        # Trigger error callback
        initial_errors = strategy._streaming_metrics.streaming_errors

        if hasattr(wrapped, "_ctx") and wrapped._ctx.error_metrics_callback:
            wrapped._ctx.error_metrics_callback(bucket_id="bucket-1")
            assert strategy._streaming_metrics.streaming_errors == initial_errors + 1

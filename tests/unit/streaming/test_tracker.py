"""
Unit tests for StreamingInFlightEntry and StreamingInFlightTracker.

Tests cover:
- StreamingInFlightEntry dataclass initialization
- StreamingInFlightTracker initialization
- active_count property
- start_cleanup() / stop_cleanup() lifecycle
- register() / deregister() methods
- update_activity() method
- _cleanup_stale() method
- get_stats() method
"""

from __future__ import annotations

import asyncio
import gc
import time
import weakref
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.streaming.tracker import (
    StreamingInFlightEntry,
    StreamingInFlightTracker,
)


class StaleWrapper:
    """A wrapper class that mimics a stale iterator without _ctx attribute."""

    def __init__(self) -> None:
        self._released = False
        # Note: No _ctx attribute - simulates iterator that hasn't started


class TestStreamingInFlightEntry:
    """Tests for StreamingInFlightEntry dataclass."""

    def test_init_with_all_fields(self) -> None:
        """Verify initialization with all required fields."""
        wrapper = Mock()
        now = time.time()

        entry = StreamingInFlightEntry(
            reservation_id="res-1",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=now,
            last_activity_at=now,
            wrapper_ref=weakref.ref(wrapper),
        )

        assert entry.reservation_id == "res-1"
        assert entry.bucket_id == "bucket-1"
        assert entry.reserved_tokens == 1000
        assert entry.started_at == now
        assert entry.last_activity_at == now
        assert entry.wrapper_ref() is wrapper

    def test_wrapper_ref_is_weak_reference(self) -> None:
        """Verify wrapper_ref is a weak reference."""
        wrapper = Mock()
        now = time.time()

        entry = StreamingInFlightEntry(
            reservation_id="res-1",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=now,
            last_activity_at=now,
            wrapper_ref=weakref.ref(wrapper),
        )

        assert entry.wrapper_ref() is wrapper

        # Delete wrapper and force GC
        del wrapper
        gc.collect()

        # Weak reference should now return None
        assert entry.wrapper_ref() is None


class TestStreamingInFlightTrackerInit:
    """Tests for StreamingInFlightTracker initialization."""

    def test_init_with_backend(self) -> None:
        """Verify initialization with backend."""
        backend = Mock()
        tracker = StreamingInFlightTracker(backend=backend)

        assert tracker._backend is backend

    def test_init_with_default_intervals(self) -> None:
        """Verify default cleanup_interval and activity_timeout."""
        backend = Mock()
        tracker = StreamingInFlightTracker(backend=backend)

        assert tracker._cleanup_interval == 60
        assert tracker._activity_timeout == 300

    def test_init_with_custom_intervals(self) -> None:
        """Verify custom cleanup_interval and activity_timeout."""
        backend = Mock()
        tracker = StreamingInFlightTracker(
            backend=backend,
            cleanup_interval=30,
            activity_timeout=120,
        )

        assert tracker._cleanup_interval == 30
        assert tracker._activity_timeout == 120

    def test_init_with_metrics_callback(self) -> None:
        """Verify metrics_callback is set."""
        backend = Mock()
        metrics_callback = Mock()
        tracker = StreamingInFlightTracker(
            backend=backend,
            metrics_callback=metrics_callback,
        )

        assert tracker._metrics_callback is metrics_callback

    def test_init_starts_with_empty_tracking(self) -> None:
        """Verify tracker starts with no entries."""
        backend = Mock()
        tracker = StreamingInFlightTracker(backend=backend)

        assert len(tracker._streaming_in_flight) == 0
        assert tracker.active_count == 0

    def test_init_not_running(self) -> None:
        """Verify tracker starts not running."""
        backend = Mock()
        tracker = StreamingInFlightTracker(backend=backend)

        assert tracker._running is False
        assert tracker._cleanup_task is None


class TestStreamingInFlightTrackerActiveCount:
    """Tests for StreamingInFlightTracker.active_count property."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def tracker(self, mock_backend: Mock) -> StreamingInFlightTracker:
        """Create a test tracker."""
        return StreamingInFlightTracker(backend=mock_backend)

    def test_returns_zero_when_empty(self, tracker: StreamingInFlightTracker) -> None:
        """Verify returns 0 when no entries."""
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_returns_count_after_register(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify returns correct count after registering."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        assert tracker.active_count == 1

        await tracker.register("res-2", "bucket-1", 2000, wrapper)
        assert tracker.active_count == 2

    @pytest.mark.asyncio
    async def test_returns_count_after_deregister(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify returns correct count after deregistering."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        await tracker.register("res-2", "bucket-1", 2000, wrapper)
        assert tracker.active_count == 2

        await tracker.deregister("res-1")
        assert tracker.active_count == 1

        await tracker.deregister("res-2")
        assert tracker.active_count == 0


class TestStreamingInFlightTrackerLifecycle:
    """Tests for start_cleanup() and stop_cleanup() methods."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def tracker(self, mock_backend: Mock) -> StreamingInFlightTracker:
        """Create a test tracker with short intervals."""
        return StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=2,
        )

    @pytest.mark.asyncio
    async def test_start_cleanup_creates_task(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify start_cleanup() creates cleanup task."""
        assert tracker._cleanup_task is None

        await tracker.start_cleanup()

        assert tracker._cleanup_task is not None
        assert tracker._running is True

        await tracker.stop_cleanup()

    @pytest.mark.asyncio
    async def test_stop_cleanup_cancels_task(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify stop_cleanup() cancels cleanup task."""
        await tracker.start_cleanup()
        assert tracker._cleanup_task is not None

        await tracker.stop_cleanup()

        assert tracker._cleanup_task is None
        assert tracker._running is False

    @pytest.mark.asyncio
    async def test_start_cleanup_is_idempotent(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify multiple start_cleanup() calls are safe."""
        await tracker.start_cleanup()
        task1 = tracker._cleanup_task

        await tracker.start_cleanup()
        task2 = tracker._cleanup_task

        # Should be same task (no double creation)
        assert task1 is task2

        await tracker.stop_cleanup()

    @pytest.mark.asyncio
    async def test_stop_cleanup_is_idempotent(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify multiple stop_cleanup() calls are safe."""
        await tracker.start_cleanup()

        await tracker.stop_cleanup()
        await tracker.stop_cleanup()
        await tracker.stop_cleanup()

        assert tracker._cleanup_task is None


class TestStreamingInFlightTrackerRegister:
    """Tests for StreamingInFlightTracker.register() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def tracker(self, mock_backend: Mock) -> StreamingInFlightTracker:
        """Create a test tracker."""
        return StreamingInFlightTracker(backend=mock_backend)

    @pytest.mark.asyncio
    async def test_adds_entry_to_tracking(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify register() adds entry to tracking dict."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        assert "res-1" in tracker._streaming_in_flight
        entry = tracker._streaming_in_flight["res-1"]
        assert entry.reservation_id == "res-1"
        assert entry.bucket_id == "bucket-1"
        assert entry.reserved_tokens == 1000

    @pytest.mark.asyncio
    async def test_creates_weak_reference_to_wrapper(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify wrapper is stored as weak reference."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        entry = tracker._streaming_in_flight["res-1"]
        assert entry.wrapper_ref() is wrapper

    @pytest.mark.asyncio
    async def test_sets_started_at_and_last_activity_at(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify timestamps are set on register."""
        wrapper = Mock()
        before = time.time()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        after = time.time()
        entry = tracker._streaming_in_flight["res-1"]

        assert before <= entry.started_at <= after
        assert before <= entry.last_activity_at <= after
        assert entry.started_at == entry.last_activity_at

    @pytest.mark.asyncio
    async def test_can_register_multiple_entries(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify multiple entries can be registered."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        await tracker.register("res-2", "bucket-2", 2000, wrapper)
        await tracker.register("res-3", "bucket-1", 3000, wrapper)

        assert len(tracker._streaming_in_flight) == 3


class TestStreamingInFlightTrackerDeregister:
    """Tests for StreamingInFlightTracker.deregister() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def tracker(self, mock_backend: Mock) -> StreamingInFlightTracker:
        """Create a test tracker."""
        return StreamingInFlightTracker(backend=mock_backend)

    @pytest.mark.asyncio
    async def test_removes_entry_from_tracking(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify deregister() removes entry from tracking dict."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        assert "res-1" in tracker._streaming_in_flight

        await tracker.deregister("res-1")

        assert "res-1" not in tracker._streaming_in_flight

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_is_safe(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify deregistering nonexistent entry is safe."""
        await tracker.deregister("nonexistent")

        # Should not raise
        assert tracker.active_count == 0


class TestStreamingInFlightTrackerUpdateActivity:
    """Tests for StreamingInFlightTracker.update_activity() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def tracker(self, mock_backend: Mock) -> StreamingInFlightTracker:
        """Create a test tracker."""
        return StreamingInFlightTracker(backend=mock_backend)

    @pytest.mark.asyncio
    async def test_updates_last_activity_at(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify update_activity() updates last_activity_at."""
        wrapper = Mock()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        entry = tracker._streaming_in_flight["res-1"]
        original_time = entry.last_activity_at

        # Small delay
        await asyncio.sleep(0.01)

        await tracker.update_activity("res-1")

        assert entry.last_activity_at >= original_time

    @pytest.mark.asyncio
    async def test_update_nonexistent_is_safe(
        self, tracker: StreamingInFlightTracker
    ) -> None:
        """Verify updating nonexistent entry is safe."""
        await tracker.update_activity("nonexistent")

        # Should not raise
        assert tracker.active_count == 0


class TestStreamingInFlightTrackerCleanupStale:
    """Tests for StreamingInFlightTracker._cleanup_stale() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.mark.asyncio
    async def test_removes_entries_with_gc_wrappers(self, mock_backend: Mock) -> None:
        """Verify entries with GC'd wrappers are removed."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=300,  # High timeout to ensure not stale by time
        )

        # Register with a wrapper that will be GC'd
        wrapper: Any = Mock()
        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        assert tracker.active_count == 1

        # Delete wrapper and force GC
        del wrapper
        gc.collect()

        # Run cleanup
        cleaned = await tracker._cleanup_stale()

        assert cleaned == 1
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_removes_entries_already_released(self, mock_backend: Mock) -> None:
        """Verify entries with _released=True are removed."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=300,
        )

        wrapper = Mock()
        wrapper._released = True

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        assert tracker.active_count == 1

        _cleaned = await tracker._cleanup_stale()

        # Entry should be removed (but not counted as cleaned since already released)
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_removes_entries_inactive_past_timeout(
        self, mock_backend: Mock
    ) -> None:
        """Verify entries inactive > timeout are removed."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,  # 1 second timeout for fast test
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        assert tracker.active_count == 1

        # Wait for timeout
        await asyncio.sleep(1.5)

        cleaned = await tracker._cleanup_stale()

        assert cleaned == 1
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_calls_backend_release_with_conservative_fallback(
        self, mock_backend: Mock
    ) -> None:
        """Verify backend release is called with actual=reserved (zero refund)."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        # Wait for timeout
        await asyncio.sleep(1.5)

        await tracker._cleanup_stale()

        mock_backend.release_streaming_reservation.assert_called_once_with(
            "bucket-1",
            "res-1",
            reserved_tokens=1000,
            actual_tokens=1000,  # Conservative fallback
        )

    @pytest.mark.asyncio
    async def test_calls_metrics_callback(self, mock_backend: Mock) -> None:
        """Verify metrics_callback is called on stale cleanup."""
        metrics_callback = Mock()
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,
            metrics_callback=metrics_callback,
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        await asyncio.sleep(1.5)
        await tracker._cleanup_stale()

        metrics_callback.assert_called_once_with("bucket-1")

    @pytest.mark.asyncio
    async def test_handles_metrics_callback_exception(self, mock_backend: Mock) -> None:
        """Verify metrics_callback exception is handled gracefully."""
        metrics_callback = Mock(side_effect=RuntimeError("Callback error"))
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,
            metrics_callback=metrics_callback,
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        await asyncio.sleep(1.5)

        # Should not raise
        cleaned = await tracker._cleanup_stale()
        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_handles_backend_exception(self, mock_backend: Mock) -> None:
        """Verify backend exception is handled gracefully."""
        mock_backend.release_streaming_reservation.side_effect = RuntimeError(
            "Backend error"
        )
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        await asyncio.sleep(1.5)

        # Should not raise
        cleaned = await tracker._cleanup_stale()
        assert cleaned == 1
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_syncs_activity_from_wrapper_context(
        self, mock_backend: Mock
    ) -> None:
        """Verify activity is synced from wrapper._ctx.last_chunk_at."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=2,
        )

        wrapper = Mock()
        wrapper._released = False
        wrapper._ctx = Mock()
        wrapper._ctx.last_chunk_at = time.time() + 1  # Future time

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        # Wait a bit
        await asyncio.sleep(0.5)

        # Entry should NOT be cleaned because wrapper context has recent activity
        cleaned = await tracker._cleanup_stale()

        assert cleaned == 0
        assert tracker.active_count == 1

    @pytest.mark.asyncio
    async def test_does_not_remove_active_entries(self, mock_backend: Mock) -> None:
        """Verify active entries are not removed."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=10,  # Long timeout
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)

        cleaned = await tracker._cleanup_stale()

        assert cleaned == 0
        assert tracker.active_count == 1


class TestStreamingInFlightTrackerGetStats:
    """Tests for StreamingInFlightTracker.get_stats() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    def test_returns_correct_stats(self, mock_backend: Mock) -> None:
        """Verify get_stats() returns correct statistics."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=30,
            activity_timeout=120,
        )

        stats = tracker.get_stats()

        assert stats["active_streams"] == 0
        assert stats["cleanup_interval"] == 30
        assert stats["activity_timeout"] == 120
        assert stats["cleanup_running"] is False

    @pytest.mark.asyncio
    async def test_returns_correct_active_count(self, mock_backend: Mock) -> None:
        """Verify get_stats() returns correct active_streams count."""
        tracker = StreamingInFlightTracker(backend=mock_backend)

        wrapper = Mock()
        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        await tracker.register("res-2", "bucket-1", 2000, wrapper)

        stats = tracker.get_stats()

        assert stats["active_streams"] == 2

    @pytest.mark.asyncio
    async def test_returns_cleanup_running_true(self, mock_backend: Mock) -> None:
        """Verify get_stats() shows cleanup_running when started."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=60,
            activity_timeout=300,
        )

        await tracker.start_cleanup()

        stats = tracker.get_stats()
        assert stats["cleanup_running"] is True

        await tracker.stop_cleanup()


class TestStreamingInFlightTrackerCleanupLoop:
    """Tests for StreamingInFlightTracker cleanup loop behavior."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.mark.asyncio
    async def test_cleanup_loop_runs_periodically(self, mock_backend: Mock) -> None:
        """Verify cleanup loop runs at cleanup_interval."""
        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,  # 1 second interval
            activity_timeout=1,  # 1 second timeout
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        await tracker.start_cleanup()

        # Wait for cleanup to run
        await asyncio.sleep(2.5)

        await tracker.stop_cleanup()

        # Entry should have been cleaned
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_loop_handles_exceptions(self, mock_backend: Mock) -> None:
        """Verify cleanup loop continues after exception."""
        mock_backend.release_streaming_reservation.side_effect = RuntimeError("Error")

        tracker = StreamingInFlightTracker(
            backend=mock_backend,
            cleanup_interval=1,
            activity_timeout=1,
        )

        wrapper = StaleWrapper()

        await tracker.register("res-1", "bucket-1", 1000, wrapper)
        await tracker.start_cleanup()

        # Wait for cleanup to run
        await asyncio.sleep(2.5)

        # Tracker should still be running
        assert tracker._running

        await tracker.stop_cleanup()

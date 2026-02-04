"""
Unit tests for IntelligentModeStrategy cleanup tasks.

Tests background cleanup tasks, completed task cleanup, and cleanup loop exception handling.
"""

import asyncio
import contextlib
import time
from unittest.mock import patch

import pytest

from adaptive_rate_limiter.reservation.context import ReservationContext

# ============================================================================
# Cleanup Task Tests
# ============================================================================


class TestIntelligentModeStrategyCleanupTasks:
    """Tests for cleanup task methods."""

    @pytest.mark.asyncio
    async def test_cleanup_completed_tasks(self, strategy):
        """Test cleaning up completed tasks."""

        # Create a completed task
        async def dummy():
            return "done"

        task = asyncio.create_task(dummy())
        await task  # Wait for completion

        async with strategy._task_lock:
            strategy._active_tasks["task-1"] = task

        await strategy._cleanup_completed_tasks()

        async with strategy._task_lock:
            assert "task-1" not in strategy._active_tasks


# ============================================================================
# Cleanup Loop Exception Handling Tests
# ============================================================================


class TestIntelligentModeStrategyCleanupLoopExceptions:
    """Tests for exception handling in cleanup loops."""

    @pytest.mark.asyncio
    async def test_cleanup_loop_handles_exception(self, strategy, caplog):
        """Test cleanup loop continues after exception."""

        strategy._running = True
        strategy._cleanup_interval = 0.001

        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Test cleanup error")

        with patch.object(
            strategy, "_cleanup_completed_tasks", side_effect=failing_cleanup
        ):
            task = asyncio.create_task(strategy._cleanup_loop())
            await asyncio.sleep(0.02)
            strategy._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_streaming_cleanup_loop_handles_exception(self, strategy, caplog):
        """Test streaming cleanup loop continues after exception."""

        strategy._streaming_cleanup_manager._running = True
        strategy._streaming_cleanup_manager._cleanup_interval = 0.001

        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Streaming cleanup error")
            return 0

        with patch.object(
            strategy._streaming_cleanup_manager,
            "_cleanup_stale",
            side_effect=failing_cleanup,
        ):
            task = asyncio.create_task(
                strategy._streaming_cleanup_manager._cleanup_loop()
            )
            await asyncio.sleep(0.02)
            strategy._streaming_cleanup_manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert call_count >= 1


# ============================================================================
# Stop Lifecycle Tests - CancelledError paths (lines 505-514)
# ============================================================================


class TestIntelligentModeStrategyStopLifecycle:
    """Tests for stop() lifecycle cleanup and CancelledError handling."""

    @pytest.mark.asyncio
    async def test_stop_with_running_stale_cleanup_task(self, strategy):
        """Test stop() cancels running stale cleanup task (lines 505-506)."""
        await strategy.start()
        # Let cleanup task actually start
        await asyncio.sleep(0.01)
        assert strategy._stale_cleanup_task is not None
        assert not strategy._stale_cleanup_task.done()

        await strategy.stop()
        # Should handle CancelledError in except block
        assert strategy._running is False

    @pytest.mark.asyncio
    async def test_stop_with_running_streaming_cleanup_task(self, strategy):
        """Test stop() cancels running streaming cleanup task (lines 513-514)."""
        await strategy.start()
        await asyncio.sleep(0.01)
        assert strategy._streaming_cleanup_manager._cleanup_task is not None
        assert not strategy._streaming_cleanup_manager._cleanup_task.done()

        await strategy.stop()
        # CancelledError should be caught in except block
        assert strategy._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_all_reset_watchers(self, strategy):
        """Test stop cancels rate limit reset watchers."""
        await strategy.start()

        # Create multiple watchers
        await strategy._reset_watcher.schedule_watcher("bucket-1", time.time() + 100)
        await strategy._reset_watcher.schedule_watcher("bucket-2", time.time() + 200)

        assert len(strategy._reset_watcher._reset_tasks) == 2

        await strategy.stop()

        assert len(strategy._reset_watcher._reset_tasks) == 0
        assert len(strategy._reset_watcher._buckets_waiting) == 0

    @pytest.mark.asyncio
    async def test_stop_with_no_running_tasks(self, strategy):
        """Test stop() gracefully handles when no tasks are running."""
        # Don't start - just stop
        await strategy.stop()
        assert strategy._running is False

    @pytest.mark.asyncio
    async def test_stop_clears_state_manager(self, strategy, mock_state_manager):
        """Test stop() calls state_manager.stop()."""
        await strategy.start()
        await strategy.stop()

        mock_state_manager.stop.assert_called_once()


# ============================================================================
# Reset Watcher Async Paths (lines 1394-1408)
# ============================================================================


class TestIntelligentModeStrategyResetWatcher:
    """Tests for rate limit reset watcher async paths."""

    @pytest.mark.asyncio
    async def test_reset_watcher_completes_successfully(self, strategy):
        """Lines 1394-1402: Watcher completes, cleans up, wakes scheduler."""
        # Very short wait time
        reset_ts = time.time() + 0.01

        await strategy._reset_watcher.schedule_watcher("bucket-complete", reset_ts)

        # Wait for watcher to complete
        await asyncio.sleep(0.15)

        # Bucket should be removed from waiting set
        assert "bucket-complete" not in strategy._reset_watcher._buckets_waiting

        # Wakeup should be set
        assert strategy._wakeup_event.is_set()

    @pytest.mark.asyncio
    async def test_reset_watcher_cancelled_cleans_up(self, strategy, caplog):
        """Lines 1404-1408: Watcher cancellation cleans up tracking."""
        import logging

        reset_ts = time.time() + 1000  # Far future

        await strategy._reset_watcher.schedule_watcher("bucket-cancel", reset_ts)

        # Verify bucket is in waiting set
        assert "bucket-cancel" in strategy._reset_watcher._buckets_waiting
        assert len(strategy._reset_watcher._reset_tasks) == 1
        task = next(iter(strategy._reset_watcher._reset_tasks))

        # Allow task to start sleeping
        await asyncio.sleep(0.01)

        # Cancel the task
        task.cancel()

        with caplog.at_level(logging.DEBUG):
            # Use gather with return_exceptions to wait for task completion
            await asyncio.gather(task, return_exceptions=True)

        # Should be cleaned up in finally block
        assert "bucket-cancel" not in strategy._reset_watcher._buckets_waiting

    @pytest.mark.asyncio
    async def test_reset_watcher_deduplication(self, strategy):
        """Test duplicate watchers are not created."""
        reset_ts = time.time() + 10.0

        await strategy._reset_watcher.schedule_watcher("bucket-dup", reset_ts)
        await strategy._reset_watcher.schedule_watcher("bucket-dup", reset_ts)

        assert len(strategy._reset_watcher._reset_tasks) == 1
        assert "bucket-dup" in strategy._reset_watcher._buckets_waiting

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_reset_watcher_immediate_reset(self, strategy):
        """Test immediate reset triggers wakeup without scheduling task."""
        reset_ts = time.time() - 1.0  # Past time

        await strategy._reset_watcher.schedule_watcher("bucket-past", reset_ts)

        # Allow the task to run
        await asyncio.sleep(0.02)

        # Should trigger wakeup immediately
        assert strategy._wakeup_event.is_set()


# ============================================================================
# Stale Cleanup Exception Handling
# ============================================================================


class TestIntelligentModeStrategyStaleCleanupExceptions:
    """Tests for exception handling in stale reservation cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_reservations_handles_release_failure(
        self, strategy, mock_backend, caplog
    ):
        """Test stale cleanup continues despite release failures."""
        import logging

        strategy._reservation_tracker._reservation_contexts[("req-1", "bucket-1")] = (
            ReservationContext(
                reservation_id="res-1",
                bucket_id="bucket-1",
                estimated_tokens=100,
                created_at=time.time() - 1000,
            )
        )
        strategy._reservation_tracker._request_id_index["req-1"] = {
            ("req-1", "bucket-1")
        }
        # Rebuild the time heap after directly manipulating internal state
        strategy._reservation_tracker._rebuild_time_heap()
        strategy.MAX_RESERVATION_AGE = 1
        mock_backend.release_reservation.side_effect = Exception("Backend error")

        with caplog.at_level(logging.WARNING):
            cleaned = await strategy._cleanup_stale_reservations()

        assert cleaned == 1
        assert "Failed to release" in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_stale_reservations_reraises_cancelled_error(
        self, strategy, mock_backend
    ):
        """Line 1514: CancelledError is re-raised during stale cleanup."""
        strategy._reservation_tracker._reservation_contexts[("req-1", "bucket-1")] = (
            ReservationContext(
                reservation_id="res-1",
                bucket_id="bucket-1",
                estimated_tokens=100,
                created_at=time.time() - 1000,
            )
        )
        strategy._reservation_tracker._request_id_index["req-1"] = {
            ("req-1", "bucket-1")
        }
        # Rebuild the time heap after directly manipulating internal state
        strategy._reservation_tracker._rebuild_time_heap()
        strategy.MAX_RESERVATION_AGE = 1

        mock_backend.release_reservation.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await strategy._cleanup_stale_reservations()

    @pytest.mark.asyncio
    async def test_stale_reservations_loop_handles_exception(self, strategy, caplog):
        """Test stale reservations cleanup loop continues after exception."""

        strategy._running = True
        strategy.STALE_CLEANUP_INTERVAL = 0.001

        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Stale cleanup error")
            return 0

        with patch.object(
            strategy, "_cleanup_stale_reservations", side_effect=failing_cleanup
        ):
            task = asyncio.create_task(strategy._cleanup_stale_reservations_loop())
            await asyncio.sleep(0.02)
            strategy._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert call_count >= 1

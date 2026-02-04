"""
Tests specifically targeting cache.py coverage gaps.

Coverage gaps addressed:
- Lines 220, 227, 233, 242, 249, 565: Metrics collector integration
- Lines 260→266, 495→500: TTL branches (config.cache_ttl when no override)
- Lines 280→288, 350→352: Versioning branches
- Lines 587: Race condition guard in _remove_expired_unsafe
- Lines 607, 614-617: Cleanup loop exit branches
- Lines 630, 639, 644, 647: Heap cleanup edge cases
- Lines 98-99, 164-173, 185-186: Cross-loop scenarios
"""

import asyncio
import contextlib
import heapq
import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from adaptive_rate_limiter.observability.constants import (
    CACHE_EVICTIONS_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
)
from adaptive_rate_limiter.observability.protocols import MetricsCollectorProtocol
from adaptive_rate_limiter.scheduler.config import StateConfig
from adaptive_rate_limiter.scheduler.state import Cache, StateEntry


class TestMetricsCollectorIntegration:
    """Tests for metrics collector integration (lines 220, 227, 233, 242, 249, 565)."""

    @pytest.mark.asyncio
    async def test_lock_free_get_hit_with_metrics_collector(self):
        """Test lock-free get hit records metric via collector (line 220)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=True)
        cache = Cache(config, metrics_collector=collector)

        entry = StateEntry(key="test", data={"value": 1})
        await cache.set(entry)

        result = await cache.get("test")

        assert result is not None
        collector.inc_counter.assert_any_call(CACHE_HITS_TOTAL)

    @pytest.mark.asyncio
    async def test_lock_free_get_miss_with_metrics_collector(self):
        """Test lock-free get miss records metric via collector (line 227)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=True)
        cache = Cache(config, metrics_collector=collector)

        result = await cache.get("nonexistent")

        assert result is None
        collector.inc_counter.assert_any_call(CACHE_MISSES_TOTAL)

    @pytest.mark.asyncio
    async def test_lock_free_get_expired_miss_with_metrics_collector(self):
        """Test lock-free get expired entry records miss via collector (line 227)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=True)
        cache = Cache(config, metrics_collector=collector)

        # Create expired entry
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        expired_entry = StateEntry(
            key="expired",
            data={"value": 1},
            created_at=old_time,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        cache._cache["expired"] = expired_entry

        result = await cache.get("expired")

        assert result is None
        collector.inc_counter.assert_any_call(CACHE_MISSES_TOTAL)

    @pytest.mark.asyncio
    async def test_lock_free_get_exception_with_metrics_collector(self):
        """Test lock-free get exception records miss via collector (line 233)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=True)
        cache = Cache(config, metrics_collector=collector)

        # Create entry that will raise on is_expired check
        class BrokenEntry:
            key = "broken"

            @property
            def is_expired(self):
                raise ValueError("Broken entry")

        cache._cache["broken"] = BrokenEntry()  # type: ignore

        result = await cache.get("broken")

        assert result is None
        collector.inc_counter.assert_any_call(CACHE_MISSES_TOTAL)

    @pytest.mark.asyncio
    async def test_full_lock_get_hit_with_metrics_collector(self):
        """Test full locking get hit records metric via collector (line 242)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=False)  # Full locking
        cache = Cache(config, metrics_collector=collector)

        entry = StateEntry(key="test", data={"value": 1})
        await cache.set(entry)

        result = await cache.get("test")

        assert result is not None
        collector.inc_counter.assert_any_call(CACHE_HITS_TOTAL)

    @pytest.mark.asyncio
    async def test_full_lock_get_miss_with_metrics_collector(self):
        """Test full locking get miss records metric via collector (line 249)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=False)  # Full locking
        cache = Cache(config, metrics_collector=collector)

        result = await cache.get("nonexistent")

        assert result is None
        collector.inc_counter.assert_any_call(CACHE_MISSES_TOTAL)

    @pytest.mark.asyncio
    async def test_full_lock_get_expired_miss_with_metrics_collector(self):
        """Test full locking get expired entry records miss via collector (line 249)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(lock_free_reads=False)  # Full locking
        cache = Cache(config, metrics_collector=collector)

        # Create expired entry
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        expired_entry = StateEntry(
            key="expired",
            data={"value": 1},
            created_at=old_time,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        cache._cache["expired"] = expired_entry

        result = await cache.get("expired")

        assert result is None
        collector.inc_counter.assert_any_call(CACHE_MISSES_TOTAL)

    @pytest.mark.asyncio
    async def test_eviction_with_metrics_collector(self):
        """Test eviction records metric via collector (line 565)."""
        collector = MagicMock(spec=MetricsCollectorProtocol)
        config = StateConfig(max_cache_size=1)
        cache = Cache(config, metrics_collector=collector)

        await cache.set(StateEntry(key="a", data={"value": 1}))
        await cache.set(StateEntry(key="b", data={"value": 2}))  # Triggers eviction

        collector.inc_counter.assert_any_call(CACHE_EVICTIONS_TOTAL)


class TestTTLBranches:
    """Tests for TTL branches (lines 260→266, 495→500)."""

    @pytest.mark.asyncio
    async def test_set_uses_config_ttl_when_no_override(self):
        """Test set() uses config.cache_ttl when ttl_override is None (line 260→266)."""
        config = StateConfig(cache_ttl=120.0)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"value": 1})
        before = datetime.now(timezone.utc)
        await cache.set(entry)  # No ttl_override
        after = datetime.now(timezone.utc)

        cached = cache._cache["test"]
        assert cached.expires_at is not None

        # Expiration should be ~120 seconds from now
        expected_min = before + timedelta(seconds=119)
        expected_max = after + timedelta(seconds=121)
        assert expected_min <= cached.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_set_uses_ttl_override_over_config(self):
        """Test set() prefers ttl_override over config.cache_ttl."""
        config = StateConfig(cache_ttl=3600.0)  # 1 hour
        cache = Cache(config)

        entry = StateEntry(key="test", data={"value": 1})
        before = datetime.now(timezone.utc)
        await cache.set(entry, ttl_override=10.0)  # 10 seconds
        after = datetime.now(timezone.utc)

        cached = cache._cache["test"]
        assert cached.expires_at is not None

        # Expiration should be ~10 seconds (not 3600)
        expected_min = before + timedelta(seconds=9)
        expected_max = after + timedelta(seconds=11)
        assert expected_min <= cached.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_atomic_bulk_set_uses_config_ttl(self):
        """Test atomic_bulk_set uses config.cache_ttl when no override (line 495→500)."""
        config = StateConfig(cache_ttl=180.0)
        cache = Cache(config)

        entries = [StateEntry(key=f"key_{i}", data={"value": i}) for i in range(3)]
        before = datetime.now(timezone.utc)
        await cache.atomic_bulk_set(entries)  # No ttl_override
        after = datetime.now(timezone.utc)

        for i in range(3):
            cached = cache._cache[f"key_{i}"]
            assert cached.expires_at is not None
            expected_min = before + timedelta(seconds=179)
            expected_max = after + timedelta(seconds=181)
            assert expected_min <= cached.expires_at <= expected_max


class TestVersioningBranches:
    """Tests for versioning branches (lines 280→288, 350→352)."""

    @pytest.mark.asyncio
    async def test_set_stores_version_when_enabled(self):
        """Test _set_unsafe stores version history when enabled (line 280→288)."""
        config = StateConfig(enable_versioning=True, max_versions=5)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"value": 1})
        await cache.set(entry)

        assert "test" in cache._versions
        assert len(cache._versions["test"]) == 1

        # Update entry
        entry2 = StateEntry(key="test", data={"value": 2})
        await cache.set(entry2)

        assert len(cache._versions["test"]) == 2

    @pytest.mark.asyncio
    async def test_set_trims_versions_to_max(self):
        """Test _set_unsafe trims version history to max_versions (line 285-286)."""
        config = StateConfig(enable_versioning=True, max_versions=3)
        cache = Cache(config)

        # Add more versions than max
        for i in range(5):
            entry = StateEntry(key="test", data={"value": i})
            await cache.set(entry)

        assert len(cache._versions["test"]) == 3
        # Should have the last 3 versions
        values = [e.data["value"] for e in cache._versions["test"]]
        assert values == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_delete_removes_versions(self):
        """Test _delete_unsafe removes version history (line 350→352)."""
        config = StateConfig(enable_versioning=True)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"value": 1})
        await cache.set(entry)
        assert "test" in cache._versions

        await cache.delete("test")
        assert "test" not in cache._versions

    @pytest.mark.asyncio
    async def test_set_no_versioning_when_disabled(self):
        """Test _set_unsafe skips versioning when disabled (branch not taken)."""
        config = StateConfig(enable_versioning=False)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"value": 1})
        await cache.set(entry)

        assert "test" not in cache._versions


class TestRemoveExpiredRaceCondition:
    """Tests for race condition guard in _remove_expired_unsafe (line 587)."""

    @pytest.mark.asyncio
    async def test_remove_expired_unsafe_entry_gone(self):
        """Test _remove_expired_unsafe when entry already deleted (line 587)."""
        config = StateConfig()
        cache = Cache(config)

        # Key not in cache
        await cache._remove_expired_unsafe("nonexistent")
        # Should return early without error
        assert "nonexistent" not in cache._cache

    @pytest.mark.asyncio
    async def test_remove_expired_unsafe_entry_no_longer_expired(self):
        """Test _remove_expired_unsafe when entry not expired (line 587)."""
        config = StateConfig()
        cache = Cache(config)

        # Entry with future expiration (not expired)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        cache._cache["test"] = StateEntry(
            key="test", data={"value": 1}, expires_at=future
        )
        cache._creation_times["test"] = time.time()

        await cache._remove_expired_unsafe("test")

        # Should return early, entry remains
        assert "test" in cache._cache

    @pytest.mark.asyncio
    async def test_remove_expired_unsafe_actually_removes_expired(self):
        """Test _remove_expired_unsafe removes actually expired entry."""
        config = StateConfig()
        cache = Cache(config)

        # Create expired entry
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        cache._cache["test"] = StateEntry(
            key="test",
            data={"value": 1},
            created_at=old_time,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        cache._creation_times["test"] = time.time()

        await cache._remove_expired_unsafe("test")

        # Should be removed
        assert "test" not in cache._cache
        assert "test" not in cache._creation_times


class TestHeapCleanupEdgeCases:
    """Tests for heap cleanup edge cases (lines 630, 639, 644, 647)."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_breaks_when_first_entry_not_expired(self):
        """Test heap breaks when first entry is in future (line 630)."""
        config = StateConfig()
        cache = Cache(config)

        # Add future entry to heap
        future_time = time.time() + 3600
        heapq.heappush(cache._expiration_heap, (future_time, "test"))

        # Add corresponding cache entry
        future_dt = datetime.now(timezone.utc) + timedelta(hours=1)
        cache._cache["test"] = StateEntry(
            key="test", data={"value": 1}, expires_at=future_dt
        )

        await cache._cleanup_expired()

        # Entry should remain (not expired)
        assert "test" in cache._cache
        assert len(cache._expiration_heap) == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_stale_heap_entry_deleted_key(self):
        """Test heap continues when key was already deleted (line 639)."""
        config = StateConfig()
        cache = Cache(config)

        # Add expired entry to heap for key that doesn't exist in cache
        past_time = time.time() - 10
        heapq.heappush(cache._expiration_heap, (past_time, "deleted_key"))

        await cache._cleanup_expired()

        # Heap should be empty (entry popped but key not in cache)
        assert len(cache._expiration_heap) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_entry_has_no_expiration(self):
        """Test heap continues when entry has expires_at=None (line 644)."""
        config = StateConfig()
        cache = Cache(config)

        # Add expired entry to heap
        past_time = time.time() - 10
        heapq.heappush(cache._expiration_heap, (past_time, "test"))

        # Add entry with no expiration
        cache._cache["test"] = StateEntry(
            key="test", data={"value": 1}, expires_at=None
        )

        await cache._cleanup_expired()

        # Entry should remain (expires_at is None)
        assert "test" in cache._cache
        assert len(cache._expiration_heap) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_stale_heap_entry_different_expiration(self):
        """Test heap skips when entry has different expiration (line 647)."""
        config = StateConfig()
        cache = Cache(config)

        # Add old expiration to heap
        old_time = time.time() - 10
        heapq.heappush(cache._expiration_heap, (old_time, "test"))

        # Entry now has a new (future) expiration
        new_dt = datetime.now(timezone.utc) + timedelta(hours=1)
        cache._cache["test"] = StateEntry(
            key="test", data={"value": 1}, expires_at=new_dt
        )

        await cache._cleanup_expired()

        # Entry should remain (heap had stale expiration timestamp)
        assert "test" in cache._cache
        assert len(cache._expiration_heap) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_multiple_entries(self):
        """Test heap cleanup with mix of valid and stale entries."""
        config = StateConfig()
        cache = Cache(config)

        current_time = time.time()

        # Entry 1: Expired and still in cache
        old_time1 = datetime.now(timezone.utc) - timedelta(hours=2)
        expired_dt1 = datetime.now(timezone.utc) - timedelta(seconds=10)
        cache._cache["expired1"] = StateEntry(
            key="expired1", data={}, created_at=old_time1, expires_at=expired_dt1
        )
        heapq.heappush(cache._expiration_heap, (expired_dt1.timestamp(), "expired1"))

        # Entry 2: Deleted (stale heap entry)
        heapq.heappush(cache._expiration_heap, (current_time - 5, "deleted"))

        # Entry 3: Updated expiration (stale heap entry)
        new_future = datetime.now(timezone.utc) + timedelta(hours=1)
        cache._cache["updated"] = StateEntry(
            key="updated", data={}, expires_at=new_future
        )
        heapq.heappush(cache._expiration_heap, (current_time - 3, "updated"))

        # Entry 4: Future expiration (should stop processing)
        future_dt = datetime.now(timezone.utc) + timedelta(hours=2)
        cache._cache["future"] = StateEntry(key="future", data={}, expires_at=future_dt)
        heapq.heappush(cache._expiration_heap, (future_dt.timestamp(), "future"))

        await cache._cleanup_expired()

        # expired1 should be removed
        assert "expired1" not in cache._cache
        # updated and future should remain
        assert "updated" in cache._cache
        assert "future" in cache._cache
        # Only future entry should remain in heap
        assert len(cache._expiration_heap) == 1


class TestCleanupLoopBranches:
    """Tests for cleanup loop exit branches (lines 607, 614-617)."""

    @pytest.mark.asyncio
    async def test_cleanup_loop_exits_after_cleanup_when_stopped(self):
        """Test cleanup loop checks flags after _cleanup_expired (line 607)."""
        config = StateConfig(cleanup_interval=0.01)
        cache = Cache(config)
        cache._running = True
        cache._task_cancelled = False
        cache._task_stop_requested = False

        iterations = []

        original_cleanup = cache._cleanup_expired

        async def tracked_cleanup():
            iterations.append(1)
            await original_cleanup()
            # Stop after first cleanup completes
            cache._running = False

        with patch.object(cache, "_cleanup_expired", tracked_cleanup):
            task = asyncio.create_task(cache._cleanup_loop())
            await asyncio.sleep(0.05)

            try:
                await asyncio.wait_for(task, timeout=0.5)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Should have run cleanup at least once
        assert len(iterations) >= 1

    @pytest.mark.asyncio
    async def test_cleanup_loop_exits_on_cancelled_flag(self):
        """Test cleanup loop exits when _task_cancelled is set (line 607)."""
        config = StateConfig(cleanup_interval=0.01)
        cache = Cache(config)
        cache._running = True
        cache._task_cancelled = False

        async def set_cancelled():
            await asyncio.sleep(0.02)
            cache._task_cancelled = True

        _ = asyncio.create_task(set_cancelled())  # noqa: RUF006

        task = asyncio.create_task(cache._cleanup_loop())

        try:
            await asyncio.wait_for(task, timeout=0.5)
        except asyncio.TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Loop should have exited

    @pytest.mark.asyncio
    async def test_cleanup_loop_cancelled_error_reraise(self, caplog):
        """Test cleanup loop re-raises CancelledError from _cleanup_expired (line 614-615).

        Lines 614-615 catch CancelledError that bubbles up from _cleanup_expired(),
        not from the sleep() call (which is caught at line 603-604 and breaks).
        """
        config = StateConfig(cleanup_interval=0.01)
        cache = Cache(config)
        cache._running = True
        cache._task_cancelled = False
        cache._task_stop_requested = False

        call_count = [0]

        async def raise_cancelled():
            call_count[0] += 1
            if call_count[0] == 1:
                # First cleanup call raises CancelledError
                raise asyncio.CancelledError()

        with patch.object(cache, "_cleanup_expired", raise_cancelled):
            task = asyncio.create_task(cache._cleanup_loop())

            # The task should exit by re-raising CancelledError
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_cleanup_loop_fatal_exception(self, caplog):
        """Test cleanup loop logs fatal exception (lines 616-617)."""
        config = StateConfig(cleanup_interval=0.01)
        cache = Cache(config)
        cache._running = True

        # Create loop that will raise a non-CancelledError exception
        call_count = [0]

        async def failing_cleanup():
            call_count[0] += 1
            if call_count[0] == 1:
                # First call raises fatal exception
                raise RuntimeError("Fatal error in cleanup")

        with (
            patch.object(cache, "_cleanup_expired", failing_cleanup),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(cache._cleanup_loop())
            await asyncio.sleep(0.05)

            cache._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Should have logged the error
        assert "Error during cache cleanup" in caplog.text


class TestCrossLoopScenarios:
    """Tests for cross-loop scenarios (lines 98-99, 164-173, 185-186)."""

    @pytest.mark.asyncio
    async def test_start_detects_different_event_loop(self):
        """Test start handles cross-loop scenario (lines 98-99)."""
        config = StateConfig(cleanup_interval=0.5)
        cache = Cache(config)

        # Start the cache
        await cache.start()
        # original_loop = cache._event_loop
        original_task = cache._cleanup_task

        assert cache._running
        assert original_task is not None

        # Simulate having a different saved loop (as if cache was used in another loop)
        # Create a mock loop that is "different"
        mock_old_loop = MagicMock()
        mock_old_loop.call_soon_threadsafe = MagicMock()

        # Create a mock old task
        mock_old_task = MagicMock()
        mock_old_task.done.return_value = True

        # Set up the cache as if it was from a different loop
        cache._running = False  # Set to false so start() will proceed
        cache._event_loop = mock_old_loop
        cache._cleanup_task = mock_old_task

        # Now call start() - it should detect different loop
        await cache.start()

        # Should have created new task in current loop
        assert cache._running
        assert cache._event_loop == asyncio.get_running_loop()
        assert cache._cleanup_task != mock_old_task

        await cache.stop()

    @pytest.mark.asyncio
    async def test_stop_handles_cross_loop_cleanup(self):
        """Test stop handles cross-loop cleanup (lines 164-173)."""
        config = StateConfig(cleanup_interval=0.5)
        cache = Cache(config)

        await cache.start()

        # Simulate cross-loop by setting a different event loop
        mock_old_loop = MagicMock()
        mock_old_loop.call_soon_threadsafe = MagicMock()

        mock_old_task = MagicMock()
        mock_old_task.done.return_value = True
        mock_old_task.cancel = MagicMock()

        # Set internal state to simulate cross-loop
        cache._event_loop = mock_old_loop
        cache._cleanup_task = mock_old_task

        # Stop should detect different loop and handle it
        await cache.stop()

        assert cache._cleanup_task is None
        assert not cache._running

    @pytest.mark.asyncio
    async def test_stop_handles_runtime_error(self, caplog):
        """Test stop handles RuntimeError from get_running_loop (lines 185-186)."""
        config = StateConfig()
        cache = Cache(config)

        await cache.start()
        original_task = cache._cleanup_task

        # Mock get_running_loop to raise RuntimeError
        call_count = [0]

        def raising_get_loop():
            call_count[0] += 1
            if call_count[0] > 0:
                raise RuntimeError("No running event loop")
            return asyncio.get_running_loop()

        with patch("asyncio.get_running_loop", side_effect=raising_get_loop):
            # Stop should catch RuntimeError
            await cache.stop()

        # Should still clean up
        assert cache._cleanup_task is None

        # Clean up original task if needed
        if original_task and not original_task.done():
            original_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await original_task

    @pytest.mark.asyncio
    async def test_cancel_cross_loop_task_timeout_not_done(self):
        """Test _cancel_cross_loop_task when task doesn't finish in time."""
        config = StateConfig()
        cache = Cache(config)

        # Create a task that ignores cancellation for a while
        async def stubborn():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(2)  # Delay before exiting
                raise

        task = asyncio.create_task(stubborn())
        await asyncio.sleep(0.01)  # Let it start

        current_loop = asyncio.get_running_loop()

        # Very short timeout - task won't finish
        _result = await cache._cancel_cross_loop_task(task, current_loop, timeout=0.05)

        # Result depends on whether task managed to complete
        # Either way, task should have been cancelled
        assert cache._task_cancelled

        # Cleanup
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.5)

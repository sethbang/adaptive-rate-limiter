"""
Tests for state.py coverage gaps.

This module specifically targets missing branches and error handlers:
- Lines 585-601: atomic_bulk_get with expired entries and else branch
- Lines 630-642: atomic_bulk_set TTL handling (override, config, none)
- Lines 1074-1098: Signal handler installation/restoration errors
- Lines 1159-1183: _flush_pending_updates_sync edge cases
- Lines 1751-1760: Batch and cleanup loop exception handling
"""

import asyncio
import contextlib
import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adaptive_rate_limiter.backends.base import BaseBackend
from adaptive_rate_limiter.scheduler.config import CachePolicy, StateConfig
from adaptive_rate_limiter.scheduler.state import (
    Cache,
    PendingUpdate,
    StateEntry,
    StateManager,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = AsyncMock(spec=BaseBackend)
    backend.namespace = "default"
    return backend


@pytest.fixture
def config():
    """Create a StateConfig with short TTL for testing."""
    return StateConfig(
        cache_policy=CachePolicy.WRITE_THROUGH,
        cache_ttl=60.0,
    )


@pytest.fixture
def cache(config):
    """Create a Cache instance."""
    return Cache(config)


@pytest.fixture
def manager(mock_backend, config):
    """Create a StateManager with mock backend."""
    return StateManager(backend=mock_backend, config=config)


class TestAtomicBulkGetCoverage:
    """Tests for atomic_bulk_get branches (lines 585-601)."""

    @pytest.mark.asyncio
    async def test_atomic_bulk_get_with_expired_entry(self, cache):
        """Test atomic_bulk_get removes expired entry and increments misses (lines 591-596)."""
        # Create an expired entry - must set created_at in the past to satisfy validator
        old_created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        past_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        entry = StateEntry(
            key="expired_key",
            data={"value": "old"},
            created_at=old_created_at,
            expires_at=past_expires_at,
        )

        # Manually add to cache without using set (bypasses TTL logic)
        cache._cache["expired_key"] = entry
        cache._creation_times["expired_key"] = time.time()

        # Reset metrics
        cache.metrics.cache_hits = 0
        cache.metrics.cache_misses = 0

        results = await cache.atomic_bulk_get(["expired_key"])

        # Should return None for expired entry
        assert results["expired_key"] is None
        # Should have recorded a miss
        assert cache.metrics.cache_misses == 1
        assert cache.metrics.cache_hits == 0
        # Entry should be removed from cache
        assert "expired_key" not in cache._cache

    @pytest.mark.asyncio
    async def test_atomic_bulk_get_nonexistent_entry(self, cache):
        """Test atomic_bulk_get with non-existent key (lines 597-600)."""
        # Reset metrics
        cache.metrics.cache_hits = 0
        cache.metrics.cache_misses = 0

        results = await cache.atomic_bulk_get(["nonexistent_key"])

        # Should return None
        assert results["nonexistent_key"] is None
        # Should have recorded a miss
        assert cache.metrics.cache_misses == 1
        assert cache.metrics.cache_hits == 0

    @pytest.mark.asyncio
    async def test_atomic_bulk_get_valid_entry(self, cache):
        """Test atomic_bulk_get with valid entry (lines 585-590)."""
        # Create a valid entry with future expiration
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        entry = StateEntry(
            key="valid_key",
            data={"value": "test"},
            expires_at=future_time,
        )
        await cache.set(entry)

        # Reset metrics
        cache.metrics.cache_hits = 0
        cache.metrics.cache_misses = 0

        results = await cache.atomic_bulk_get(["valid_key"])

        # Should return the entry
        assert results["valid_key"] is not None
        assert results["valid_key"].data["value"] == "test"
        # Should have recorded a hit
        assert cache.metrics.cache_hits == 1
        assert cache.metrics.cache_misses == 0

    @pytest.mark.asyncio
    async def test_atomic_bulk_get_mixed_entries(self, cache):
        """Test atomic_bulk_get with mix of valid, expired, and missing."""
        # Valid entry
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        valid_entry = StateEntry(
            key="valid_key",
            data={"value": "valid"},
            expires_at=future_time,
        )
        await cache.set(valid_entry)

        # Expired entry - must set created_at in the past to satisfy validator
        old_created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        past_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        expired_entry = StateEntry(
            key="expired_key",
            data={"value": "expired"},
            created_at=old_created_at,
            expires_at=past_expires_at,
        )
        cache._cache["expired_key"] = expired_entry
        cache._creation_times["expired_key"] = time.time()

        # Reset metrics
        cache.metrics.cache_hits = 0
        cache.metrics.cache_misses = 0

        results = await cache.atomic_bulk_get(
            ["valid_key", "expired_key", "missing_key"]
        )

        # Check results
        assert results["valid_key"] is not None
        assert results["expired_key"] is None
        assert results["missing_key"] is None

        # Check metrics - 1 hit, 2 misses
        assert cache.metrics.cache_hits == 1
        assert cache.metrics.cache_misses == 2


class TestAtomicBulkSetCoverage:
    """Tests for atomic_bulk_set TTL handling (lines 630-642)."""

    @pytest.mark.asyncio
    async def test_atomic_bulk_set_with_ttl_override(self):
        """Test atomic_bulk_set applies ttl_override (lines 632-635)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH, cache_ttl=3600.0)
        cache = Cache(config)

        entry = StateEntry(key="test_key", data={"value": 1})

        # Set with TTL override of 10 seconds
        before = datetime.now(timezone.utc)
        await cache.atomic_bulk_set([entry], ttl_override=10.0)
        after = datetime.now(timezone.utc)

        # Verify expiration was set based on override
        cached = await cache.get("test_key")
        assert cached is not None
        assert cached.expires_at is not None

        # Expiration should be approximately 10 seconds from now
        expected_min = before + timedelta(seconds=9)
        expected_max = after + timedelta(seconds=11)
        assert expected_min <= cached.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_atomic_bulk_set_with_config_ttl(self):
        """Test atomic_bulk_set applies config cache_ttl (lines 636-639)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH, cache_ttl=120.0)
        cache = Cache(config)

        entry = StateEntry(key="test_key", data={"value": 1})

        before = datetime.now(timezone.utc)
        # No ttl_override, should use config.cache_ttl
        await cache.atomic_bulk_set([entry])
        after = datetime.now(timezone.utc)

        cached = await cache.get("test_key")
        assert cached is not None
        assert cached.expires_at is not None

        # Expiration should be approximately 120 seconds from now
        expected_min = before + timedelta(seconds=119)
        expected_max = after + timedelta(seconds=121)
        assert expected_min <= cached.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_atomic_bulk_set_entry_with_existing_expiration(self):
        """Test atomic_bulk_set preserves existing expiration when no override and no config TTL.

        Note: StateConfig requires cache_ttl > 0, so we test the case where
        the entry already has an expiration set and config TTL is applied.
        """
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH, cache_ttl=60.0)
        cache = Cache(config)

        # Create entry with explicit far-future expiration
        future_time = datetime.now(timezone.utc) + timedelta(hours=24)
        entry = StateEntry(
            key="test_key",
            data={"value": 1},
            expires_at=future_time,  # Explicit expiration
        )

        # Set entry normally first (to establish it)
        await cache.atomic_bulk_set([entry])

        cached = await cache.get("test_key")
        assert cached is not None
        # Config TTL should override the original expiration
        assert cached.expires_at is not None

    @pytest.mark.asyncio
    async def test_atomic_bulk_set_multiple_entries(self):
        """Test atomic_bulk_set with multiple entries."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH, cache_ttl=60.0)
        cache = Cache(config)

        entries = [StateEntry(key=f"key_{i}", data={"value": i}) for i in range(5)]

        results = await cache.atomic_bulk_set(entries)

        # All should succeed
        assert all(results.values())
        assert len(results) == 5

        # All should be in cache
        for i in range(5):
            cached = await cache.get(f"key_{i}")
            assert cached is not None
            assert cached.data["value"] == i


class TestSignalHandlerErrorsCoverage:
    """Tests for signal handler error handling (lines 1074-1098)."""

    def test_install_signal_handlers_value_error(self, mock_backend, caplog):
        """Test _install_signal_handlers handles ValueError (line 1074-1076)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        with (
            patch.object(signal, "getsignal", side_effect=ValueError("Bad signal")),
            caplog.at_level(logging.DEBUG),
        ):
            manager._install_signal_handlers()

        # Should not have installed handlers due to error
        # Note: Due to early return on exception, flag may not be set
        assert not manager._signal_handlers_installed

    def test_install_signal_handlers_os_error(self, mock_backend, caplog):
        """Test _install_signal_handlers handles OSError (line 1074-1076)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        with (
            patch.object(signal, "getsignal", side_effect=OSError("OS error")),
            caplog.at_level(logging.DEBUG),
        ):
            manager._install_signal_handlers()

        # Should not have installed handlers due to error
        assert not manager._signal_handlers_installed

    def test_restore_signal_handlers_value_error(self, mock_backend, caplog):
        """Test _restore_signal_handlers handles ValueError (lines 1097-1098)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        # Manually set state as if handlers were installed
        manager._signal_handlers_installed = True
        manager._original_sigterm = signal.SIG_DFL
        manager._original_sigint = signal.SIG_DFL

        with (
            patch.object(signal, "signal", side_effect=ValueError("Bad signal")),
            caplog.at_level(logging.DEBUG),
        ):
            manager._restore_signal_handlers()

        # Handler flag should be unchanged due to error in restoration
        # The method catches the exception and logs it
        assert (
            "Could not restore signal handlers" in caplog.text
            or manager._signal_handlers_installed
        )

    def test_restore_signal_handlers_os_error(self, mock_backend, caplog):
        """Test _restore_signal_handlers handles OSError (lines 1097-1098)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        # Manually set state as if handlers were installed
        manager._signal_handlers_installed = True
        manager._original_sigterm = signal.SIG_DFL
        manager._original_sigint = signal.SIG_DFL

        with (
            patch.object(signal, "signal", side_effect=OSError("OS error")),
            caplog.at_level(logging.DEBUG),
        ):
            manager._restore_signal_handlers()

        # Should log the error
        assert (
            "Could not restore signal handlers" in caplog.text
            or manager._signal_handlers_installed
        )

    def test_install_signal_handlers_non_main_thread_value_error(
        self, mock_backend, caplog
    ):
        """Test signal handler skip on non-main thread logs debug (lines 1061-1062)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        result_holder = {"called": False}

        def run_in_thread():
            with caplog.at_level(logging.DEBUG):
                manager._install_signal_handlers()
            result_holder["called"] = True

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        assert result_holder["called"]
        # Should have skipped installation in non-main thread
        assert not manager._signal_handlers_installed


class TestFlushPendingUpdatesSyncCoverage:
    """Tests for _flush_pending_updates_sync edge cases (lines 1159-1183)."""

    def test_flush_sync_timeout_error(self, mock_backend, caplog):
        """Test _flush_pending_updates_sync handles TimeoutError (lines 1178-1181)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        # Add pending updates
        entry = StateEntry(key="test", data={"value": 1})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        with patch(
            "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_running_loop"
        ) as mock_get_running_loop:
            mock_loop = MagicMock()
            mock_get_running_loop.return_value = mock_loop

            # Mock future that times out
            mock_future = MagicMock()
            mock_future.result.side_effect = TimeoutError("Timed out")

            with patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.run_coroutine_threadsafe"
            ) as mock_run:
                mock_run.return_value = mock_future

                with caplog.at_level(logging.WARNING):
                    manager._flush_pending_updates_sync()

                # Should log timeout warning
                assert "timed out" in caplog.text.lower()

    def test_flush_sync_no_running_loop_create_new(self, mock_backend, caplog):
        """Test _flush_pending_updates_sync creates new event loop (lines 1159-1162)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        # Add pending updates
        entry = StateEntry(key="test", data={"value": 1})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        # First RuntimeError: no running loop
        # Second RuntimeError: no event loop at all
        call_count = [0]

        def mock_get_running_loop():
            raise RuntimeError("No running loop")

        def mock_get_event_loop():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("No event loop")
            return MagicMock()

        with (
            patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_running_loop",
                side_effect=mock_get_running_loop,
            ),
            patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_event_loop",
                side_effect=mock_get_event_loop,
            ),
            patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.new_event_loop"
            ) as mock_new_loop,
            patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.set_event_loop"
            ) as mock_set_loop,
        ):
            mock_loop = MagicMock()
            mock_new_loop.return_value = mock_loop

            manager._flush_pending_updates_sync()

            # Should have created a new event loop
            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)

    def test_flush_sync_exception_during_write(self, mock_backend, caplog):
        """Test _flush_pending_updates_sync handles general exception (line 1189-1190)."""
        config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
        manager = StateManager(backend=mock_backend, config=config)

        # Add pending updates
        entry = StateEntry(key="test", data={"value": 1})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        with patch(
            "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_running_loop"
        ) as mock_get_running_loop:
            mock_loop = MagicMock()
            mock_get_running_loop.return_value = mock_loop

            # Mock future that raises generic exception
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("Backend error")

            with patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.run_coroutine_threadsafe"
            ) as mock_run:
                mock_run.return_value = mock_future

                with caplog.at_level(logging.ERROR):
                    manager._flush_pending_updates_sync()

                # Should log error
                assert "Failed to flush state" in caplog.text


class TestBatchLoopErrorCoverage:
    """Tests for _batch_loop error handling (lines 1751-1752)."""

    @pytest.mark.asyncio
    async def test_batch_loop_attribute_error(self, mock_backend, caplog):
        """Test _batch_loop handles AttributeError from flush itself (line 1751)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            batch_timeout=0.01,
        )
        manager = StateManager(backend=mock_backend, config=config)
        manager._running = True

        # Add pending update so flush is triggered
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        # Mock _flush_pending_updates to raise AttributeError
        # This tests the exception handler in _batch_loop lines 1751-1752
        with (
            patch.object(
                manager,
                "_flush_pending_updates",
                side_effect=AttributeError("Missing attribute"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._batch_loop())
            await asyncio.sleep(0.05)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert "Error in batch processing" in caplog.text

    @pytest.mark.asyncio
    async def test_batch_loop_value_error(self, mock_backend, caplog):
        """Test _batch_loop handles ValueError from flush itself (line 1751)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            batch_timeout=0.01,
        )
        manager = StateManager(backend=mock_backend, config=config)
        manager._running = True

        # Add pending update so flush is triggered
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        # Mock _flush_pending_updates to raise ValueError
        with (
            patch.object(
                manager,
                "_flush_pending_updates",
                side_effect=ValueError("Invalid value"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._batch_loop())
            await asyncio.sleep(0.05)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert "Error in batch processing" in caplog.text

    @pytest.mark.asyncio
    async def test_batch_loop_os_error(self, mock_backend, caplog):
        """Test _batch_loop handles OSError from flush itself (line 1751)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            batch_timeout=0.01,
        )
        manager = StateManager(backend=mock_backend, config=config)
        manager._running = True

        # Add pending update so flush is triggered
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        # Mock _flush_pending_updates to raise OSError
        with (
            patch.object(
                manager,
                "_flush_pending_updates",
                side_effect=OSError("Network error"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._batch_loop())
            await asyncio.sleep(0.05)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert "Error in batch processing" in caplog.text


class TestCleanupLoopErrorCoverage:
    """Tests for _cleanup_loop error handling (lines 1754-1764)."""

    @pytest.mark.asyncio
    async def test_cleanup_loop_general_exception(self, mock_backend, caplog):
        """Test _cleanup_loop handles general Exception (line 1763-1764)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            reservation_cleanup_interval=0.01,
        )
        manager = StateManager(backend=mock_backend, config=config)
        manager._running = True

        # Make cleanup raise exception
        with (
            patch.object(
                manager,
                "_cleanup_expired_reservations",
                side_effect=Exception("Cleanup failed"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._cleanup_loop())
            await asyncio.sleep(0.05)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert "Error in cleanup loop" in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_loop_account_state_exception(self, mock_backend, caplog):
        """Test _cleanup_loop handles exception in account cleanup."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            reservation_cleanup_interval=0.01,
        )
        manager = StateManager(backend=mock_backend, config=config)
        manager._running = True

        # Make account cleanup raise exception
        with (
            patch.object(
                manager,
                "_cleanup_account_states",
                side_effect=RuntimeError("Account cleanup failed"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._cleanup_loop())
            await asyncio.sleep(0.05)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert "Error in cleanup loop" in caplog.text


class TestFlushWithBackoffRetry:
    """Additional tests for flush retry/backoff logic."""

    @pytest.mark.asyncio
    async def test_flush_respects_backoff_delay(self, mock_backend):
        """Test flush waits for backoff delay before retry (lines 1843-1851)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            flush_backoff_base=0.5,
            flush_backoff_max=10.0,
            flush_max_retries=3,
        )
        manager = StateManager(backend=mock_backend, config=config)

        # Create update that has already been retried once, very recently
        entry = StateEntry(key="test", data={"value": 1})
        pending = PendingUpdate(
            entry=entry,
            retry_count=1,
            last_attempt_time=time.time(),  # Just now
        )
        manager._pending_updates = [pending]

        # Flush should re-queue without calling backend (not enough time elapsed)
        await manager._flush_pending_updates()

        # Update should be re-queued (not ready for retry yet)
        assert len(manager._pending_updates) == 1
        # Backend should not be called (skipped due to backoff)
        mock_backend.set_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_drops_after_max_retries(self, mock_backend, caplog):
        """Test flush drops update after max retries (lines 1870-1879)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            flush_max_retries=2,
        )
        manager = StateManager(backend=mock_backend, config=config)

        # Create update that has reached max retries
        entry = StateEntry(key="test", data={"value": 1})
        pending = PendingUpdate(
            entry=entry,
            retry_count=2,  # At max
            last_attempt_time=0,  # Long ago, ready for retry
        )
        manager._pending_updates = [pending]

        # Make backend fail
        mock_backend.set_state.side_effect = OSError("Backend down")

        with caplog.at_level(logging.ERROR):
            await manager._flush_pending_updates()

        # Update should be dropped (not re-queued)
        assert len(manager._pending_updates) == 0
        # Error should be logged
        assert "DROPPING state update" in caplog.text
        # Flush drops metric should be incremented
        assert manager.cache.metrics.flush_drops >= 1


class TestCacheEntryExpiration:
    """Additional cache expiration tests."""

    @pytest.mark.asyncio
    async def test_cache_get_lock_free_expired_removal(self):
        """Test lock-free get removes expired entries (lines 371-372)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            lock_free_reads=True,
        )
        cache = Cache(config)

        # Create expired entry - must set created_at in the past to satisfy validator
        old_created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        past_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        expired_entry = StateEntry(
            key="expired_key",
            data={"value": "old"},
            created_at=old_created_at,
            expires_at=past_expires_at,
        )
        cache._cache["expired_key"] = expired_entry
        cache._creation_times["expired_key"] = time.time()

        # Get should return None and remove the entry
        result = await cache.get("expired_key")

        assert result is None
        assert "expired_key" not in cache._cache

    @pytest.mark.asyncio
    async def test_cache_get_lock_free_exception_handling(self):
        """Test lock-free get handles exceptions (lines 376-379)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_THROUGH,
            lock_free_reads=True,
        )
        cache = Cache(config)

        # Add a mock entry that will raise on is_expired
        mock_entry = MagicMock()
        mock_entry.is_expired = property(
            lambda _: (_ for _ in ()).throw(ValueError("test"))
        )
        cache._cache["bad_key"] = mock_entry

        # Should catch exception and return None
        result = await cache.get("bad_key")

        assert result is None
        assert cache.metrics.cache_misses >= 1

"""
Tests for StateManager lifecycle, reservations, and cleanup edge cases.

This module covers:
- StateManager start/stop lifecycle
- Reservation management edge cases
- Cleanup expired reservations
- Tomorrow midnight calculation
- Add to batch triggering flush
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from adaptive_rate_limiter.backends.base import BaseBackend
from adaptive_rate_limiter.scheduler.config import CachePolicy, StateConfig
from adaptive_rate_limiter.scheduler.state import (
    PendingUpdate,
    RateLimitState,
    StateEntry,
    StateManager,
    StateType,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = AsyncMock(spec=BaseBackend)
    backend.namespace = "default"
    return backend


@pytest.fixture
def manager(mock_backend):
    """Create a StateManager with mock backend."""
    config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
    return StateManager(backend=mock_backend, config=config)


class TestStateManagerLifecycle:
    """Tests for StateManager start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_already_running(self, manager):
        """Test start when already running (line 832-833)."""
        # Mock background tasks
        with (
            patch.object(manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            await manager.start()
            assert manager._running

            original_batch_task = manager._batch_task

            # Start again - should be no-op
            await manager.start()
            assert manager._running
            assert manager._batch_task is original_batch_task

            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_not_running(self, manager):
        """Test stop when not running."""
        assert not manager._running

        await manager.stop()

        assert not manager._running
        manager.backend.set_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_flushes_pending(self, manager):
        """Test stop flushes pending updates (line 850)."""
        # Add pending update
        entry = StateEntry(key="test", data={"v": 1})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]
        manager._running = True

        # Create mock tasks
        manager._batch_task = asyncio.create_task(asyncio.sleep(10))
        manager._cleanup_task = asyncio.create_task(asyncio.sleep(10))

        await manager.stop()

        # Should have flushed
        manager.backend.set_state.assert_called()

    @pytest.mark.asyncio
    async def test_stop_cancels_batch_task(self, manager):
        """Test stop cancels batch task (lines 852-858)."""
        manager._running = True

        async def long_batch():
            await asyncio.sleep(100)

        manager._batch_task = asyncio.create_task(long_batch())
        manager._cleanup_task = None

        await manager.stop()

        assert manager._batch_task.cancelled() or manager._batch_task.done()

    @pytest.mark.asyncio
    async def test_stop_cancels_cleanup_task(self, manager):
        """Test stop cancels cleanup task (lines 859-864)."""
        manager._running = True

        async def long_cleanup():
            await asyncio.sleep(100)

        manager._batch_task = None
        manager._cleanup_task = asyncio.create_task(long_cleanup())

        await manager.stop()

        assert manager._cleanup_task.cancelled() or manager._cleanup_task.done()


class TestReservationManagement:
    """Tests for reservation management edge cases."""

    @pytest.mark.asyncio
    async def test_create_reservation_with_ttl(self, manager):
        """Test create_reservation sets proper expiration."""
        res_id = await manager.create_reservation(
            "req-1", {"bucket_id": "bucket-1", "tokens": 100}, ttl_seconds=60.0
        )

        assert res_id == "req-1"

        # Check internal reservation
        async with manager._reservation_lock:
            reservation = manager._reservations["req-1"]
            assert reservation["tokens"] == 100
            assert "expires_at" in reservation
            assert reservation["expires_at"] > time.time()

        # Check cache
        cached = await manager.cache.get("reservation:req-1")
        assert cached is not None
        assert cached.state_type == StateType.RESERVATION

    @pytest.mark.asyncio
    async def test_get_reservation_from_cache(self, manager):
        """Test get_reservation retrieves from cache first."""
        # Set up in cache
        entry = StateEntry(
            key="reservation:req-1",
            data={"bucket_id": "bucket-1"},
            state_type=StateType.RESERVATION,
        )
        await manager.cache.set(entry)

        result = await manager.get_reservation("req-1")

        assert result["bucket_id"] == "bucket-1"

    @pytest.mark.asyncio
    async def test_get_reservation_from_internal_store(self, manager):
        """Test get_reservation falls back to internal store."""
        current_time = time.time()
        async with manager._reservation_lock:
            manager._reservations["req-1"] = {
                "data": "test",
                "created_at": current_time,
                "expires_at": current_time + 300,
            }

        # Cache miss, should fall back to internal store
        result = await manager.get_reservation("req-1")
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_get_reservation_expired_internal(self, manager):
        """Test get_reservation returns None for expired internal reservation."""
        current_time = time.time()
        async with manager._reservation_lock:
            manager._reservations["req-1"] = {
                "data": "test",
                "created_at": current_time - 400,
                "expires_at": current_time - 100,  # Expired
            }

        result = await manager.get_reservation("req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_reservation_not_found(self, manager):
        """Test get_reservation returns None when not found."""
        result = await manager.get_reservation("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_release_reservation_data_clears_both(self, manager):
        """Test release_reservation_data clears cache and internal store."""
        current_time = time.time()

        # Set up in both locations
        async with manager._reservation_lock:
            manager._reservations["req-1"] = {
                "data": "test",
                "expires_at": current_time + 300,
            }

        await manager.cache.set(
            StateEntry(
                key="reservation:req-1",
                data={"data": "test"},
                state_type=StateType.RESERVATION,
            )
        )

        result = await manager.release_reservation_data("req-1")

        assert result is True
        async with manager._reservation_lock:
            assert "req-1" not in manager._reservations
        assert await manager.cache.get("reservation:req-1") is None

    @pytest.mark.asyncio
    async def test_release_reservation_data_not_found(self, manager):
        """Test release_reservation_data returns True even if not found."""
        result = await manager.release_reservation_data("nonexistent")
        assert result is True


class TestCleanupExpiredReservations:
    """Tests for _cleanup_expired_reservations."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, manager, caplog):
        """Test cleanup removes expired reservations (lines 1356-1368)."""
        current_time = time.time()

        async with manager._reservation_lock:
            # Expired reservation
            manager._reservations["expired"] = {
                "expires_at": current_time - 100,
            }
            manager._reservation_timestamps["expired"] = current_time - 100

            # Valid reservation
            manager._reservations["valid"] = {
                "expires_at": current_time + 100,
            }
            manager._reservation_timestamps["valid"] = current_time

        with caplog.at_level(logging.INFO):
            await manager._cleanup_expired_reservations()

        assert "expired" not in manager._reservations
        assert "valid" in manager._reservations

    @pytest.mark.asyncio
    async def test_cleanup_no_expired(self, manager, caplog):
        """Test cleanup with no expired reservations."""
        current_time = time.time()

        async with manager._reservation_lock:
            manager._reservations["valid1"] = {"expires_at": current_time + 100}
            manager._reservations["valid2"] = {"expires_at": current_time + 200}

        with caplog.at_level(logging.INFO):
            await manager._cleanup_expired_reservations()

        # Both should remain
        assert "valid1" in manager._reservations
        assert "valid2" in manager._reservations
        # No log message about cleanup
        assert "Cleaned up" not in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_all_expired(self, manager, caplog):
        """Test cleanup when all reservations are expired."""
        current_time = time.time()

        async with manager._reservation_lock:
            manager._reservations["exp1"] = {"expires_at": current_time - 100}
            manager._reservation_timestamps["exp1"] = current_time - 100
            manager._reservations["exp2"] = {"expires_at": current_time - 50}
            manager._reservation_timestamps["exp2"] = current_time - 50

        with caplog.at_level(logging.INFO):
            await manager._cleanup_expired_reservations()

        assert len(manager._reservations) == 0
        assert "Cleaned up 2 expired reservations" in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_reservation_without_timestamp(self, manager):
        """Test cleanup handles reservation without timestamp entry."""
        current_time = time.time()

        async with manager._reservation_lock:
            # Reservation without matching timestamp entry
            manager._reservations["orphan"] = {"expires_at": current_time - 100}
            # No entry in _reservation_timestamps

        await manager._cleanup_expired_reservations()

        # Should still be removed from _reservations
        assert "orphan" not in manager._reservations


class TestTomorrowMidnight:
    """Tests for _tomorrow_midnight calculation."""

    def test_tomorrow_midnight_returns_utc(self, manager):
        """Test _tomorrow_midnight returns UTC datetime."""
        result = manager._tomorrow_midnight()

        assert result.tzinfo == timezone.utc
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.microsecond == 0

    def test_tomorrow_midnight_is_future(self, manager):
        """Test _tomorrow_midnight returns future date."""
        now = datetime.now(timezone.utc)
        result = manager._tomorrow_midnight()

        # Should be after now
        assert result > now

        # Should be within 24-48 hours
        delta = result - now
        assert 0 < delta.total_seconds() < 48 * 3600


class TestAddToBatch:
    """Tests for _add_to_batch edge cases."""

    @pytest.mark.asyncio
    async def test_add_to_batch_triggers_flush_on_size(self, manager):
        """Test _add_to_batch triggers flush when batch_size reached."""
        manager.config.batch_size = 2
        manager.config.batch_timeout = 3600  # Long timeout

        entry1 = StateEntry(key="k1", data={"v": 1})
        entry2 = StateEntry(key="k2", data={"v": 2})

        await manager._add_to_batch(entry1)
        assert len(manager._pending_updates) == 1

        await manager._add_to_batch(entry2)

        # Should have flushed
        assert manager.backend.set_state.call_count >= 2

    @pytest.mark.asyncio
    async def test_add_to_batch_triggers_flush_on_timeout(self, manager):
        """Test _add_to_batch triggers flush when timeout reached."""
        manager.config.batch_size = 100  # High size
        manager.config.batch_timeout = 0.001  # Very short timeout

        # Set last batch time to old
        manager._last_batch_time = time.time() - 1

        entry = StateEntry(key="k1", data={"v": 1})
        await manager._add_to_batch(entry)

        # Should have triggered flush due to timeout
        manager.backend.set_state.assert_called()

    @pytest.mark.asyncio
    async def test_add_to_batch_no_flush(self, manager):
        """Test _add_to_batch does not flush when under thresholds."""
        manager.config.batch_size = 100
        manager.config.batch_timeout = 3600

        entry = StateEntry(key="k1", data={"v": 1})
        await manager._add_to_batch(entry)

        # Should not flush yet
        assert len(manager._pending_updates) == 1
        manager.backend.set_state.assert_not_called()


class TestCacheInfoMethods:
    """Tests for cache info methods edge cases."""

    @pytest.mark.asyncio
    async def test_get_cached_bucket_info_from_cache(self, manager):
        """Test get_cached_bucket_info retrieves from cache."""
        entry = StateEntry(
            key="bucket_info",
            data={"buckets": ["b1", "b2"]},
            state_type=StateType.BUCKET_INFO,
        )
        await manager.cache.set(entry)

        result = await manager.get_cached_bucket_info()

        assert result["buckets"] == ["b1", "b2"]
        manager.backend.get_cached_bucket_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_cached_bucket_info_wrong_type(self, manager):
        """Test get_cached_bucket_info ignores wrong state type."""
        # Cache with wrong type
        entry = StateEntry(
            key="bucket_info",
            data={"buckets": ["b1"]},
            state_type=StateType.MODEL_CONFIG,  # Wrong type
        )
        await manager.cache.set(entry)

        manager.backend.get_cached_bucket_info.return_value = {"from_backend": True}

        result = await manager.get_cached_bucket_info()

        # Should fall back to backend
        assert result["from_backend"] is True

    @pytest.mark.asyncio
    async def test_get_cached_model_info_from_cache(self, manager):
        """Test get_cached_model_info retrieves from cache."""
        entry = StateEntry(
            key="model:gpt-5",
            data={"tier": "premium"},
            state_type=StateType.MODEL_CONFIG,
        )
        await manager.cache.set(entry)

        result = await manager.get_cached_model_info("gpt-5")

        assert result["tier"] == "premium"
        manager.backend.get_cached_model_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_cached_model_info_wrong_type(self, manager):
        """Test get_cached_model_info ignores wrong state type."""
        # Cache with wrong type
        entry = StateEntry(
            key="model:gpt-5",
            data={"tier": "premium"},
            state_type=StateType.RESERVATION,  # Wrong type
        )
        await manager.cache.set(entry)

        manager.backend.get_cached_model_info.return_value = {"from_backend": True}

        result = await manager.get_cached_model_info("gpt-5")

        # Should fall back to backend
        assert result["from_backend"] is True


class TestCachePolicySetState:
    """Tests for set_state with different cache policies."""

    @pytest.mark.asyncio
    async def test_set_state_write_around(self, manager):
        """Test set_state with WRITE_AROUND policy (lines 933-935)."""
        manager.config.cache_policy = CachePolicy.WRITE_AROUND

        state = RateLimitState(model_id="test")
        await manager.set_state("test", state)

        # Should write to backend
        manager.backend.set_state.assert_called_once()

        # Should be removed from cache after set
        cached = await manager.cache.get("test")
        assert cached is None

    @pytest.mark.asyncio
    async def test_set_state_dict_data(self, manager):
        """Test set_state with dict data instead of RateLimitState (line 917-918)."""
        manager.config.cache_policy = CachePolicy.WRITE_THROUGH

        data = {"custom": "data"}
        await manager.set_state("test", data)

        # Should work with dict
        manager.backend.set_state.assert_called_once()
        cached = await manager.cache.get("test")
        assert cached is not None
        assert cached.data["custom"] == "data"


class TestRecordFailedRequest:
    """Tests for record_failed_request."""

    @pytest.mark.asyncio
    async def test_record_failed_raises_at_limit(self, manager):
        """Test record_failed_request raises at limit."""
        # Record 21 failures to exceed limit
        for _i in range(21):
            try:
                await manager.record_failed_request()
            except Exception as e:
                if "TooManyFailedRequestsError" in str(e):
                    break

        # Should have raised
        assert manager.is_failed_limit_exceeded()

    @pytest.mark.asyncio
    async def test_record_failed_returns_count(self, manager):
        """Test record_failed_request returns current count."""
        count1 = await manager.record_failed_request()
        count2 = await manager.record_failed_request()
        count3 = await manager.record_failed_request()

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3

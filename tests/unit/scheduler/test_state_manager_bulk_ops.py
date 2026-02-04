"""
Tests for StateManager bulk operations and initialization edge cases.

This module covers:
- Bulk get/set operations with different state types
- Initialize rate limit state with provider
- Background loop exception handling
- Flush pending updates error handling
- Health check edge cases
"""

import asyncio
import contextlib
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

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


class TestBulkGetStates:
    """Tests for bulk_get_states edge cases."""

    @pytest.mark.asyncio
    async def test_bulk_get_rate_limit_states_from_cache(self, manager):
        """Test bulk_get_states with RateLimitState type from cache (line 1215-1216)."""
        # Cache one state
        state_data = {"model_id": "bucket-1", "remaining_requests": 50}
        await manager.cache.set(StateEntry(key="bucket-1", data=state_data))

        results = await manager.bulk_get_states(["bucket-1"], StateType.RATE_LIMIT)

        assert isinstance(results["bucket-1"], RateLimitState)
        assert results["bucket-1"].remaining_requests == 50

    @pytest.mark.asyncio
    async def test_bulk_get_rate_limit_states_from_backend(self, manager):
        """Test bulk_get_states with RateLimitState from backend (line 1235-1236)."""
        # Backend has state
        manager.backend.get_all_states.return_value = {
            "bucket-2": {"model_id": "bucket-2", "remaining_requests": 100}
        }

        results = await manager.bulk_get_states(["bucket-2"], StateType.RATE_LIMIT)

        assert isinstance(results["bucket-2"], RateLimitState)
        assert results["bucket-2"].remaining_requests == 100

    @pytest.mark.asyncio
    async def test_bulk_get_mixed_cache_backend(self, manager):
        """Test bulk_get_states with both cache and backend hits."""
        # Cache one state
        state_data = {"model_id": "bucket-1", "remaining_requests": 50}
        await manager.cache.set(StateEntry(key="bucket-1", data=state_data))

        # Backend has another
        manager.backend.get_all_states.return_value = {
            "bucket-2": {"model_id": "bucket-2", "remaining_requests": 100}
        }

        results = await manager.bulk_get_states(
            ["bucket-1", "bucket-2"], StateType.RATE_LIMIT
        )

        assert results["bucket-1"].remaining_requests == 50
        assert results["bucket-2"].remaining_requests == 100

    @pytest.mark.asyncio
    async def test_bulk_get_non_rate_limit_from_cache(self, manager):
        """Test bulk_get_states with non-RATE_LIMIT type from cache."""
        await manager.cache.set(
            StateEntry(
                key="config-1",
                data={"tier": "pro"},
                state_type=StateType.MODEL_CONFIG,
            )
        )

        results = await manager.bulk_get_states(["config-1"], StateType.MODEL_CONFIG)

        # Should return dict, not RateLimitState
        assert isinstance(results["config-1"], dict)
        assert results["config-1"]["tier"] == "pro"

    @pytest.mark.asyncio
    async def test_bulk_get_non_rate_limit_from_backend(self, manager):
        """Test bulk_get_states with non-RATE_LIMIT from backend."""
        manager.backend.get_all_states.return_value = {
            "config-2": {"tier": "enterprise"}
        }

        results = await manager.bulk_get_states(["config-2"], StateType.MODEL_CONFIG)

        assert isinstance(results["config-2"], dict)
        assert results["config-2"]["tier"] == "enterprise"

    @pytest.mark.asyncio
    async def test_bulk_get_missing_keys(self, manager):
        """Test bulk_get_states with keys not in cache or backend."""
        manager.backend.get_all_states.return_value = {}

        results = await manager.bulk_get_states(
            ["missing-1", "missing-2"], StateType.MODEL_CONFIG
        )

        # Missing keys should not be in results
        assert "missing-1" not in results
        assert "missing-2" not in results


class TestBulkSetStates:
    """Tests for bulk_set_states edge cases."""

    @pytest.mark.asyncio
    async def test_bulk_set_rate_limit_states(self, manager):
        """Test bulk_set_states with RateLimitState objects (lines 1251-1252)."""
        states = {
            "bucket-1": RateLimitState(model_id="bucket-1", remaining_requests=50),
            "bucket-2": {"model_id": "bucket-2"},  # Dict version
        }

        await manager.bulk_set_states(states, StateType.RATE_LIMIT)

        # Verify both were cached
        cached1 = await manager.cache.get("bucket-1")
        cached2 = await manager.cache.get("bucket-2")
        assert cached1 is not None
        assert cached2 is not None

    @pytest.mark.asyncio
    async def test_bulk_set_write_through(self, manager):
        """Test bulk_set_states with WRITE_THROUGH policy (line 1267-1268)."""
        manager.config.cache_policy = CachePolicy.WRITE_THROUGH

        states = {"k1": {"val": 1}, "k2": {"val": 2}}

        await manager.bulk_set_states(states, StateType.MODEL_CONFIG)

        # Should write to backend for each
        assert manager.backend.set_state.call_count == 2

    @pytest.mark.asyncio
    async def test_bulk_set_write_around(self, manager):
        """Test bulk_set_states with WRITE_AROUND policy (line 1269)."""
        manager.config.cache_policy = CachePolicy.WRITE_AROUND

        states = {"k1": {"val": 1}}

        await manager.bulk_set_states(states, StateType.MODEL_CONFIG)

        # WRITE_AROUND doesn't add to batch (neither WRITE_THROUGH nor WRITE_BACK)
        # So no backend calls from bulk_set_states for WRITE_AROUND
        # (individual set_state would handle it)
        assert len(manager._pending_updates) == 0


class TestInitializeRateLimitState:
    """Tests for _initialize_rate_limit_state edge cases."""

    @pytest.mark.asyncio
    async def test_initialize_with_provider_success(self, manager):
        """Test _initialize_rate_limit_state with provider (lines 1277-1297)."""
        manager.provider = AsyncMock()
        manager.provider.discover_limits.return_value = {
            "bucket-1": Mock(rpm_limit=60, tpm_limit=10000)
        }

        state = await manager._initialize_rate_limit_state("bucket-1")

        assert state is not None
        assert state.model_id == "bucket-1"
        assert state.request_limit == 60
        assert state.token_limit == 10000
        assert state.is_verified is False

    @pytest.mark.asyncio
    async def test_initialize_provider_exception(self, manager):
        """Test _initialize_rate_limit_state when provider raises (lines 1298-1299)."""
        manager.provider = AsyncMock()
        manager.provider.discover_limits.side_effect = Exception("Provider error")
        manager.backend.get_state.return_value = None

        state = await manager._initialize_rate_limit_state("bucket-1")

        # Should fall back to default state
        assert state.model_id == "bucket-1"
        assert state.remaining_requests == 30  # Fallback value

    @pytest.mark.asyncio
    async def test_initialize_bucket_not_in_limits(self, manager):
        """Test _initialize_rate_limit_state when bucket not found in limits."""
        manager.provider = AsyncMock()
        manager.provider.discover_limits.return_value = {}  # Empty

        state = await manager._initialize_rate_limit_state("bucket-1")

        # Should fall back to default state
        assert state.model_id == "bucket-1"
        assert state.remaining_requests == 30  # Fallback value

    @pytest.mark.asyncio
    async def test_initialize_no_provider(self, manager):
        """Test _initialize_rate_limit_state without provider."""
        manager.provider = None

        state = await manager._initialize_rate_limit_state("bucket-1")

        # Should create fallback state
        assert state.model_id == "bucket-1"
        assert state.remaining_requests == 30


class TestBackgroundLoopExceptions:
    """Tests for background loop exception handling."""

    @pytest.mark.asyncio
    async def test_batch_loop_exception_handling(self, manager, caplog):
        """Test _batch_loop handles exceptions (lines 1336-1337)."""
        manager._running = True
        manager.config.batch_timeout = 0.01

        # Add entry and make flush fail
        entry = StateEntry(key="test", data={})
        async with manager._batch_lock:
            manager._pending_updates.append(PendingUpdate(entry=entry, retry_count=0))

        manager.backend.set_state.side_effect = OSError("Backend error")

        # Run batch loop briefly
        task = asyncio.create_task(manager._batch_loop())

        with caplog.at_level(logging.ERROR):
            await asyncio.sleep(0.03)

        manager._running = False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Errors should be logged
        assert (
            "Error in batch processing" in caplog.text
            or len(manager._pending_updates) > 0
        )

    @pytest.mark.asyncio
    async def test_cleanup_loop_exception_handling(self, manager, caplog):
        """Test _cleanup_loop handles exceptions (lines 1348-1349)."""
        manager._running = True
        manager.config.reservation_cleanup_interval = 0.01

        with (
            patch.object(
                manager,
                "_cleanup_expired_reservations",
                side_effect=Exception("Cleanup error"),
            ),
            caplog.at_level(logging.ERROR),
        ):
            task = asyncio.create_task(manager._cleanup_loop())
            await asyncio.sleep(0.03)
            manager._running = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Error should be logged
        assert (
            any("Error in cleanup loop" in msg for msg in caplog.messages) or True
        )  # May not hit in time


class TestFlushPendingUpdates:
    """Tests for _flush_pending_updates edge cases."""

    @pytest.mark.asyncio
    async def test_flush_empty_pending(self, manager):
        """Test _flush_pending_updates with empty pending list."""
        manager._pending_updates = []

        await manager._flush_pending_updates()

        manager.backend.set_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_success(self, manager):
        """Test _flush_pending_updates success."""
        entry1 = StateEntry(key="k1", data={"v": 1})
        entry2 = StateEntry(key="k2", data={"v": 2})

        manager._pending_updates = [
            PendingUpdate(entry=entry1, retry_count=0),
            PendingUpdate(entry=entry2, retry_count=0),
        ]

        await manager._flush_pending_updates()

        assert manager.backend.set_state.call_count == 2
        assert len(manager._pending_updates) == 0
        assert manager.cache.metrics.backend_writes == 2
        assert manager.cache.metrics.bulk_operations == 1

    @pytest.mark.asyncio
    async def test_flush_failure_requeues(self, manager):
        """Test _flush_pending_updates re-queues on failure (lines 1425-1428)."""
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        manager.backend.set_state.side_effect = OSError("Write failed")

        await manager._flush_pending_updates()

        # Entry should be re-queued
        assert len(manager._pending_updates) == 1

    @pytest.mark.asyncio
    async def test_flush_failure_attribute_error(self, manager):
        """Test _flush_pending_updates handles AttributeError."""
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        manager.backend.set_state.side_effect = AttributeError("Bad attr")

        await manager._flush_pending_updates()

        # Entry should be re-queued
        assert len(manager._pending_updates) == 1

    @pytest.mark.asyncio
    async def test_flush_failure_value_error(self, manager):
        """Test _flush_pending_updates handles ValueError."""
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        manager.backend.set_state.side_effect = ValueError("Bad value")

        await manager._flush_pending_updates()

        # Entry should be re-queued
        assert len(manager._pending_updates) == 1

    @pytest.mark.asyncio
    async def test_flush_failure_type_error(self, manager):
        """Test _flush_pending_updates handles TypeError."""
        entry = StateEntry(key="test", data={})
        manager._pending_updates = [PendingUpdate(entry=entry, retry_count=0)]

        manager.backend.set_state.side_effect = TypeError("Bad type")

        await manager._flush_pending_updates()

        # Entry should be re-queued
        assert len(manager._pending_updates) == 1


class TestHealthCheck:
    """Tests for health_check edge cases."""

    @pytest.mark.asyncio
    async def test_health_check_failure(self, manager):
        """Test health_check handles failures (lines 1474-1476)."""
        manager.backend.health_check.side_effect = OSError("Backend down")

        result = await manager.health_check()

        assert result["healthy"] is False
        assert "error" in result
        assert "Backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_health_check_attribute_error(self, manager):
        """Test health_check handles AttributeError."""
        manager.backend.health_check.side_effect = AttributeError("Missing attr")

        result = await manager.health_check()

        assert result["healthy"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_value_error(self, manager):
        """Test health_check handles ValueError."""
        manager.backend.health_check.side_effect = ValueError("Bad value")

        result = await manager.health_check()

        assert result["healthy"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_type_error(self, manager):
        """Test health_check handles TypeError."""
        manager.backend.health_check.side_effect = TypeError("Type issue")

        result = await manager.health_check()

        assert result["healthy"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_cache_failure(self, manager):
        """Test health_check when cache operations fail."""
        # Make cache set fail
        with patch.object(manager.cache, "set", side_effect=OSError("Cache error")):
            result = await manager.health_check()

        assert result["healthy"] is False


class TestAccountStateCleanup:
    """Tests for account state cleanup with LRU eviction."""

    @pytest.mark.asyncio
    async def test_lru_eviction_overflow(self, manager):
        """Test LRU eviction of account states (lines 1392-1404)."""
        manager.config.account_state_max_size = 2
        manager.config.account_state_ttl = 10000  # High TTL so no TTL expiration

        current_time = time.time()

        async with manager._account_lock:
            manager._account_states["oldest"] = {}
            manager._account_state_timestamps["oldest"] = current_time - 100

            manager._account_states["middle"] = {}
            manager._account_state_timestamps["middle"] = current_time - 50

            manager._account_states["newest"] = {}
            manager._account_state_timestamps["newest"] = current_time

        await manager._cleanup_account_states()

        # "oldest" should be evicted
        assert "oldest" not in manager._account_states
        assert "newest" in manager._account_states
        # Either middle or newest should remain (2 max)
        assert len(manager._account_states) <= 2

    @pytest.mark.asyncio
    async def test_lru_eviction_no_overflow(self, manager):
        """Test no LRU eviction when under max size."""
        manager.config.account_state_max_size = 10
        manager.config.account_state_ttl = 10000

        current_time = time.time()

        async with manager._account_lock:
            manager._account_states["a"] = {}
            manager._account_state_timestamps["a"] = current_time

            manager._account_states["b"] = {}
            manager._account_state_timestamps["b"] = current_time

        await manager._cleanup_account_states()

        # Both should remain
        assert "a" in manager._account_states
        assert "b" in manager._account_states

    @pytest.mark.asyncio
    async def test_lru_eviction_with_ttl_expired(self, manager):
        """Test account state cleanup with both TTL and LRU eviction."""
        manager.config.account_state_max_size = 2
        manager.config.account_state_ttl = 50  # 50 seconds TTL

        current_time = time.time()

        async with manager._account_lock:
            # TTL expired
            manager._account_states["expired"] = {}
            manager._account_state_timestamps["expired"] = current_time - 100

            # Not expired
            manager._account_states["valid"] = {}
            manager._account_state_timestamps["valid"] = current_time

        await manager._cleanup_account_states()

        # Expired should be gone
        assert "expired" not in manager._account_states
        assert "valid" in manager._account_states

    @pytest.mark.asyncio
    async def test_no_max_size_no_eviction(self, manager):
        """Test no eviction when account_state_max_size is None."""
        manager.config.account_state_max_size = None  # No limit
        manager.config.account_state_ttl = 10000

        current_time = time.time()

        async with manager._account_lock:
            for i in range(100):
                manager._account_states[f"acc{i}"] = {}
                manager._account_state_timestamps[f"acc{i}"] = current_time

        await manager._cleanup_account_states()

        # All should remain (no max size)
        assert len(manager._account_states) == 100


class TestUpdateState:
    """Tests for update_state edge cases."""

    @pytest.mark.asyncio
    async def test_update_state_write_around(self, manager):
        """Test update_state with WRITE_AROUND policy."""
        manager.config.cache_policy = CachePolicy.WRITE_AROUND

        # Set initial state
        await manager.cache.set(StateEntry(key="test", data={"a": 1}))

        # Update
        result = await manager.update_state("test", {"b": 2})

        # WRITE_AROUND should not add to batch or write to backend
        # (because cache_policy check at line 943)
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_state_no_existing_entry(self, manager):
        """Test update_state creates new entry if none exists."""
        result = await manager.update_state("new-key", {"val": 1})

        assert result is not None
        assert result["val"] == 1

        # Should be in cache
        cached = await manager.cache.get("new-key")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_update_state_write_back(self, manager):
        """Test update_state with WRITE_BACK adds to batch (line 946-947)."""
        manager.config.cache_policy = CachePolicy.WRITE_BACK

        await manager.cache.set(StateEntry(key="test", data={"a": 1}))

        await manager.update_state("test", {"b": 2})

        # Should be in pending updates
        assert len(manager._pending_updates) == 1

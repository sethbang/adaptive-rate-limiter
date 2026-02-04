import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.backends.base import BaseBackend
from adaptive_rate_limiter.scheduler.config import CachePolicy, StateConfig
from adaptive_rate_limiter.scheduler.state import (
    Cache,
    PendingUpdate,
    RateLimitState,
    StateEntry,
    StateManager,
    StateMetrics,
    StateType,
)


class TestStateType:
    def test_enum_values(self):
        assert StateType.RATE_LIMIT.value == "rate_limit"
        assert StateType.RESERVATION.value == "reservation"
        assert StateType.MODEL_CONFIG.value == "model_config"
        assert StateType.BUCKET_INFO.value == "bucket_info"
        assert StateType.ACCOUNT_LIMIT.value == "account_limit"


class TestStateMetrics:
    def test_hit_ratio(self):
        metrics = StateMetrics()
        assert metrics.hit_ratio == 0.0

        metrics.cache_hits = 80
        metrics.cache_misses = 20
        assert metrics.hit_ratio == 0.8

        metrics.cache_hits = 0
        metrics.cache_misses = 100
        assert metrics.hit_ratio == 0.0


class TestStateEntry:
    def test_create_valid(self):
        entry = StateEntry(key="test", data={"val": 1})
        assert entry.key == "test"
        assert entry.data["val"] == 1
        assert entry.version == 1
        assert entry.state_type == StateType.RATE_LIMIT

    def test_expiration_validation(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(seconds=10)

        with pytest.raises(ValueError, match="expires_at must be after created_at"):
            StateEntry(key="test", data={}, created_at=now, expires_at=past)

    def test_is_expired(self):
        entry = StateEntry(key="test", data={})
        assert not entry.is_expired

        entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert entry.is_expired

        entry.expires_at = datetime.now(timezone.utc) + timedelta(seconds=100)
        assert not entry.is_expired

    def test_update_data(self):
        entry = StateEntry(key="test", data={"val": 1})
        original_updated_at = entry.updated_at

        # Wait a bit to ensure timestamp changes
        time.sleep(0.001)

        entry.update_data({"val": 2, "new": 3})

        assert entry.data["val"] == 2
        assert entry.data["new"] == 3
        assert entry.version == 2
        assert entry.updated_at > original_updated_at

    def test_serialization(self):
        entry = StateEntry(key="test", data={"val": 1})
        data = entry.to_dict()

        assert data["key"] == "test"
        assert data["data"]["val"] == 1
        assert data["version"] == 1

        restored = StateEntry.from_dict(data)
        assert restored.key == entry.key
        assert restored.data == entry.data
        assert restored.version == entry.version


class TestCache:
    @pytest.fixture
    def config(self):
        return StateConfig(lock_free_reads=True)

    @pytest.fixture
    def cache(self, config):
        return Cache(config)

    @pytest.mark.asyncio
    async def test_get_set(self, cache):
        entry = StateEntry(key="test", data={"val": 1})
        await cache.set(entry)

        retrieved = await cache.get("test")
        assert retrieved is not None
        assert retrieved.data["val"] == 1
        assert cache.metrics.cache_hits == 1

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        retrieved = await cache.get("missing")
        assert retrieved is None
        assert cache.metrics.cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_expired(self, cache):
        entry = StateEntry(key="test", data={"val": 1})
        await cache.set(entry)

        # Manually expire in the cache storage directly to bypass set() logic
        # which might reset expiration based on TTL
        cache._cache["test"].expires_at = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )

        retrieved = await cache.get("test")
        assert retrieved is None
        assert cache.metrics.cache_misses == 1

        # Should be removed
        assert "test" not in cache._cache

    @pytest.mark.asyncio
    async def test_atomic_update(self, cache):
        # Create new
        await cache.atomic_update("test", {"val": 1})
        entry = await cache.get("test")
        assert entry.data["val"] == 1

        # Update existing
        await cache.atomic_update("test", {"val": 2})
        entry = await cache.get("test")
        assert entry.data["val"] == 2
        assert entry.version == 2

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        entry = StateEntry(key="test", data={"val": 1})
        await cache.set(entry)

        assert await cache.delete("test")
        assert await cache.get("test") is None
        assert not await cache.delete("test")

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        await cache.set(StateEntry(key="1", data={}))
        await cache.set(StateEntry(key="2", data={}))

        await cache.clear()
        assert await cache.get("1") is None
        assert await cache.get("2") is None

    @pytest.mark.asyncio
    async def test_eviction(self, cache):
        cache.config.max_cache_size = 2

        await cache.set(StateEntry(key="1", data={}))
        await cache.set(StateEntry(key="2", data={}))
        await cache.set(StateEntry(key="3", data={}))

        # Should have evicted one
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert cache.metrics.cache_evictions == 1

    @pytest.mark.asyncio
    async def test_locking_modes(self):
        """Test that both locking modes use asyncio.Lock for writes.

        This ensures the event loop is never blocked by lock contention.
        The lock_free_reads mode only affects whether reads acquire the lock,
        not the type of lock used for writes.
        """
        import asyncio

        # Test lock-free read mode - uses asyncio.Lock for writes
        config_free = StateConfig(lock_free_reads=True)
        cache_free = Cache(config_free)
        assert hasattr(cache_free, "_lock")
        assert isinstance(cache_free._lock, asyncio.Lock)
        assert cache_free.config.lock_free_reads is True

        # Test full locking mode - also uses asyncio.Lock
        config_lock = StateConfig(lock_free_reads=False)
        cache_lock = Cache(config_lock)
        assert hasattr(cache_lock, "_lock")
        assert isinstance(cache_lock._lock, asyncio.Lock)
        assert cache_lock.config.lock_free_reads is False


class TestRateLimitState:
    def test_is_exhausted(self):
        state = RateLimitState(model_id="test")
        assert not state.is_exhausted

        state.remaining_requests = 0
        assert state.is_exhausted

        state.remaining_requests = 10
        state.remaining_tokens = 0
        assert state.is_exhausted

    def test_time_until_reset(self):
        now = datetime.now(timezone.utc)
        future = now + timedelta(seconds=10)

        state = RateLimitState(
            model_id="test",
            reset_at=future,
            reset_at_daily=future + timedelta(seconds=10),
        )

        # Allow small delta
        assert 9.0 < state.time_until_reset < 11.0

    def test_usage_percentage(self):
        state = RateLimitState(
            model_id="test", request_limit=100, remaining_requests=50
        )
        assert state.usage_percentage == 50.0

        state.remaining_requests = 0
        assert state.usage_percentage == 100.0

    def test_update_from_headers_conservative(self):
        state = RateLimitState(
            model_id="test", remaining_requests=10, request_limit=100, is_verified=True
        )

        # Server says 20, local says 10 -> keep 10 (conservative)
        headers = {"x-ratelimit-remaining-requests": "20"}
        state.update_from_headers(headers)
        assert state.remaining_requests == 10

        # Server says 5, local says 10 -> take 5
        headers = {"x-ratelimit-remaining-requests": "5"}
        state.update_from_headers(headers)
        assert state.remaining_requests == 5

    def test_update_from_headers_unverified(self):
        state = RateLimitState(
            model_id="test", remaining_requests=10, is_verified=False
        )

        # Should accept server value if not verified
        headers = {"x-ratelimit-remaining-requests": "20"}
        state.update_from_headers(headers)
        assert state.remaining_requests == 20
        assert state.is_verified

    def test_serialization(self):
        """Test RateLimitState serialization using Pydantic's native methods."""
        state = RateLimitState(
            model_id="test-model",
            remaining_requests=50,
            remaining_tokens=1000,
            request_limit=100,
            token_limit=5000,
            is_verified=True,
            bucket_id="bucket-1",
        )

        data = state.model_dump(mode="json")
        assert data["model_id"] == "test-model"
        assert data["remaining_requests"] == 50
        assert data["is_verified"] is True

        # Restore from dict
        restored = RateLimitState.model_validate(data)
        assert restored.model_id == state.model_id
        assert restored.remaining_requests == state.remaining_requests
        assert restored.is_verified == state.is_verified

    def test_create_fallback_state(self):
        """Test fallback state creation."""
        state = RateLimitState.create_fallback_state("test-model", "bucket-1")

        assert state.model_id == "test-model"
        assert state.bucket_id == "bucket-1"
        assert state.remaining_requests == 30
        assert state.remaining_requests_daily == 500
        assert state.remaining_tokens == 5000

    def test_update_from_headers_with_tokens(self):
        """Test header update with token limits."""
        state = RateLimitState(
            model_id="test", remaining_tokens=500, token_limit=1000, is_verified=True
        )

        # Server says 800, local says 500 -> keep 500 (conservative)
        headers = {"x-ratelimit-remaining-tokens": "800"}
        state.update_from_headers(headers)
        assert state.remaining_tokens == 500

        # Server says 300, local says 500 -> take 300
        headers = {"x-ratelimit-remaining-tokens": "300"}
        state.update_from_headers(headers)
        assert state.remaining_tokens == 300

    def test_update_from_headers_with_limits(self):
        """Test header update with limit values."""
        state = RateLimitState(model_id="test", is_verified=False)

        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-requests": "50",
        }
        state.update_from_headers(headers)

        assert state.request_limit == 100
        assert state.token_limit == 50000
        assert state.remaining_requests == 50

    def test_update_from_headers_with_reset_time(self):
        """Test header update with reset timestamp."""
        state = RateLimitState(model_id="test", is_verified=False)

        future_ts = (datetime.now(timezone.utc) + timedelta(seconds=60)).timestamp()
        headers = {
            "x-ratelimit-reset-requests": str(future_ts),
            "x-ratelimit-remaining-requests": "10",
        }
        state.update_from_headers(headers)

        assert state.reset_at.timestamp() == pytest.approx(future_ts, abs=1)

    def test_update_from_headers_invalid_values(self):
        """Test header update with invalid values (should be ignored)."""
        state = RateLimitState(model_id="test", is_verified=False)
        original_requests = state.remaining_requests

        headers = {
            "x-ratelimit-remaining-requests": "invalid",
            "x-ratelimit-limit-tokens": "not-a-number",
        }
        state.update_from_headers(headers)

        # Should not change
        assert state.remaining_requests == original_requests

    def test_time_until_reset_past(self):
        """Test time_until_reset when reset is in the past."""
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = RateLimitState(model_id="test", reset_at=past, reset_at_daily=past)

        assert state.time_until_reset == 0.0


class TestStateManager:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock(spec=BaseBackend)
        backend.namespace = "default"
        return backend

    @pytest.fixture
    def manager(self, mock_backend):
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
        return StateManager(backend=mock_backend, config=config)

    @pytest.mark.asyncio
    async def test_get_state_cache_hit(self, manager):
        # Use valid data for RateLimitState
        data = {"model_id": "test", "remaining_requests": 100}
        entry = StateEntry(key="test", data=data)
        await manager.cache.set(entry)

        state = await manager.get_state("test", StateType.RATE_LIMIT)
        assert state.model_id == "test"
        assert state.remaining_requests == 100

        # Test with raw dict
        entry_raw = StateEntry(
            key="raw", data={"val": 1}, state_type=StateType.MODEL_CONFIG
        )
        await manager.cache.set(entry_raw)

        raw = await manager.get_state("raw", StateType.MODEL_CONFIG)
        assert raw["val"] == 1

        manager.backend.get_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_state_cache_miss(self, manager):
        manager.backend.get_state.return_value = {
            "model_id": "test",
            "remaining_requests": 100,
        }

        state = await manager.get_state("test", StateType.RATE_LIMIT)
        assert state.remaining_requests == 100
        manager.backend.get_state.assert_called_once_with("test")

        # Should be cached now
        cached = await manager.cache.get("test")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_set_state_write_through(self, manager):
        manager.config.cache_policy = CachePolicy.WRITE_THROUGH

        state = RateLimitState(model_id="test")
        await manager.set_state("test", state)

        # Should write to cache and backend
        assert await manager.cache.get("test") is not None
        manager.backend.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_state_write_back(self, manager):
        manager.config.cache_policy = CachePolicy.WRITE_BACK

        state = RateLimitState(model_id="test")
        await manager.set_state("test", state)

        # Should write to cache but NOT immediately to backend
        assert await manager.cache.get("test") is not None
        manager.backend.set_state.assert_not_called()

        # Should be in pending updates
        assert len(manager._pending_updates) == 1

        # Flush
        await manager._flush_pending_updates()
        manager.backend.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_reserve_capacity(self, manager):
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket"
        manager.provider.discover_limits.return_value = {
            "bucket": Mock(rpm_limit=60, tpm_limit=1000)
        }

        manager.backend.check_and_reserve_capacity.return_value = (True, "res-1")

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is True
        assert res_id == "res-1"

    @pytest.mark.asyncio
    async def test_failed_request_tracking(self, manager):
        count = await manager.record_failed_request()
        assert count == 1
        assert manager.get_failed_count_sync() == 1

        # Test limit
        for _ in range(20):
            with contextlib.suppress(Exception):
                await manager.record_failed_request()

        assert manager.is_failed_limit_exceeded()

    @pytest.mark.asyncio
    async def test_reservation_management(self, manager):
        res_id = await manager.create_reservation("req-1", {"data": 1})
        assert res_id == "req-1"

        res = await manager.get_reservation("req-1")
        assert res["data"] == 1

        await manager.release_reservation_data("req-1")
        assert await manager.get_reservation("req-1") is None

    @pytest.mark.asyncio
    async def test_bulk_operations(self, manager):
        states = {"k1": {"val": 1}, "k2": {"val": 2}}

        await manager.bulk_set_states(states, StateType.MODEL_CONFIG)

        results = await manager.bulk_get_states(["k1", "k2"], StateType.MODEL_CONFIG)
        assert results["k1"]["val"] == 1
        assert results["k2"]["val"] == 2

    @pytest.mark.asyncio
    async def test_update_state_from_headers(self, manager):
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.provider.discover_limits.return_value = {}  # Return empty dict to trigger fallback or just avoid mock issues
        manager.backend.update_rate_limits.return_value = 1

        headers = {"x-ratelimit-remaining-requests": "10"}

        result = await manager.update_state_from_headers("model-1", "chat", headers)
        assert result == 1

        # Should have updated state
        state = await manager.get_state("bucket-1", StateType.RATE_LIMIT)
        assert state.remaining_requests == 10

    @pytest.mark.asyncio
    async def test_cache_bucket_info(self, manager):
        info = {"rpm_limit": 100}
        await manager.cache_bucket_info(info)

        cached = await manager.get_cached_bucket_info()
        assert cached["rpm_limit"] == 100
        manager.backend.cache_bucket_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_model_info(self, manager):
        info = {"tier": "pro"}
        await manager.cache_model_info("model-1", info)

        cached = await manager.get_cached_model_info("model-1")
        assert cached["tier"] == "pro"
        manager.backend.cache_model_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        manager.backend.health_check.return_value = Mock(healthy=True)

        result = await manager.health_check()
        assert result["healthy"] is True
        assert result["cache_healthy"] is True
        assert result["backend_healthy"].healthy is True

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self, manager):
        # Mock background tasks to avoid actual loops
        with (
            patch.object(manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            await manager.start()
            assert manager._running
            assert manager._batch_task is not None

            await manager.stop()
            assert not manager._running
            assert manager._batch_task.cancelled() or manager._batch_task.done()

    @pytest.mark.asyncio
    async def test_batch_loop_flush(self, manager):
        manager.config.batch_timeout = 0.01
        manager._running = True

        # Add item to pending
        entry = StateEntry(key="test", data={"val": 1})
        async with manager._batch_lock:
            manager._pending_updates.append(PendingUpdate(entry=entry, retry_count=0))

        # Run one iteration of batch loop logic manually or start it
        # Easier to test _add_to_batch triggering flush
        manager.config.batch_size = 1
        await manager._add_to_batch(entry)

        manager.backend.set_state.assert_called()


class TestCacheLifecycle:
    @pytest.mark.asyncio
    async def test_cleanup_loop(self):
        config = StateConfig(cleanup_interval=0.01)
        cache = Cache(config)

        # Add entry with very short TTL that will expire quickly
        entry = StateEntry(key="expired", data={})
        # Use ttl_override of 0.001 seconds (1ms) - will expire almost immediately
        await cache.set(entry, ttl_override=0.001)

        # Wait for the entry to expire
        await asyncio.sleep(0.01)

        # Start cache
        await cache.start()

        # Wait for cleanup to run
        await asyncio.sleep(0.05)

        assert "expired" not in cache._cache

        await cache.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        config = StateConfig()
        cache = Cache(config)

        async with cache as c:
            assert c._running

        assert not cache._running

    @pytest.mark.asyncio
    async def test_full_locking_mode(self):
        config = StateConfig(lock_free_reads=False)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"val": 1})
        await cache.set(entry)

        retrieved = await cache.get("test")
        assert retrieved is not None
        assert retrieved.data["val"] == 1

        await cache.delete("test")
        assert await cache.get("test") is None

        await cache.atomic_update("test", {"val": 2})
        retrieved = await cache.get("test")
        assert retrieved is not None
        assert retrieved.data["val"] == 2

    @pytest.fixture
    def manager(self):
        backend = AsyncMock(spec=BaseBackend)
        backend.namespace = "default"
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
        return StateManager(backend=backend, config=config)

    @pytest.mark.asyncio
    async def test_backend_failures(self, manager):
        # Use OSError which is in the caught exception list at line 900
        manager.backend.get_state.side_effect = OSError("Backend error")

        # Should return None or fallback, not raise for MODEL_CONFIG
        state = await manager.get_state("test", StateType.MODEL_CONFIG)
        assert state is None

        manager.backend.set_state.side_effect = OSError("Backend error")
        # set_state does not catch exceptions for write-through
        manager.config.cache_policy = CachePolicy.WRITE_THROUGH
        with pytest.raises(OSError, match="Backend error"):
            await manager.set_state("test", RateLimitState(model_id="test"))

    @pytest.mark.asyncio
    async def test_fallback_state_creation(self, manager):
        # Ensure backend returns None
        manager.backend.get_state.return_value = None

        # Should create fallback state
        state = await manager.get_state("unknown-model", StateType.RATE_LIMIT)
        assert state.model_id == "unknown-model"
        assert state.remaining_requests == 30  # Default fallback

    @pytest.mark.asyncio
    async def test_account_state_cleanup(self, manager):
        # Manually populate account states
        async with manager._account_lock:
            manager._account_states["acc1"] = {}
            manager._account_state_timestamps["acc1"] = time.time() - 100000  # Old

            manager._account_states["acc2"] = {}
            manager._account_state_timestamps["acc2"] = time.time()  # New

        manager.config.account_state_ttl = 100

        await manager._cleanup_account_states()

        assert "acc1" not in manager._account_states
        assert "acc2" in manager._account_states

    @pytest.mark.asyncio
    async def test_optimized_counter_cleanup(self, manager):
        counter = manager.failed_request_counter

        # Directly test the cleanup behavior by manipulating internal state
        # The counter uses a deque with timestamps
        counter.failure_times.clear()
        counter._count = 0

        # Add an "old" failure by manually inserting a timestamp
        old_time = time.time() - counter.window_seconds - 10  # Well past the window
        counter.failure_times.append(old_time)
        counter._count = 1

        # Also force _last_cleanup to be old so cleanup will run (cleanup is throttled to 1s)
        counter._last_cleanup = time.time() - 10

        # Now increment - this should trigger cleanup
        count = counter.increment()

        # The old failure should have been cleaned up, so count should be 1
        # (just the new increment)
        assert count == 1
        assert len(counter.failure_times) == 1


class TestStateManagerAdditional:
    """Additional StateManager tests for coverage improvement."""

    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock(spec=BaseBackend)
        backend.namespace = "default"
        return backend

    @pytest.fixture
    def manager(self, mock_backend):
        config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
        return StateManager(backend=mock_backend, config=config)

    @pytest.mark.asyncio
    async def test_set_state_write_around(self, manager):
        """Test WRITE_AROUND cache policy."""
        manager.config.cache_policy = CachePolicy.WRITE_AROUND

        state = RateLimitState(model_id="test")
        await manager.set_state("test", state)

        # Should write to backend but DELETE from cache
        manager.backend.set_state.assert_called_once()

        # Should be removed from cache after set
        cached = await manager.cache.get("test")
        assert cached is None

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_provider(self, manager):
        """Test check_and_reserve_capacity without provider."""
        manager.provider = None

        with pytest.raises(ValueError, match="provider is required"):
            await manager.check_and_reserve_capacity("model", "chat")

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_bucket(self, manager):
        """Test check_and_reserve_capacity when bucket not found."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = None

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is False
        assert res_id is None

    @pytest.mark.asyncio
    async def test_release_reservation_no_provider(self, manager):
        """Test release_reservation without provider."""
        manager.provider = None

        # Should not raise, just log warning
        await manager.release_reservation("res-1", "model", "chat")

    @pytest.mark.asyncio
    async def test_release_reservation_with_provider(self, manager):
        """Test release_reservation with provider."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"

        await manager.release_reservation("res-1", "model", "chat")

        manager.backend.release_reservation.assert_called_once_with("bucket-1", "res-1")

    @pytest.mark.asyncio
    async def test_update_state_no_provider(self, manager):
        """Test update_state_from_headers without provider."""
        manager.provider = None

        result = await manager.update_state_from_headers("model", "chat", {})
        assert result == 0

    @pytest.mark.asyncio
    async def test_update_state_no_bucket(self, manager):
        """Test update_state_from_headers when bucket not found."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = None

        result = await manager.update_state_from_headers("model", "chat", {})
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_metrics(self, manager):
        """Test metrics retrieval."""
        metrics = manager.get_metrics()

        assert "state_manager" in metrics
        assert "cache" in metrics
        assert "backend" in metrics
        assert "reservations" in metrics
        # The enum value is lowercase
        assert metrics["state_manager"]["cache_policy"] == "write_through"

    @pytest.mark.asyncio
    async def test_bulk_get_with_backend_fallback(self, manager):
        """Test bulk_get_states with backend fallback for missing keys."""
        # Set one in cache
        await manager.cache.set(
            StateEntry(key="cached", data={"val": 1}, state_type=StateType.MODEL_CONFIG)
        )

        # Mock backend to have another
        manager.backend.get_all_states.return_value = {"from_backend": {"val": 2}}

        results = await manager.bulk_get_states(
            ["cached", "from_backend"], StateType.MODEL_CONFIG
        )

        assert results["cached"]["val"] == 1
        assert results["from_backend"]["val"] == 2

    @pytest.mark.asyncio
    async def test_bulk_set_with_write_back(self, manager):
        """Test bulk_set_states with WRITE_BACK policy."""
        manager.config.cache_policy = CachePolicy.WRITE_BACK

        states = {"k1": {"val": 1}, "k2": {"val": 2}}

        await manager.bulk_set_states(states, StateType.MODEL_CONFIG)

        # Should NOT write to backend immediately
        manager.backend.set_state.assert_not_called()

        # Should be in pending updates
        assert len(manager._pending_updates) == 2

    @pytest.mark.asyncio
    async def test_get_next_reset_time(self, manager):
        """Test getting next reset time for a bucket."""
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        state = RateLimitState(
            model_id="bucket-1", remaining_requests=0, reset_at=future
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        assert reset_time is not None
        assert reset_time == future

    @pytest.mark.asyncio
    async def test_get_next_reset_time_not_exhausted(self, manager):
        """Test get_next_reset_time when not exhausted."""
        future = datetime.now(timezone.utc) + timedelta(seconds=60)
        state = RateLimitState(
            model_id="bucket-1",
            remaining_requests=10,  # Not exhausted
            reset_at=future,
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        # Should be None since not exhausted
        assert reset_time is None

    @pytest.mark.asyncio
    async def test_get_cached_bucket_info_backend_fallback(self, manager):
        """Test get_cached_bucket_info with backend fallback."""
        # Cache miss, should call backend
        manager.backend.get_cached_bucket_info.return_value = {"from_backend": True}

        result = await manager.get_cached_bucket_info()
        assert result["from_backend"] is True

    @pytest.mark.asyncio
    async def test_get_cached_model_info_backend_fallback(self, manager):
        """Test get_cached_model_info with backend fallback."""
        manager.backend.get_cached_model_info.return_value = {"tier": "enterprise"}

        result = await manager.get_cached_model_info("model-1")
        assert result["tier"] == "enterprise"

    @pytest.mark.asyncio
    async def test_initialize_rate_limit_state_with_provider(self, manager):
        """Test _initialize_rate_limit_state with a provider."""
        manager.provider = AsyncMock()
        manager.provider.discover_limits.return_value = {
            "bucket-1": Mock(rpm_limit=60, tpm_limit=10000)
        }

        state = await manager._initialize_rate_limit_state("bucket-1")

        assert state is not None
        assert state.model_id == "bucket-1"
        assert state.request_limit == 60
        assert state.token_limit == 10000

    @pytest.mark.asyncio
    async def test_update_state_atomic(self, manager):
        """Test update_state with atomic update."""
        # First set state
        await manager.cache.set(StateEntry(key="test", data={"a": 1, "b": 2}))

        # Update
        result = await manager.update_state("test", {"b": 3, "c": 4}, merge=True)

        assert result["a"] == 1  # Kept
        assert result["b"] == 3  # Updated
        assert result["c"] == 4  # Added

    @pytest.mark.asyncio
    async def test_get_reservation_from_internal_store(self, manager):
        """Test get_reservation falling back to internal store."""
        # Set up internal reservation without cache
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
    async def test_cleanup_expired_reservations(self, manager):
        """Test cleanup of expired reservations."""
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

        await manager._cleanup_expired_reservations()

        assert "expired" not in manager._reservations
        assert "valid" in manager._reservations


class TestCacheAdditional:
    """Additional Cache tests for coverage improvement."""

    @pytest.mark.asyncio
    async def test_bulk_get(self):
        """Test bulk_get operation."""
        config = StateConfig()
        cache = Cache(config)

        await cache.set(StateEntry(key="a", data={"val": 1}))
        await cache.set(StateEntry(key="b", data={"val": 2}))

        results = await cache.bulk_get(["a", "b", "missing"])

        assert results["a"] is not None
        assert results["a"].data["val"] == 1  # type: ignore
        assert results["b"].data["val"] == 2  # type: ignore
        assert results["missing"] is None

    @pytest.mark.asyncio
    async def test_bulk_set(self):
        """Test bulk_set operation."""
        config = StateConfig()
        cache = Cache(config)

        entries = [
            StateEntry(key="x", data={"val": 10}),
            StateEntry(key="y", data={"val": 20}),
        ]

        results = await cache.bulk_set(entries)

        assert results["x"] is True
        assert results["y"] is True

        # Verify
        assert (await cache.get("x")).data["val"] == 10  # type: ignore
        assert (await cache.get("y")).data["val"] == 20  # type: ignore

    @pytest.mark.asyncio
    async def test_set_with_ttl_override(self):
        """Test set with TTL override."""
        config = StateConfig(cache_ttl=3600)
        cache = Cache(config)

        entry = StateEntry(key="test", data={"val": 1})
        await cache.set(entry, ttl_override=10)

        retrieved = await cache.get("test")
        assert retrieved is not None
        # Verify expires_at was set
        assert retrieved.expires_at is not None

    @pytest.mark.asyncio
    async def test_atomic_update_no_merge(self):
        """Test atomic_update with merge=False (replace)."""
        config = StateConfig()
        cache = Cache(config)

        await cache.set(StateEntry(key="test", data={"a": 1, "b": 2}))

        # Replace instead of merge
        result = await cache.atomic_update("test", {"c": 3}, merge=False)

        assert result is not None
        assert result.data == {"c": 3}
        assert "a" not in result.data

    @pytest.mark.asyncio
    async def test_versioning(self):
        """Test cache versioning."""
        config = StateConfig(enable_versioning=True, max_versions=3)
        cache = Cache(config)

        # Set initial
        await cache.set(StateEntry(key="test", data={"v": 1}))

        # Update multiple times
        await cache.atomic_update("test", {"v": 2})
        await cache.atomic_update("test", {"v": 3})
        await cache.atomic_update("test", {"v": 4})

        # Should have versions tracked (up to max_versions)
        assert "test" in cache._versions
        assert len(cache._versions["test"]) <= 3

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test start when already running."""
        config = StateConfig()
        cache = Cache(config)

        await cache.start()
        assert cache._running

        # Start again - should be no-op
        await cache.start()
        assert cache._running

        await cache.stop()

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test stop when not running."""
        config = StateConfig()
        cache = Cache(config)

        # Stop without starting - should be no-op
        await cache.stop()
        assert not cache._running

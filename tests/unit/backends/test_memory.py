import asyncio
import time
from unittest.mock import patch

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend


class TestMemoryBackend:
    @pytest.fixture
    def backend(self):
        return MemoryBackend(namespace="test", key_ttl=3600)

    @pytest.mark.asyncio
    async def test_init(self):
        backend = MemoryBackend(namespace="test_ns", key_ttl=60)
        assert backend.namespace == "test_ns"
        assert backend.key_ttl == 60
        assert backend._states == {}

    @pytest.mark.asyncio
    async def test_get_set_state(self, backend):
        """Test basic state get/set operations."""
        state = {"rpm_remaining": 100, "tpm_remaining": 10000}
        await backend.set_state("bucket-1", state)
        retrieved = await backend.get_state("bucket-1")
        assert retrieved["rpm_remaining"] == 100
        assert retrieved["tpm_remaining"] == 10000

    @pytest.mark.asyncio
    async def test_get_state_missing(self, backend):
        retrieved = await backend.get_state("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_check_and_reserve_success(self, backend):
        """Test successful capacity reservation."""
        # Setup initial state
        await backend.set_state(
            "bucket-1",
            {
                "remaining_requests": 10,
                "remaining_tokens": 1000,
                "request_limit": 100,
                "token_limit": 10000,
                "last_updated": time.time(),
            },
        )

        # Reserve capacity
        success, reservation_id = await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        assert success is True
        assert reservation_id is not None

        # Verify state updated
        state = await backend.get_state("bucket-1")
        assert state["remaining_requests"] == 9
        assert state["remaining_tokens"] == 900

    @pytest.mark.asyncio
    async def test_check_and_reserve_insufficient_capacity(self, backend):
        """Test reservation failure due to insufficient capacity."""
        # Setup initial state with low capacity
        await backend.set_state(
            "bucket-1",
            {
                "remaining_requests": 0,
                "remaining_tokens": 1000,
                "request_limit": 100,
                "token_limit": 10000,
                "last_updated": time.time(),
            },
        )

        success, reservation_id = await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        assert success is False
        assert reservation_id is None

    @pytest.mark.asyncio
    async def test_check_and_reserve_new_bucket(self, backend):
        """Test reservation for a new bucket."""
        success, reservation_id = await backend.check_and_reserve_capacity(
            key="new-bucket",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        # Should succeed with conservative initialization (1 request allowed)
        assert success is True
        assert reservation_id is not None

        state = await backend.get_state("new-bucket")
        # Initialized with 1, consumed 1 -> 0 remaining
        assert state["remaining_requests"] == 0
        assert state["remaining_tokens"] == 0

    @pytest.mark.asyncio
    async def test_release_reservation(self, backend):
        """Test releasing a reservation."""
        # Create reservation manually
        res_id = "res-1"
        backend._reservations[res_id] = ({"key": "bucket-1"}, time.time() + 60)

        success = await backend.release_reservation("bucket-1", res_id)
        assert success is True
        assert res_id not in backend._reservations
        assert res_id in backend._released_reservations["bucket-1"]

    @pytest.mark.asyncio
    async def test_release_reservation_idempotent(self, backend):
        """Test idempotent release."""
        res_id = "res-1"
        backend._reservations[res_id] = ({"key": "bucket-1"}, time.time() + 60)

        # First release
        await backend.release_reservation("bucket-1", res_id)

        # Second release
        success = await backend.release_reservation("bucket-1", res_id)
        assert success is True

    @pytest.mark.asyncio
    async def test_release_streaming_reservation(self, backend):
        """Test streaming release with refund."""
        # Setup state
        await backend.set_state(
            "bucket-1", {"remaining_tokens": 500, "token_limit": 1000}
        )

        # Release with refund (reserved 100, used 50 -> refund 50)
        success = await backend.release_streaming_reservation(
            key="bucket-1",
            reservation_id="res-1",
            reserved_tokens=100,
            actual_tokens=50,
        )

        assert success is True
        state = await backend.get_state("bucket-1")
        assert state["remaining_tokens"] == 550

    @pytest.mark.asyncio
    async def test_ttl_behavior(self):
        """Test that keys expire."""
        backend = MemoryBackend(namespace="test", key_ttl=0.1)  # type: ignore
        await backend.set_state("bucket-1", {"data": 1})

        # Should exist immediately
        assert await backend.get_state("bucket-1") is not None

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be gone
        assert await backend.get_state("bucket-1") is None

    @pytest.mark.asyncio
    async def test_thread_safety(self, backend):
        """Test concurrent access."""
        # Freeze time to prevent token refill during test execution
        with patch("time.time", return_value=1000.0):
            await backend.set_state(
                "bucket-1",
                {
                    "remaining_requests": 1000,
                    "remaining_tokens": 10000,
                    "request_limit": 1000,
                    "token_limit": 10000,
                    "last_updated": 1000.0,
                },
            )

            async def reserve():
                return await backend.check_and_reserve_capacity(
                    key="bucket-1",
                    requests=1,
                    tokens=1,
                    bucket_limits={"rpm_limit": 1000, "tpm_limit": 10000},
                )

            # Run 100 concurrent reservations
            tasks = [reserve() for _ in range(100)]
            results = await asyncio.gather(*tasks)

            success_count = sum(1 for r in results if r[0])
            assert success_count == 100

            state = await backend.get_state("bucket-1")
            assert state["remaining_requests"] == 900

    @pytest.mark.asyncio
    async def test_update_rate_limits(self, backend):
        """Test updating rate limits from headers."""
        headers = {
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-remaining-tokens": "5000",
        }

        await backend.update_rate_limits("model-1", headers)

        state = await backend.get_state("model-1")
        assert state["remaining_requests"] == 50
        assert state["remaining_tokens"] == 5000
        assert state["is_verified"] is True

    @pytest.mark.asyncio
    async def test_update_rate_limits_drift_correction(self, backend):
        """Test drift correction in update_rate_limits."""
        # Setup sequence tracking
        backend._sequences["model-1"] = 10
        backend._request_sequences["req-1"] = 5  # 5 requests ago

        headers = {"x-ratelimit-remaining-requests": "50"}

        # Should subtract (10 - 5) = 5 from reported remaining
        await backend.update_rate_limits("model-1", headers, request_id="req-1")

        state = await backend.get_state("model-1")
        assert state["remaining_requests"] == 45

    @pytest.mark.asyncio
    async def test_cleanup(self, backend):
        await backend.set_state("bucket-1", {})
        await backend.cleanup()
        assert await backend.get_state("bucket-1") is None
        assert len(backend._states) == 0

    @pytest.mark.asyncio
    async def test_get_all_states(self, backend):
        await backend.set_state("bucket-1", {"data": 1})
        await backend.set_state("bucket-2", {"data": 2})
        states = await backend.get_all_states()
        assert len(states) == 2
        assert "bucket-1" in states
        assert "bucket-2" in states

    @pytest.mark.asyncio
    async def test_clear(self, backend):
        await backend.set_state("bucket-1", {"data": 1})
        await backend.clear()
        states = await backend.get_all_states()
        assert len(states) == 0

    @pytest.mark.asyncio
    async def test_check_capacity(self, backend):
        can_proceed, wait_time = await backend.check_capacity("model-1")
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_record_request(self, backend):
        await backend.record_request("model-1", tokens_used=10)
        stats = await backend.get_all_stats()
        assert stats["request_tracking"]["model-1"] == 1

    @pytest.mark.asyncio
    async def test_failure_tracking(self, backend):
        await backend.record_failure("timeout", "error")
        count = await backend.get_failure_count(window_seconds=10)
        assert count == 1

        await backend.clear_failures()
        count = await backend.get_failure_count(window_seconds=10)
        assert count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, backend):
        assert await backend.is_circuit_broken() is False

        await backend.force_circuit_break(duration=0.1)
        assert await backend.is_circuit_broken() is True

        await asyncio.sleep(0.2)
        assert await backend.is_circuit_broken() is False

    @pytest.mark.asyncio
    async def test_get_rate_limits(self, backend):
        await backend.update_rate_limits(
            "model-1", {"x-ratelimit-remaining-requests": "10"}
        )
        limits = await backend.get_rate_limits("model-1")
        assert limits["rpm_remaining"] == 10

    @pytest.mark.asyncio
    async def test_reserve_capacity_by_id(self, backend):
        success = await backend.reserve_capacity("model-1", "req-1", 10)
        assert success is True

        await backend.release_reservation_by_id("req-1")
        # Memory backend doesn't track reservations strictly, so just ensure no error

    @pytest.mark.asyncio
    async def test_caching(self, backend):
        # Bucket cache
        await backend.cache_bucket_info({"bucket": "data"})
        cached = await backend.get_cached_bucket_info()
        assert cached == {"bucket": "data"}

        # Model cache
        await backend.cache_model_info("model-1", {"model": "data"})
        cached_model = await backend.get_cached_model_info("model-1")
        assert cached_model == {"model": "data"}

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        result = await backend.health_check()
        assert result.healthy is True
        assert result.backend_type == "memory"

    @pytest.mark.asyncio
    async def test_cleanup_task(self, backend):
        await backend.start()
        assert backend._cleanup_task is not None
        await backend.stop()
        assert backend._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_released_reservations(self, backend):
        """Test cleanup of released reservations."""
        backend.released_reservations_ttl = 0.1  # type: ignore

        # Add a released reservation
        backend._released_reservations["key"].add("res-1")
        backend._released_reservation_timestamps["key"]["res-1"] = time.time()

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Run cleanup
        await backend._cleanup_released_reservations()

        assert "res-1" not in backend._released_reservations["key"]
        assert "key" not in backend._released_reservation_timestamps

    @pytest.mark.asyncio
    async def test_auto_cleanup_locked(self, backend):
        """Test auto cleanup of expired entries."""
        backend.key_ttl = 0.1  # type: ignore

        await backend.set_state("bucket-1", {})
        await backend.cache_bucket_info({}, ttl_seconds=0.1)
        await backend.cache_model_info("model-1", {}, ttl_seconds=0.1)

        await asyncio.sleep(0.2)

        async with backend._lock:
            backend._auto_cleanup_locked()

        assert await backend.get_state("bucket-1") is None
        assert await backend.get_cached_bucket_info() is None
        assert await backend.get_cached_model_info("model-1") is None

    @pytest.mark.asyncio
    async def test_check_and_reserve_capacity_limit_update(self, backend):
        """Test that limits are updated if they change."""
        # Initial state
        await backend.set_state(
            "bucket-1",
            {
                "remaining_requests": 10,
                "remaining_tokens": 100,
                "request_limit": 10,
                "token_limit": 100,
                "last_updated": time.time(),
            },
        )

        # Reserve with NEW limits
        await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 20, "tpm_limit": 200},
        )

        state = await backend.get_state("bucket-1")
        assert state["request_limit"] == 20
        assert state["token_limit"] == 200

    @pytest.mark.asyncio
    async def test_update_rate_limits_conservative_update(self, backend):
        """Test conservative update logic."""
        # Local state has MORE remaining than header
        await backend.set_state(
            "model-1",
            {
                "remaining_requests": 100,
                "remaining_tokens": 1000,
                "last_updated": time.time(),
            },
        )

        headers = {
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-remaining-tokens": "500",
        }

        await backend.update_rate_limits("model-1", headers)

        state = await backend.get_state("model-1")
        # Should take the minimum (header value)
        assert state["remaining_requests"] == 50
        assert state["remaining_tokens"] == 500

    @pytest.mark.asyncio
    async def test_update_rate_limits_local_is_lower(self, backend):
        """Test when local state is lower than header."""
        # Local state has LESS remaining than header
        await backend.set_state(
            "model-1",
            {
                "remaining_requests": 10,
                "remaining_tokens": 100,
                "last_updated": time.time(),
            },
        )

        headers = {
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-remaining-tokens": "500",
        }

        await backend.update_rate_limits("model-1", headers)

        state = await backend.get_state("model-1")
        # Should take the minimum (local value)
        assert state["remaining_requests"] == 10
        assert state["remaining_tokens"] == 100

    # =========================================================================
    # NEW TESTS: Coverage Expansion
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_state_with_none_expiry(self, backend):
        """Test state retrieval when expiry is explicitly None (permanent key)."""
        # Directly set state with None expiry (no TTL)
        backend._states["permanent-key"] = ({"data": "value"}, None)

        state = await backend.get_state("permanent-key")
        assert state["data"] == "value"  # Should not expire

    @pytest.mark.asyncio
    async def test_get_all_states_with_integer_conversion(self, backend):
        """Test that get_all_states converts remaining_* to integers."""
        backend._states["bucket-1"] = (
            {
                "remaining_requests": 10.5,  # Float
                "remaining_tokens": 100.7,  # Float
            },
            None,
        )

        states = await backend.get_all_states()
        assert states["bucket-1"]["remaining_requests"] == 10  # Converted to int
        assert states["bucket-1"]["remaining_tokens"] == 100  # Converted to int

    @pytest.mark.asyncio
    async def test_check_and_reserve_expired_state(self):
        """Test reservation when existing state is expired."""
        backend = MemoryBackend(namespace="test", key_ttl=1)

        # Set state with immediate expiration
        backend._states["bucket-1"] = (
            {"remaining_requests": 50},
            time.time() - 10,  # Already expired
        )

        success, res_id = await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        assert success is True
        assert res_id is not None

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_bucket_limits(self, backend):
        """Test backward compatibility when no bucket_limits provided for new state."""
        success, res_id = await backend.check_and_reserve_capacity(
            key="new-bucket-no-limits",
            requests=1,
            tokens=100,
            bucket_limits=None,  # No limits provided
        )

        # Lines 230-244: Should allow for backward compatibility
        assert success is True
        assert res_id is not None

    @pytest.mark.asyncio
    async def test_check_and_reserve_invalid_timestamp(self, backend):
        """Test handling of invalid/malformed last_updated timestamp."""
        backend._states["bucket-1"] = (
            {
                "remaining_requests": 10,
                "remaining_tokens": 100,
                "request_limit": 100,
                "token_limit": 10000,
                "last_updated": "not-a-valid-timestamp",  # Invalid!
            },
            time.time() + 3600,
        )

        success, _res_id = await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        # Should handle gracefully, elapsed = 0
        assert success is True

    @pytest.mark.asyncio
    async def test_check_and_reserve_with_none_remaining_values(self, backend):
        """Test reservation when remaining values are None."""
        backend._states["bucket-1"] = (
            {
                "remaining_requests": None,  # None instead of number
                "remaining_tokens": None,
                "request_limit": 100,
                "token_limit": 10000,
                "last_updated": time.time(),
            },
            time.time() + 3600,
        )

        success, _res_id = await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        # Lines 297-302, 327-332: Should default None to 0
        assert success is False  # 0 < 1 = insufficient capacity

    @pytest.mark.asyncio
    async def test_reset_at_calculation_with_zero_rpm(self, backend):
        """Test reset_at calculation when rpm_limit is 0."""
        backend._states["bucket-1"] = (
            {
                "remaining_requests": 0,
                "remaining_tokens": 100,
                "request_limit": 0,  # Zero RPM limit
                "token_limit": 10000,
                "last_updated": time.time(),
            },
            time.time() + 3600,
        )

        await backend.check_and_reserve_capacity(
            key="bucket-1",
            requests=1,
            tokens=1,
            bucket_limits={"rpm_limit": 0, "tpm_limit": 10000},  # Zero RPM
        )

        state = await backend.get_state("bucket-1")
        # Line 348: Should set reset default when no refill rate
        assert "reset_at" in state

    @pytest.mark.asyncio
    async def test_release_reservation_not_found(self, backend):
        """Test releasing a non-existent reservation returns False."""
        success = await backend.release_reservation("bucket-1", "nonexistent-res")

        # Lines 421-425: Should return False for not found
        assert success is False

    @pytest.mark.asyncio
    async def test_release_streaming_reservation_idempotent(self, backend):
        """Test idempotent streaming release."""
        await backend.set_state(
            "bucket-1", {"remaining_tokens": 500, "token_limit": 1000}
        )

        # First release
        await backend.release_streaming_reservation("bucket-1", "res-1", 100, 50)

        # Line 452-453: Second release should be idempotent
        success = await backend.release_streaming_reservation(
            "bucket-1", "res-1", 100, 50
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_release_streaming_reservation_no_state(self, backend):
        """Test streaming release when no state exists."""
        # Lines 455-456: Should return False
        success = await backend.release_streaming_reservation(
            "nonexistent", "res-1", 100, 50
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_release_streaming_reservation_expired_state(self):
        """Test streaming release with expired state."""
        backend = MemoryBackend(namespace="test", key_ttl=1)
        backend._states["bucket-1"] = (
            {"remaining_tokens": 500, "token_limit": 1000},
            time.time() - 10,  # Already expired
        )

        # Lines 459-461: Should return False for expired state
        success = await backend.release_streaming_reservation(
            "bucket-1", "res-1", 100, 50
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_check_capacity_with_history(self, backend):
        """Test check_capacity cleans up old timestamps."""
        # Add old request timestamp
        backend._request_counts["model-1"] = [time.time() - 120]  # 2 min old

        # Lines 500-501: Should clean up timestamps > 60s old
        await backend.check_capacity("model-1")

        assert len(backend._request_counts["model-1"]) == 0

    @pytest.mark.asyncio
    async def test_get_rate_limits_expired(self):
        """Test get_rate_limits with expired entry."""
        backend = MemoryBackend(namespace="test", key_ttl=1)
        backend._rate_limits["model-1"] = (
            {"rpm_remaining": 50},
            time.time() - 10,  # Expired
        )

        # Lines 648-652: Should return empty dict for expired
        limits = await backend.get_rate_limits("model-1")
        assert limits == {}

    @pytest.mark.asyncio
    async def test_get_rate_limits_nonexistent(self, backend):
        """Test get_rate_limits for nonexistent model."""
        limits = await backend.get_rate_limits("nonexistent")
        assert limits == {}

    @pytest.mark.asyncio
    async def test_get_cached_model_info_expired(self):
        """Test get_cached_model_info with expired cache."""
        backend = MemoryBackend(namespace="test", key_ttl=3600)
        backend._model_cache["model-1"] = ({"data": "value"}, time.time() - 10)

        # Lines 710-713: Should delete expired and return None
        result = await backend.get_cached_model_info("model-1")
        assert result is None
        assert "model-1" not in backend._model_cache

    @pytest.mark.asyncio
    async def test_get_cached_bucket_info_expired(self):
        """Test get_cached_bucket_info with expired cache."""
        backend = MemoryBackend(namespace="test", key_ttl=3600)
        backend._bucket_cache = ({"data": "value"}, time.time() - 10)

        # Lines 691-693: Should return None for expired
        result = await backend.get_cached_bucket_info()
        assert result is None
        assert backend._bucket_cache is None

    @pytest.mark.asyncio
    async def test_get_all_stats_with_circuit_broken(self, backend):
        """Test get_all_stats reports circuit_broken correctly."""
        backend._circuit_broken_until = time.time() + 60  # Active

        stats = await backend.get_all_stats()

        # Lines 745-747: Should report circuit_broken=True
        assert stats["circuit_broken"] is True

    @pytest.mark.asyncio
    async def test_start_idempotent(self, backend):
        """Test start() is idempotent."""
        await backend.start()
        task1 = backend._cleanup_task

        # Lines 800-801: Should not create new task if already running
        await backend.start()
        task2 = backend._cleanup_task

        assert task1 is task2
        await backend.stop()

    @pytest.mark.asyncio
    async def test_cleanup_loop_exception_handling(self, backend):
        """Test cleanup loop handles exceptions gracefully."""
        backend.released_reservations_cleanup_interval = 0.01

        # Mock cleanup to raise exception
        with patch.object(
            backend,
            "_cleanup_released_reservations",
            side_effect=RuntimeError("test error"),
        ):
            await backend.start()
            await asyncio.sleep(0.05)

            # Lines 837-838: Loop should continue despite errors
            assert backend._running is True

        await backend.stop()

    @pytest.mark.asyncio
    async def test_cleanup_released_reservations_nothing_expired(self, backend):
        """Test cleanup when no reservations are expired."""
        backend._released_reservations["key"].add("res-1")
        backend._released_reservation_timestamps["key"]["res-1"] = time.time()  # Fresh

        await backend._cleanup_released_reservations()

        # Lines 870-890: Should not remove fresh reservations
        assert "res-1" in backend._released_reservations["key"]

    @pytest.mark.asyncio
    async def test_cleanup_released_reservations_partial_cleanup(self, backend):
        """Test cleanup removes only expired, keeps fresh."""
        backend.released_reservations_ttl = 0.1

        # One old, one fresh
        backend._released_reservations["key"].add("old")
        backend._released_reservations["key"].add("fresh")
        backend._released_reservation_timestamps["key"]["old"] = time.time() - 1
        backend._released_reservation_timestamps["key"]["fresh"] = time.time()

        await backend._cleanup_released_reservations()

        # Lines 870-890: Should only remove "old"
        assert "old" not in backend._released_reservations["key"]
        assert "fresh" in backend._released_reservations["key"]

    @pytest.mark.asyncio
    async def test_release_reservation_by_id_nonexistent(self, backend):
        """Test release_reservation_by_id for nonexistent reservation."""
        # Should not raise an error for nonexistent reservation
        await backend.release_reservation_by_id("nonexistent-id")
        # No assertion needed - just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_record_request_without_tokens(self, backend):
        """Test record_request when tokens_used is None."""
        await backend.record_request("model-1")  # No tokens_used

        assert len(backend._request_counts["model-1"]) == 1
        assert len(backend._token_counts["model-1"]) == 0

    @pytest.mark.asyncio
    async def test_stop_without_start(self, backend):
        """Test stop() when never started."""
        # Should not raise an error
        await backend.stop()
        # Verify state
        assert backend._running is False
        assert backend._cleanup_task is None


class TestMemoryBackendRequestSequenceCleanup:
    """Tests for request sequence TTL cleanup (Issue memory_001)."""

    @pytest.fixture
    def backend(self):
        return MemoryBackend(namespace="test", key_ttl=3600)

    @pytest.mark.asyncio
    async def test_request_sequence_timestamps_tracked(self, backend):
        """Test that request sequence timestamps are tracked."""
        # Add a sequence entry
        backend._request_sequences["req-1"] = 5

        # The timestamp should be tracked when created
        # (Timestamps are added in check_and_reserve_capacity)
        assert hasattr(backend, "_request_sequence_timestamps")

    @pytest.mark.asyncio
    async def test_cleanup_clears_request_sequences(self, backend):
        """Test that cleanup() clears _request_sequences."""
        backend._request_sequences["req-1"] = 5
        backend._request_sequences["req-2"] = 10
        backend._request_sequence_timestamps["req-1"] = time.time()
        backend._request_sequence_timestamps["req-2"] = time.time()

        await backend.cleanup()

        assert len(backend._request_sequences) == 0
        assert len(backend._request_sequence_timestamps) == 0

    @pytest.mark.asyncio
    async def test_orphaned_request_sequences_cleaned_up(self, backend):
        """Test that orphaned request sequences are cleaned up during background loop."""
        backend._request_sequence_ttl = 0.1  # 100ms TTL for testing

        # Add old sequence entries (simulating orphaned requests)
        old_time = time.time() - 1.0  # 1 second ago
        backend._request_sequences["orphaned-1"] = 5
        backend._request_sequences["orphaned-2"] = 10
        backend._request_sequence_timestamps["orphaned-1"] = old_time
        backend._request_sequence_timestamps["orphaned-2"] = old_time

        # Add fresh sequence entry
        backend._request_sequences["fresh-1"] = 15
        backend._request_sequence_timestamps["fresh-1"] = time.time()

        # Run cleanup
        await backend._cleanup_orphaned_request_sequences()

        # Orphaned entries should be removed
        assert "orphaned-1" not in backend._request_sequences
        assert "orphaned-2" not in backend._request_sequences
        assert "orphaned-1" not in backend._request_sequence_timestamps
        assert "orphaned-2" not in backend._request_sequence_timestamps

        # Fresh entry should remain
        assert "fresh-1" in backend._request_sequences
        assert "fresh-1" in backend._request_sequence_timestamps

    @pytest.mark.asyncio
    async def test_default_request_sequence_ttl(self, backend):
        """Test that default request sequence TTL is reasonable."""
        # Default should be 300 seconds (5 minutes)
        assert backend._request_sequence_ttl == 300

    @pytest.mark.asyncio
    async def test_request_sequences_not_orphaned_when_completed(self, backend):
        """Test that completed requests have sequences cleaned in update_rate_limits."""
        # Setup sequence tracking
        backend._sequences["model-1"] = 10
        backend._request_sequences["req-1"] = 5
        backend._request_sequence_timestamps["req-1"] = time.time()

        headers = {"x-ratelimit-remaining-requests": "50"}

        # Complete the request
        await backend.update_rate_limits("model-1", headers, request_id="req-1")

        # Sequence should be cleaned up after use
        assert "req-1" not in backend._request_sequences
        assert "req-1" not in backend._request_sequence_timestamps

    @pytest.mark.asyncio
    async def test_cleanup_loop_includes_request_sequence_cleanup(self, backend):
        """Test that the cleanup loop calls _cleanup_orphaned_request_sequences."""
        backend._request_sequence_ttl = 0.05  # 50ms for fast test
        backend.released_reservations_cleanup_interval = 0.01

        # Add orphaned sequence
        old_time = time.time() - 1.0
        backend._request_sequences["orphan"] = 5
        backend._request_sequence_timestamps["orphan"] = old_time

        # Start the backend (starts cleanup loop)
        await backend.start()

        # Wait for cleanup to run
        await asyncio.sleep(0.1)

        # Orphan should be cleaned up
        assert "orphan" not in backend._request_sequences

        await backend.stop()

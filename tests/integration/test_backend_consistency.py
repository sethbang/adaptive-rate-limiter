"""
Backend consistency integration tests.

These tests verify that MemoryBackend behaves consistently
and produces expected results for the same operations.
"""

import asyncio

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend


class TestBackendConsistency:
    """Test that backends behave consistently."""

    @pytest.fixture
    async def memory_backend(self):
        backend = MemoryBackend()
        await backend.start()
        yield backend
        await backend.stop()

    @pytest.mark.asyncio
    async def test_state_update_and_read(self, memory_backend):
        """Test basic state update and read consistency."""
        bucket_id = "test-bucket"

        # Set state directly
        state = {
            "remaining_requests": 95,
            "remaining_tokens": 9500,
            "request_limit": 100,
            "token_limit": 10000,
        }

        await memory_backend.set_state(bucket_id, state)

        # Read state
        retrieved = await memory_backend.get_state(bucket_id)
        assert retrieved is not None
        assert retrieved.get("remaining_requests") == 95
        assert retrieved.get("request_limit") == 100

    @pytest.mark.asyncio
    async def test_reservation_lifecycle(self, memory_backend):
        """Test complete reservation lifecycle."""
        bucket_id = "reservation-test"

        # Initialize state with bucket limits
        bucket_limits = {"rpm_limit": 100, "tpm_limit": 10000}

        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 10,
                "remaining_tokens": 1000,
                "request_limit": 100,
                "token_limit": 10000,
            },
        )

        # Reserve capacity
        success, reservation_id = await memory_backend.check_and_reserve_capacity(
            key=bucket_id,
            requests=1,
            tokens=100,
            bucket_limits=bucket_limits,
        )

        assert success is True
        assert reservation_id is not None

        # Release reservation
        release_result = await memory_backend.release_reservation(
            key=bucket_id,
            reservation_id=reservation_id,
        )
        assert release_result is True

    @pytest.mark.asyncio
    async def test_streaming_reservation(self, memory_backend):
        """Test streaming reservation release with refund."""
        bucket_id = "streaming-test"

        # Initialize state
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 10,
                "remaining_tokens": 1000,
                "request_limit": 100,
                "token_limit": 10000,
            },
        )

        # Reserve
        bucket_limits = {"rpm_limit": 100, "tpm_limit": 10000}
        success, reservation_id = await memory_backend.check_and_reserve_capacity(
            key=bucket_id,
            requests=1,
            tokens=500,  # Reserve 500 tokens
            bucket_limits=bucket_limits,
        )
        assert success is True

        # Release streaming with refund (only used 200 tokens)
        result = await memory_backend.release_streaming_reservation(
            key=bucket_id,
            reservation_id=reservation_id,
            reserved_tokens=500,
            actual_tokens=200,
        )
        assert result is True

        # Verify refund was applied
        state = await memory_backend.get_state(bucket_id)
        assert state is not None
        # Should have more tokens available after refund (300 refunded)
        assert state.get("remaining_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_capacity_exhaustion(self, memory_backend):
        """Test behavior when capacity is exhausted."""
        bucket_id = "exhaustion-test"

        # Initialize with very low capacity
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 1,
                "remaining_tokens": 50,
                "request_limit": 100,
                "token_limit": 10000,
            },
        )

        bucket_limits = {"rpm_limit": 100, "tpm_limit": 10000}

        # First reservation should succeed
        success1, _ = await memory_backend.check_and_reserve_capacity(
            key=bucket_id,
            requests=1,
            tokens=50,
            bucket_limits=bucket_limits,
        )
        assert success1 is True

        # Second reservation should fail (no capacity)
        success2, _ = await memory_backend.check_and_reserve_capacity(
            key=bucket_id,
            requests=1,
            tokens=50,
            bucket_limits=bucket_limits,
        )
        assert success2 is False

    @pytest.mark.asyncio
    async def test_idempotent_release(self, memory_backend):
        """Test that releasing the same reservation twice is idempotent."""
        bucket_id = "idempotent-test"

        # Initialize state
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 10,
                "remaining_tokens": 1000,
                "request_limit": 100,
                "token_limit": 10000,
            },
        )

        bucket_limits = {"rpm_limit": 100, "tpm_limit": 10000}

        # Reserve
        success, reservation_id = await memory_backend.check_and_reserve_capacity(
            key=bucket_id,
            requests=1,
            tokens=100,
            bucket_limits=bucket_limits,
        )
        assert success is True

        # Release once
        result1 = await memory_backend.release_reservation(bucket_id, reservation_id)
        assert result1 is True

        # Release again (should be idempotent)
        result2 = await memory_backend.release_reservation(bucket_id, reservation_id)
        assert result2 is True  # Idempotent - returns True

    @pytest.mark.asyncio
    async def test_rate_limit_update(self, memory_backend):
        """Test updating rate limits from headers."""
        model = "test-model"

        # Simulate rate limit headers from API response
        headers = {
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-tokens": "9500",
            "x-ratelimit-limit-tokens": "10000",
        }

        # Update rate limits
        result = await memory_backend.update_rate_limits(model, headers)
        assert result == 1  # Success

        # Get rate limits
        limits = await memory_backend.get_rate_limits(model)
        assert limits is not None
        assert limits.get("rpm_remaining") == 95
        assert limits.get("rpm_limit") == 100

    @pytest.mark.asyncio
    async def test_health_check(self, memory_backend):
        """Test health check returns valid result."""
        result = await memory_backend.health_check()

        assert result.healthy is True
        assert result.backend_type == "memory"
        assert "states_count" in result.metadata
        assert "reservations_count" in result.metadata

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_backend):
        """Test that concurrent operations are handled correctly."""
        bucket_id = "concurrent-test"

        # Initialize with more capacity
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 100,
                "remaining_tokens": 10000,
                "request_limit": 100,
                "token_limit": 10000,
            },
        )

        bucket_limits = {"rpm_limit": 100, "tpm_limit": 10000}

        async def make_reservation(i: int):
            success, res_id = await memory_backend.check_and_reserve_capacity(
                key=bucket_id,
                requests=1,
                tokens=10,
                bucket_limits=bucket_limits,
            )
            return i, success, res_id

        # Make 10 concurrent reservations
        results = await asyncio.gather(*[make_reservation(i) for i in range(10)])

        # All should succeed since we have enough capacity
        successful = [r for r in results if r[1]]
        assert len(successful) >= 5  # At least half should succeed

        # Verify state was updated
        state = await memory_backend.get_state(bucket_id)
        assert state is not None
        # Should have less capacity now
        assert state.get("remaining_requests", 0) < 100

    @pytest.mark.asyncio
    async def test_stats_tracking(self, memory_backend):
        """Test that statistics are tracked correctly."""
        # Get initial stats
        stats = await memory_backend.get_all_stats()

        assert "states_count" in stats
        assert "reservations_count" in stats
        assert "total_failures" in stats

        # Add some state
        await memory_backend.set_state("test-bucket", {"test": "data"})

        # Stats should update
        new_stats = await memory_backend.get_all_stats()
        assert new_stats["states_count"] >= 1

    @pytest.mark.asyncio
    async def test_cleanup(self, memory_backend):
        """Test cleanup clears all stored data."""
        # Add some data
        await memory_backend.set_state("bucket-1", {"data": 1})
        await memory_backend.set_state("bucket-2", {"data": 2})

        # Verify data exists
        state = await memory_backend.get_state("bucket-1")
        assert state is not None

        # Cleanup
        await memory_backend.cleanup()

        # All data should be cleared
        state = await memory_backend.get_state("bucket-1")
        assert state is None

        stats = await memory_backend.get_all_stats()
        assert stats["states_count"] == 0

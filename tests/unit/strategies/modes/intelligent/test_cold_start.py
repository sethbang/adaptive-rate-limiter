"""
Unit tests for IntelligentModeStrategy cold start protection.

Tests cold start protection, probe logic, and double-check pattern.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Cold Start Protection Tests
# ============================================================================


class TestIntelligentModeStrategyColdStart:
    """Tests for cold start protection."""

    @pytest.mark.asyncio
    async def test_ensure_bucket_initialized_first_call(self, strategy):
        """Test first call initializes bucket."""
        await strategy._ensure_bucket_initialized("bucket-1")

        assert "bucket-1" in strategy._initialized_buckets

    @pytest.mark.asyncio
    async def test_ensure_bucket_initialized_fast_path(self, strategy):
        """Test subsequent calls use fast path."""
        strategy._initialized_buckets.add("bucket-1")

        # Should not create lock
        await strategy._ensure_bucket_initialized("bucket-1")

        assert "bucket-1" not in strategy._bucket_init_locks


# ============================================================================
# Double-Check Pattern Tests
# ============================================================================


class TestIntelligentModeStrategyDoubleCheckPattern:
    """Tests for double-check pattern in bucket initialization."""

    @pytest.mark.asyncio
    async def test_ensure_bucket_initialized_concurrent_double_check(self, strategy):
        """Test double-check pattern inside lock."""
        # Simulate race: bucket already initialized when we check after acquiring lock
        lock = asyncio.Lock()
        strategy._bucket_init_locks["bucket-race"] = lock

        # Pre-mark as initialized to trigger double-check return
        strategy._initialized_buckets.add("bucket-race")

        # Now call ensure - should hit the double-check return inside lock
        await strategy._ensure_bucket_initialized("bucket-race")
        assert "bucket-race" in strategy._initialized_buckets

    @pytest.mark.asyncio
    async def test_ensure_bucket_initialized_race_inside_lock(self, strategy):
        """Test line 359: Double-check return when bucket initialized while waiting for lock."""
        # Don't pre-add to initialized_buckets - let concurrent task do it

        completed_order = []

        async def slow_first_initializer():
            """First caller - holds lock and initializes."""
            await strategy._ensure_bucket_initialized("bucket-concurrent")
            completed_order.append("first")

        async def waiting_second_initializer():
            """Second caller - waits for lock, then hits double-check return."""
            await asyncio.sleep(0.005)  # Ensure first caller gets lock first
            await strategy._ensure_bucket_initialized("bucket-concurrent")
            completed_order.append("second")

        # Run both concurrently
        await asyncio.gather(slow_first_initializer(), waiting_second_initializer())

        # Both should complete and bucket should be initialized
        assert "bucket-concurrent" in strategy._initialized_buckets
        assert len(completed_order) == 2

    @pytest.mark.asyncio
    async def test_ensure_bucket_creates_lock_if_not_exists(self, strategy):
        """Test lock creation when bucket not yet in _bucket_init_locks."""
        # Ensure bucket is NOT in initialized_buckets or init_locks
        assert "new-bucket" not in strategy._initialized_buckets
        assert "new-bucket" not in strategy._bucket_init_locks

        await strategy._ensure_bucket_initialized("new-bucket")

        # Lock should have been created
        assert "new-bucket" in strategy._bucket_init_locks
        assert "new-bucket" in strategy._initialized_buckets


# ============================================================================
# Cold Start Probe Logic Tests
# ============================================================================


class TestIntelligentModeStrategyColdStartProbe:
    """Tests for cold-start probe logic in _try_process_next_request_intelligent."""

    @pytest.mark.asyncio
    async def test_try_process_waits_for_active_probe(
        self, strategy, mock_state_manager
    ):
        """Test request waits when probe is active."""
        state = Mock()
        state.is_verified = False
        mock_state_manager.get_state.return_value = state

        strategy._bucket_probes.add("bucket-1")  # Probe already active

        metadata = RequestMetadata(
            request_id="req-wait",
            model_id="test-model",
            resource_type="chat",
        )
        queue = deque(
            [
                QueuedRequest(
                    metadata=metadata,
                    request_func=AsyncMock(return_value="success"),
                    future=asyncio.Future(),
                    queue_entry_time=datetime.now(timezone.utc),
                )
            ]
        )

        # Use queue key format that matches our naming convention: "{bucket_id}:{resource_type}"
        result = await strategy._try_process_next_request_intelligent(
            queue, "bucket-1:chat"
        )

        assert result is False
        assert len(queue) == 1  # Still in queue

    @pytest.mark.asyncio
    async def test_try_process_starts_probe(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Test request starts probe for unverified bucket."""
        state = Mock()
        state.is_verified = False
        mock_state_manager.get_state.return_value = state
        mock_backend.check_and_reserve_capacity.return_value = (True, "res-123")

        metadata = RequestMetadata(
            request_id="req-probe",
            model_id="test-model",
            resource_type="chat",
        )
        queue = deque(
            [
                QueuedRequest(
                    metadata=metadata,
                    request_func=AsyncMock(return_value="success"),
                    future=asyncio.Future(),
                    queue_entry_time=datetime.now(timezone.utc),
                )
            ]
        )
        # Use queue key format that matches our naming convention: "{bucket_id}:{resource_type}"
        queue_key = "bucket-1:chat"
        strategy.queue_info[queue_key] = Mock()
        strategy.queue_info[queue_key].update_on_dequeue = AsyncMock()

        _result = await strategy._try_process_next_request_intelligent(queue, queue_key)

        # Probe should have been started
        assert "bucket-1" in strategy._bucket_probes

        # Allow task to run and clean up
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_try_process_clears_probe_on_failure(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Test probe flag is cleared when capacity check fails."""
        state = Mock()
        state.is_verified = False
        state.remaining_requests = None
        state.reset_at = None
        mock_state_manager.get_state.return_value = state
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        metadata = RequestMetadata(
            request_id="req-probe-fail",
            model_id="test-model",
            resource_type="chat",
        )
        queue = deque(
            [
                QueuedRequest(
                    metadata=metadata,
                    request_func=AsyncMock(return_value="success"),
                    future=asyncio.Future(),
                    queue_entry_time=datetime.now(timezone.utc),
                )
            ]
        )

        # Use queue key format that matches our naming convention: "{bucket_id}:{resource_type}"
        result = await strategy._try_process_next_request_intelligent(
            queue, "bucket-1:chat"
        )

        assert result is False
        # Probe should be cleared on failure
        assert "bucket-1" not in strategy._bucket_probes

"""
Unit tests for IntelligentModeStrategy capacity reservation.

Tests capacity checking, reservation tracking, and retry logic.
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.exceptions import ReservationCapacityError
from adaptive_rate_limiter.reservation.context import ReservationContext
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Capacity Reservation Tests
# ============================================================================


class TestIntelligentModeStrategyCapacityReservation:
    """Tests for capacity reservation methods."""

    @pytest.mark.asyncio
    async def test_check_and_reserve_success(self, strategy, metadata, mock_backend):
        """Test successful capacity reservation."""
        mock_backend.check_and_reserve_capacity.return_value = (True, "res-123")

        success = await strategy._check_and_reserve_capacity_intelligent(metadata)

        assert success is True
        mock_backend.check_and_reserve_capacity.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_reserve_failure(self, strategy, metadata, mock_backend):
        """Test failed capacity reservation."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        success = await strategy._check_and_reserve_capacity_intelligent(
            metadata, schedule_watcher=False
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_check_and_reserve_stores_context(
        self, strategy, metadata, mock_backend
    ):
        """Test reservation context is stored on success."""
        mock_backend.check_and_reserve_capacity.return_value = (True, "res-123")

        await strategy._check_and_reserve_capacity_intelligent(metadata)

        # Verify context was stored via the tracker
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 1

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_provider(self, strategy, metadata):
        """Test reservation succeeds when no provider."""
        strategy.provider = None

        success = await strategy._check_and_reserve_capacity_intelligent(metadata)

        assert success is True

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_state_manager(self, strategy, metadata):
        """Test reservation succeeds when no state manager."""
        strategy.state_manager = None

        success = await strategy._check_and_reserve_capacity_intelligent(metadata)

        assert success is True


# ============================================================================
# Reservation Tracking Tests
# ============================================================================


class TestIntelligentModeStrategyReservationTracking:
    """Tests for reservation tracking methods."""

    @pytest.mark.asyncio
    async def test_store_reservation_context(self, strategy):
        """Test storing reservation context."""
        await strategy._store_reservation_context(
            request_id="req-1",
            bucket_id="bucket-1",
            reservation_id="res-1",
            estimated_tokens=100,
        )

        # Verify via the tracker's internal storage
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 1
            key = ("req-1", "bucket-1")
            assert key in strategy._reservation_tracker._reservation_contexts

    @pytest.mark.asyncio
    async def test_store_reservation_capacity_limit(self, strategy):
        """Test reservation capacity limit."""
        # Set a low limit for testing on the tracker
        strategy._reservation_tracker._max_reservations = 2

        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)
        await strategy._store_reservation_context("req-2", "bucket-1", "res-2", 100)

        # Third should fail
        with pytest.raises(ReservationCapacityError):
            await strategy._store_reservation_context("req-3", "bucket-1", "res-3", 100)

    @pytest.mark.asyncio
    async def test_get_and_clear_reservation(self, strategy):
        """Test get and clear reservation atomically."""
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)

        ctx = await strategy._get_and_clear_reservation("req-1", "bucket-1")

        assert ctx is not None
        assert ctx.reservation_id == "res-1"

        # Should be cleared - verify via tracker's internal storage
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 0

    @pytest.mark.asyncio
    async def test_get_and_clear_reservation_not_found(self, strategy):
        """Test get reservation that doesn't exist."""
        ctx = await strategy._get_and_clear_reservation("nonexistent", "bucket-1")

        assert ctx is None

    @pytest.mark.asyncio
    async def test_clear_all_reservations_for_request(self, strategy):
        """Test clearing all reservations for a request."""
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)
        await strategy._store_reservation_context("req-1", "bucket-2", "res-2", 100)

        cleared = await strategy._clear_all_reservations_for_request("req-1")

        assert len(cleared) == 2

        # Verify via tracker's internal storage
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 0

    @pytest.mark.asyncio
    async def test_get_reservation_without_clearing(self, strategy):
        """Test get reservation without clearing (for streaming)."""
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)

        ctx = await strategy._streaming_handler.get_reservation("req-1", "bucket-1")

        assert ctx is not None
        assert ctx.reservation_id == "res-1"

        # Should still exist - verify via tracker's internal storage
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 1


# ============================================================================
# Capacity Retry Tests
# ============================================================================


class TestIntelligentModeStrategyCapacityRetry:
    """Tests for retry logic after capacity check failure."""

    @pytest.mark.asyncio
    async def test_try_process_retries_after_state_refresh(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Test retry succeeds after state refresh shows capacity."""
        # First call fails, second succeeds
        call_count = 0

        async def reserve_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (False, None)
            return (True, "res-456")

        mock_backend.check_and_reserve_capacity.side_effect = reserve_mock

        state = Mock()
        state.remaining_requests = 5
        state.reset_at = None
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-retry",
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

        result = await strategy._try_process_next_request_intelligent(queue, queue_key)

        assert call_count == 2  # Retried
        assert result is True  # Succeeded on retry

        # Allow task to run
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_try_process_schedules_watcher_on_retry_failure(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Test watcher is scheduled when retry also fails."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        state = Mock()
        state.remaining_requests = 5
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-fail",
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
        assert "bucket-1" in strategy._reset_watcher._buckets_waiting

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()


# ============================================================================
# Stale Reservation Cleanup Tests
# ============================================================================


class TestIntelligentModeStrategyStaleReservationCleanup:
    """Tests for stale reservation cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_reservations(self, strategy, mock_backend):
        """Test stale reservation cleanup."""
        # Store a context with old timestamp via the tracker's internal storage
        strategy._reservation_tracker._reservation_contexts[("req-1", "bucket-1")] = (
            ReservationContext(
                reservation_id="res-1",
                bucket_id="bucket-1",
                estimated_tokens=100,
                created_at=time.time() - 1000,  # Very old
            )
        )
        strategy._reservation_tracker._request_id_index["req-1"] = {
            ("req-1", "bucket-1")
        }
        # Rebuild the time heap after directly manipulating internal state
        # (required for heap-based cleanup to work correctly)
        strategy._reservation_tracker._rebuild_time_heap()

        # Set short timeout for test
        strategy.MAX_RESERVATION_AGE = 1

        cleaned = await strategy._cleanup_stale_reservations()

        assert cleaned == 1
        mock_backend.release_reservation.assert_called_once()


# ============================================================================
# No Bucket / No Bucket Info Tests (lines 948, 954)
# ============================================================================


class TestIntelligentModeStrategyNoBucket:
    """Tests for capacity check when bucket or bucket info is unavailable."""

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_bucket_id(self, strategy, mock_provider):
        """Line 947-948: Return True when no bucket_id from provider."""
        mock_provider.get_bucket_for_model.return_value = None

        metadata = RequestMetadata(
            request_id="req-no-bucket",
            model_id="test-model",
            resource_type="chat",
        )
        result = await strategy._check_and_reserve_capacity_intelligent(metadata)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_bucket_info(self, strategy, mock_provider):
        """Line 953-954: Return True when bucket not in discovered limits."""
        mock_provider.get_bucket_for_model.return_value = "unknown-bucket"
        mock_provider.discover_limits.return_value = {}  # Empty - no bucket info

        metadata = RequestMetadata(
            request_id="req-no-info",
            model_id="test-model",
            resource_type="chat",
        )
        result = await strategy._check_and_reserve_capacity_intelligent(metadata)

        assert result is True


# ============================================================================
# Schedule Watcher on Capacity Failure Tests (lines 989-993)
# ============================================================================


class TestIntelligentModeStrategyScheduleWatcherOnFailure:
    """Tests for scheduling watcher when capacity reservation fails."""

    @pytest.mark.asyncio
    async def test_check_and_reserve_schedules_watcher_on_failure(
        self, strategy, mock_backend, mock_state_manager
    ):
        """Lines 987-995: Schedule watcher when reservation fails."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        state = Mock()
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-fail-watcher",
            model_id="test-model",
            resource_type="chat",
        )

        # schedule_watcher=True (default)
        result = await strategy._check_and_reserve_capacity_intelligent(
            metadata, schedule_watcher=True
        )

        assert result is False
        assert "bucket-1" in strategy._reset_watcher._buckets_waiting

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_watcher_when_already_waiting(
        self, strategy, mock_backend, mock_state_manager
    ):
        """Lines 992-993: Don't schedule duplicate watcher."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        state = Mock()
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        mock_state_manager.get_state.return_value = state

        # Pre-add bucket to waiting set
        strategy._reset_watcher._buckets_waiting.add("bucket-1")
        original_task_count = len(strategy._reset_watcher._reset_tasks)

        metadata = RequestMetadata(
            request_id="req-dup-watcher",
            model_id="test-model",
            resource_type="chat",
        )

        await strategy._check_and_reserve_capacity_intelligent(
            metadata, schedule_watcher=True
        )

        # Should not have added a new task
        assert len(strategy._reset_watcher._reset_tasks) == original_task_count

    @pytest.mark.asyncio
    async def test_check_and_reserve_no_watcher_when_no_reset_time(
        self, strategy, mock_backend, mock_state_manager
    ):
        """Lines 990-991: Don't schedule watcher when no reset time."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        state = Mock()
        state.reset_at = None  # No reset time
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-no-reset",
            model_id="test-model",
            resource_type="chat",
        )

        result = await strategy._check_and_reserve_capacity_intelligent(
            metadata, schedule_watcher=True
        )

        assert result is False
        # No watcher should be scheduled
        assert len(strategy._reset_watcher._reset_tasks) == 0


# ============================================================================
# Capacity Retry Failure Paths (lines 892-900)
# ============================================================================


class TestIntelligentModeStrategyCapacityRetryFailurePaths:
    """Tests for capacity retry failure paths in _try_process_next_request_intelligent."""

    @pytest.mark.asyncio
    async def test_try_process_schedules_watcher_when_state_shows_no_capacity(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Lines 890-896: Schedule watcher when state refresh still shows no capacity."""
        mock_backend.check_and_reserve_capacity.return_value = (False, None)

        state = Mock()
        state.remaining_requests = 0  # No capacity even after refresh
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-no-cap",
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
        assert "bucket-1" in strategy._reset_watcher._buckets_waiting

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_try_process_allows_request_no_bucket_id(
        self, strategy, mock_state_manager, mock_backend, mock_provider
    ):
        """Lines 947-948: Request proceeds when no bucket_id (no capacity check needed)."""
        # Use default queue key format which has no bucket_id
        metadata = RequestMetadata(
            request_id="req-no-bucket",
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
        # Use default_* format which extracts to None bucket_id
        queue_key = "default_chat"
        strategy.queue_info[queue_key] = Mock()
        strategy.queue_info[queue_key].update_on_dequeue = AsyncMock()

        result = await strategy._try_process_next_request_intelligent(queue, queue_key)

        # Should succeed - no capacity check when no bucket_id
        assert result is True

        # Allow task to run
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_try_process_allows_request_no_provider(
        self, strategy, mock_state_manager, mock_backend
    ):
        """Lines 940-941: Request proceeds when no provider (no capacity check needed)."""
        strategy.provider = None

        metadata = RequestMetadata(
            request_id="req-no-prov",
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
        # Use default_* format since provider is None
        queue_key = "default_chat"
        strategy.queue_info[queue_key] = Mock()
        strategy.queue_info[queue_key].update_on_dequeue = AsyncMock()

        result = await strategy._try_process_next_request_intelligent(queue, queue_key)

        # Should succeed - no capacity check when no provider
        assert result is True

        # Allow task to run
        await asyncio.sleep(0.05)

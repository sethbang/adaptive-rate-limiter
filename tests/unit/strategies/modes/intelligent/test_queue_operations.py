"""
Unit tests for IntelligentModeStrategy queue operations.

Tests submit_request, queue management, eligible queue finding, and queue processing.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.exceptions import RateLimiterError
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Submit Request Tests
# ============================================================================


class TestIntelligentModeStrategySubmitRequest:
    """Tests for submit_request method."""

    @pytest.mark.asyncio
    async def test_submit_request_returns_schedule_result(self, strategy, metadata):
        """Test submit_request returns ScheduleResult."""
        request_func = AsyncMock(return_value="success")

        result = await strategy.submit_request(metadata, request_func)

        assert result.request is not None
        assert result.wait_time == 0.0
        assert result.should_retry is True

    @pytest.mark.asyncio
    async def test_submit_request_creates_queue(self, strategy, metadata):
        """Test queue is created for new bucket."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        # Queue should be created
        assert len(strategy.fast_queues) == 1

    @pytest.mark.asyncio
    async def test_submit_request_adds_to_queue(self, strategy, metadata):
        """Test request is added to queue."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        # Find the queue that was created
        queue_key = next(iter(strategy.fast_queues.keys()))
        assert len(strategy.fast_queues[queue_key]) == 1

    @pytest.mark.asyncio
    async def test_submit_request_sets_queue_has_items(self, strategy, metadata):
        """Test _queue_has_items is set to True."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        queue_key = next(iter(strategy.fast_queues.keys()))
        assert strategy._queue_has_items[queue_key] is True

    @pytest.mark.asyncio
    async def test_submit_request_overflow_reject_policy(
        self, strategy, metadata, mock_config
    ):
        """Test queue overflow with reject policy."""
        mock_config.overflow_policy = "reject"
        mock_config.max_queue_size = 1

        request_func = AsyncMock(return_value="success")

        # First request should succeed
        await strategy.submit_request(metadata, request_func)

        # Second request should raise
        with pytest.raises(RateLimiterError):
            await strategy.submit_request(metadata, request_func)

    @pytest.mark.asyncio
    async def test_submit_request_overflow_drop_oldest_policy(
        self, strategy, metadata, mock_config
    ):
        """Test queue overflow with drop_oldest policy."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        request_func = AsyncMock(return_value="success")

        # First request
        result1 = await strategy.submit_request(metadata, request_func)

        # Second request should drop oldest
        meta2 = RequestMetadata(
            request_id="req-456",
            model_id="test-model",
            resource_type="chat",
            estimated_tokens=100,
        )
        _result2 = await strategy.submit_request(meta2, request_func)

        # First future should have exception
        with pytest.raises(RateLimiterError):
            result1.request.future.result()


# ============================================================================
# Queue Management Tests
# ============================================================================


class TestIntelligentModeStrategyQueueManagement:
    """Tests for queue management methods."""

    @pytest.mark.asyncio
    async def test_get_queue_key_with_provider(self, strategy, metadata, mock_provider):
        """Test queue key is based on bucket discovery."""
        mock_provider.get_bucket_for_model.return_value = "my-bucket"

        queue_key = await strategy._get_queue_key(metadata)

        assert queue_key == "my-bucket:chat"

    @pytest.mark.asyncio
    async def test_get_queue_key_no_provider(self, strategy, metadata):
        """Test queue key fallback when no provider."""
        strategy.provider = None

        queue_key = await strategy._get_queue_key(metadata)

        assert queue_key == "default_chat"

    @pytest.mark.asyncio
    async def test_get_queue_key_unknown_model(self, strategy, metadata):
        """Test queue key fallback for unknown model."""
        metadata.model_id = "unknown"

        queue_key = await strategy._get_queue_key(metadata)

        assert "default" in queue_key

    @pytest.mark.asyncio
    async def test_create_fast_queue(self, strategy, metadata):
        """Test _create_fast_queue creates queue structures."""
        await strategy._create_fast_queue("test-queue", metadata)

        assert "test-queue" in strategy.fast_queues
        assert "test-queue" in strategy.queue_info
        assert strategy._queue_has_items["test-queue"] is False

    @pytest.mark.asyncio
    async def test_check_queue_overflow_under_limit(self, strategy, mock_config):
        """Test no overflow when under limit."""
        mock_config.max_queue_size = 100

        result = await strategy._check_queue_overflow(50, "queue-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_queue_overflow_at_limit_reject(self, strategy, mock_config):
        """Test overflow at limit with reject policy."""
        mock_config.max_queue_size = 100
        mock_config.overflow_policy = "reject"

        with pytest.raises(RateLimiterError):
            await strategy._check_queue_overflow(100, "queue-1")


# ============================================================================
# Eligible Queue Finding Tests
# ============================================================================


class TestIntelligentModeStrategyEligibleQueues:
    """Tests for _find_eligible_queues_intelligent."""

    @pytest.mark.asyncio
    async def test_find_eligible_excludes_empty_queues(self, strategy, metadata):
        """Test empty queues are excluded."""
        # Create a queue but mark it as empty
        await strategy._create_fast_queue("empty-queue", metadata)
        strategy._queue_has_items["empty-queue"] = False

        eligible = await strategy._find_eligible_queues_intelligent()

        assert "empty-queue" not in eligible

    @pytest.mark.asyncio
    async def test_find_eligible_includes_non_empty_queues(self, strategy, metadata):
        """Test non-empty queues are included."""
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_find_eligible_excludes_circuit_broken(
        self, strategy, metadata, mock_scheduler
    ):
        """Test circuit broken queues are excluded."""
        mock_scheduler._circuit_breaker_always_closed = False
        mock_scheduler.circuit_breaker = Mock()
        mock_scheduler.circuit_breaker.can_execute = AsyncMock(return_value=False)

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 0


# ============================================================================
# Process Queue Tests
# ============================================================================


class TestIntelligentModeStrategyProcessQueue:
    """Tests for queue processing methods."""

    @pytest.mark.asyncio
    async def test_process_single_queue_empty(self, strategy, metadata):
        """Test processing empty queue."""
        await strategy._create_fast_queue("test-queue", metadata)

        result = await strategy._process_single_queue_intelligent("test-queue")

        assert result is False
        assert strategy._queue_has_items["test-queue"] is False

    @pytest.mark.asyncio
    async def test_process_selected_queues(self, strategy, metadata):
        """Test processing selected queues."""
        await strategy._create_fast_queue("test-queue", metadata)

        # Add a request to the queue
        request_func = AsyncMock(return_value="success")
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["test-queue"].append(queued_request)
        strategy._queue_has_items["test-queue"] = True

        await strategy._process_selected_queues_intelligent(["test-queue"])

        # Allow task to run
        await asyncio.sleep(0.05)


# ============================================================================
# Queue Info Warning Tests
# ============================================================================


class TestIntelligentModeStrategyQueueInfoWarning:
    """Tests for warning when queue_info is missing."""

    @pytest.mark.asyncio
    async def test_find_eligible_warns_missing_queue_info(self, strategy, caplog):
        """Test warning when queue_info is missing."""
        import logging

        strategy._queue_has_items["orphan-queue"] = True
        # Don't create queue_info - intentionally missing

        with caplog.at_level(logging.WARNING):
            await strategy._find_eligible_queues_intelligent()

        assert "has items but no queue_info" in caplog.text


# ============================================================================
# Drop Oldest with Queue Info Update Tests
# ============================================================================


class TestIntelligentModeStrategyDropOldestQueueInfo:
    """Tests for drop_oldest policy with queue_info update."""

    @pytest.mark.asyncio
    async def test_submit_drops_oldest_and_updates_queue_info(
        self, strategy, mock_config
    ):
        """Test drop_oldest updates queue_info on dequeue."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        metadata1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )
        metadata2 = RequestMetadata(
            request_id="req-2",
            model_id="test-model",
            resource_type="chat",
            priority=1,
        )

        request_func = AsyncMock(return_value="success")

        # First request
        _result1 = await strategy.submit_request(metadata1, request_func)

        # Get queue key
        queue_key = next(iter(strategy.queue_info.keys()))
        queue_info = strategy.queue_info[queue_key]

        # Mock update_on_dequeue to track calls
        queue_info.update_on_dequeue = AsyncMock()

        # Second request should drop first
        _result2 = await strategy.submit_request(metadata2, request_func)

        # Queue info should have been updated
        queue_info.update_on_dequeue.assert_called_once()


# ============================================================================
# Queue Selection Edge Cases (lines 653, 765)
# ============================================================================


class TestIntelligentModeStrategyQueueSelectionEdgeCases:
    """Tests for queue selection edge cases."""

    @pytest.mark.asyncio
    async def test_safe_set_wakeup_handles_exception(self, strategy, caplog):
        """Line 653-654: Logs warning on wakeup event failure."""
        import logging

        strategy._wakeup_lock = Mock()
        strategy._wakeup_lock.__aenter__ = AsyncMock(
            side_effect=RuntimeError("Lock error")
        )
        strategy._wakeup_lock.__aexit__ = AsyncMock()

        with caplog.at_level(logging.WARNING):
            await strategy._safe_set_wakeup_event()

        assert "Failed to set wakeup event" in caplog.text

    @pytest.mark.asyncio
    async def test_select_queues_returns_empty_when_strategy_returns_none(
        self, strategy
    ):
        """Line 764-765: Break loop when strategy returns None."""

        # Create eligible queue
        metadata = RequestMetadata(
            request_id="req-select",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("test-queue", metadata)
        strategy._queue_has_items["test-queue"] = True
        strategy.fast_queues["test-queue"].append(
            QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
        )

        # Mock scheduling strategy to return None
        strategy.scheduling_strategy.select = AsyncMock(return_value=None)

        eligible = await strategy._find_eligible_queues_intelligent()
        selected = await strategy._select_queues_for_processing(eligible)

        assert selected == []


# ============================================================================
# Process Queue Items Remaining Tests (lines 801, 808)
# ============================================================================


class TestIntelligentModeStrategyProcessQueueRemaining:
    """Tests for queue has_items tracking during processing."""

    @pytest.mark.asyncio
    async def test_process_queue_sets_has_items_false_when_empty(self, strategy):
        """Line 805-806: Queue has_items set False when empty."""
        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("test-queue", metadata)

        # Add one request
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["test-queue"].append(queued_request)
        strategy._queue_has_items["test-queue"] = True
        strategy.queue_info["test-queue"].update_on_dequeue = AsyncMock()

        # Process - capacity check should fail, not removing request
        with patch.object(
            strategy, "_check_and_reserve_capacity_intelligent", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = False

            result = await strategy._process_single_queue_intelligent("test-queue")

        # Line 801: break when capacity check fails
        assert result is False
        # Queue should still have items since processing failed
        assert strategy._queue_has_items["test-queue"] is True

    @pytest.mark.asyncio
    async def test_process_queue_leaves_has_items_true_with_remaining(
        self, strategy, mock_backend
    ):
        """Line 807-808: Queue has_items stays True when items remain."""
        metadata = RequestMetadata(
            request_id="req-remain",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("test-queue", metadata)
        strategy.queue_info["test-queue"].update_on_dequeue = AsyncMock()

        # Add 5 requests
        for i in range(5):
            queued_request = QueuedRequest(
                metadata=RequestMetadata(
                    request_id=f"req-{i}",
                    model_id="test-model",
                    resource_type="chat",
                ),
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
            strategy.fast_queues["test-queue"].append(queued_request)
        strategy._queue_has_items["test-queue"] = True

        # Process with capacity available - should process max_from_queue (3)
        with patch.object(
            strategy, "_check_and_reserve_capacity_intelligent", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True

            result = await strategy._process_single_queue_intelligent("test-queue")

        # Should have processed some
        assert result is True
        # Should still have items remaining
        assert len(strategy.fast_queues["test-queue"]) < 5
        assert strategy._queue_has_items["test-queue"] is True

        # Clean up any background tasks
        await asyncio.sleep(0.05)


# ============================================================================
# Imminent Reset Threshold Tests (lines 711-715, 719-723)
# ============================================================================


class TestIntelligentModeStrategyImminentReset:
    """Tests for imminent reset threshold handling in _find_eligible_queues_intelligent."""

    @pytest.mark.asyncio
    async def test_find_eligible_allows_imminent_reset(
        self, strategy, metadata, mock_state_manager, mock_provider
    ):
        """Lines 711-715: Allow processing when reset is imminent."""
        # Create queue with pending request
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        # Set up state with imminent reset (within 2x loop_sleep_time)
        state = Mock()
        state.remaining_requests = 0  # No capacity
        state.reset_at = datetime.now(timezone.utc) + timedelta(
            milliseconds=1  # Very imminent
        )
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        eligible = await strategy._find_eligible_queues_intelligent()

        # Queue should be eligible due to imminent reset
        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_find_eligible_schedules_watcher_for_non_imminent_reset(
        self, strategy, metadata, mock_state_manager, mock_provider
    ):
        """Lines 716-723: Schedule watcher when reset is NOT imminent."""
        # Create queue with pending request
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        # Set up state with reset far in future
        state = Mock()
        state.remaining_requests = 0  # No capacity
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        eligible = await strategy._find_eligible_queues_intelligent()

        # Queue should NOT be eligible
        assert len(eligible) == 0

        # Should have scheduled a watcher
        assert len(strategy._reset_watcher._buckets_waiting) == 1

        # Clean up watchers
        for task in list(strategy._reset_watcher._reset_tasks):
            task.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_find_eligible_skips_when_no_reset_time(
        self, strategy, metadata, mock_state_manager, mock_provider
    ):
        """Lines 724-726: Skip queue when no reset time available."""
        # Create queue with pending request
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        # Set up state with no reset time
        state = Mock()
        state.remaining_requests = 0  # No capacity
        state.reset_at = None  # No reset time
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        eligible = await strategy._find_eligible_queues_intelligent()

        # Queue should NOT be eligible
        assert len(eligible) == 0

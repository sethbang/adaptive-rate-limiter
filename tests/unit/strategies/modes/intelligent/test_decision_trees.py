"""
Unit tests for IntelligentModeStrategy decision trees.

Tests for specific decision branches in:
- Lines 416-430: Queue overflow drop_oldest policy handling
- Lines 703-750: Eligible queue finding with circuit breaker and rate limits
- Lines 874-920: Capacity check failure retry logic
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.exceptions import RateLimiterError
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Lines 416-430: Queue Overflow Drop Oldest Policy Tests
# ============================================================================


class TestQueueOverflowDropOldestDecisions:
    """Tests for drop_oldest overflow policy decision branches."""

    @pytest.mark.asyncio
    async def test_drop_oldest_with_queue_info_calls_update_on_dequeue(
        self, strategy, mock_config
    ):
        """Line 422-423: When queue_info exists, update_on_dequeue is called."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        meta1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )
        meta2 = RequestMetadata(
            request_id="req-2",
            model_id="test-model",
            resource_type="chat",
            priority=1,
        )

        request_func = AsyncMock(return_value="success")

        # First request - creates queue and queue_info
        await strategy.submit_request(meta1, request_func)

        # Get queue key and mock update_on_dequeue
        queue_key = next(iter(strategy.queue_info.keys()))
        strategy.queue_info[queue_key].update_on_dequeue = AsyncMock()

        # Second request triggers drop_oldest
        await strategy.submit_request(meta2, request_func)

        # queue_info.update_on_dequeue should have been called for dropped request
        strategy.queue_info[queue_key].update_on_dequeue.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_oldest_without_queue_info_skips_update(
        self, strategy, mock_config
    ):
        """Line 422: When queue_info is None, skip update_on_dequeue."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        meta1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )
        meta2 = RequestMetadata(
            request_id="req-2",
            model_id="test-model",
            resource_type="chat",
            priority=1,
        )

        request_func = AsyncMock(return_value="success")

        # First request
        await strategy.submit_request(meta1, request_func)

        # Remove queue_info to simulate missing info
        queue_key = next(iter(strategy.queue_info.keys()))
        del strategy.queue_info[queue_key]

        # Second request should still work (no exception)
        await strategy.submit_request(meta2, request_func)

        # verify the new request was added
        assert len(strategy.fast_queues[queue_key]) == 1

    @pytest.mark.asyncio
    async def test_drop_oldest_with_cancelled_future(self, strategy, mock_config):
        """Line 424: When future is cancelled, don't set_exception."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        meta1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )

        request_func = AsyncMock(return_value="success")

        # First request
        result1 = await strategy.submit_request(meta1, request_func)

        # Cancel the future BEFORE overflow
        result1.request.future.cancel()

        # Second request triggers drop_oldest
        meta2 = RequestMetadata(
            request_id="req-2",
            model_id="test-model",
            resource_type="chat",
            priority=1,
        )
        await strategy.submit_request(meta2, request_func)

        # Future should still be cancelled, not have an exception
        assert result1.request.future.cancelled()

    @pytest.mark.asyncio
    async def test_drop_oldest_sets_exception_on_non_cancelled_future(
        self, strategy, mock_config
    ):
        """Line 425-427: set_exception called when future is not cancelled."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 1

        meta1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )

        request_func = AsyncMock(return_value="success")

        # First request
        result1 = await strategy.submit_request(meta1, request_func)

        # Second request triggers drop_oldest
        meta2 = RequestMetadata(
            request_id="req-2",
            model_id="test-model",
            resource_type="chat",
            priority=1,
        )
        await strategy.submit_request(meta2, request_func)

        # First future should have exception
        with pytest.raises(RateLimiterError, match="queue overflow"):
            result1.request.future.result()


# ============================================================================
# Lines 703-750: Eligible Queue Finding Decision Trees
# ============================================================================


class TestEligibleQueueFindingDecisions:
    """Tests for decision branches in _find_eligible_queues_intelligent."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_excludes_queue_when_cannot_execute(
        self, strategy, metadata, mock_scheduler
    ):
        """Lines 703-706: Skip queue when circuit breaker rejects."""
        mock_scheduler._circuit_breaker_always_closed = False
        mock_scheduler.circuit_breaker = Mock()
        mock_scheduler.circuit_breaker.can_execute = AsyncMock(return_value=False)

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 0
        mock_scheduler.circuit_breaker.can_execute.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_queue_when_can_execute(
        self, strategy, metadata, mock_scheduler, mock_state_manager
    ):
        """Lines 703-706: Include queue when circuit breaker allows."""
        mock_scheduler._circuit_breaker_always_closed = False
        mock_scheduler.circuit_breaker = Mock()
        mock_scheduler.circuit_breaker.can_execute = AsyncMock(return_value=True)

        # No rate limit state - allows processing
        mock_state_manager.get_state.return_value = None

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_null_state_allows_queue_for_bootstrap(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 717-720: Null state allows processing to bootstrap."""
        mock_state_manager.get_state.return_value = None

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_state_with_remaining_requests_allows_queue(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 717-720: State with remaining_requests > 0 allows queue."""
        state = Mock()
        state.remaining_requests = 10
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_state_with_null_remaining_allows_queue(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 717-720: State with remaining_requests=None allows queue."""
        state = Mock()
        state.remaining_requests = None  # Never set
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_exhausted_capacity_with_imminent_reset_allows_queue(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 722-733: Zero remaining but imminent reset allows queue."""
        state = Mock()
        state.remaining_requests = 0
        # Reset within 2x loop_sleep_time (imminent)
        state.reset_at = datetime.now(timezone.utc) + timedelta(milliseconds=1)
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        # Should be eligible due to imminent reset
        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_exhausted_capacity_non_imminent_schedules_watcher(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 734-744: Non-imminent reset schedules watcher and skips queue."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        # Queue should NOT be eligible
        assert len(eligible) == 0

        # Watcher should have been scheduled
        assert len(strategy._reset_watcher._buckets_waiting) == 1

        # Clean up watchers
        for task in list(strategy._reset_watcher._reset_tasks):
            task.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_exhausted_capacity_avoids_duplicate_watchers(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 737-743: Don't schedule duplicate watcher for same bucket."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        # Pre-add the bucket to waiting set
        strategy._reset_watcher._buckets_waiting.add("bucket-1")

        with patch.object(
            strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
        ) as mock_schedule:
            await strategy._find_eligible_queues_intelligent()

            # Should NOT schedule another watcher since bucket already waiting
            mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_exhausted_capacity_no_reset_time_skips_queue(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 745-747: Skip queue when no reset time available."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = None  # No reset time
        mock_state_manager.get_state.return_value = state

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 0

    @pytest.mark.asyncio
    async def test_default_queue_bypasses_state_check(
        self, strategy, mock_state_manager
    ):
        """Lines 711-713: Default queues (no bucket_id) bypass state check."""
        # Create metadata that will use default queue key
        metadata = RequestMetadata(
            request_id="req-default",
            model_id="unknown",  # "unknown" triggers default queue in _get_queue_key
            resource_type="chat",
            requires_model=False,  # This triggers default queue
        )
        strategy.provider = None  # Forces default queue

        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()

        # Should be eligible even without state check
        assert len(eligible) == 1
        # State manager should not have been called for default queue
        mock_state_manager.get_state.assert_not_called()


# ============================================================================
# Lines 874-920: Capacity Check Failure Retry Decision Trees
# ============================================================================


class TestCapacityCheckFailureRetryDecisions:
    """Tests for decision branches in _try_process_next_request_intelligent failure path."""

    @pytest.fixture
    def setup_queue_with_request(self, strategy, metadata):
        """Helper to set up a queue with a pending request."""

        async def _setup():
            await strategy._create_fast_queue("bucket-1:chat", metadata)
            request = QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
            strategy.fast_queues["bucket-1:chat"].append(request)
            strategy._queue_has_items["bucket-1:chat"] = True
            strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()
            return request

        return _setup

    @pytest.mark.asyncio
    async def test_failure_path_force_refreshes_state(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Line 881-883: Force refresh state on failure."""
        await setup_queue_with_request()

        # First capacity check fails
        call_count = 0

        async def failing_reserve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return False

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            side_effect=failing_reserve,
        ):
            # State shows no capacity
            state = Mock()
            state.remaining_requests = 0
            state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
            mock_state_manager.get_state.return_value = state

            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        assert result is False
        # Verify force_refresh was called
        mock_state_manager.get_state.assert_called_with("bucket-1", force_refresh=True)

    @pytest.mark.asyncio
    async def test_failure_retries_when_state_shows_capacity(
        self,
        strategy,
        metadata,
        mock_state_manager,
        mock_backend,
        setup_queue_with_request,
    ):
        """Lines 886-900: Retry reservation when refreshed state shows capacity."""
        await setup_queue_with_request()

        # Track reservation calls
        reservation_calls = []

        async def tracking_reserve(*args, **kwargs):
            reservation_calls.append(kwargs.get("bucket_id", args[0] if args else None))
            # First call fails, second succeeds
            return len(reservation_calls) >= 2

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            side_effect=tracking_reserve,
        ):
            # State shows capacity after refresh
            state = Mock()
            state.remaining_requests = 5  # Has capacity
            state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
            mock_state_manager.get_state.return_value = state

            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        # Should have retried and succeeded
        assert result is True
        assert len(reservation_calls) == 2

        # Clean up background task
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_failure_retry_fails_schedules_watcher(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Lines 901-907: Schedule watcher when retry also fails."""
        await setup_queue_with_request()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State shows capacity but reservation still fails
            state = Mock()
            state.remaining_requests = 5
            state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
            mock_state_manager.get_state.return_value = state

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_no_capacity_schedules_watcher(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Lines 909-916: Schedule watcher when state still shows no capacity."""
        await setup_queue_with_request()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State shows no capacity after refresh
            state = Mock()
            state.remaining_requests = 0
            state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
            mock_state_manager.get_state.return_value = state

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_no_reset_time_returns_false(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Lines 909-916: Return False when no reset_at even after refresh."""
        await setup_queue_with_request()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State has no reset time
            state = Mock()
            state.remaining_requests = 0
            state.reset_at = None
            mock_state_manager.get_state.return_value = state

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        # No watcher scheduled since no reset time
        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_no_bucket_id_returns_false(
        self, strategy, mock_state_manager
    ):
        """Lines 917-918: Return False when no bucket_id (default queue)."""
        # Create default queue
        metadata = RequestMetadata(
            request_id="req-default",
            model_id="unknown",  # "unknown" triggers default queue in _get_queue_key
            resource_type="chat",
            requires_model=False,
        )
        strategy.provider = None

        await strategy._create_fast_queue("default_chat", metadata)
        request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["default_chat"].append(request)
        strategy._queue_has_items["default_chat"] = True

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["default_chat"], "default_chat"
            )

        # Should return False immediately without retry logic
        assert result is False

    @pytest.mark.asyncio
    async def test_failure_no_provider_returns_false(
        self, strategy, metadata, setup_queue_with_request
    ):
        """Lines 919-920: Return False when no provider."""
        await setup_queue_with_request()

        # Remove provider
        strategy.provider = None

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_failure_no_state_manager_returns_false(
        self, strategy, metadata, setup_queue_with_request
    ):
        """Lines 919-920: Return False when no state_manager."""
        await setup_queue_with_request()

        # Remove state_manager
        strategy.state_manager = None

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_failure_clears_probe_flag_on_reservation_failure(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Lines 867-868: Clear probe flag when reservation fails."""
        await setup_queue_with_request()

        # Mark bucket as being probed
        strategy._bucket_probes.add("bucket-1")

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State shows no capacity
            state = Mock()
            state.remaining_requests = 0
            state.reset_at = None
            mock_state_manager.get_state.return_value = state

            await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        # Probe flag should be cleared
        assert "bucket-1" not in strategy._bucket_probes


# ============================================================================
# Additional Branch Coverage: Cold Start Probing (Lines 850-861)
# ============================================================================


class TestColdStartProbingDecisions:
    """Tests for cold start probing decision branches."""

    @pytest.fixture
    def setup_queue_with_request(self, strategy, metadata):
        """Helper to set up a queue with a pending request."""

        async def _setup():
            await strategy._create_fast_queue("bucket-1:chat", metadata)
            request = QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
            strategy.fast_queues["bucket-1:chat"].append(request)
            strategy._queue_has_items["bucket-1:chat"] = True
            strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()
            return request

        return _setup

    @pytest.mark.asyncio
    async def test_cold_start_probe_initiated_for_unverified_state(
        self,
        strategy,
        metadata,
        mock_state_manager,
        mock_backend,
        setup_queue_with_request,
        caplog,
    ):
        """Lines 853-860: Start probe for unverified bucket state."""
        import logging

        await setup_queue_with_request()

        # State is unverified
        state = Mock()
        state.remaining_requests = 10
        state.is_verified = False
        mock_state_manager.get_state.return_value = state

        with (
            caplog.at_level(logging.INFO),
            patch.object(
                strategy,
                "_check_and_reserve_capacity_intelligent",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        # Should have logged probe start
        assert "cold start probe" in caplog.text.lower()

        # Clean up
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_cold_start_probe_waits_when_already_active(
        self, strategy, metadata, mock_state_manager, setup_queue_with_request
    ):
        """Lines 854-856: Wait when probe already active for bucket."""
        await setup_queue_with_request()

        # Mark bucket as already being probed
        strategy._bucket_probes.add("bucket-1")

        # State is unverified
        state = Mock()
        state.remaining_requests = 10
        state.is_verified = False
        mock_state_manager.get_state.return_value = state

        # Should return False immediately without trying to reserve
        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
        ) as mock_reserve:
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        assert result is False
        # Capacity check should not be called since we're waiting
        mock_reserve.assert_not_called()

    @pytest.mark.asyncio
    async def test_verified_state_bypasses_probe_check(
        self,
        strategy,
        metadata,
        mock_state_manager,
        mock_backend,
        setup_queue_with_request,
    ):
        """Lines 853: Skip probe for verified state."""
        await setup_queue_with_request()

        # State is verified
        state = Mock()
        state.remaining_requests = 10
        state.is_verified = True
        mock_state_manager.get_state.return_value = state

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        assert result is True
        # Bucket should not be in probes
        assert "bucket-1" not in strategy._bucket_probes

        # Clean up
        await asyncio.sleep(0.05)


# ============================================================================
# Additional Edge Case Coverage
# ============================================================================


class TestAdditionalEdgeCases:
    """Tests for additional edge case coverage in decision trees."""

    @pytest.mark.asyncio
    async def test_overflow_with_empty_queue_does_not_drop(
        self, strategy, mock_config, mock_provider
    ):
        """Line 418: When overflow=True but queue is empty, skip drop logic."""
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_queue_size = 0  # Force overflow immediately

        metadata = RequestMetadata(
            request_id="req-edge",
            model_id="test-model",
            resource_type="chat",
        )

        request_func = AsyncMock(return_value="success")

        # Queue should be created empty then overflow detected
        # Since queue is empty, the drop_oldest block won't execute the pop
        # But we need to ensure the request is still added
        await strategy.submit_request(metadata, request_func)

        queue_key = next(iter(strategy.fast_queues.keys()))
        # The new request should still have been added
        assert len(strategy.fast_queues[queue_key]) == 1

    @pytest.mark.asyncio
    async def test_overflow_with_non_drop_oldest_policy_skips_block(
        self, strategy, mock_config
    ):
        """Line 416-418: Overflow True but policy != drop_oldest skips inner block."""
        # Custom policy that triggers overflow but isn't drop_oldest
        mock_config.overflow_policy = (
            "allow_overflow"  # Not "reject" and not "drop_oldest"
        )
        mock_config.max_queue_size = 1

        meta1 = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )

        request_func = AsyncMock(return_value="success")

        # First request
        await strategy.submit_request(meta1, request_func)

        # Manually trigger overflow condition by patching _check_queue_overflow
        # to return True (indicating overflow occurred)
        with patch.object(
            strategy, "_check_queue_overflow", new_callable=AsyncMock, return_value=True
        ):
            meta2 = RequestMetadata(
                request_id="req-2",
                model_id="test-model",
                resource_type="chat",
            )
            await strategy.submit_request(meta2, request_func)

        # Both requests should be in queue (no dropping occurred)
        queue_key = next(iter(strategy.fast_queues.keys()))
        assert len(strategy.fast_queues[queue_key]) == 2

    @pytest.mark.asyncio
    async def test_eligible_queue_with_missing_fast_queue_entry(
        self, strategy, metadata, mock_state_manager
    ):
        """Line 750-753: Queue info exists but fast_queue entry is missing."""
        # Submit request to create queue structures
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        queue_key = next(iter(strategy.fast_queues.keys()))

        # Remove the fast_queues entry but keep queue_has_items True
        del strategy.fast_queues[queue_key]
        strategy.fast_queues = {queue_key: None}  # Set to None instead of removing

        eligible = await strategy._find_eligible_queues_intelligent()

        # Queue should not be in eligible since fast_queues[queue_key] is falsy
        assert len(eligible) == 0

    @pytest.mark.asyncio
    async def test_retry_succeeds_and_processes_request(
        self, strategy, metadata, mock_state_manager, mock_backend
    ):
        """Lines 898-900: Retry succeeds (the 'pass' branch) and processes request."""
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["bucket-1:chat"].append(request)
        strategy._queue_has_items["bucket-1:chat"] = True
        strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()

        # State shows capacity after refresh
        state = Mock()
        state.remaining_requests = 5
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        mock_state_manager.get_state.return_value = state

        call_count = [0]

        async def reserve_with_retry(*args, **kwargs):
            call_count[0] += 1
            # First call fails, second succeeds
            return call_count[0] >= 2

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            side_effect=reserve_with_retry,
        ):
            result = await strategy._try_process_next_request_intelligent(
                strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
            )

        # Should have succeeded on retry
        assert result is True
        # Request should have been dequeued
        assert len(strategy.fast_queues["bucket-1:chat"]) == 0

        # Clean up background task
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_failure_with_null_current_state_returns_false(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 909-916: Return False when current_state is None after refresh."""
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["bucket-1:chat"].append(request)
        strategy._queue_has_items["bucket-1:chat"] = True
        strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # get_state returns None (no state exists)
            mock_state_manager.get_state.return_value = None

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        # No watcher scheduled since no state
        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_with_null_remaining_requests_returns_false(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 886-889: Return False when remaining_requests is None after refresh."""
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["bucket-1:chat"].append(request)
        strategy._queue_has_items["bucket-1:chat"] = True
        strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State exists but remaining_requests is None
            state = Mock()
            state.remaining_requests = None
            state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
            mock_state_manager.get_state.return_value = state

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        # Watcher should be scheduled since state exists with reset_at
        mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_fails_no_reset_at_does_not_schedule_watcher(
        self, strategy, metadata, mock_state_manager
    ):
        """Lines 903: No watcher when reset_at is None after retry fails."""
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        strategy.fast_queues["bucket-1:chat"].append(request)
        strategy._queue_has_items["bucket-1:chat"] = True
        strategy.queue_info["bucket-1:chat"].update_on_dequeue = AsyncMock()

        with patch.object(
            strategy,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # State shows capacity but no reset_at
            state = Mock()
            state.remaining_requests = 5  # Triggers retry path
            state.reset_at = None  # No reset time
            mock_state_manager.get_state.return_value = state

            with patch.object(
                strategy._reset_watcher, "schedule_watcher", new_callable=AsyncMock
            ) as mock_schedule:
                result = await strategy._try_process_next_request_intelligent(
                    strategy.fast_queues["bucket-1:chat"], "bucket-1:chat"
                )

        assert result is False
        # No watcher scheduled since reset_at is None
        mock_schedule.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_probe_check_and_add_is_atomic_for_same_bucket(self, strategy):
        """Concurrent probe attempts for the same bucket must not both start."""
        bucket_id = "bucket-x"

        results = await asyncio.gather(
            *(strategy._try_acquire_probe(bucket_id) for _ in range(10))
        )
        assert sum(results) == 1, "exactly one concurrent probe may start"

    @pytest.mark.asyncio
    async def test_execute_request_with_tracking_does_not_release_probe_when_reclaim_failed(
        self, strategy
    ):
        """
        Regression: if retry re-claim returns False (another coroutine owns the
        probe slot), _execute_request_with_tracking must NOT release that other
        coroutine's probe slot.

        Scenario:
          1. Request A acquires probe for bucket-1.
          2. First capacity check fails; A releases the probe.
          3. State refreshes and shows capacity; A retries — but between steps 2
             and 3, Request B has taken the probe slot (simulated by pre-loading
             it).  A's re-claim therefore returns False.
          4. A calls _execute_request_with_tracking with owns_probe=False.
          5. After A's cleanup finally block runs, bucket-1 must STILL be in
             _bucket_probes (B's slot is intact).
        """
        bucket_id = "bucket-1"

        # Simulate request B having taken the probe slot.
        strategy._bucket_probes.add(bucket_id)

        # owns_probe=False: request A does NOT own the slot.
        request = QueuedRequest(
            metadata=RequestMetadata(
                request_id="req-A",
                model_id="test-model",
                resource_type="chat",
            ),
            request_func=AsyncMock(return_value="ok"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        task_id = "bucket-1:chat:req-A"

        # Track task so finally block can clean it up properly.
        async with strategy._task_lock:
            strategy._active_request_count += 1

        await strategy._execute_request_with_tracking(
            request, task_id, bucket_id=bucket_id, owns_probe=False
        )

        # After A finishes, bucket-1 must still be held (B's slot is intact).
        assert bucket_id in strategy._bucket_probes, (
            "Request A must not release a probe slot it does not own"
        )

    @pytest.mark.asyncio
    async def test_execute_request_with_tracking_releases_probe_when_owned(
        self, strategy
    ):
        """
        Positive case: when owns_probe=True the finally block MUST release the
        probe slot so future requests are not permanently blocked.
        """
        bucket_id = "bucket-1"
        strategy._bucket_probes.add(bucket_id)

        request = QueuedRequest(
            metadata=RequestMetadata(
                request_id="req-owner",
                model_id="test-model",
                resource_type="chat",
            ),
            request_func=AsyncMock(return_value="ok"),
            future=asyncio.Future(),
            queue_entry_time=datetime.now(timezone.utc),
        )
        task_id = "bucket-1:chat:req-owner"

        async with strategy._task_lock:
            strategy._active_request_count += 1

        await strategy._execute_request_with_tracking(
            request, task_id, bucket_id=bucket_id, owns_probe=True
        )

        # The owning request must have released the slot.
        assert bucket_id not in strategy._bucket_probes, (
            "Request that owns the probe must release it in its finally block"
        )

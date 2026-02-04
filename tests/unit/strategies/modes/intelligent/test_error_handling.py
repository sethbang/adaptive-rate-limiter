"""
Unit tests for IntelligentModeStrategy error handling.

Tests release failures, stale cleanup exceptions, and metrics update error paths.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import ClassVar
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.reservation.context import ReservationContext
from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Release Failure Handling Tests
# ============================================================================


class TestIntelligentModeStrategyReleaseFailure:
    """Tests for exception handling in release paths."""

    @pytest.fixture
    def mock_backend_with_failure(self, mock_backend):
        """Backend that fails on release."""
        mock_backend.release_reservation = AsyncMock(
            side_effect=Exception("Release failed")
        )
        return mock_backend

    @pytest.fixture
    def strategy_with_failing_backend(
        self,
        mock_scheduler,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
        mock_backend_with_failure,
    ):
        """Strategy with a backend that fails on release."""
        state_manager = Mock()
        state_manager.backend = mock_backend_with_failure
        state_manager.get_state = AsyncMock(return_value=None)
        state_manager.update_state_from_headers = AsyncMock(return_value=1)
        state_manager.stop = AsyncMock()

        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
        }

        return IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=state_manager,
        )

    @pytest.mark.asyncio
    async def test_update_state_handles_release_failure(
        self, strategy_with_failing_backend, caplog
    ):
        """Test release failure is logged but doesn't crash."""
        import logging

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test",
            resource_type="chat",
        )
        await strategy_with_failing_backend._store_reservation_context(
            "req-1", "bucket-1", "res-1", 100
        )

        with caplog.at_level(logging.WARNING):
            # Should not raise
            await strategy_with_failing_backend._update_rate_limit_state(
                metadata,
                result=None,
                status_code=None,
                headers={},
                bucket_id_override="bucket-1",
            )

        assert "Release failed" in caplog.text or "failed" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_update_state_handles_lua_fallback_release_failure(
        self,
        mock_scheduler,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
        caplog,
    ):
        """Test fallback release failure after Lua returns 0."""
        import logging

        mock_backend = Mock()
        mock_backend.check_and_reserve_capacity = AsyncMock(
            return_value=(True, "res-123")
        )
        mock_backend.release_reservation = AsyncMock(
            side_effect=Exception("Fallback release failed")
        )

        mock_state_manager = Mock()
        mock_state_manager.backend = mock_backend
        mock_state_manager.get_state = AsyncMock(return_value=None)
        mock_state_manager.update_state_from_headers = AsyncMock(return_value=0)
        mock_state_manager.stop = AsyncMock()

        # Full headers so it goes through update_state_from_headers path
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-reset-tokens": "60",
        }

        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )

        metadata = RequestMetadata(
            request_id="req-2",
            model_id="test",
            resource_type="chat",
        )
        await strategy._store_reservation_context("req-2", "bucket-1", "res-2", 100)

        with caplog.at_level(logging.WARNING):
            await strategy._update_rate_limit_state(
                metadata,
                result=Mock(),
                status_code=200,
                bucket_id_override="bucket-1",
            )

        assert "fallback" in caplog.text.lower() or "failed" in caplog.text.lower()


# ============================================================================
# Stale Cleanup Exception Handling Tests
# ============================================================================


class TestIntelligentModeStrategyStaleCleanupExceptions:
    """Tests for exception handling in stale reservation cleanup."""

    @pytest.fixture
    def strategy_with_backend_error(
        self,
        mock_scheduler,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
    ):
        """Strategy with backend that errors on release."""
        mock_backend = Mock()
        mock_backend.check_and_reserve_capacity = AsyncMock(
            return_value=(True, "res-123")
        )
        mock_backend.release_reservation = AsyncMock(
            side_effect=Exception("Backend error")
        )
        mock_backend.release_streaming_reservation = AsyncMock(
            side_effect=Exception("Backend error")
        )

        mock_state_manager = Mock()
        mock_state_manager.backend = mock_backend
        mock_state_manager.get_state = AsyncMock(return_value=None)
        mock_state_manager.update_state_from_headers = AsyncMock(return_value=1)
        mock_state_manager.stop = AsyncMock()

        return IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )

    @pytest.mark.asyncio
    async def test_cleanup_stale_reservations_handles_release_failure(
        self, strategy_with_backend_error, caplog
    ):
        """Test stale cleanup continues despite release failures."""
        import logging

        strategy_with_backend_error._reservation_tracker._reservation_contexts[
            ("req-1", "bucket-1")
        ] = ReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            estimated_tokens=100,
            created_at=time.time() - 1000,
        )
        strategy_with_backend_error._reservation_tracker._request_id_index["req-1"] = {
            ("req-1", "bucket-1")
        }
        # Rebuild the time heap after directly manipulating internal state
        strategy_with_backend_error._reservation_tracker._rebuild_time_heap()
        strategy_with_backend_error.MAX_RESERVATION_AGE = 1

        with caplog.at_level(logging.WARNING):
            cleaned = await strategy_with_backend_error._cleanup_stale_reservations()

        assert cleaned == 1
        assert "Failed to release" in caplog.text

    @pytest.mark.asyncio
    async def test_cleanup_stale_streaming_handles_release_failure(
        self, strategy_with_backend_error, caplog
    ):
        """Test stale streaming cleanup continues despite release failures."""
        import logging
        import weakref

        from adaptive_rate_limiter.streaming.tracker import StreamingInFlightEntry

        class FakeWrapper:
            _released = False

        wrapper = FakeWrapper()
        entry = StreamingInFlightEntry(
            reservation_id="res-stale",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=time.time() - 1000,
            last_activity_at=time.time() - 1000,
            wrapper_ref=weakref.ref(wrapper),
        )

        strategy_with_backend_error._streaming_cleanup_manager._streaming_in_flight[
            "res-stale"
        ] = entry
        strategy_with_backend_error._streaming_cleanup_manager._activity_timeout = 0.001

        with caplog.at_level(logging.WARNING):
            cleaned = await strategy_with_backend_error._streaming_cleanup_manager._cleanup_stale()

        assert cleaned == 1
        assert "Failed to release" in caplog.text


# ============================================================================
# Metrics Update When Enabled Tests
# ============================================================================


class TestIntelligentModeStrategyMetricsUpdate:
    """Tests for metrics update paths when metrics_enabled is True."""

    @pytest.fixture
    def strategy_with_metrics_overflow(
        self,
        mock_client,
        mock_provider,
        mock_classifier,
        mock_state_manager,
    ):
        """Strategy configured for overflow testing with metrics."""
        mock_scheduler = Mock()
        mock_scheduler.circuit_breaker = None
        mock_scheduler._circuit_breaker_always_closed = True
        mock_scheduler.metrics_enabled = True
        mock_scheduler.metrics = {"queue_overflows": 0}
        mock_scheduler.extract_response_headers = Mock(return_value={})

        mock_config = Mock()
        mock_config.batch_size = 50
        mock_config.scheduler_interval = 0.001
        mock_config.rate_limit_buffer_ratio = 0.9
        mock_config.max_queue_size = 0  # Trigger overflow
        mock_config.overflow_policy = "drop_oldest"
        mock_config.max_concurrent_executions = 100
        mock_config.request_timeout = 30.0

        return IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        ), mock_scheduler

    @pytest.mark.asyncio
    async def test_queue_overflow_updates_metrics(self, strategy_with_metrics_overflow):
        """Test queue overflow increments metrics."""
        strategy, mock_scheduler = strategy_with_metrics_overflow

        # Create empty queue so drop_oldest won't crash
        strategy.fast_queues["test-queue"] = []

        _result = await strategy._check_queue_overflow(1, "test-queue")

        assert mock_scheduler.metrics["queue_overflows"] == 1


# ============================================================================
# Execute Request Cleanup Failure Tests (lines 1244-1269, 1290-1305)
# ============================================================================


class TestIntelligentModeStrategyExecuteCleanupFailures:
    """Tests for cleanup failure handling in _execute_request_with_tracking."""

    @pytest.fixture
    def strategy_with_cleanup_failure(
        self,
        mock_scheduler,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
    ):
        """Strategy where _update_rate_limit_state raises."""
        mock_backend = Mock()
        mock_backend.check_and_reserve_capacity = AsyncMock(
            return_value=(True, "res-123")
        )
        mock_backend.release_reservation = AsyncMock(return_value=True)

        mock_state_manager = Mock()
        mock_state_manager.backend = mock_backend
        mock_state_manager.get_state = AsyncMock(return_value=None)
        mock_state_manager.update_state_from_headers = AsyncMock(return_value=1)
        mock_state_manager.stop = AsyncMock()

        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )
        strategy._request_timeout = 0.01  # Short timeout for testing
        return strategy

    @pytest.mark.asyncio
    async def test_execute_timeout_cleanup_failure_logged(
        self, strategy_with_cleanup_failure, caplog
    ):
        """Lines 1244-1245: Cleanup failure during timeout is logged."""
        strategy = strategy_with_cleanup_failure

        async def failing_cleanup(*args, **kwargs):
            raise RuntimeError("Cleanup failed")

        metadata = RequestMetadata(
            request_id="req-timeout",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._store_reservation_context(
            "req-timeout", "bucket-1", "res-1", 100
        )

        async def slow_request():
            await asyncio.sleep(1.0)

        request_func = AsyncMock(side_effect=slow_request)
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        with (
            patch.object(
                strategy, "_update_rate_limit_state", side_effect=failing_cleanup
            ),
            caplog.at_level(logging.WARNING),
        ):
            await strategy._execute_request_with_tracking(
                queued_request, "task-1", "bucket-1"
            )

        assert "Cleanup failed during timeout" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_cancellation_cleanup_failure_logged(
        self, strategy_with_cleanup_failure, caplog
    ):
        """Lines 1267-1268: Cleanup failure during cancellation is logged."""
        strategy = strategy_with_cleanup_failure
        strategy._request_timeout = 30.0  # Long enough to cancel

        async def failing_cleanup(*args, **kwargs):
            raise RuntimeError("Cleanup failed")

        metadata = RequestMetadata(
            request_id="req-cancel",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._store_reservation_context(
            "req-cancel", "bucket-1", "res-1", 100
        )

        async def cancellable():
            await asyncio.sleep(10.0)

        request_func = AsyncMock(side_effect=cancellable)
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        with patch.object(
            strategy, "_update_rate_limit_state", side_effect=failing_cleanup
        ):
            task = asyncio.create_task(
                strategy._execute_request_with_tracking(
                    queued_request, "task-1", "bucket-1"
                )
            )
            await asyncio.sleep(0.01)
            task.cancel()

            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(asyncio.CancelledError),
            ):
                await task

        assert "Cleanup failed during cancellation" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_rate_limit_error_cleanup_failure(
        self, strategy_with_cleanup_failure, caplog
    ):
        """Lines 1290-1291: Cleanup failure during 429 handling."""
        strategy = strategy_with_cleanup_failure
        strategy._request_timeout = 30.0

        async def failing_cleanup(*args, **kwargs):
            raise RuntimeError("429 cleanup failed")

        class RateLimitError(Exception):
            status_code = 429
            cached_rate_limit_headers: ClassVar[dict] = {}

        metadata = RequestMetadata(
            request_id="req-429",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._store_reservation_context("req-429", "bucket-1", "res-1", 100)

        request_func = AsyncMock(side_effect=RateLimitError("rate limited"))
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        with (
            patch.object(
                strategy, "_update_rate_limit_state", side_effect=failing_cleanup
            ),
            caplog.at_level(logging.WARNING),
        ):
            await strategy._execute_request_with_tracking(
                queued_request, "task-1", "bucket-1"
            )

        assert "Cleanup failed during 429 handling" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_generic_error_cleanup_failure(
        self, strategy_with_cleanup_failure, caplog
    ):
        """Lines 1304-1305: Cleanup failure during generic error handling."""
        strategy = strategy_with_cleanup_failure
        strategy._request_timeout = 30.0

        async def failing_cleanup(*args, **kwargs):
            raise RuntimeError("Generic cleanup failed")

        metadata = RequestMetadata(
            request_id="req-err",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._store_reservation_context("req-err", "bucket-1", "res-1", 100)

        request_func = AsyncMock(side_effect=ValueError("some error"))
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Track active request
        async with strategy._task_lock:
            strategy._active_request_count += 1

        with (
            patch.object(
                strategy, "_update_rate_limit_state", side_effect=failing_cleanup
            ),
            caplog.at_level(logging.WARNING),
        ):
            await strategy._execute_request_with_tracking(
                queued_request, "task-1", "bucket-1"
            )

        assert "Cleanup failed during exception handling" in caplog.text
